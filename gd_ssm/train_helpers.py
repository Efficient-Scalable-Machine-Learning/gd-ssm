from functools import partial
import jax
import jax.numpy as np
from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
import optax
from typing import Any, Tuple


# LR schedulers
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = np.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc


def constant_lr(step, base_lr, end_step, lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = np.array(lr_val,
                                                                                                dtype=np.float32)
    state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val,
                                                                                            dtype=np.float32)
    if opt_config in ["BandCdecay"]:
        # In this case we are applying the ssm learning rate to B, even though
        # we are also using weight decay on B
        state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val,
                                                                                                 dtype=np.float32)

    return state, step


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(model_cls,
                       rng,
                       in_dim=1,
                       bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="standard",
                       ssm_lr=1e-3,
                       lr=1e-3,
                       dt_global=False
                       ):
    """
    Initializes the training state using optax

    :param model_cls:
    :param rng:
    :param in_dim:
    :param bsz:
    :param seq_len:
    :param weight_decay:
    :param batchnorm:
    :param opt_config:
    :param ssm_lr:
    :param lr:
    :param dt_global:
    :return:
    """

    dummy_input = np.ones((bsz, seq_len, in_dim))
    integration_timesteps = np.ones((bsz, seq_len,))

    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init({"params": init_rng,
                            "dropout": dropout_rng},
                           dummy_input, integration_timesteps,
                           )
    if batchnorm:
        # params = variables["params"].unfreeze()
        params = variables["params"]
        batch_stats = variables["batch_stats"]
    else:
        # params = variables["params"].unfreeze()
        params = variables["params"]
        # Note: `unfreeze()` is for using Optax.

    if opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.
        """
        print("configuring standard optimization setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.
        """
        print("configuring optimization with B in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in ["B"] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in ["B"] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=ssm_lr,
                                                              weight_decay=weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply 
           faster global learning rate to B also.
        """
        print("configuring optimization with B in AdamW setup with lr")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included 
            with the SSM parameters and uses ssm learning rate.
         """
        print("configuring optimization with C not in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")

    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any

        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    else:
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# Train and eval steps
@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


# @partial(np.vectorize, signature="(c),()->()")
def mse_loss(preds, targets):
    # assert preds.shape == targets.shape
    return 0.5 * np.sum((targets - preds) ** 2) # / targets.shape[0]


@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label

# @partial(np.vectorize, signature="(c),()->()")
def compute_rmse(preds, targets):
    # assert preds.shape == targets.shape
    # return np.sqrt(0.5 * np.sum((targets - preds) ** 2) / targets.shape[0])
    return np.sqrt(0.5 * np.sum((targets - preds) ** 2))


def prep_batch(batch: tuple,
               seq_len: int,
               in_dim: int) -> Tuple[np.ndarray, np.ndarray, np.array]:
    """
    Take a batch and convert it to a standard x/y format.
    :param batch:       (x, y, aux_data) as returned from dataloader.
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Err... not sure what I should do... Unhandled data type. ")

    # Convert to JAX.
    inputs = np.asarray(inputs.numpy())

    # Grab lengths from aux if it is there.
    lengths = aux_data.get('lengths', None)

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        # Assuming vocab padding value is zero
        inputs = np.pad(inputs, ((0, 0), (0, num_pad)), 'constant', constant_values=(0,))

    # Inputs is either [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(np.asarray(inputs), in_dim)

    # If there are lengths, bundle them up.
    if lengths is not None:
        lengths = np.asarray(lengths.numpy())
        full_inputs = (inputs.astype(float), lengths.astype(float))
    else:
        full_inputs = inputs.astype(float)

    # Convert and apply.
    targets = np.array(targets.numpy())

    # If there is an aux channel containing the integration times, then add that.
    if 'timesteps' in aux_data.keys():
        integration_timesteps = np.diff(np.asarray(aux_data['timesteps'].numpy()))
    else:
        integration_timesteps = np.ones((len(inputs), seq_len))

    return full_inputs, targets.astype(float), integration_timesteps


def train_epoch(state, rng, model, trainloader, seq_len, in_dim, batchnorm,dataset,lr_params):
    """
    Training function for an epoch that loops over batches.
    """
    # Store Metrics
    model = model(training=True)
    batch_losses = []

    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    for batch_idx, batch in enumerate(trainloader):
        #inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        inputs, labels, _ = trainloader
        integration_times = np.ones((len(inputs), seq_len))
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(
            state,
#            params,  #TODO - Testing flexible SSM models
            drop_rng,
            inputs,
            labels,
            integration_times,
            model,
            batchnorm,
            dataset,
        )
        batch_losses.append(loss)
        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
        #state, step = update_learning_rate_per_step(lr_params, state)

    # Return average loss over batches
    return state, np.mean(np.array(batch_losses)), step


def validate(state, model, testloader, seq_len, in_dim, batchnorm,dataset, step_rescale=1.0):
    """Validation function that loops over batches"""
    model = model(training=False, step_rescale=step_rescale)
    losses, accuracies, preds = np.array([]), np.array([]), np.array([])
    for batch_idx, batch in enumerate(testloader):
        inputs, labels, _ = testloader
        integration_timesteps = np.ones((len(inputs), seq_len))
        loss, _ = eval_step(inputs, labels, integration_timesteps, state, model, batchnorm,dataset)
        losses = np.append(losses, loss)

    aveloss = np.mean(losses)    
    return aveloss

def get_prediction(state, model, inputs, seq_len, in_dim, batchnorm,dataset, step_rescale=1.0):
    """Validation function that loops over batches"""
    model = model(training=False, step_rescale=step_rescale)
    integration_timesteps = np.ones((len(inputs), seq_len))
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                            inputs, integration_timesteps,
                            )
    else:
        logits = model.apply({"params": state.params},
                            inputs, integration_timesteps,
                            )
    if dataset in ["normal_token_scalar"]:
        logits = logits[0,-1,-1] * -1
    elif dataset in ["normal_token_vector"]:
        logits = logits[0,-1] * -1
    else:
        logits = logits[0]*-1
        # _, logits = eval_step(inputs, labels, integration_timesteps, state, model, batchnorm,dataset)
    # model = model(training=False, step_rescale=step_rescale)
    return logits


@partial(jax.jit, static_argnums=(5,6,7))
def train_step(state,
            rng,
            batch_inputs,
            batch_labels,
            batch_integration_timesteps,
            model,
            batchnorm,
            dataset,
            ):
    """Performs a single training step given a batch of data"""
    def loss_fn(params):

        if batchnorm:
            logits, mod_vars = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                batch_inputs, batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            logits, mod_vars = model.apply(
                {"params": params},
                batch_inputs, batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )
        if dataset in ['normal_token_scalar']:
            loss = compute_loss(logits[:,-1,-1]*-1, batch_labels[:,-1])
        elif dataset in ['normal_token_vector']:
            loss = compute_loss(logits[:,-1]*-1, batch_labels)
            loss = loss/(logits.shape[2])
        else:
            loss = compute_loss(logits[:,-1]*-1, batch_labels[:,-1])

        return loss, (mod_vars, logits)

    (loss, (mod_vars, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jax.jit, static_argnums=(4,5,6))
def eval_step(batch_inputs,
            batch_labels,
            batch_integration_timesteps,
            state,
            model,
            batchnorm,
            dataset,
            ):
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                            batch_inputs, batch_integration_timesteps,
                            )
    else:
        logits = model.apply({"params": state.params},
                            batch_inputs, batch_integration_timesteps,
                            )
    if dataset in ['normal_token_scalar']:
        losses = compute_loss(logits[:,-1,-1]*-1, batch_labels[:,-1])
    elif dataset in ['normal_token_vector']:
        losses = compute_loss(logits[:,-1]*-1, batch_labels)
        losses = losses/(logits.shape[2])
    else:
        losses = compute_loss(logits[:,-1]*-1, batch_labels[:,-1])
    return losses, logits

def compute_loss(preds, targets):
    """
    root-mean square loss
    """
    assert preds.shape == targets.shape
    return 0.5*np.sum((targets-preds)**2)/(targets.shape[0])

