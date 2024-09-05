import os
import wandb
from jax.scipy.linalg import block_diag
from s5.multi_ssm import init_multi_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.seq_model import BatchS5Model
from ml_collections import config_dict
from tqdm import tqdm
from jax import random
from flax import linen as nn
from flax.training import checkpoints
from s5.train_helpers import create_train_state,reduce_lr_on_plateau,\
    linear_warmup, cosine_annealing, constant_lr, train_epoch, validate

from transformer.src.transformer import Transformer
from transformer.src.data import create_reg_data_classic_token, create_weights, create_vec_reg_data_classic_token
from transformer.src.config import config 
from transformer.src.train import *

def train(args):
    """
    Main function to train over a certain number of epochs
    """
    if args.USE_WANDB:
        # Make wandb config dictionary
        wandb.init(project=args.wandb_project, job_type='model_training', config=vars(args), entity=args.wandb_entity)
    retrieval = False
    padded = False
    seq_len = args.dataset_size
    in_dim = 10 # Before embedding
    padded = False
    ssm_lr = args.ssm_lr_base
    lr = args.lr_factor * ssm_lr
    ### Data
    if args.dataset in ["D2_scalar"]:
        data_creator = vmap(create_reg_data_classic_token,
                            in_axes=(0, None, None, None, None, None),
                            out_axes=0)
    elif args.dataset in ["D2_vector"]:
        data_creator = vmap(create_vec_reg_data_classic_token,
                            in_axes=(0, None, None, None, None, None),
                            out_axes=0)
    else:
        raise NotImplementedError("dataset method {} not implemented".format(args.dataset))
        
    ##Multi-SSM
    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    #TODO - Currently skipping the V^-1 *B and CV matrices.
    lambda_all = []
    for _ in range(in_dim):
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
        if args.conj_sym:
            block_size = block_size // 2
            ssm_size = ssm_size // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T
    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
        Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel() # Each row is diagonal recurrent params
        V = block_diag(*([V] * args.blocks))            
        Vinv = block_diag(*([Vc] * args.blocks))
        lambda_all.append(Lambda.real)

    Lambda = jnp.vstack(lambda_all)
    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))
    ssm_init_fn = init_multi_S5SSM(H=args.d_model,
                                P=ssm_size,
                                Lambda_re_init=Lambda,
                                V=V,
                                Vinv=Vinv,
                                C_init=args.C_init,
                                discretization=args.discretization,
                                dt_min=args.dt_min,
                                dt_max=args.dt_max,
                                conj_sym=args.conj_sym,
                                clip_eigs=args.clip_eigs,
                                bidirectional=args.bidirectional)

    model_cls = partial(
                BatchS5Model,
                ssm=ssm_init_fn,
                d_model=args.d_model,
                n_layers=args.n_layers,
                activation=args.activation_fn,
                dropout=args.p_dropout,
    #            mode=args.mode,
                prenorm=args.prenorm,
                batchnorm=args.batchnorm,
            )   
    # Set randomness...
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng , data_rng,eval_rng= random.split(key, num=4)
    #Create eval data
    eval_data = data_creator(jax.random.split(eval_rng, num=10000),
                                    10, 
                                    args.dataset_size,
                                    config.size_distract,
                                    config.input_range,
                                    config.weight_scale)

    state = create_train_state(model_cls,
                                init_rng,
                                padded,
                                retrieval,
                                in_dim=in_dim,
                                bsz=args.bsz,
                                seq_len=seq_len,
                                weight_decay=args.weight_decay,
                                batchnorm=args.batchnorm,
                                opt_config=args.opt_config,
                                ssm_lr=ssm_lr,
                                lr=lr,
                                dt_global=args.dt_global)

    # Training Loop over epochs
    ls_trainloss, ls_valloss = [],[]
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    train_size= args.bsz # FIXME - hardcode
    steps_per_epoch = int(train_size/args.bsz)
    for epoch in range(args.epochs):

        rng, data_rng = jax.random.split(data_rng, 2)
        trainloader = data_creator(jax.random.split(rng, num=args.bsz), 
                                    10,
                                    args.dataset_size,
                                    config.size_distract,
                                    config.input_range,
                                    config.weight_scale)
        #print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch+1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end

        elif args.cosine_anneal:
            if (epoch+1) % 100 ==0:
                print("using cosine annealing for epoch {}".format(epoch+1))
            decay_function = cosine_annealing
            #for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
        else:
            #print("using constant lr for epoch {}".format(epoch+1))
            decay_function = constant_lr
            end_step = None

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

        train_rng, skey = random.split(train_rng)
        state, train_loss, step = train_epoch(state,
                                                skey,
                                                model_cls,
                                                trainloader,
                                                seq_len,
                                                in_dim,
                                                args.batchnorm,
                                                lr_params)
        val_loss = validate(state,
                                    model_cls,
                                    eval_data,
                                    seq_len,
                                    in_dim,
                                    args.batchnorm)
        if args.USE_WANDB:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        ls_trainloss.append(train_loss)
        ls_valloss.append(val_loss)
        if (epoch+1) % 100 ==0:
            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f}"
                )
 #save model checkpoint
    if not os.path.isdir(os.path.join(args.dir_name, 'checkpoints')):
        os.makedirs(os.path.join(args.dir_name, 'checkpoints'))
    checkpoints.save_checkpoint(ckpt_dir=os.path.join(os.path.abspath(args.dir_name),'checkpoints'), target=state, step=0)
    