import os
import wandb
from jax.scipy.linalg import block_diag
from s5.multi_ssm import init_multi_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.seq_model import BatchS5Model
from s5.model_init import model_init
from ml_collections import config_dict
from tqdm import tqdm
from jax import random
from flax import linen as nn
from flax.training import checkpoints
from s5.train_helpers import create_train_state,reduce_lr_on_plateau,\
    linear_warmup, cosine_annealing, constant_lr, train_epoch, validate
from s5.analysis import scan_lrs,analyse
from transformer.src.transformer import Transformer
from transformer.src.data import create_reg_data_classic_token, create_vec_reg_data_classic_token
from transformer.src.config import config 
from transformer.src.train import *

def train(args):
    """
    Main function to train over a certain number of epochs
    """
    if args.USE_WANDB:
        # Make wandb config dictionary
        wandb.init(project=args.wandb_project, job_type='model_training', config=vars(args), entity=args.wandb_entity)
    # Set randomness...
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng , data_rng,eval_rng= random.split(key, num=4)
    model_cls,state = model_init(args,init_rng,gd_params=False,gd_lr=None)
    retrieval = False
    padded = False
    in_dim = 10 # Before embedding
    ssm_lr = args.ssm_lr_base
    lr = args.lr_factor * ssm_lr
    ### Data
    if args.dataset in ["normal_token_scalar"]:
        seq_len = args.dataset_size
        data_creator = vmap(create_reg_data_classic_token,
                            in_axes=(0, None, None, None, None, None),
                            out_axes=0)
    elif args.dataset in ["normal_token_vector"]:
        seq_len = (args.dataset_size *2) + 1
        data_creator = vmap(create_vec_reg_data_classic_token,
                            in_axes=(0, None, None, None, None, None),
                            out_axes=0)
    else:
        raise NotImplementedError("dataset method {} not implemented".format(args.dataset))
        
    #Create eval data
    eval_data = data_creator(jax.random.split(eval_rng, num=10000),
                                    10, 
                                    args.dataset_size,
                                    config.size_distract,
                                    config.input_range,
                                    config.weight_scale)
    if args.analyse:
        gd_lr,min_loss = scan_lrs(args,data_rng, lin_diag=False, bs=10000)
        gd_model_cls,gd_state = model_init(args,init_rng,gd_params=True,gd_lr=gd_lr)
        print(f"Validation loss on the gradient-descent based construction:{min_loss} for the learning rate {gd_lr}")
    # if args.analyse and args.dataset in ["normal_token_vector"]:
    #     print("Analysis for normal token vector is not implemented")
    #     args.analyse = False
    # Training Loop over epochs
    ls_trainloss, ls_valloss, loss_ssm_list, losses_gd_list, cos_sim_list, grad_norm_list, p_norm_list= [],[],[],[],[],[],[]

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
                                                args.dataset,
                                                lr_params)
        val_loss = validate(state,
                            model_cls,
                            eval_data,
                            seq_len,
                            in_dim,
                            args.batchnorm,
                            args.dataset)
        # val_loss_gd = validate(gd_state,
        #                         gd_model_cls,
        #                         eval_data,
        #                         seq_len,
        #                         in_dim,
        #                         args.batchnorm,
        #                         args.dataset)
        if args.analyse:
            cos_sim, w_norm, p_norm = analyse(args.dataset,args.dataset_size,args.batchnorm,eval_data,state,model_cls, gd_model_cls,gd_state)
            cos_sim_list.append(cos_sim)
            grad_norm_list.append(w_norm)
            p_norm_list.append(p_norm)
            losses_gd_list.append(min_loss)
            loss_ssm_list.append(train_loss)

        if args.USE_WANDB:
            if args.analyse:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss,"Model_cos":cos_sim,"Model_diff":w_norm,"Preds_diff":p_norm,"GD_loss":min_loss})
            else:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        ls_trainloss.append(train_loss)
        ls_valloss.append(val_loss)
        if (epoch+1) % 100 ==0:
            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} -- Cos similarity: {0:.5f}"
                )
 #save model checkpoint
    if not os.path.isdir(os.path.join(args.dir_name, 'checkpoints')):
        os.makedirs(os.path.join(args.dir_name, 'checkpoints'))
    checkpoints.save_checkpoint(ckpt_dir=os.path.join(os.path.abspath(args.dir_name),'checkpoints'), target=state, step=0,overwrite=True)
    

### Analysis
    if args.analyse:
        import numpy as np
        from s5.analysis import display_learning
        cosine_low = 0.0
        loss_ssm_list =[np.array(ls_valloss)]
        cos_sim_list = [np.array(cos_sim_list)]
        grad_norm_list = [np.array(grad_norm_list)]
        p_norm_list = [np.array(p_norm_list)]
        losses_gd_list = [np.array(losses_gd_list)]
        display_learning(loss_ssm_list, test=[losses_gd_list[0]], y_lim_u=0.4, y_lim_l=0.2,
                            rw=1, title="train.pdf", allow_download=True,
                            single_seeds = True, label_title ="Loss",
                            title2='GD', title1='Trained SSM', 
                            title3='GD',  loc_first='upper right',
                            num_iter_os=len(loss_ssm_list[0]))

        display_learning(cos_sim_list, grad_norm_list, p_norm_list, 
                            title1="Model cos",
                            title2="Model diff", y_lim_u=2,
                            title3="Preds diff", second_axis=True, color_add=0.2,
                            y_lim_u2=1.19, loc_sec='center right', single_seeds = False, 
                            y_lim_l2=cosine_low, color_axis=False, width= 5, y_label2 = 'Cosine sim',
                            rw=1, num_iter_os=len(cos_sim_list[0]), title="sim.pdf",
                            allow_download=True)
        



    