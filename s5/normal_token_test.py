import os
import wandb
from jax.scipy.linalg import block_diag
from s5.multi_ssm import init_multi_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.seq_model import BatchS5Model
from s5.model_init import model_init
from ml_collections import config_dict
from tqdm import tqdm
from jax import random,jacrev,vmap
import jax.numpy as jnp
from flax import linen as nn
from flax.training import checkpoints
from s5.train_helpers import create_train_state,reduce_lr_on_plateau,\
    linear_warmup, cosine_annealing, constant_lr, train_epoch,\
        validate,get_prediction
from s5.analysis import analyse,scan_lrs
from transformer.src.transformer import Transformer
from transformer.src.data import create_constructed_reg_data,create_reg_data_classic_token,\
    create_vec_reg_data_classic_token
from transformer.src.config import config 

def test(args):
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

    # checkpoint_dir = os.path.join(os.path.abspath(args.dir_name), 'checkpoints')
    restored_state = checkpoints.restore_checkpoint(ckpt_dir='/home/tianyusq/icl-matrix/loss/checkpoints/normal_30000', target=state)
    # restored_state = checkpoints.restore_checkpoint(ckpt_dir='/home/tianyusq/icl-matrix/loss/checkpoints/swapp_30000', target=state)

    in_dim = args.input_size # In-context regression data feature size
    ssm_lr = args.ssm_lr_base
    lr = args.lr_factor * ssm_lr
    ### Data
    if args.dataset in ["normal_token_scalar"]:
        seq_len = (args.dataset_size *2) + 1
        data_creator = vmap(create_reg_data_classic_token,
                            in_axes=(0, None, None, None, None, None),
                            out_axes=0)
    elif args.dataset in ["normal_token_vector"]:
        seq_len = (args.dataset_size *2) + 1
        data_creator = vmap(create_vec_reg_data_classic_token,
                            in_axes=(0, None, None, None, None, None),
                            out_axes=0)
    elif args.dataset in ["constructed_token"]:
        seq_len = args.dataset_size
        data_creator = vmap(create_constructed_reg_data,
                        in_axes=(0, None, None, None, None, None),
                        out_axes=0)
        
    else:
        raise NotImplementedError("dataset method {} not implemented".format(args.dataset))
        
    #Create eval data
    eval_data = data_creator(random.split(eval_rng, num=10000),
                                    in_dim, 
                                    args.dataset_size,
                                    config.size_distract,
                                    config.input_range,
                                    config.weight_scale,)
    
    # run one evaluation
    val_loss, logged_params = validate(restored_state,
                            model_cls,
                            eval_data,
                            seq_len,
                            in_dim,
                            args.batchnorm,
                            args.dataset)
    print(f"Validation loss on the restored model:{val_loss}")
    import numpy as np
    np.save('/home/tianyusq/icl-matrix/normal30.npy', logged_params)  # Save the matrix in .npy format
    print(f"Saved to {'/home/tianyusq/icl-matrix'}")




    