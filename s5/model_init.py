
from jax.scipy.linalg import block_diag
from s5.multi_ssm import init_multi_S5SSM
# from s5.ssm import init_S5SSM
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.seq_model import BatchS5Model

from s5.train_helpers import create_train_state
    # reduce_lr_on_plateau,\
    # linear_warmup, cosine_annealing, constant_lr, train_epoch, validate
from functools import partial
import jax.numpy as np

def model_init(args,init_rng,gd_params,gd_lr):
    
    retrieval = False
    padded = False
    ssm_lr = args.ssm_lr_base
    lr = args.lr_factor * ssm_lr
    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    #TODO - Currently skipping the V^-1 *B and CV matrices.
    if args.dataset in ["normal_token_scalar","normal_token_vector"]:
        seq_len = (args.dataset_size *2) + 1
        in_dim = args.input_size 
    else:
        seq_len = args.dataset_size
        # args.d_model = args.d_model
        in_dim = 2 * args.input_size 
    if gd_params:
        ssm_init_fn = init_S5SSM(H=args.d_model,
                                P=ssm_size,
                                Lambda_re_init=np.ones(ssm_size),
                                V=None,
                                Vinv=None,
                                C_init=None,
                                discretization=None,
                                dt_min=None,
                                dt_max=None,
                                conj_sym=None,
                                clip_eigs=None,
                                bidirectional=None,
                                step_rescale=None,
                                gd_params=gd_params,
                                gd_lr=gd_lr                             
                                )
    else:
        if args.dataset in ["normal_token_scalar","normal_token_vector"]:
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

            Lambda = np.vstack(lambda_all)
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
                                    bidirectional=args.bidirectional,
                                    step_rescale=1,
                                    gd_params=False,
                                    gd_lr=0.01                                
                                    )

        else:
            Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
            if args.conj_sym:
                block_size = block_size // 2
                ssm_size = ssm_size // 2
            Lambda = Lambda[:block_size]
            V = V[:, :block_size]
            Vc = V.conj().T
        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
            Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel().real # Each row is diagonal recurrent params
            V = block_diag(*([V] * args.blocks))            
            Vinv = block_diag(*([Vc] * args.blocks))
            ssm_init_fn = init_S5SSM(H=args.d_model,
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
                                    bidirectional=args.bidirectional,
                                    step_rescale=1,
                                    gd_params=False,
                                    gd_lr=0.01,
                          
                                    )

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
                                dt_global=args.dt_global,
                                # log_callback=None,
                                )
    return model_cls,state
