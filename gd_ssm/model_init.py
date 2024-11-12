from functools import partial
from jax import numpy as jnp
from jax.scipy.linalg import block_diag
from gd_ssm.gd_ssm import init_GD_SSM
from gd_ssm.ssm_init import make_DPLR_HiPPO
from gd_ssm.seq_model import BatchGDSSMModel
from gd_ssm.train_helpers import create_train_state


def model_init(args,init_rng,gd_params,gd_lr):
    ssm_lr = args.ssm_lr_base
    lr = args.lr_factor * ssm_lr
    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    if args.regression in ["non-linear"]:
        use_embddings= True
    else:
        use_embddings = False
    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    #TODO - Currently skipping the V^-1 *B and CV matrices.
    if args.dataset in ["normal_token_scalar","normal_token_vector"]:
        seq_len = (args.dataset_size *2) + 1
        in_dim = args.input_size 
        if args.regression == 'linear':
            args.d_model = args.input_size
    else:
        seq_len = args.dataset_size
        args.d_model = 2* args.d_model
        in_dim = 2 * args.input_size 
    if gd_params:
        ssm_init_fn = init_GD_SSM(H=args.d_model,
                                P=ssm_size,
                                Lambda_re_init=jnp.ones(ssm_size),
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
                                gd_lr=gd_lr,
                                use_embeddings=use_embddings,
                                attn_window=args.attn_window,
                                stride=args.stride                            
                                )
    else:
        if args.dataset in ["normal_token_scalar","normal_token_vector"]:
            if (args.input_size != ssm_size) and not use_embddings:
                raise ValueError("Input size and ssm size required to be same value")
            lambda_all = []
            for _ in range(ssm_size):
                Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
                if args.conj_sym:
                    block_size = block_size // 2
                    ssm_size = ssm_size // 2
                Lambda = Lambda[:block_size]
                V = V[:, :block_size]
                Vc = V.conj().T
            # If initializing state matrix A as block-diagonal, put HiPPO approximation
            # on each block
                Lambda = (Lambda * jnp.ones((args.blocks, block_size))).ravel() # Each row is diagonal recurrent params
                V = block_diag(*([V] * args.blocks))            
                Vinv = block_diag(*([Vc] * args.blocks))
                lambda_all.append(Lambda.real)

            Lambda = jnp.vstack(lambda_all)
            ssm_init_fn = init_GD_SSM(H=args.d_model,
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
                                    use_embeddings=use_embddings,
                                    attn_window=args.attn_window,
                                    stride=args.stride                               
                                    )

    model_cls = partial(
                BatchGDSSMModel,
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
                                in_dim=in_dim,
                                bsz=args.bsz,
                                seq_len=seq_len,
                                weight_decay=args.weight_decay,
                                batchnorm=args.batchnorm,
                                opt_config=args.opt_config,
                                ssm_lr=ssm_lr,
                                lr=lr,
                                dt_global=args.dt_global)
    return model_cls,state
