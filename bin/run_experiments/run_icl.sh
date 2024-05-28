python run_train.py --C_init=lecun_normal --batchnorm=True --bidirectional=False \
                    --blocks=8 --bsz=128 --d_model=128 --dataset=icl-linreg \
                    --epochs=5 --jax_seed=3000 --lr_factor=2 --n_layers=6 \
                    --ssm_lr_base=0.001 --ssm_size_base=128
