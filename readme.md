# State-space models can learn in-context by gradient descent

This is the first version of our paper [State-space models can learn in-context by gradient descent](https://arxiv.org/abs/2410.11687). The model code is highly influenced by the [S5](https://github.com/lindermanlab/S5) repository, and the regression tasks influenced from [Transformers Learn In-Context by Gradient Descent](https://github.com/google-research/self-organising-systems/tree/master/transformers_learn_icl_by_gd).

# Installation

``` pip install -r requirements.txt ``` 

# Reproducing the results

1) For linear regression task:

    ``` python3 run_train.py --epochs=10000 --dataset=normal_token_vector --ssm_lr=0.0001 --analyse=True --n_layers=1 --lr_factor=2 --regression=linear --dataset_size=10 ```

2) For non-linear regression task:

    ``` python3 run_train.py --epochs=10000 --dataset=normal_token_vector --ssm_lr=0.0001 --analyse=True --n_layers=1 --lr_factor=2 --regression=non-linear --dataset_size=10 --activation_fn=half_glu2```

For the argument details, please have a look at ```run_train.py```.

For normal token setup, please use ```main``` branch and for the constructed token setup, please use ```constructed-token``` branch.

# Future works
We are working on a more efficient implementation of our GD-SSM model that can be used for more advanced sequence modelling task.