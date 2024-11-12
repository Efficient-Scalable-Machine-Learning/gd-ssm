import jax
import jax.numpy as np
from flax import linen as nn
from .layers import GDSSMlayer

class GDSSMModel(nn.Module):
    """ Defines a stack of S5 layers to be used as an encoder.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    d_model: int
    n_layers: int
    activation: str = "gelu"
    dropout: float = 0.0
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        """
        Initializes a linear encoder and the stack of S5 layers.
        """
        self.layers = [
            GDSSMlayer(
                ssm=self.ssm,
                dropout=self.dropout,
                d_model=self.d_model,
                activation=self.activation,
                training=self.training,
                prenorm=self.prenorm,
                batchnorm=self.batchnorm,
                bn_momentum=self.bn_momentum,
                step_rescale=self.step_rescale,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x, integration_timesteps):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
            x (float32): input sequence (L, d_input)
        Returns:
            output sequence (float32): (L, d_model)
        """
        
        if len(self.layers) == 1:
            _,_,x = self.layers[0](x)
#            x = jax.vmap(lambda u,v: u@v,in_axes=(0))(zs,os)            
        else:
            skip_input = x
            for i,layer in enumerate(self.layers):
                #skip_input = x
#                zs,os = layer(skip_input)
                if (i+1)%3==1:
                    zs,os,_ = layer(x)
                    skip_zs1 = zs #weight term
                    #x = jax.vmap(lambda u,v: u@v,in_axes=(0))(zs,os) 
                if (i+1)%3 ==2:
                    zs,os,_ = layer(x)
                    skip_zs2 = zs #second recurrence
                    #x = jax.vmap(lambda u,v: u@v,in_axes=(0))(zs,os)
                if  (i+1)%3 ==0:
                    #zs,os = layer(skip_input)
                    _,_,x = layer(x,skip_zs1,skip_zs2)
                    padding_rows = len(skip_input) - len(x)
                    padding_array = np.zeros((padding_rows, x.shape[1]))
                    x = np.vstack([padding_array, x])
                    #if i < (len(self.layers)-1):
                    x = x+skip_input #TODO: scale skip connection
                    i+=1
        return x


# Here we call vmap to parallelize across a batch of input sequences
BatchGDSSMModel = nn.vmap(
    GDSSMModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')