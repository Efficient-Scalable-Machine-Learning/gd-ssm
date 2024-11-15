from flax import linen as nn
import jax


class GDSSMlayer(nn.Module):
    """ Defines a single GD-SSM layer, with SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    dropout: float
    d_model: int
    activation: str = "gelu"
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    use_skip: bool = False

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm(step_rescale=self.step_rescale)

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)
            self.out3 = nn.Dense(1)

        if self.batchnorm:
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                    momentum=self.bn_momentum, axis_name='batch')
        else:
            self.norm = nn.LayerNorm()

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x,skip_zs1=None,skip_zs2=None):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
            x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        zs,os = self.seq(x)

        if skip_zs1 is not None:
            explicit_weight = skip_zs1 + jax.vmap(lambda u: skip_zs1[-1]@u,in_axes=(0))(skip_zs2) +zs
            x = jax.vmap(lambda u,v: u@v,in_axes=(0))(explicit_weight,os)
        else: 
            x = jax.vmap(lambda u,v: u@v,in_axes=(0))(zs,os)
        if self.activation in ["full_glu"]:
            x = self.drop(nn.gelu(x))
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
            
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x1))
            x = self.drop(x)
            x = self.out3(x)
        elif self.activation in ["gelu"]:
            x = self.drop(nn.gelu(x))
        elif self.activation in [None]:
            x = x
        else:
            raise NotImplementedError(
                "Activation: {} not implemented".format(self.activation))
        if self.use_skip:
            x = skip + x       
        return zs,os,x
