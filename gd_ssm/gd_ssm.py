from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .gd_ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

def local_self_attention(x,w_q):
    """
    Implement local self attention transformation to the input sequence
    """
    attn = x.T @ w_q @ x
    #attn = attn.at[-1,-1].set(0)
    return attn

def apply_ssm(Lambda_bar, C_tilde,w_q, D, input_sequence, conj_sym, bidirectional,attn_window,stride):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    num_windows = (len(input_sequence) - attn_window) // stride + 1     # Calculate the number of windows
    indices = np.arange(num_windows)[:, None] * stride + np.arange(attn_window) #triplet positions
    
    transformed_sequence = input_sequence[indices] #2D sequence
    lsa_sequence  = jax.vmap(local_self_attention,in_axes=(0,None))(transformed_sequence,w_q)    
    state_init = np.zeros((input_sequence.shape[1],input_sequence.shape[1])) # recurrent state matrix initialisation
    zs_ls = []
    for dim in range(input_sequence.shape[1]):
        state_vec = state_init[:,dim]
        lsa_vec = lsa_sequence[:,:,dim] 
        rec_param = Lambda_bar[:,dim]
        def f(carry, inp):
            carry = rec_param*carry+inp
            return carry,carry
        _, zs = jax.lax.scan(f, state_vec, lsa_vec)
        zs_ls.append(zs)
    zs = np.stack(zs_ls,axis=1)
    zs = jax.vmap(lambda u: C_tilde@u)(zs)
    os = jax.vmap(lambda u: u.T @ D)(transformed_sequence)
    return zs,os
    
class GD_SSM(nn.Module):
    Lambda_re_init: np.array
    #Lambda_im_init: np.array
    V: np.array
    Vinv: np.array

    H: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0
    gd_params: bool = False
    gd_lr: float = 1e-4
    use_embeddings:bool = False
    attn_window:int=3,
    stride:int=2,

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """
        if self.use_embeddings:
            self.encoder = nn.Dense(self.P)
        if self.gd_params:
            self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,)) #Dummy
            self.w_q = np.outer(np.array([1,0,0]), np.array([0,1,0]))
            self.Lambda_bar = np.ones((self.P,self.P))
            self.C_tilde = -(self.gd_lr/self.P)*np.eye(self.P)
            #self.D = -(self.gd_lr/self.P)*np.array([0,0,1])
            self.D = np.array([0,0,1])
        else:
            self.w_q = self.param('w_q', nn.initializers.normal(stddev=0.1), (3, 3)) # Local Self Attention parameter
            if self.conj_sym:
                # Need to account for case where we actually sample real B and C, and then multiply
                # by the half sized Vinv and possibly V
                local_P = 2*self.P
            else:
                local_P = self.P

            # Initialize diagonal state to state matrix Lambda (eigenvalues)
            self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
            #self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
            if self.clip_eigs:
                #self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im 
                #self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * jax.numpy.zeros_like(self.Lambda_im) # FIXME- Enforcing the complex term to zero
                self.Lambda = np.clip(self.Lambda_re, None, -1e-4)
            else:
                self.Lambda = self.Lambda_re
                #self.Lambda = self.Lambda_re + 1j * jax.numpy.zeros_like(self.Lambda_im) # FIXME- Enforcing the complex term to zero

            # Initialize input to state (B) matrix
            B_init = lecun_normal()
            B_shape = (local_P, self.H)
            self.B = self.param("B",
                                lambda rng, shape: init_VinvB(B_init,
                                                            rng,
                                                            shape,
                                                            self.Vinv),
                                B_shape)
            #B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
            #B_tilde = self.B[..., 0] + 1j * jax.numpy.zeros_like(self.B[..., 1]) #FIXME - enforcing the complex term to zero
            B_tilde = self.B

            # Initialize state to output (C) matrix
            if self.C_init in ["trunc_standard_normal"]:
                C_init = trunc_standard_normal
                C_shape = (self.H, local_P, 2)
            elif self.C_init in ["lecun_normal"]:
                C_init = lecun_normal()
                C_shape = (self.H, local_P, 2)
            elif self.C_init in ["complex_normal"]:
                C_init = normal(stddev=0.5 ** 0.5)
            else:
                raise NotImplementedError(
                    "C_init method {} not implemented".format(self.C_init))

            if self.C_init in ["complex_normal"]:
                if self.bidirectional:
                    C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                    #self.C_tilde = C[..., 0] + 1j * C[..., 1]
                    #self.C_tilde = C[..., 0] + 1j * jax.numpy.zeros_like(C[..., 1])
                    self.C_tilde = C

                else:
                    C = self.param("C", C_init, (self.H, self.P, 2))
                    #self.C_tilde = C[..., 0] + 1j * C[..., 1]
                    #self.C_tilde = C[..., 0] + 1j * jax.numpy.zeros_like(C[..., 1])
                    self.C_tilde = C

            else:
                if self.bidirectional:
                    self.C1 = self.param("C1",
                                        lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                        C_shape)
                    self.C2 = self.param("C2",
                                        lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                        C_shape)

                    C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                    C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                    self.C_tilde = np.concatenate((C1, C2), axis=-1)

                else:
                    self.C = self.param("C",
                                        lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                        C_shape)

                # self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]
                    #self.C_tilde = self.C[..., 0] + 1j * jax.numpy.zeros_like(self.C[..., 1])  #FIXME - enforcing the complex term to zero
                    self.C_tilde = self.C

            # Initialize feedthrough (D) matrix
            #self.D = self.param("D", normal(stddev=1.0), (self.H,))
            self.D = self.param("D", normal(stddev=1.0), (3,))
            lambda_bar_all = []
            log_step_all = []
            # Initialize learnable discretization timescale value
            self.log_step = self.param("log_step",
                                        init_log_steps,
                                        (self.P, self.dt_min, self.dt_max)) #TODO - seperate logstep parameter for each recurrent params
            for param_id in range(self.Lambda_re.shape[0]):
                step = self.step_rescale * np.exp(self.log_step[:, 0])

                # Discretize
                if self.discretization in ["zoh"]:
                    self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda[param_id,:], B_tilde, step)
                elif self.discretization in ["bilinear"]:
                    self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda[param_id,:], B_tilde, step)
                elif self.discretization is None:
                    #self.Lambda_bar, self.B_bar = self.Lambda[param_id,:], B_tilde
                    self.Lambda_bar = np.ones(10)
                    self.C_tilde = self.C
                else:
                    raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))
                lambda_bar_all.append(self.Lambda_bar)
            self.Lambda_bar = np.vstack(lambda_bar_all)

    def __call__(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        if self.use_embeddings:
            input_sequence = jax.vmap(self.encoder)(input_sequence)
        zs,os = apply_ssm(self.Lambda_bar,
                       self.C_tilde,
                       self.w_q,
                       self.D,
                       input_sequence,
                       self.conj_sym,
                       self.bidirectional,
                       self.attn_window,
                       self.stride,
                       )
        # Add feedthrough matrix output Du;
        #Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return zs,os


def init_GD_SSM(H,
               P,
               Lambda_re_init,
#               Lambda_im_init,
               V,
               Vinv,
               C_init,
               discretization,
               dt_min,
               dt_max,
               conj_sym,
               clip_eigs,
               bidirectional,
               step_rescale,
               gd_params,
               gd_lr,
               use_embeddings,    
               attn_window,
               stride
               ):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in GD_SSM above."""
    return partial(GD_SSM,
                    H=H,
                    P=P,
                    Lambda_re_init=Lambda_re_init,
                    V=V,
                    Vinv=Vinv,
                    C_init=C_init,
                    discretization=discretization,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    conj_sym=conj_sym,
                    clip_eigs=clip_eigs,
                    bidirectional=bidirectional,
                    step_rescale=step_rescale,
                    gd_params=gd_params,
                    gd_lr=gd_lr,
                    use_embeddings=use_embeddings,
                    attn_window=attn_window,
                    stride=stride
                  )   
