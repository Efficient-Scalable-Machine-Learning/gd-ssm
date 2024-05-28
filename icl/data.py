"""Data utils.
  Provide functions to create regression datasets.
"""

from functools import partial
from typing import Tuple, Any

import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
import torch.utils.data as data
import torch
# See https://github.com/google/jax/issues/1100
from dlpack import asdlpack

class LinRegData(data.Dataset):
    """From `create_reg_data`"""

    def __init__(self, seed, input_size, dataset_size, size_distract, input_range, w_scale):
        """Initializes the data generator.
        """

        self.rng = jax.random.PRNGKey(seed)
        self.input_size = input_size
        self.dataset_size = dataset_size
        self.size_distract = size_distract
        self.input_range = input_range
        self.w_scale = w_scale

    # @jax.jit
    def __getitem__(self, index) -> Tuple[Any, Any]:
        """
        """
        self.rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(self.rng, 5)
        w = jax.random.normal(self.rng, shape=[self.input_size]) * self.w_scale

        x = jax.random.uniform(new_rng, shape=[self.dataset_size, self.input_size],
                               minval=-self.input_range / 2, maxval=self.input_range / 2)
        x_querry = jax.random.uniform(new_rng2, shape=[1, self.input_size],
                                      minval=-self.input_range / 2, maxval=self.input_range / 2)

        y_data = jnp.squeeze(x @ w)
        choice = jax.random.choice(new_rng4, self.dataset_size, shape=[self.size_distract],
                                   replace=False)
        y_data = y_data.at[choice].set(jax.random.normal(new_rng3,
                                                         shape=[self.size_distract]))

        y_target = x_querry @ w
        y_target = y_target[..., None]

        seq = jnp.concatenate([x, y_data[..., None]], -1)
        target = jnp.concatenate([x_querry, y_target], -1)
        x_querry_init = -1 * x_querry.dot(jnp.ones_like(x_querry).T * 0.0)
        zero = jnp.concatenate([x_querry, x_querry_init], -1)
        seq = jnp.concatenate([seq, zero], 0)
        # return jnp.squeeze(seq), jnp.squeeze(target), w

        # episode = {}
        # episode['x'] = seq
        # episode['y'] = target
        # episode['w'] = w
        # print(x.shape, y_data.shape, x_querry.shape, y_target.shape, zero.shape, seq.shape, target.shape)
        return torch.from_numpy(np.asarray(seq)), torch.from_numpy(np.asarray(target))
        # return torch.from_dlpack(asdlpack(seq)), torch.from_dlpack(asdlpack(target))
        # return seq, target

    def __len__(self):  # denotes the total number of samples
        return 1000 * self.dataset_size