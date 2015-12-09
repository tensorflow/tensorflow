"""TensorFlow ops for array / tensor manipulation."""

#  Copyright 2015 Google Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import tensorflow as tf


def split_squeeze(dim, num_split, tensor_in):
    """Splits input on given dimension and then squeezes that dimension.

    Args:
        dim: Dimension to split and squeeze on.
        num_split: integer, the number of ways to split.
        tensor_in: Input tensor of shape [N1, N2, .. Ndim, .. Nx].

    Returns:
        List of tensors [N1, N2, .. Ndim-1, Ndim+1, .. Nx].
    """
    return [tf.squeeze(t, squeeze_dims=[dim]) for t in tf.split(dim, num_split, tensor_in)]


def expand_concat(dim, inputs):
    """Expands inputs on given dimension and then concatenates them.

    Args:
        dim: Dimension to expand and concatenate on.
        inputs: List of tensors of the same shape [N1, ... Nx].

    Returns:
        A tensor of shape [N1, .. Ndim, ... Nx]
    """
    return tf.concat(dim, [tf.expand_dims(t, dim) for t in inputs])

