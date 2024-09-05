# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras initializers for TF 1."""

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops


_v1_zeros_initializer = init_ops.Zeros
_v1_ones_initializer = init_ops.Ones
_v1_constant_initializer = init_ops.Constant
_v1_variance_scaling_initializer = init_ops.VarianceScaling
_v1_orthogonal_initializer = init_ops.Orthogonal
_v1_identity = init_ops.Identity
_v1_glorot_uniform_initializer = init_ops.GlorotUniform
_v1_glorot_normal_initializer = init_ops.GlorotNormal


class RandomNormal(init_ops.RandomNormal):

  def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=dtypes.float32):
    super(RandomNormal, self).__init__(
        mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class RandomUniform(init_ops.RandomUniform):

  def __init__(self, minval=-0.05, maxval=0.05, seed=None,
               dtype=dtypes.float32):
    super(RandomUniform, self).__init__(
        minval=minval, maxval=maxval, seed=seed, dtype=dtype)


class TruncatedNormal(init_ops.TruncatedNormal):

  def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=dtypes.float32):
    super(TruncatedNormal, self).__init__(
        mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class LecunNormal(init_ops.VarianceScaling):

  def __init__(self, seed=None):
    super(LecunNormal, self).__init__(
        scale=1., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


class LecunUniform(init_ops.VarianceScaling):

  def __init__(self, seed=None):
    super(LecunUniform, self).__init__(
        scale=1., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


class HeNormal(init_ops.VarianceScaling):

  def __init__(self, seed=None):
    super(HeNormal, self).__init__(
        scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


class HeUniform(init_ops.VarianceScaling):

  def __init__(self, seed=None):
    super(HeUniform, self).__init__(
        scale=2., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    return {'seed': self.seed}
