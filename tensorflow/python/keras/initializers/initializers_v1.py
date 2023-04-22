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
from tensorflow.python.util.tf_export import keras_export


_v1_zeros_initializer = init_ops.Zeros
_v1_ones_initializer = init_ops.Ones
_v1_constant_initializer = init_ops.Constant
_v1_variance_scaling_initializer = init_ops.VarianceScaling
_v1_orthogonal_initializer = init_ops.Orthogonal
_v1_identity = init_ops.Identity
_v1_glorot_uniform_initializer = init_ops.GlorotUniform
_v1_glorot_normal_initializer = init_ops.GlorotNormal

keras_export(v1=['keras.initializers.Zeros', 'keras.initializers.zeros'])(
    _v1_zeros_initializer)
keras_export(v1=['keras.initializers.Ones', 'keras.initializers.ones'])(
    _v1_ones_initializer)
keras_export(v1=['keras.initializers.Constant', 'keras.initializers.constant'])(
    _v1_constant_initializer)
keras_export(v1=['keras.initializers.VarianceScaling'])(
    _v1_variance_scaling_initializer)
keras_export(v1=['keras.initializers.Orthogonal',
                 'keras.initializers.orthogonal'])(_v1_orthogonal_initializer)
keras_export(v1=['keras.initializers.Identity',
                 'keras.initializers.identity'])(_v1_identity)
keras_export(v1=['keras.initializers.glorot_uniform'])(
    _v1_glorot_uniform_initializer)
keras_export(v1=['keras.initializers.glorot_normal'])(
    _v1_glorot_normal_initializer)


@keras_export(v1=['keras.initializers.RandomNormal',
                  'keras.initializers.random_normal',
                  'keras.initializers.normal'])
class RandomNormal(init_ops.RandomNormal):

  def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=dtypes.float32):
    super(RandomNormal, self).__init__(
        mean=mean, stddev=stddev, seed=seed, dtype=dtype)


@keras_export(v1=['keras.initializers.RandomUniform',
                  'keras.initializers.random_uniform',
                  'keras.initializers.uniform'])
class RandomUniform(init_ops.RandomUniform):

  def __init__(self, minval=-0.05, maxval=0.05, seed=None,
               dtype=dtypes.float32):
    super(RandomUniform, self).__init__(
        minval=minval, maxval=maxval, seed=seed, dtype=dtype)


@keras_export(v1=['keras.initializers.TruncatedNormal',
                  'keras.initializers.truncated_normal'])
class TruncatedNormal(init_ops.TruncatedNormal):

  def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=dtypes.float32):
    super(TruncatedNormal, self).__init__(
        mean=mean, stddev=stddev, seed=seed, dtype=dtype)


@keras_export(v1=['keras.initializers.lecun_normal'])
class LecunNormal(init_ops.VarianceScaling):

  def __init__(self, seed=None):
    super(LecunNormal, self).__init__(
        scale=1., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


@keras_export(v1=['keras.initializers.lecun_uniform'])
class LecunUniform(init_ops.VarianceScaling):

  def __init__(self, seed=None):
    super(LecunUniform, self).__init__(
        scale=1., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


@keras_export(v1=['keras.initializers.he_normal'])
class HeNormal(init_ops.VarianceScaling):

  def __init__(self, seed=None):
    super(HeNormal, self).__init__(
        scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


@keras_export(v1=['keras.initializers.he_uniform'])
class HeUniform(init_ops.VarianceScaling):

  def __init__(self, seed=None):
    super(HeUniform, self).__init__(
        scale=2., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    return {'seed': self.seed}
