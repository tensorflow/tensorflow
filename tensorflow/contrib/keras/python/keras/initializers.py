# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Keras initializer classes (soon to be replaced with core TF initializers).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.contrib.keras.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops.init_ops import Constant
from tensorflow.python.ops.init_ops import Identity
from tensorflow.python.ops.init_ops import Initializer  # pylint: disable=unused-import
from tensorflow.python.ops.init_ops import Ones
from tensorflow.python.ops.init_ops import Orthogonal
from tensorflow.python.ops.init_ops import RandomNormal
from tensorflow.python.ops.init_ops import RandomUniform
from tensorflow.python.ops.init_ops import TruncatedNormal
from tensorflow.python.ops.init_ops import VarianceScaling
from tensorflow.python.ops.init_ops import Zeros


def lecun_normal(seed=None):
  """LeCun normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(1 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
      - [Efficient
      Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  """
  return VarianceScaling(
      scale=1., mode='fan_in', distribution='normal', seed=seed)


def lecun_uniform(seed=None):
  """LeCun uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(3 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      LeCun 98, Efficient Backprop,
      http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  """
  return VarianceScaling(
      scale=1., mode='fan_in', distribution='uniform', seed=seed)


def glorot_normal(seed=None):
  """Glorot normal initializer, also called Xavier normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(2 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      Glorot & Bengio, AISTATS 2010
      http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
  """
  return VarianceScaling(
      scale=1., mode='fan_avg', distribution='normal', seed=seed)


def glorot_uniform(seed=None):
  """Glorot uniform initializer, also called Xavier uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      Glorot & Bengio, AISTATS 2010
      http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
  """
  return VarianceScaling(
      scale=1., mode='fan_avg', distribution='uniform', seed=seed)


def he_normal(seed=None):
  """He normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(2 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      He et al., http://arxiv.org/abs/1502.01852
  """
  return VarianceScaling(
      scale=2., mode='fan_in', distribution='normal', seed=seed)


def he_uniform(seed=None):
  """He uniform variance scaling initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      He et al., http://arxiv.org/abs/1502.01852
  """
  return VarianceScaling(
      scale=2., mode='fan_in', distribution='uniform', seed=seed)


# Compatibility aliases

# pylint: disable=invalid-name
zero = zeros = Zeros
one = ones = Ones
constant = Constant
uniform = random_uniform = RandomUniform
normal = random_normal = RandomNormal
truncated_normal = TruncatedNormal
identity = Identity
orthogonal = Orthogonal

# pylint: enable=invalid-name

# Utility functions


def serialize(initializer):
  return serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='initializer')


def get(identifier):
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret initializer identifier:', identifier)
