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
"""Keras initializer serialization / deserialization.
"""
# pylint: disable=unused-import
# pylint: disable=line-too-long
# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python import tf2
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.util.tf_export import keras_export

# These imports are brought in so that keras.initializers.deserialize
# has them available in module_objects.
from tensorflow.python.keras.initializers.initializers_v2 import Constant as ConstantV2
from tensorflow.python.keras.initializers.initializers_v2 import GlorotNormal as GlorotNormalV2
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform as GlorotUniformV2
from tensorflow.python.keras.initializers.initializers_v2 import HeNormal as HeNormalV2
from tensorflow.python.keras.initializers.initializers_v2 import HeUniform as HeUniformV2
from tensorflow.python.keras.initializers.initializers_v2 import Identity as IdentityV2
from tensorflow.python.keras.initializers.initializers_v2 import Initializer
from tensorflow.python.keras.initializers.initializers_v2 import LecunNormal as LecunNormalV2
from tensorflow.python.keras.initializers.initializers_v2 import LecunUniform  as LecunUniformV2
from tensorflow.python.keras.initializers.initializers_v2 import Ones as OnesV2
from tensorflow.python.keras.initializers.initializers_v2 import Orthogonal as OrthogonalV2
from tensorflow.python.keras.initializers.initializers_v2 import RandomNormal as RandomNormalV2
from tensorflow.python.keras.initializers.initializers_v2 import RandomUniform as RandomUniformV2
from tensorflow.python.keras.initializers.initializers_v2 import TruncatedNormal as TruncatedNormalV2
from tensorflow.python.keras.initializers.initializers_v2 import VarianceScaling as VarianceScalingV2
from tensorflow.python.keras.initializers.initializers_v2 import Zeros as ZerosV2

if tf2.enabled():
  Constant = ConstantV2
  GlorotNormal = GlorotNormalV2
  GlorotUniform = GlorotUniformV2
  HeNormal = HeNormalV2
  HeUniform = HeUniformV2
  Identity = IdentityV2
  LecunNormal = LecunNormalV2
  LecunUniform = LecunUniformV2
  Ones = OnesV2
  Orthogonal = OrthogonalV2
  RandomNormal = RandomNormalV2
  RandomUniform = RandomUniformV2
  TruncatedNormal = TruncatedNormalV2
  VarianceScaling = VarianceScalingV2
  Zeros = ZerosV2
else:
  from tensorflow.python.ops.init_ops import Constant
  from tensorflow.python.ops.init_ops import GlorotNormal
  from tensorflow.python.ops.init_ops import GlorotUniform
  from tensorflow.python.ops.init_ops import Identity
  from tensorflow.python.ops.init_ops import Ones
  from tensorflow.python.ops.init_ops import Orthogonal
  from tensorflow.python.ops.init_ops import VarianceScaling
  from tensorflow.python.ops.init_ops import Zeros
  from tensorflow.python.keras.initializers.initializers_v1 import HeNormal
  from tensorflow.python.keras.initializers.initializers_v1 import HeUniform
  from tensorflow.python.keras.initializers.initializers_v1 import LecunNormal
  from tensorflow.python.keras.initializers.initializers_v1 import LecunUniform
  from tensorflow.python.keras.initializers.initializers_v1 import RandomNormal
  from tensorflow.python.keras.initializers.initializers_v1 import RandomUniform
  from tensorflow.python.keras.initializers.initializers_v1 import TruncatedNormal


# Compatibility aliases
glorot_normal = GlorotNormal
glorot_uniform = GlorotUniform
he_normal = HeNormal
he_uniform = HeUniform
lecun_normal = LecunNormal
lecun_uniform = LecunUniform
zero = zeros = Zeros
one = ones = Ones
constant = Constant
uniform = random_uniform = RandomUniform
normal = random_normal = RandomNormal
truncated_normal = TruncatedNormal
identity = Identity
orthogonal = Orthogonal

# For unit tests
glorot_normalV2 = GlorotNormalV2
glorot_uniformV2 = GlorotUniformV2
he_normalV2 = HeNormalV2
he_uniformV2 = HeUniformV2
lecun_normalV2 = LecunNormalV2
lecun_uniformV2 = LecunUniformV2

# Utility functions


@keras_export('keras.initializers.serialize')
def serialize(initializer):
  return serialize_keras_object(initializer)


@keras_export('keras.initializers.deserialize')
def deserialize(config, custom_objects=None):
  """Return an `Initializer` object from its config."""
  module_objects = globals()
  return deserialize_keras_object(
      config,
      module_objects=module_objects,
      custom_objects=custom_objects,
      printable_module_name='initializer')


@keras_export('keras.initializers.get')
def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret initializer identifier: ' +
                     str(identifier))
