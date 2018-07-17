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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object

# These imports are brought in so that keras.initializers.deserialize
# has them available in module_objects.
from tensorflow.python.ops.init_ops import Constant
from tensorflow.python.ops.init_ops import glorot_normal_initializer
from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from tensorflow.python.ops.init_ops import he_normal  # pylint: disable=unused-import
from tensorflow.python.ops.init_ops import he_uniform  # pylint: disable=unused-import
from tensorflow.python.ops.init_ops import Identity
from tensorflow.python.ops.init_ops import Initializer  # pylint: disable=unused-import
from tensorflow.python.ops.init_ops import lecun_normal  # pylint: disable=unused-import
from tensorflow.python.ops.init_ops import lecun_uniform  # pylint: disable=unused-import
from tensorflow.python.ops.init_ops import Ones
from tensorflow.python.ops.init_ops import Orthogonal
from tensorflow.python.ops.init_ops import RandomNormal
from tensorflow.python.ops.init_ops import RandomUniform
from tensorflow.python.ops.init_ops import TruncatedNormal
from tensorflow.python.ops.init_ops import VarianceScaling  # pylint: disable=unused-import
from tensorflow.python.ops.init_ops import Zeros

from tensorflow.python.util.tf_export import tf_export


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
glorot_normal = glorot_normal_initializer
glorot_uniform = glorot_uniform_initializer

# pylint: enable=invalid-name

# Utility functions


@tf_export('keras.initializers.serialize')
def serialize(initializer):
  return serialize_keras_object(initializer)


@tf_export('keras.initializers.deserialize')
def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='initializer')


@tf_export('keras.initializers.get')
def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret initializer identifier: ' +
                     str(identifier))
