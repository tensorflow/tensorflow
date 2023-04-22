# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for serializing Python objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import wrapt

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util.compat import collections_abc


def get_json_type(obj):
  """Serializes any object to a JSON-serializable structure.

  Args:
      obj: the object to serialize

  Returns:
      JSON-serializable structure representing `obj`.

  Raises:
      TypeError: if `obj` cannot be serialized.
  """
  # if obj is a serializable Keras class instance
  # e.g. optimizer, layer
  if hasattr(obj, 'get_config'):
    return {'class_name': obj.__class__.__name__, 'config': obj.get_config()}

  # if obj is any numpy type
  if type(obj).__module__ == np.__name__:
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return obj.item()

  # misc functions (e.g. loss function)
  if callable(obj):
    return obj.__name__

  # if obj is a python 'type'
  if type(obj).__name__ == type.__name__:
    return obj.__name__

  if isinstance(obj, tensor_shape.Dimension):
    return obj.value

  if isinstance(obj, tensor_shape.TensorShape):
    return obj.as_list()

  if isinstance(obj, dtypes.DType):
    return obj.name

  if isinstance(obj, collections_abc.Mapping):
    return dict(obj)

  if obj is Ellipsis:
    return {'class_name': '__ellipsis__'}

  if isinstance(obj, wrapt.ObjectProxy):
    return obj.__wrapped__

  raise TypeError('Not JSON Serializable:', obj)
