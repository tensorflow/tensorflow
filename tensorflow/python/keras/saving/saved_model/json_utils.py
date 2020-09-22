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
"""Utils for creating and loading the Layer metadata for SavedModel.

These are required to retain the original format of the build input shape, since
layers and models may have different build behaviors depending on if the shape
is a list, tuple, or TensorShape. For example, Network.build() will create
separate inputs if the given input_shape is a list, and will create a single
input if the given shape is a tuple.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import wrapt

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import serialization

try:
  # This import only works on python 3.3 and above.
  import collections.abc as collections_abc  # pylint: disable=unused-import, g-import-not-at-top
except ImportError:
  import collections as collections_abc  # pylint: disable=unused-import, g-import-not-at-top


class Encoder(json.JSONEncoder):
  """JSON encoder and decoder that handles TensorShapes and tuples."""

  def default(self, obj):
    if isinstance(obj, tensor_shape.TensorShape):
      items = obj.as_list() if obj.rank is not None else None
      return {'class_name': 'TensorShape', 'items': items}
    return serialization.get_json_type(obj)

  def encode(self, obj):
    return super(Encoder, self).encode(_encode_tuple(obj))


def _encode_tuple(x):
  if isinstance(x, tuple):
    return {'class_name': '__tuple__',
            'items': tuple(_encode_tuple(i) for i in x)}
  elif isinstance(x, list):
    return [_encode_tuple(i) for i in x]
  elif isinstance(x, dict):
    return {key: _encode_tuple(value) for key, value in x.items()}
  else:
    return x


def decode(json_string):
  return json.loads(json_string, object_hook=_decode_helper)


def _decode_helper(obj):
  """A decoding helper that is TF-object aware."""
  if isinstance(obj, dict) and 'class_name' in obj:
    if obj['class_name'] == 'TensorShape':
      return tensor_shape.TensorShape(obj['items'])
    elif obj['class_name'] == '__tuple__':
      return tuple(_decode_helper(i) for i in obj['items'])
    elif obj['class_name'] == '__ellipsis__':
      return Ellipsis
  return obj


def get_json_type(obj):
  """Serializes any object to a JSON-serializable structure.

  Arguments:
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
