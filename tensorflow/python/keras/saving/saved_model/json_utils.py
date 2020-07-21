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

from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import serialization


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
  if isinstance(obj, dict) and 'class_name' in obj:
    if obj['class_name'] == 'TensorShape':
      return tensor_shape.TensorShape(obj['items'])
    elif obj['class_name'] == '__tuple__':
      return tuple(_decode_helper(i) for i in obj['items'])
  return obj
