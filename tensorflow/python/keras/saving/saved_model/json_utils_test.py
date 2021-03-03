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
# pylint: disable=protected-access
"""Tests the JSON encoder and decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.platform import test


class JsonUtilsTest(test.TestCase):

  def test_encode_decode_tensor_shape(self):
    metadata = {
        'key1': tensor_shape.TensorShape(None),
        'key2': [tensor_shape.TensorShape([None]),
                 tensor_shape.TensorShape([3, None, 5])]}
    string = json_utils.Encoder().encode(metadata)
    loaded = json_utils.decode(string)

    self.assertEqual(set(loaded.keys()), {'key1', 'key2'})
    self.assertAllEqual(loaded['key1'].rank, None)
    self.assertAllEqual(loaded['key2'][0].as_list(), [None])
    self.assertAllEqual(loaded['key2'][1].as_list(), [3, None, 5])

  def test_encode_decode_tuple(self):
    metadata = {
        'key1': (3, 5),
        'key2': [(1, (3, 4)), (1,)]}
    string = json_utils.Encoder().encode(metadata)
    loaded = json_utils.decode(string)

    self.assertEqual(set(loaded.keys()), {'key1', 'key2'})
    self.assertAllEqual(loaded['key1'], (3, 5))
    self.assertAllEqual(loaded['key2'], [(1, (3, 4)), (1,)])

  def test_encode_decode_type_spec(self):
    spec = tensor_spec.TensorSpec((1, 5), dtypes.float32)
    string = json_utils.Encoder().encode(spec)
    loaded = json_utils.decode(string)
    self.assertEqual(spec, loaded)

    invalid_type_spec = {'class_name': 'TypeSpec', 'type_spec': 'Invalid Type',
                         'serialized': None}
    string = json_utils.Encoder().encode(invalid_type_spec)
    with self.assertRaisesRegexp(ValueError, 'No TypeSpec has been registered'):
      loaded = json_utils.decode(string)

  def test_encode_decode_enum(self):
    class Enum(enum.Enum):
      CLASS_A = 'a'
      CLASS_B = 'b'
    config = {'key': Enum.CLASS_A, 'key2': Enum.CLASS_B}
    string = json_utils.Encoder().encode(config)
    loaded = json_utils.decode(string)
    self.assertAllEqual({'key': 'a', 'key2': 'b'}, loaded)

if __name__ == '__main__':
  test.main()
