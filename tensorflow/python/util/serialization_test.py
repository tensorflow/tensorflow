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
"""Tests for serialization functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test
from tensorflow.python.util import serialization


class SerializationTests(test.TestCase):

  def test_serialize_shape(self):
    round_trip = json.loads(json.dumps(
        tensor_shape.TensorShape([None, 2, 3]),
        default=serialization.get_json_type))
    self.assertIs(round_trip[0], None)
    self.assertEqual(round_trip[1], 2)


if __name__ == "__main__":
  test.main()
