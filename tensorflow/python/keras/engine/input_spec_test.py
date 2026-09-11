# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""InputSpec tests."""

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.platform import test


class DisplayShapeTest(test.TestCase):

  def test_display_shape_known_rank(self):
    shape = tensor_shape.TensorShape([1, None, 2, 3])
    self.assertEqual(input_spec.display_shape(shape), '(1, None, 2, 3)')

  def test_display_shape_unknown_rank(self):
    shape = tensor_shape.TensorShape(None)
    self.assertEqual(input_spec.display_shape(shape), str(shape))


if __name__ == '__main__':
  test.main()
