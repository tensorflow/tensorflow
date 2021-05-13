# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.engine import input_spec
from tensorflow.python.platform import test


class InputSpecTest(test.TestCase):

  def test_axes_initialization(self):
    input_spec.InputSpec(shape=[1, None, 2, 3], axes={3: 5, '2': 2})
    with self.assertRaisesRegex(ValueError, 'Axis 4 is greater than'):
      input_spec.InputSpec(shape=[1, None, 2, 3], axes={4: 5})
    with self.assertRaisesRegex(TypeError, 'keys in axes must be integers'):
      input_spec.InputSpec(shape=[1, None, 2, 3], axes={'string': 5})


class InputSpecToTensorShapeTest(test.TestCase):

  def test_defined_shape(self):
    spec = input_spec.InputSpec(shape=[1, None, 2, 3])
    self.assertAllEqual(
        [1, None, 2, 3], input_spec.to_tensor_shape(spec).as_list())

  def test_defined_ndims(self):
    spec = input_spec.InputSpec(ndim=5)
    self.assertAllEqual(
        [None] * 5, input_spec.to_tensor_shape(spec).as_list())

    spec = input_spec.InputSpec(ndim=0)
    self.assertAllEqual(
        [], input_spec.to_tensor_shape(spec).as_list())

    spec = input_spec.InputSpec(ndim=3, axes={1: 3, -1: 2})
    self.assertAllEqual(
        [None, 3, 2], input_spec.to_tensor_shape(spec).as_list())

  def test_undefined_shapes(self):
    spec = input_spec.InputSpec(max_ndim=5)
    with self.assertRaisesRegex(ValueError, 'unknown TensorShape'):
      input_spec.to_tensor_shape(spec).as_list()

    spec = input_spec.InputSpec(min_ndim=5, max_ndim=5)
    with self.assertRaisesRegex(ValueError, 'unknown TensorShape'):
      input_spec.to_tensor_shape(spec).as_list()


if __name__ == '__main__':
  test.main()
