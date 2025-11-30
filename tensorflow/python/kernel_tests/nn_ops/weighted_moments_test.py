# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_impl
from tensorflow.python.platform import test


class WeightedMomentsTest(test.TestCase):

  def test_tensor_axes_keepdims_false_shape_mismatch(self):
    x = constant_op.constant([[1., 2., 3., 4., 5.],
                              [6., 7., 8., 9., 10.]])
    weights = constant_op.constant([[1.], [1.]])
    axes = constant_op.constant([0], dtype=constant_op.dtypes.int32)
    mean, var = nn_impl.weighted_moments(
        x, axes=axes, frequency_weights=weights, keepdims=False)
    self.assertEqual(mean.shape.as_list(), [5])
    self.assertEqual(var.shape.as_list(), [5])


if __name__ == "__main__":
  test.main()

