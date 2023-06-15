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
"""Tests for tensorflow.ops.math_ops.linspace."""

# Using distutils.version.LooseVersion was resulting in an error, so importing
# directly.
from distutils.version import LooseVersion  # pylint: disable=g-importing-member

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class LinspaceTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.parameters([
      {
          "start_shape": start_shape,
          "stop_shape": stop_shape,
          "dtype": dtype,
          "num": num
      }
      for start_shape in [(), (2,), (2, 2)]
      for stop_shape in [(), (2,), (2, 2)]
      for dtype in [np.float64, np.int64]
      for num in [0, 1, 2, 20]
  ])
  # pylint: enable=g-complex-comprehension
  def testLinspaceBroadcasts(self, start_shape, stop_shape, dtype, num):
    if LooseVersion(np.version.version) < LooseVersion("1.16.0"):
      self.skipTest("numpy doesn't support axes before version 1.16.0")

      ndims = max(len(start_shape), len(stop_shape))
      for axis in range(-ndims, ndims):
        start = np.ones(start_shape, dtype)
        stop = 10 * np.ones(stop_shape, dtype)

        np_ans = np.linspace(start, stop, num, axis=axis)
        tf_ans = self.evaluate(
            math_ops.linspace_nd(start, stop, num, axis=axis))

        self.assertAllClose(np_ans, tf_ans)

  def testShapeInformationPeserved(self):
    @def_function.function
    def linspace(start, stop, num, axis):
      return math_ops.linspace_nd(start, stop, num=num, axis=axis)

    # Constant num and axis leads to preserved known shape.
    output_shape = linspace.get_concrete_function(
        start=tensor.TensorSpec(shape=[64, None], dtype=dtypes.float32),
        stop=tensor.TensorSpec(shape=[64, None], dtype=dtypes.float32),
        num=10,
        axis=-1,
    ).output_shapes
    expected_shape = (64, None, 10)
    self.assertEqual(output_shape, expected_shape)


if __name__ == "__main__":
  googletest.main()
