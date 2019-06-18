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
"""XLA tests for pfor."""
# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class PForTest(PForTestCase):

  def test_xla(self):

    def compute(x):
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    def vectorized_compute(x):
      return pfor_control_flow_ops.vectorized_map(compute, x)

    result = xla.compile(vectorized_compute,
                         inputs=[array_ops.ones((10, 5, 3))])
    self.run_and_assert_equal(result, array_ops.ones((10, 1, 3)))


if __name__ == "__main__":
  test.main()
