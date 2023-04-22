# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class TraceTest(test.TestCase):

  def setUp(self):
    x = np.random.seed(0)

  def compare(self, x):
    np_ans = np.trace(x, axis1=-2, axis2=-1)
    with self.cached_session():
      tf_ans = math_ops.trace(x).eval()
    self.assertAllClose(tf_ans, np_ans)

  @test_util.run_deprecated_v1
  def testTrace(self):
    for dtype in [np.int32, np.float32, np.float64]:
      for shape in [[2, 2], [2, 3], [3, 2], [2, 3, 2], [2, 2, 2, 3]]:
        x = np.random.rand(np.prod(shape)).astype(dtype).reshape(shape)
        self.compare(x)


if __name__ == "__main__":
  test.main()
