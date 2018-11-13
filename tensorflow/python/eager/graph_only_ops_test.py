# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for graph_only_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import graph_only_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class GraphOnlyOpsTest(test_util.TensorFlowTestCase):

  def testGraphZerosLike(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    z_tf = graph_only_ops.graph_zeros_like(x)
    with self.cached_session():
      self.assertAllClose(np.zeros((2, 3)), z_tf.eval())

  def testGraphPlaceholder(self):
    x_tf = graph_only_ops.graph_placeholder(dtypes.int32, shape=(1,))
    y_tf = math_ops.square(x_tf)
    with self.cached_session() as sess:
      x = np.array([42])
      y = sess.run(y_tf, feed_dict={x_tf: np.array([42])})
      self.assertAllClose(np.square(x), y)


if __name__ == '__main__':
  test.main()
