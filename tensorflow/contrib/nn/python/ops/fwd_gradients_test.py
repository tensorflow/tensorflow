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
"""Tests for forward_ad.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.nn.python.ops import fwd_gradients
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ForwardAdTest(test.TestCase):

  def testSquare(self):
    x = constant_op.constant(1.)
    y = math_ops.square(x)
    grad_x = 3.

    dydx_tf = fwd_gradients.fwd_gradients([y], [x], [grad_x])[0]
    dydx_py = 2. * grad_x

    with self.test_session() as sess:
      self.assertAllClose(sess.run(dydx_tf), dydx_py, 1e-6)

  def testGather(self):
    x = constant_op.constant([1., 2., 3.])
    y = array_ops.gather(x, [0, 1])
    y.set_shape([2])
    dydx = fwd_gradients.fwd_gradients([y], [x], assert_unused=True)

    with self.test_session() as sess:
      sess.run(dydx)


if __name__ == '__main__':
  test.main()
