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
"""Tests for tf.subscribe."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import subscribe
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import googletest


class SubscribeTest(test_util.TensorFlowTestCase):

  def testSideEffect(self):
    a = constant_op.constant(1)
    b = constant_op.constant(1)
    c = math_ops.add(a, b)
    with ops.control_dependencies([c]):
      d = constant_op.constant(42)
    n = math_ops.negative(c)

    shared = []

    def sub(t):
      shared.append(t)
      return t

    c = subscribe.subscribe(c,
                            lambda t: script_ops.py_func(sub, [t], [t.dtype]))

    with self.test_session() as sess:
      c_out = sess.run([c])
      n_out = sess.run([n])
      d_out = sess.run([d])

    self.assertEquals(n_out, [-2])
    self.assertEquals(c_out, [2])
    self.assertEquals(d_out, [42])
    self.assertEquals(shared, [2, 2, 2])

  def testCaching(self):
    """Confirm caching of control output is recacluated between calls."""
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.control_dependencies([a]):
      c = constant_op.constant(42)

    shared = {}

    def sub(t):
      shared[t] = shared.get(t, 0) + 1
      return t

    a = subscribe.subscribe(a,
                            lambda t: script_ops.py_func(sub, [t], [t.dtype]))

    with ops.control_dependencies([b]):
      d = constant_op.constant(11)

    # If it was using outdated cached control_outputs then
    # evaling would not trigger the new subscription.
    b = subscribe.subscribe(b,
                            lambda t: script_ops.py_func(sub, [t], [t.dtype]))

    with self.test_session() as sess:
      c_out = sess.run([c])
      d_out = sess.run([d])

    self.assertEquals(c_out, [42])
    self.assertEquals(d_out, [11])
    self.assertEquals(shared, {2: 1, 1: 1})


if __name__ == '__main__':
  googletest.main()
