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
"""Tests for operator dispatch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


class CustomTensor(object):
  """A fake composite tensor class, for testing type-based dispatching."""

  def __init__(self, tensor, score):
    self.tensor = ops.convert_to_tensor(tensor)
    self.score = score


@tf_export("test_op")
@dispatch.add_dispatch_support
def test_op(x, y, z):
  """A fake op for testing dispatch of Python ops."""
  return x + (2 * y) + (3 * z)


@test_util.run_all_in_graph_and_eager_modes
class DispatchTest(test_util.TensorFlowTestCase):

  def testAddDispatchForTypes_With_CppOp(self):
    original_handlers = gen_math_ops.add._tf_dispatchers[:]

    # Override the behavior of gen_math_ops.add.
    @dispatch.dispatch_for_types(gen_math_ops.add, CustomTensor)
    def custom_add(x, y, name=None):  # pylint: disable=unused-variable
      return CustomTensor(gen_math_ops.add(x.tensor, y.tensor, name),
                          (x.score+y.score) / 2.0)
    self.assertEqual(len(math_ops.add._tf_dispatchers),
                     len(original_handlers) + 1)

    # Test that we see the overridden behavior when using CustomTensors.
    x = CustomTensor([1, 2, 3], 2.0)
    y = CustomTensor([7, 8, 2], 0.0)
    x_plus_y = gen_math_ops.add(x, y)
    self.assertAllEqual(self.evaluate(x_plus_y.tensor), [8, 10, 5])
    self.assertNear(x_plus_y.score, 1.0, 0.001)

    # Test that we still get the right behavior when using normal Tensors.
    a = [1, 2, 3]
    b = [4, 5, 6]
    a_plus_b = gen_math_ops.add(a, b)
    self.assertAllEqual(a_plus_b, [5, 7, 9])

    # Test that we still get a TypeError or ValueError if we pass some
    # type that's not supported by any dispatcher.
    with self.assertRaises((TypeError, ValueError)):
      gen_math_ops.add(a, None)

    # Clean up
    gen_math_ops.add._tf_dispatchers = original_handlers

  def testAddDispatchForTypes_With_PythonOp(self):
    original_handlers = test_op._tf_dispatchers[:]

    @dispatch.dispatch_for_types(test_op, CustomTensor)
    def override_for_test_op(x, y, z):  # pylint: disable=unused-variable
      return CustomTensor(test_op(x.tensor, y.tensor, z.tensor),
                          (x.score + y.score + z.score) / 3.0)

    x = CustomTensor([1, 2, 3], 0.2)
    y = CustomTensor([7, 8, 2], 0.4)
    z = CustomTensor([0, 1, 2], 0.6)

    result = test_op(x, y, z)
    self.assertAllEqual(self.evaluate(result.tensor), [15, 21, 13])
    self.assertNear(result.score, 0.4, 0.001)

    # Clean up
    test_op._tf_dispatchers = original_handlers

  def testDispatchForTypes_SignatureMismatch(self):
    with self.assertRaisesRegexp(AssertionError, "The decorated function's "
                                 "signature must exactly match.*"):
      @dispatch.dispatch_for_types(test_op, CustomTensor)
      def override_for_test_op(a, b, c):  # pylint: disable=unused-variable
        return CustomTensor(test_op(a.tensor, b.tensor, c.tensor),
                            (a.score + b.score + c.score) / 3.0)

  def testDispatchForTypes_OpDoesNotSupportDispatch(self):
    def some_op(x, y):
      return x + y

    with self.assertRaisesRegexp(AssertionError, "Dispatching not enabled for"):
      @dispatch.dispatch_for_types(some_op, CustomTensor)
      def override_for_some_op(x, y):  # pylint: disable=unused-variable
        return x if x.score > 0 else y


if __name__ == "__main__":
  googletest.main()


