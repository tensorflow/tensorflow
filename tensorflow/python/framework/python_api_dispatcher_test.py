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
"""Tests for tensorflow.python.framework.python_api_dispatcher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import _pywrap_python_api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class Trace(object):
  """A dispatchable type that builds traces of ops it's called with."""

  log = []

  def __init__(self, api_name, *args):
    self.api_name = api_name
    self.args = args

  @classmethod
  def __tf_dispatch__(cls, api_name, api_func, args):
    Trace.log.append("__tf_dispatch__%s" % ((cls.__name__, api_name),))
    if "disabled" in str(args) or api_name == "disabled":
      return NotImplemented
    del api_func  # not used
    return cls(api_name, *args)

  def __repr__(self):
    return "%s%s" % (type(self).__name__, (self.api_name,) + self.args)

  def __eq__(self, other):
    return (type(self) is type(other) and self.api_name == other.api_name and
            self.args == other.args)


class Trace2(Trace):
  pass


class Trace2B(Trace2):
  pass


class Trace3(Trace):
  pass


class Trace4(Trace):
  pass


class WeightedTensor(object):

  def __init__(self, tensor, weight):
    self.tensor = ops.convert_to_tensor(tensor)
    self.weight = weight  # Python float

  @classmethod
  def __tf_dispatch__(cls, api_name, api_func, args):
    del api_name  # unused
    weights = [arg.weight for arg in args if isinstance(arg, WeightedTensor)]
    tensors = [
        arg.tensor if isinstance(arg, WeightedTensor) else arg for arg in args
    ]
    tensor_result = api_func(*tensors)
    avg_weight = sum(weights) / len(weights)
    return cls(tensor_result, avg_weight)


@test_util.run_all_in_graph_and_eager_modes
class PythonAPIDispatcherTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

  def testNoDispatchableTypes(self):
    add_dispatcher = _pywrap_python_api_dispatcher.PythonAPIDispatcher(
        "tf.math.add", math_ops.add, 2, [0, 1], [], False)
    self.assertEqual(add_dispatcher.Dispatch(1, 2), NotImplemented)

    concat_dispatcher = _pywrap_python_api_dispatcher.PythonAPIDispatcher(
        "tf.concat", array_ops.concat, 2, [1], [0], False)
    self.assertEqual(concat_dispatcher.Dispatch([1], 0), NotImplemented)

  def testSimpleDispatchWithTrace(self):
    dispatcher = _pywrap_python_api_dispatcher.PythonAPIDispatcher(
        "tf.math.add", math_ops.add, 2, [0, 1], [], False)
    x = 5
    y = Trace("constant", "y")
    z = Trace("constant", "z")

    Trace.log.clear()
    self.assertEqual(dispatcher.Dispatch(x, y), Trace("tf.math.add", x, y))
    self.assertEqual(dispatcher.Dispatch(y, x), Trace("tf.math.add", y, x))
    self.assertEqual(dispatcher.Dispatch(y, z), Trace("tf.math.add", y, z))
    self.assertEqual(Trace.log, [
        "__tf_dispatch__('Trace', 'tf.math.add')",
        "__tf_dispatch__('Trace', 'tf.math.add')",
        "__tf_dispatch__('Trace', 'tf.math.add')"
    ])

  def testDispatcherReturnsNotImplemented(self):
    dispatcher = _pywrap_python_api_dispatcher.PythonAPIDispatcher(
        "tf.math.add", math_ops.add, 2, [0, 1], [], False)
    x = 5
    y = Trace("constant", "disabled")
    z = Trace("constant", "z")

    self.assertEqual(dispatcher.Dispatch(x, y), NotImplemented)
    self.assertEqual(dispatcher.Dispatch(y, x), NotImplemented)
    self.assertEqual(dispatcher.Dispatch(y, z), NotImplemented)
    self.assertEqual(dispatcher.Dispatch(z, z), Trace("tf.math.add", z, z))

  def testSimpleDispatchWithWeightedTensor(self):
    dispatcher = _pywrap_python_api_dispatcher.PythonAPIDispatcher(
        "tf.math.add", math_ops.add, 2, [0, 1], [], False)
    x = 5
    y = WeightedTensor([1, 2, 3], 0.6)
    z = WeightedTensor([10, 20, 30], 0.2)

    x_plus_y = dispatcher.Dispatch(x, y)
    y_plus_x = dispatcher.Dispatch(y, x)
    y_plus_z = dispatcher.Dispatch(y, z)

    self.assertAllEqual(x_plus_y.tensor, [6, 7, 8])
    self.assertAllEqual(y_plus_x.tensor, [6, 7, 8])
    self.assertAllEqual(y_plus_z.tensor, [11, 22, 33])

    self.assertEqual(x_plus_y.weight, 0.6)
    self.assertEqual(y_plus_x.weight, 0.6)
    self.assertEqual(y_plus_z.weight, 0.4)

  def testDispatchPrecedence(self):
    # We use an API for which dispatch is disabled, so all dispatchers get
    # called (since this test checks the order of the dispatcher list).
    dispatcher = _pywrap_python_api_dispatcher.PythonAPIDispatcher(
        "disabled", None, 5, [0, 1, 4], [2, 3], False)

    t = Trace("constant", "t")
    t2_1 = Trace2("constant", "t2_1")
    t2_2 = Trace2("constant", "t2_2")
    t2b = Trace2B("constant", "t2b")
    t3 = Trace3("constant", "t3")
    t4 = Trace4("constant", "t4")

    # Three dispatchable types, none of which is a subclass of the other:
    # * precedence is left-to-right.
    # * duplicates are removed.
    Trace.log.clear()
    result = dispatcher.Dispatch(t2_1, t3, [], [t2_2, t3], t4)
    self.assertEqual(result, NotImplemented)
    self.assertEqual(Trace.log, [
        "__tf_dispatch__('Trace2', 'disabled')",
        "__tf_dispatch__('Trace3', 'disabled')",
        "__tf_dispatch__('Trace4', 'disabled')"
    ])

    # Subtypes are moved before their base types.
    Trace.log.clear()
    result = dispatcher.Dispatch(t2_1, t3, [t], [t2_2, t, t3, t4], t2b)
    self.assertEqual(result, NotImplemented)
    self.assertEqual(Trace.log, [
        "__tf_dispatch__('Trace2B', 'disabled')",
        "__tf_dispatch__('Trace2', 'disabled')",
        "__tf_dispatch__('Trace3', 'disabled')",
        "__tf_dispatch__('Trace4', 'disabled')",
        "__tf_dispatch__('Trace', 'disabled')"
    ])

  def testDispatchPrecedenceRightToLeft(self):
    # We use an API for which dispatch is disabled, so all dispatchers get
    # called (since this test checks the order of the dispatcher list).
    dispatcher = _pywrap_python_api_dispatcher.PythonAPIDispatcher(
        "disabled", None, 5, [4, 0, 1], [2, 3], True)

    t = Trace("constant", "t")
    t2_1 = Trace2("constant", "t2_1")
    t2_2 = Trace2("constant", "t2_2")
    t2b = Trace2B("constant", "t2b")
    t3 = Trace3("constant", "t3")
    t4 = Trace4("constant", "t4")

    # Three dispatchable types, none of which is a subclass of the other:
    # * precedence is right_to_left (since we set right_to_left=True in the
    #   PtyonAPIDispatcher constructor).  (Note: arguments are scanned
    #   right-to-left, but the elements of list arguments are still scanned
    #   left-to-right.)
    # * duplicates are removed.
    Trace.log.clear()
    result = dispatcher.Dispatch(t2_1, t3, [], [t2_2, t3], t4)
    self.assertEqual(result, NotImplemented)
    self.assertEqual(Trace.log, [
        "__tf_dispatch__('Trace4', 'disabled')",
        "__tf_dispatch__('Trace2', 'disabled')",
        "__tf_dispatch__('Trace3', 'disabled')"
    ])

    # Subtypes are moved before their base types.  (Note: moving subtypes occurs
    # *after* we swap the order to be right-to-left; so the dispatch order here
    # is not what we'd get by just reversing the final dispatch order if
    # right_to_left were false.)
    Trace.log.clear()
    result = dispatcher.Dispatch(t2_1, t3, [t], [t2_2, t, t3, t4], t2b)
    self.assertEqual(result, NotImplemented)
    self.assertEqual(Trace.log, [
        "__tf_dispatch__('Trace2B', 'disabled')",
        "__tf_dispatch__('Trace2', 'disabled')",
        "__tf_dispatch__('Trace3', 'disabled')",
        "__tf_dispatch__('Trace4', 'disabled')",
        "__tf_dispatch__('Trace', 'disabled')"
    ])

  def testDispatchParamOutOfRange(self):
    with self.assertRaisesRegex(ValueError, "index out of range"):
      _pywrap_python_api_dispatcher.PythonAPIDispatcher("some_api", None, 5,
                                                        [0, 1, 5], [2, 3], True)
    with self.assertRaisesRegex(ValueError, "index out of range"):
      _pywrap_python_api_dispatcher.PythonAPIDispatcher("some_api", None, 5,
                                                        [0, -3], [2, 3], True)
    with self.assertRaisesRegex(ValueError, "index out of range"):
      _pywrap_python_api_dispatcher.PythonAPIDispatcher("some_api", None, 5,
                                                        [0, 1], [10, 3], True)


if __name__ == "__main__":
  googletest.main()
