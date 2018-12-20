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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class WrapFunctionTest(test.TestCase):

  def testDocString(self):

    def f(x, do_add):
      v = variables.Variable(5.0)
      if do_add:
        op = v.assign_add(x)
      else:
        op = v.assign_sub(x)
      with ops.control_dependencies([op]):
        return v.read_value()

    f_add = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtypes.float32), True])

    self.assertAllEqual(f_add(1.0), 6.0)
    self.assertAllEqual(f_add(1.0), 7.0)

    # Can call tf.compat.v1.wrap_function again to get a new trace, a new set
    # of variables, and possibly different non-template arguments.
    f_sub = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtypes.float32), False])

    self.assertAllEqual(f_sub(1.0), 4.0)
    self.assertAllEqual(f_sub(1.0), 3.0)

  def testPrune(self):

    x_in = []
    x_out = []

    def f(x, y):
      x_in.append(x)
      xx = x * x
      x_out.append(xx)
      return xx, 2 * y*y

    f_wrapped = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtypes.float32)] * 2)

    f_pruned = f_wrapped.prune(x_in[0], [x_out[0]])
    self.assertAllEqual(f_pruned(ops.convert_to_tensor(2.0)), [4.0])

  def testNoArguments(self):

    def f():
      return constant_op.constant(1.)

    f_wrapped = wrap_function.wrap_function(f, [])
    self.assertAllEqual(1.0, f_wrapped())

  def testPruneCaptures(self):

    v1 = variables.Variable(2.)

    def f():
      v2 = variables.Variable(3.)
      return array_ops.identity(v1 * v2 * constant_op.constant(1.), 'fetch')

    f_wrapped = wrap_function.wrap_function(f, [])
    self.assertAllEqual(6.0, f_wrapped())
    pruned = f_wrapped.prune(
        feeds=(),
        fetches=f_wrapped.graph.get_tensor_by_name('fetch:0'))
    self.assertAllEqual(6.0, pruned())

  def testPruneOperations(self):

    v = variables.Variable(0)

    def f():
      v.assign_add(1, name='increment', read_value=False)

    f_wrapped = wrap_function.wrap_function(f, [])
    pruned = f_wrapped.prune(
        feeds=(),
        fetches=(f_wrapped.graph.get_operation_by_name('increment'),))
    self.assertEqual((None,), pruned())
    self.assertEqual(1, self.evaluate(v))

    del f, f_wrapped

    def f1():
      v.assign_add(
          array_ops.placeholder(shape=[], dtype=dtypes.int32, name='step'),
          name='increment', read_value=False)
      return constant_op.constant(1, name='other')

    f_wrapped = wrap_function.wrap_function(f1, [])
    increments = f_wrapped.prune(
        feeds=(f_wrapped.graph.get_tensor_by_name('step:0')),
        fetches=(f_wrapped.graph.get_operation_by_name('increment'),
                 f_wrapped.graph.get_tensor_by_name('other:0')))
    first_output, second_output = increments(constant_op.constant(2))
    self.assertEqual(['Placeholder:0', 'Placeholder_1:0'],
                     [t.name for t in increments.inputs])
    self.assertIs(None, first_output)
    self.assertEqual(1, second_output.numpy())
    self.assertEqual(3, v.numpy())
    does_not_increment = f_wrapped.prune(
        feeds=(f_wrapped.graph.get_tensor_by_name('step:0')),
        fetches=f_wrapped.graph.get_tensor_by_name('other:0'))
    self.assertEqual(1, does_not_increment(constant_op.constant(3)).numpy())
    self.assertEqual(3, v.numpy())


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
