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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@parameterized.named_parameters(
    dict(testcase_name='Defun', function_decorator=function.defun),
    dict(testcase_name='DefFunction', function_decorator=def_function.function))
class ArgumentNamingTests(test.TestCase, parameterized.TestCase):
  """Tests for recognizable export signatures from concrete functions."""

  def testBasic(self, function_decorator):
    @function_decorator
    def fn(a, b):
      return a + b, a * b
    # Call the function to make def_function happy
    fn(array_ops.ones([]), array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['a', 'b'],
        [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'a', b'b'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
    self.assertEqual(2, len(fn_op.graph.structured_outputs))
    self.assertAllClose(
        [3., 2.],
        fn_op(constant_op.constant(1.), constant_op.constant(2.)))
    self.assertAllClose(
        [3., 2.],
        fn_op(a=constant_op.constant(1.), b=constant_op.constant(2.)))

  def testVariable(self, function_decorator):
    @function_decorator
    def fn(a, b):
      return a + b, a * b
    # Call the function to make def_function happy
    fn(array_ops.ones([]), array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32),
        variables.Variable(1.))
    self.assertEqual(
        ['a', 'b'],
        [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'a', b'b'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
    self.assertEqual(2, len(fn_op.graph.structured_outputs))

  def testDictReturned(self, function_decorator):
    @function_decorator
    def fn(x, z=(1., 2.), y=3.):
      z1, z2 = z
      return {'alpha': x + y + z1, 'beta': x * y + z2}
    # Call the function to make def_function happy
    fn(array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        x=tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32),
        y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['x', 'y'],
        [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'x', b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
    self.assertEqual({'alpha', 'beta'},
                     set(fn_op.graph.structured_outputs.keys()))

    fn_op2 = fn.get_concrete_function(
        z=(tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32,
                                  name='z_first'),
           tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32,
                                  name='z_second')),
        y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'),
        x=4.)
    self.assertEqual(
        ['z_first', 'z_second', 'custom'],
        [inp.op.name for inp in fn_op2.inputs])
    self.assertEqual(
        [b'z_first', b'z_second', b'custom'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op2.inputs])

    fn_op3 = fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'),
        z=(tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32,
                                  name='z1'),
           tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='z2')),
        y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['custom', 'z1', 'z2', 'y'],
        [inp.op.name for inp in fn_op3.inputs])
    self.assertEqual(
        [b'custom', b'z1', b'z2', b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op3.inputs])

  def testMethod(self, function_decorator):
    class HasMethod(object):

      @function_decorator
      def method(self, x):
        return x

    has_method = HasMethod()
    # Call the function to make def_function happy
    HasMethod.method(has_method, array_ops.ones([]))
    class_op = HasMethod.method.get_concrete_function(
        has_method, tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['x'],
        [inp.op.name for inp in class_op.inputs])
    self.assertEqual(
        [b'x'],
        [inp.op.get_attr('_user_specified_name') for inp in class_op.inputs])
    # Call the function to make def_function happy
    has_method.method(array_ops.ones([]))
    method_op = has_method.method.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['x'],
        [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'x'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])
    # TODO(allenl): It should be possible to override names when exporting. Do
    # TensorSpec names need to go in cache keys? Or maybe get_concrete_function
    # should always retrace?
    self.skipTest('Not working')
    method_op = has_method.method.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='y'))
    self.assertEqual(
        ['y'],
        [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])

  def testMethodSignature(self, function_decorator):

    class HasMethod(object):

      @function_decorator(
          input_signature=(tensor_spec.TensorSpec(
              shape=None, dtype=dtypes.float64, name='y'),))
      def method(self, x):
        hash(self)  # No weak proxies passed as `self`
        return x

    has_method = HasMethod()
    # Call the function to make def_function happy
    has_method.method(array_ops.ones([], dtype=dtypes.float64))
    method_op = has_method.method.get_concrete_function()
    self.assertEqual(
        ['y'],
        [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])
    method_op2 = has_method.method.get_concrete_function()
    self.assertEqual(
        ['y'],
        [inp.op.name for inp in method_op2.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op2.inputs])

  def testVariadic(self, function_decorator):
    @function_decorator
    def variadic_fn(x, *args, **kwargs):
      return x + math_ops.add_n(list(args) + list(kwargs.values()))

    # Call the function to make def_function happy
    variadic_fn(array_ops.ones([]), array_ops.ones([]))
    variadic_op = variadic_fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32, name='y'),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32,
                               name='second_variadic'),
        z=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
        zz=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='cust'))
    self.assertEqual(
        ['x', 'y', 'args_1', 'second_variadic', 'z', 'cust'],
        [inp.op.name for inp in variadic_op.inputs])
    self.assertEqual(
        [b'x', b'y', b'args_1', b'second_variadic', b'z', b'cust'],
        [inp.op.get_attr('_user_specified_name')
         for inp in variadic_op.inputs])

  def testVariadicInputSignature(self, function_decorator):
    @function_decorator(
        input_signature=(
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32, name='y'),
            tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='z'),
        ))
    def variadic_fn(x, *args):
      return x + math_ops.add_n(list(args))

    # Call the function to make def_function happy
    variadic_fn(array_ops.ones([]), array_ops.ones([]),
                array_ops.ones([]), array_ops.ones([]))
    variadic_op = variadic_fn.get_concrete_function()
    self.assertIn(b'variadic_fn', variadic_op.name)
    self.assertEqual(
        ['x', 'y', 'args_1', 'z'],
        [inp.op.name for inp in variadic_op.inputs])
    self.assertEqual(
        [b'x', b'y', b'args_1', b'z'],
        [inp.op.get_attr('_user_specified_name')
         for inp in variadic_op.inputs])


if __name__ == '__main__':
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={'CPU': 4}))
  test.main()
