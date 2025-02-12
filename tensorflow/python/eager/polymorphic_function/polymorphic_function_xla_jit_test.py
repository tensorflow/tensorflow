# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import gc
import re

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@test_util.with_eager_op_as_function
class FunctionTest(xla_test.XLATestCase):

  def _compareTwoMethodsCompilerIROutput(self, f, args, kwargs):
    """Assert the two differnet methods (tensor_spec inputs or tensor inputs) experimental_get_compiler give same HLO text."""
    flat_args = list(args) + list(kwargs.values())
    if not all([isinstance(x, tensor.Tensor) for x in flat_args]):
      self.skipTest('It only support args and kwargs are all tf.Tensor types.')

    args_spec = nest.map_structure(tensor.TensorSpec.from_tensor, args)
    kwargs_spec = nest.map_structure(tensor.TensorSpec.from_tensor, kwargs)

    hlo_1 = f.experimental_get_compiler_ir(*args, **kwargs)()
    hlo_2 = f.experimental_get_compiler_ir(*args_spec, **kwargs_spec)()

    if hlo_1 != hlo_2:
      self.fail(
          'The tensor_spec way experimental_get_compiler_ir give diff result to'
          f' normal experimental_get_compiler_ir. \nhlo_1:\n{hlo_1}'
          f'\nhlo_2:\n{hlo_2}\n'
      )

  def testAutoclusteringWithTfFunction(self):
    if 'tpu' in self.device.lower():
      self.skipTest('Autoclustering does not run on TPU')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=False)
      def outer(a, b, c):
        return a * inner(b, c) + c

      @polymorphic_function.function(jit_compile=True)
      def inner(b, c):
        return b + c * b

      i1 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0])
      i2 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0])
      i3 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0])

      with context.collect_graphs(optimized=True) as graphs:
        outer(i1, i2, i3)

      if test_util.is_xla_enabled():
        self.assertIn('_XlaRun', [n.op for n in graphs[0].node])
      else:
        self.assertNotIn('_XlaRun', [n.op for n in graphs[0].node])

  def testBasic(self):
    with ops.device('device:{}:0'.format(self.device)):

      def fn(x, a):
        return x + a

      func = polymorphic_function.function(fn, jit_compile=False)
      xla_func = polymorphic_function.function(fn, jit_compile=True)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      self.assertAllClose([2, 3, 3, 4, 4], func(inputs, 1))
      self.assertAllClose([2, 3, 3, 4, 4], xla_func(inputs, 1))

  def testBasicInt32(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x, a):
        return x + a

      inputs = constant_op.constant([1, 2, 2, 3, 3], dtype=dtypes.int32)
      self.assertAllClose([2, 3, 3, 4, 4], fn(inputs, 1))

  def testDerivative(self):
    with ops.device('device:{}:0'.format(self.device)):

      def fn(x, a):
        return 2 * x + a

      xla_func = polymorphic_function.function(fn, jit_compile=True)

      with backprop.GradientTape() as tape:
        inputs = constant_op.constant([1., 2., 2., 3., 3.])
        tape.watch(inputs)
        outputs = xla_func(inputs, 1)

      self.assertAllClose([2, 2, 2, 2, 2], tape.gradient(outputs, inputs))

      # pylint: disable=protected-access
      (forward, backward) = xla_func.get_concrete_function(
          inputs, 1)._delayed_rewrite_functions.forward_backward()

      # Check that the must-compile attribute gets correctly propagated to the
      # created derivatives.
      self.assertTrue(backward.function_def.attr['_XlaMustCompile'])
      self.assertTrue(forward.cached_definition.attr['_XlaMustCompile'])

  # Calling function with jit_compile=True from
  # jit_compile=False should compile the inner func.
  def testNestedCall(self):
    if 'tpu' in self.device.lower():
      self.skipTest('b/162800687: Inner function runs on host')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x, a):
        return x + a

      @polymorphic_function.function(jit_compile=False)
      def fn2(x, a):
        return fn(x, a)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      self.assertAllClose([2, 3, 3, 4, 4], fn2(inputs, 1))

  def testNestedCallUnsupportedOps(self):
    if 'tpu' in self.device.lower():
      self.skipTest('Outside compilation will extract string_length to CPU')

    with ops.device('device:{}:0'.format(self.device)):

      def fn(x):
        return string_ops.string_length(
            string_ops.string_format('{}', x))

      xla_func = polymorphic_function.function(fn, jit_compile=True)

      def fn2(x):
        return xla_func(x)

      func = polymorphic_function.function(fn2, jit_compile=False)
      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError, 'unsupported operations'
      ):
        func(inputs)

  def testUnsupportedOps(self):
    with ops.device('device:{}:0'.format(self.device)):

      def fn(x):
        return string_ops.string_length(
            string_ops.string_format('{}', x))

      xla_func = polymorphic_function.function(fn, jit_compile=True)

      with self.assertRaisesRegex(
          errors.InvalidArgumentError, 'unsupported operations'
      ):
        xla_func(constant_op.constant([3.1, 3.2]))

  def testCollectiveReduceChannelId(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x, y):
        t0 = collective_ops.all_reduce_v2(
            t=x, group_size=2, group_key=1, instance_key=1)
        t1 = collective_ops.all_reduce_v2(
            t=y, group_size=2, group_key=1, instance_key=1)
        return t0 + t1

      inputs = constant_op.constant([1.0, 2.0, 3.0])
      # Make sure 2 different channel ids are assigned to the 2 all-reduce
      # instructions generated by XLA.
      hlo_str = fn.experimental_get_compiler_ir(inputs, inputs)()
      matches = re.findall('channel_id=([0-9]*),', hlo_str)
      self.assertLen(matches, 2)
      self.assertNotEqual(matches[0], matches[1])
      self._compareTwoMethodsCompilerIROutput(fn, [inputs, inputs], {})

  def testCollectiveReduceReplicaGroups(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x):
        t0 = collective_ops.all_reduce_v2(
            t=x, group_size=2, group_key=1, instance_key=1)
        return t0

      inputs = constant_op.constant([1.0, 2.0, 3.0])
      # Make sure replica groups are assigned
      hlo_str = fn.experimental_get_compiler_ir(inputs)()
      self.assertIn('replica_groups={{', hlo_str)
      self._compareTwoMethodsCompilerIROutput(fn, [inputs], {})

  def testCollectiveReduceGroupAssignment(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x):
        group_size, group_key = collective_ops.assign_group_v2(
            group_assignment=[[0]], device_index=0, base_key=1000)
        t0 = collective_ops.all_reduce_v2(
            t=x, group_size=group_size, group_key=group_key, instance_key=1)
        return t0

      inputs = constant_op.constant([1.0, 2.0, 3.0])
      # Make sure 2 different channel ids are assigned to the 2 all-reduce
      # instructions generated by XLA.
      hlo_str = fn.experimental_get_compiler_ir(inputs)()
      self.assertIn('replica_groups={{0}}', hlo_str)
      self._compareTwoMethodsCompilerIROutput(fn, [inputs], {})

  @test_util.disable_mlir_bridge('TODO(b/155782411): MLIR bridge does not'
                                 'support stack traces')
  def testPythonLocationInMetadata(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def add_fn(x, y):
        return x + y

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      self.assertIn(
          'add_fn', add_fn.experimental_get_compiler_ir(inputs, inputs)()
      )
      self._compareTwoMethodsCompilerIROutput(add_fn, [inputs, inputs], {})

  @test_util.disable_mlir_bridge('TODO(b/155782411): MLIR bridge does not'
                                 'support stack traces')
  def testPythonLocationNestedInMetadata(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def add_f(x, y):
        return x + y

      @polymorphic_function.function(jit_compile=True)
      def add_g(x, y):
        return add_f(x, y)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      self.assertIn(
          'add_g', add_g.experimental_get_compiler_ir(inputs, inputs)()
      )
      self._compareTwoMethodsCompilerIROutput(add_g, [inputs, inputs], {})

  def testPythonStackTrace(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def failure_fn(x):
        return string_ops.string_length(string_ops.string_format('{}', x))

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaisesRegex(errors.InvalidArgumentError, 'failure_fn'):
        failure_fn(inputs)

  def testPythonStackTraceUncompiledWithinCompiled(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function
      def failure_fn(x):
        return string_ops.string_length(string_ops.string_format('{}', x))

      @polymorphic_function.function(jit_compile=True)
      def outer(x):
        return failure_fn(x)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaisesRegex(errors.InvalidArgumentError, 'outer'):
        outer(inputs)

  @test_util.disable_mlir_bridge('TODO(b/155782411): MLIR bridge does not'
                                 'support stack traces')
  def testPythonStackTraceCompiledWithinUncompiled(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def failure_fn(x):
        return string_ops.string_length(string_ops.string_format('{}', x))

      @polymorphic_function.function
      def outer(x):
        return failure_fn(x)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaisesRegex(errors.InvalidArgumentError, 'failure_fn'):
        outer(inputs)

  @test_util.disable_mlir_bridge('TODO(b/155782411): MLIR bridge does not'
                                 'support stack traces')
  def testPythonStackTraceCompiledWithinCompiled(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def failure_fn(x):
        return string_ops.string_length(string_ops.string_format('{}', x))

      @polymorphic_function.function
      def outer(x):
        return failure_fn(x)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaisesRegex(errors.InvalidArgumentError, 'failure_fn'):
        outer(inputs)

  def testFunctionGradient(self):
    with ops.device('device:{}:0'.format(self.device)):
      v = resource_variable_ops.ResourceVariable(2.0)

      def fn(x):
        return v * x

      func = polymorphic_function.function(fn, jit_compile=False)
      xla_func = polymorphic_function.function(fn, jit_compile=True)

      def run_and_check(test_func):
        x = constant_op.constant(3.0)
        with backprop.GradientTape() as tape:
          y = test_func(x)
        dy = tape.gradient(y, v)

        self.assertAllClose(6.0, y)
        self.assertAllClose(3.0, dy)

      run_and_check(func)
      run_and_check(xla_func)

  @test_util.disable_mlir_bridge('TODO(b/162521846): MLIR bridge fails'
                                 ' msan, function library not found')
  def testControlFlow(self):

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        assert control_flow_util.GraphOrParentsInXlaContext(
            ops.get_default_graph())
        x = ops.convert_to_tensor(x)

        def body(i, a):
          return i + 1, cond.cond(i > 2, lambda: a + (x**2),
                                  lambda: a + 3)

        return while_loop.while_loop(
            lambda i, *_: i < 10,
            body, (constant_op.constant(0), constant_op.constant(3.)),
            maximum_iterations=10)[1]

      @polymorphic_function.function(jit_compile=True)
      def g(x):
        x = ops.convert_to_tensor(x)
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = f(x)
        return y, tape.gradient(y, x)

      # Test that XLA context gets correctly propagated.
      g._get_concrete_function_garbage_collected(2.0)(2.0)

      self.assertAllClose(40.0, f(2.0))
      self.assertAllClose([40.0, 28.0], g(2.0))
      self.assertAllClose(40.0, f.get_concrete_function(2.0)(2.0))
      self.assertAllClose([40.0, 28.0], g.get_concrete_function(2.0)(2.0))

  def testWhileLoopWithUnmodifiedCarriedShape(self):
    with ops.device('device:{}:0'.format(self.device)):
      signature = [tensor.TensorSpec(shape=[None], dtype=dtypes.float32)]

      # We define a signature that specifies unknown vector shape, then test
      # that tf.shape constness gets properly propagated into the while_loop
      # even when carried as part of the loop state.
      @polymorphic_function.function(
          input_signature=signature, jit_compile=True)
      def g(x):
        return while_loop.while_loop_v2(
            lambda *_: True,
            lambda y, shp: (y + random_ops.random_normal(shp)**2, shp),
            (x, array_ops.shape(x)),
            maximum_iterations=3)[0]

      self.assertAllGreater(g(array_ops.zeros([7])), 0.)

  def testNestedWhileLoopWithUnmodifiedCarriedShape(self):
    with ops.device('device:{}:0'.format(self.device)):
      signature = [tensor.TensorSpec(shape=[None], dtype=dtypes.float32)]

      @polymorphic_function.function(
          input_signature=signature, jit_compile=True)
      def g(x):

        def inner(z, shp):
          return z + random_ops.random_normal(shp)**2, shp

        def outer(y, shp):
          y, shp = while_loop.while_loop_v2(
              lambda *_: True, inner, (y, shp), maximum_iterations=3)
          y, shp = array_ops.identity_n([y, shp])
          return while_loop.while_loop_v2(
              lambda *_: True, inner, (y, shp), maximum_iterations=5)

        shp = array_ops.shape(x, name='x_shp')
        return while_loop.while_loop_v2(
            lambda *_: True, outer, (x, shp), maximum_iterations=4)[0]

      self.assertAllGreater(g(array_ops.zeros([7])), 0.)

  def testNestedWhileLoopWithUnmodifiedCarriedShapeSlice(self):
    with ops.device('device:{}:0'.format(self.device)):
      signature = [
          tensor.TensorSpec(shape=[None, None], dtype=dtypes.float32)
      ]

      @polymorphic_function.function(
          input_signature=signature, jit_compile=True)
      def g(x):

        def inner(z, shp):
          return z + random_ops.random_normal(shp)**2, shp

        def outer(y, shp):
          y, shp = while_loop.while_loop_v2(
              lambda *_: True, inner, (y, shp), maximum_iterations=3)
          return while_loop.while_loop_v2(
              lambda *_: True, inner, (y, shp), maximum_iterations=4)

        shp = array_ops.shape(x, name='x_shp')
        x = while_loop.while_loop_v2(
            lambda *_: True, outer, (x, shp), maximum_iterations=5)[0]

        shp2 = array_ops.shape(x, name='x_shp_after')[1:]
        w = while_loop.while_loop_v2(
            lambda *_: True,
            outer, (array_ops.zeros_like(x[0]), shp2),
            maximum_iterations=6)[0]
        return x + w

      self.assertAllGreater(g(array_ops.zeros([7, 13])), 0.)

  def testMethodCompilation(self):

    with ops.device('device:{}:0'.format(self.device)):

      class C(object):

        @polymorphic_function.function(jit_compile=True)
        def f1(self, x, a):
          return x + a

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      c = C()
      self.assertAllClose([2, 3, 3, 4, 4], c.f1(inputs, 1))

  def testMethodCompilationUnsupportedFunc(self):
    with ops.device('device:{}:0'.format(self.device)):

      class C(object):

        @polymorphic_function.function(jit_compile=True)
        def f1(self, x):
          return string_ops.string_length(
              string_ops.string_format('{}', x))

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      c = C()
      with self.assertRaisesRegex(
          errors.InvalidArgumentError, 'unsupported operations'
      ):
        c.f1(inputs)

  def testMustBeConstantPropagation(self):
    if 'tpu' in self.device.lower():
      self.skipTest('b/162799319: Cannot resolve constant on TPU')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f():
        return constant_op.constant([0, 2, 1], dtype=dtypes.int32)

      @polymorphic_function.function(jit_compile=True)
      def g(a, b):
        return array_ops.transpose(a, b)

      @polymorphic_function.function
      def z():
        return g(array_ops.ones([3, 4, 3], dtype=dtypes.float32), f())

      z()

  def testArgMinMax(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def argmax(x):
        return math_ops.argmax(x)

      @polymorphic_function.function(jit_compile=True)
      def argmin(x):
        return math_ops.argmin(x)

      self.assertAllClose(0, argmax(array_ops.ones([10], dtype=dtypes.float32)))
      self.assertAllClose(0, argmax(array_ops.ones([10])))
      self.assertAllClose(0, argmin(array_ops.ones([10], dtype=dtypes.float32)))
      self.assertAllClose(0, argmin(array_ops.ones([10])))

  @test_util.disable_mlir_bridge('TensorArray support not implemented')
  def testErrorMessagePassingTensorArray(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=1, element_shape=[])
        ta = ta.write(0, 2 * x)
        y = ta.read(0)
        return y

      x = constant_op.constant(3.14)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        with self.assertRaisesRegex(errors.UnimplementedError,
                                    'TensorList crossing the XLA/TF boundary'):
          y = f(x)
          tape.gradient(y, x)

  @test_util.disable_mlir_bridge('TODO(b/162281863): MLIR bridge errors out'
                                 ' lowering TensorListConcatV2')
  def testTensorListConcatV2(self):
    with ops.device('device:{}:0'.format(self.device)):

      def f(x):
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, element_shape=[3])
        ta = ta.write(0, 2 * x)
        ta = ta.write(1, 3 * x)
        return ta.concat()

      compiled_f = polymorphic_function.function(jit_compile=True)(f)

      inputs = constant_op.constant([3.14, 2.68, 7.69])

      self.assertAllClose([6.28, 5.36, 15.38, 9.42, 8.04, 23.07], f(inputs))

      self.assertAllClose(compiled_f(inputs), f(inputs))

  @test_util.disable_mlir_bridge('TODO(b/162281863): MLIR bridge errors out'
                                 ' lowering TensorListConcatV2')
  def testTensorListConcatV2Multidim(self):
    with ops.device('device:{}:0'.format(self.device)):

      def f(x):
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, element_shape=[3, 2])
        ta = ta.write(0, 2 * x)
        ta = ta.write(1, 3 * x)
        return ta.concat()

      compiled_f = polymorphic_function.function(jit_compile=True)(f)

      inputs = constant_op.constant([[3.14, 21.1], [2.68, 22.2], [7.69, 23.3]])
      self.assertAllClose(f(inputs), compiled_f(inputs))

  @test_util.disable_mlir_bridge('TODO(b/162281863): MLIR bridge errors out'
                                 ' lowering TensorListConcatV2')
  def testTensorListConcatV2Scalars(self):
    with ops.device('device:{}:0'.format(self.device)):

      def f(x):
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, element_shape=[1])
        ta = ta.write(0, 2 * x)
        ta = ta.write(1, 3 * x)
        return ta.concat()

      compiled_f = polymorphic_function.function(jit_compile=True)(f)
      inputs = constant_op.constant([3.14])
      self.assertAllClose(f(inputs), compiled_f(inputs))

  @test_util.disable_mlir_bridge('TODO(b/162281863): MLIR bridge errors out'
                                 ' lowering TensorListConcatV2')
  def testTensorListConcatGrad(self):
    with ops.device('device:{}:0'.format(self.device)):

      def f(x):
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, element_shape=[3])
        ta = ta.write(0, 2 * x)
        ta = ta.write(1, 3 * x)
        return ta.concat()

      def g():
        x = constant_op.constant([3.14, 2.68, 7.69])
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = f(x)
          return tape.gradient(y, x)

      compiled_g = polymorphic_function.function(jit_compile=True)(g)

      self.assertAllClose([5.0, 5.0, 5.0], g())
      self.assertAllClose(compiled_g(), g())

  @test_util.disable_mlir_bridge('TODO(b/162281863): MLIR bridge errors out'
                                 ' lowering TensorListConcatV2')
  def testTensorListConcatGradNestedCompile(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, element_shape=[3])
        ta = ta.write(0, 2 * x)
        ta = ta.write(1, 3 * x)
        return ta.concat()

      @polymorphic_function.function(jit_compile=True)
      def g():
        x = constant_op.constant([3.14, 2.68, 7.69])
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = f(x)
          out = tape.gradient(y, x)
        return out

      self.assertAllClose([5.0, 5.0, 5.0], g())

  def testCumsum(self):
    if 'tpu' in self.device.lower():
      self.skipTest('b/162771302: 64bit rewrite of cumsum not supported')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        return math_ops.cumsum(x)

      f64_input = constant_op.constant([1.1, 2.2, 3.3], dtype=dtypes.float64)
      self.assertAllClose([1.1, 3.3, 6.6], f(f64_input))

  def testNoExcessiveRetracing(self):
    with ops.device('device:{}:0'.format(self.device)):
      inner_retracings = 0

      @polymorphic_function.function(jit_compile=True)
      def inner(a, b):
        nonlocal inner_retracings
        inner_retracings += 1
        return a * b + a

      def outer(a, b):
        return inner(a, b)

      func_input = random_ops.random_normal([10, 10])
      for _ in range(2):
        polymorphic_function.function(outer)(func_input, func_input)

      self.assertEqual(inner_retracings, 1)

  def testUpdateVariable(self):
    with ops.device('device:{}:0'.format(self.device)):
      v = variables.Variable([0.0, 0.0])

      @polymorphic_function.function(jit_compile=True)
      def f():
        v.assign([3.1, 2.3])

      f()
      self.assertAllClose(v, [3.1, 2.3])

  @test_util.disable_mlir_bridge('MLIR does not support resource update for'
                                 ' signature with compile-time constant.')
  def testUniqueDifferentSizes(self):
    if not 'gpu' in self.device.lower():
      self.skipTest('Currently works only on GPU')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x, y):
        return array_ops.unique(x).y + array_ops.unique(y).y

      f(constant_op.constant([3.1, 3.2]), constant_op.constant([3.3, 3.2]))

      with self.assertRaisesRegex(errors.InternalError, 'different size'):
        f(
            constant_op.constant([3.1, 3.2]),
            constant_op.constant([3.1, 3.2, 3.3]))

  def testUniqueCompilability(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        return array_ops.unique(x).y

      self.assertAllClose(f(constant_op.constant([3.1, 3.2, 3.2])), [3.1, 3.2])

  def testUpdateVariableMemoryUsage(self):
    with ops.device('device:{}:0'.format(self.device)):

      on_gpu = 'gpu' in self.device.lower()
      v = variables.Variable([3.1, 3.2])

      @polymorphic_function.function(jit_compile=True)
      def update_var(a, b):
        v.assign_add(a * b)

      arg1 = random_ops.random_normal([2])
      arg2 = random_ops.random_normal([2])

      gc.collect()
      initial_usage = context.context().get_memory_info(
          v.device)['current'] if on_gpu else 0
      update_var(arg1, arg2)
      gc.collect()
      final_usage = context.context().get_memory_info(
          v.device)['current'] if on_gpu else 0
      self.assertEqual(initial_usage, final_usage)

  @test_util.disable_mlir_bridge('TODO(b/162381930): MLIR bridge renames '
                                 ' functions')
  def testUpdateVariableInClass(self):
    with ops.device('device:{}:0'.format(self.device)):

      class C(object):

        @polymorphic_function.function(jit_compile=True)
        def update_var(self, a, b):
          if not hasattr(self, 'v'):
            self.v = variables.Variable(3.1)
          self.v.assign_add(a * b)

      c = C()

      @polymorphic_function.function
      def outer():
        c.update_var(constant_op.constant(0.7), constant_op.constant(0.6))

      outer()
      self.assertAllClose(c.v, 3.52)

  def testUpdateVariableMultipleOutputs(self):
    with ops.device('device:{}:0'.format(self.device)):
      v = variables.Variable(3.1)

      @polymorphic_function.function(jit_compile=True)
      def update_var(a, b):
        v.assign_add(a * b)
        return a * b + v

      out = update_var(constant_op.constant(0.7), constant_op.constant(0.6))
      self.assertAllClose(v, 3.52)
      self.assertAllClose(out, 3.94)

  def testReturnIdentity(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        return (a, b)

      a = random_ops.random_normal([10, 10])
      b = random_ops.random_normal([10, 10])

      on_gpu = 'gpu' in self.device.lower()
      gc.collect()
      initial_usage = context.context().get_memory_info(
          b.backing_device)['current'] if on_gpu else 0

      f(a, b)

      gc.collect()
      final_usage = context.context().get_memory_info(
          b.backing_device)['current'] if on_gpu else 0
      self.assertEqual(initial_usage, final_usage)

  def testGetCompilerIrConstants(self):
    if 'tpu' in self.device.lower():
      self.skipTest('TPU generates different HLO')

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        return array_ops.transpose(a, b)

      a = array_ops.ones([3, 4, 3], dtype=dtypes.float32)
      b = constant_op.constant([0, 2, 1], dtype=dtypes.int32)

      self.assertIn('{2,1,0}',
                    f.experimental_get_compiler_ir(a, b)(stage='optimized_hlo'))

  @test_util.disable_mlir_bridge('TODO(b/168732524): MLIR bridge does not '
                                 ' optimize single-element tuples to scalars')
  def testGetCompilerIrResourceVars(self):
    with ops.device('device:{}:0'.format(self.device)):

      v = variables.Variable([3.1, 3.2])

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        v.assign_add(a * b)

      a = random_ops.random_normal([2])
      b = random_ops.random_normal([2])

      self.assertIn('input_output_alias={ {}: (2, {}, may-alias) }',
                    f.experimental_get_compiler_ir(a, b)('optimized_hlo'))

  def testGetCompilerIrNotCompiled(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function
      def f(x):
        return x + 1

      a = random_ops.random_normal([10, 10])
      with self.assertRaisesRegex(ValueError,
                                  'marked with \'jit_compile'):
        f.experimental_get_compiler_ir(a)()

  def testGetCompilerIrNested(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x, a):
        return x + a

      @polymorphic_function.function(jit_compile=False)
      def fn2(x, a):
        fn.experimental_get_compiler_ir(x, a)()
        return fn(x, a)

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      with self.assertRaises(TypeError):
        fn2(inputs, 1)

  def testGetCompilerIrKwargs(self):
    with ops.device('device:{}:0'.format(self.device)):

      v = variables.Variable([0.1, 0.1])

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        return (a + b) * v

      a = constant_op.constant([1.1, 1.1])
      b = constant_op.constant([2.2, 2.2])

      self.assertIn('multiply',
                    f.experimental_get_compiler_ir(b=a, a=b)(stage='hlo'))

  def testGetCompilerIrDot(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        return a + b

      a = constant_op.constant([1.1, 1.1])
      b = constant_op.constant([2.2, 2.2])

      self.assertIn(
          'label',
          f.experimental_get_compiler_ir(a, b)(stage='optimized_hlo_dot'))
      self._compareTwoMethodsCompilerIROutput(f, [a, b], {})

  def testGetCompilerIrNoDevicePlacement(self):
    if 'gpu' not in self.device.lower():
      self.skipTest('Testing get_compiler_ir on GPUs without placement')

    @polymorphic_function.function(jit_compile=True)
    def f(a, b):
      return a + b

    a = constant_op.constant([1.1, 1.1])
    b = constant_op.constant([2.2, 2.2])

    self.assertIn(
        'label',
        f.experimental_get_compiler_ir(a, b)(stage='optimized_hlo_dot'))
    self._compareTwoMethodsCompilerIROutput(f, [a, b], {})

  def testGetCompilerIrNonTensors(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(l):
        return l[0] + l[1]

      l = [constant_op.constant(1.1), constant_op.constant(2.2)]

      self.assertIn('tuple',
                    f.experimental_get_compiler_ir(l)())
      self._compareTwoMethodsCompilerIROutput(f, [l], {})

  def testGetCompilerIrSerialized(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def fn(x):
        return x - x

      inputs = constant_op.constant([1, 2, 2, 3, 3])
      for stage in ('hlo_serialized', 'optimized_hlo_serialized'):
        hlo = fn.experimental_get_compiler_ir(inputs)(
            stage=stage, device_name=f'/device:{self.device}:0')
        self.assertIsInstance(hlo, bytes)
      self._compareTwoMethodsCompilerIROutput(fn, [inputs], {})

  def testDotOptimizedHlo(self):
    with ops.device('device:{}:0'.format(self.device)):

      a = random_ops.random_normal([100, 100])
      b = random_ops.random_normal([100, 100])

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        return math_ops.matmul(a, b)

      if not test_util.IsMklEnabled():
        self.assertRegex(
            f.experimental_get_compiler_ir(a, b)('optimized_hlo'),
            '(dot)|(convolution)',
        )
      else:
        self.assertRegex(
            f.experimental_get_compiler_ir(a, b)('optimized_hlo'),
            '(dot)|(convolution)|(custom-call)',
        )

  def testConstantOnWrongDevice(self):
    with ops.device('device:{}:0'.format(self.device)):

      s = random_ops.random_uniform([2], 1, 10, dtypes.int32)
      l = random_ops.random_normal([s[0] * s[1]])

      @polymorphic_function.function(jit_compile=True)
      def f(l):
        return array_ops.reshape(l, s)

      self.assertIn('tuple',
                    f.experimental_get_compiler_ir(l)())

  @test_util.disable_mlir_bridge('TODO(b/172845417): MLIR bridge does not '
                                 'support getting constants out of resources')
  def testGetConstantOutOfResourceVariable(self):
    with ops.device('device:{}:0'.format(self.device)):

      # Use floats to force device placement.
      a = variables.Variable(50.0)
      b = variables.Variable(2.0)

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        return array_ops.reshape(
            x, [math_ops.cast(a, dtypes.int32),
                math_ops.cast(b, dtypes.int32)])

      # OK since the value is known at compile time.
      out = f(random_ops.random_normal([10, 10]))
      self.assertEqual(out.shape[0], 50)
      self.assertEqual(out.shape[1], 2)

  @test_util.disable_mlir_bridge('TODO(b/172845417): MLIR bridge does not '
                                 'support getting constants out of resources')
  def testGetConstantOutOfResourceVariableAfterWrite(self):
    with ops.device('device:{}:0'.format(self.device)):

      # Use floats to force device placement.
      a = variables.Variable(50.0)
      b = variables.Variable(2.0)

      @polymorphic_function.function(jit_compile=True)
      def f(x, val1, val2):
        a.assign(math_ops.cast(val1, dtypes.float32))
        b.assign(math_ops.cast(val2, dtypes.float32))
        return array_ops.reshape(
            x, [math_ops.cast(a, dtypes.int32),
                math_ops.cast(b, dtypes.int32)])

      val1 = constant_op.constant(2)
      val2 = constant_op.constant(50)

      # Returns an error, since the value known at compile time was overriden.
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'concrete values at compile time'):
        f(random_ops.random_normal([10, 10]), val1, val2)

  @test_util.disable_mlir_bridge('TODO(b/172845417): MLIR bridge does not '
                                 'support getting constants out of resources')
  def testGetConstantOutOfResourceVariableBeforeWrite(self):
    with ops.device('device:{}:0'.format(self.device)):

      # Use floats to force device placement.
      a = variables.Variable(50.0)
      b = variables.Variable(2.0)

      @polymorphic_function.function(jit_compile=True)
      def f(x, val1, val2):
        out = array_ops.reshape(
            x, [math_ops.cast(a, dtypes.int32),
                math_ops.cast(b, dtypes.int32)])
        a.assign(math_ops.cast(val1, dtypes.float32))
        b.assign(math_ops.cast(val2, dtypes.float32))
        return out

      val1 = constant_op.constant(2)
      val2 = constant_op.constant(50)

      # OK since the write happens after the reshape.
      out = f(random_ops.random_normal([10, 10]), val1, val2)
      self.assertEqual(out.shape[0], 50)
      self.assertEqual(out.shape[1], 2)

  def testTfAssert(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        control_flow_assert.Assert(x == 1, ['Wrong value'])

      f(constant_op.constant(1))

  def testTensorArrayErrorMessage(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def failure_fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            size=2,
            dynamic_size=True,
            element_shape=(None,),
        )
        return ta.concat()

      with self.assertRaisesRegex(errors.InvalidArgumentError, 'failure_fn'):
        failure_fn()

  def testCounter(self):
    cell_nojit = polymorphic_function._tf_function_counter.get_cell('0')
    cell_jit = polymorphic_function._tf_function_counter.get_cell('1')
    orig_nojit = cell_nojit.value()
    orig_jit = cell_jit.value()

    with ops.device('device:{}:0'.format(self.device)):
      @polymorphic_function.function
      def f(a):
        return a + a
      f(constant_op.constant(1))
      self.assertEqual(cell_nojit.value(), orig_nojit + 1)
      self.assertEqual(cell_jit.value(), orig_jit)
      f(constant_op.constant(1.))  # Calling again does not increment
      self.assertEqual(cell_nojit.value(), orig_nojit + 1)

      @polymorphic_function.function(jit_compile=True)
      def f1(a):
        return a + a
      f1(constant_op.constant(1))
      self.assertEqual(cell_nojit.value(), orig_nojit + 1)
      self.assertEqual(cell_jit.value(), orig_jit + 1)

      @polymorphic_function.function
      def f2(a):
        @polymorphic_function.function
        def g(a):
          return a + a
        @polymorphic_function.function(jit_compile=True)
        def h(a):
          return a + a
        return g(a) + h(a)
      f2(constant_op.constant(1))
      self.assertEqual(cell_nojit.value(), orig_nojit + 2)
      self.assertEqual(cell_jit.value(), orig_jit + 2)

      @polymorphic_function.function(jit_compile=True)
      def f3(a):
        @polymorphic_function.function
        def g(a):
          return a + a
        @polymorphic_function.function(jit_compile=True)
        def h(a):
          return a + a
        return g(a) + h(a)
      f3(constant_op.constant(1))
      self.assertEqual(cell_nojit.value(), orig_nojit + 2)
      self.assertEqual(cell_jit.value(), orig_jit + 3)

  @test_util.disable_mlir_bridge('TODO(b/162272821): MLIR bridge returns '
                                 ' wrong status type')
  def testResourceWrongDevice(self):
    if 'gpu' not in self.device.lower():
      self.skipTest('Need a GPU to have non-trivial device placement')

    with ops.device('device:CPU:0'):
      v = variables.Variable([3.1, 3.2])

    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(experimental_compile=True)
      def update_var(a):
        v.assign_add(a)

      arg = random_ops.random_normal([2])
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'Trying to access resource .*'):
        update_var(arg)

  def testMustBeConstantInsideCondition(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x, d):
        if math_ops.reduce_all(
            math_ops.greater(x, random_ops.random_normal([10, 10]))):
          return array_ops.reshape(x * 2, constant_op.constant([100]))
        else:
          return array_ops.reshape(x * 3, d)

      f(random_ops.random_normal([10, 10]), constant_op.constant([100]))

  def testConditionalGradientTapeMathRegression(self):
    with ops.device('device:{}:0'.format(self.device)):
      with backprop.GradientTape():

        @polymorphic_function.function(jit_compile=True, autograph=False)
        def f(x):
          return cond.cond(
              math_ops.reduce_all(x > 1), lambda: 1. / x, lambda: x)

        v = variables.Variable([[2.]])
        self.assertAllClose(f(v), constant_op.constant([[0.5]]))

  @test_util.disable_mlir_bridge('TODO(b/190444466): MLIR bridge seems to '
                                 'ignore resource assignments')
  def testErrMsgAssignWrongShape(self):
    with ops.device('device:{}:0'.format(self.device)):

      v = variables.Variable([3.1, 3.2])

      @polymorphic_function.function(jit_compile=True)
      def failure_fn(samples):
        v.assign(array_ops.zeros(samples))

      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          'Shape .* cannot be changed after initialization'):
        failure_fn(constant_op.constant(6))

      with self.assertRaisesRegex(errors.InvalidArgumentError, 'failure_fn'):
        failure_fn(constant_op.constant(6))

  def testTfSummaryErrMsg(self):
    if 'gpu' not in self.device.lower():
      self.skipTest('Only runs on GPU')

    with ops.device('device:{}:0'.format(self.device)):
      writer = summary_ops_v2.create_file_writer(self.get_temp_dir())

      @polymorphic_function.function(jit_compile=True)
      def my_func_temp():
        with writer.as_default():
          summary_ops_v2.scalar('my_metric', 0.5, step=10)

      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'Trying to access resource .*'):
        my_func_temp()

  def testSinglePassArgmax(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def f(x):
        return math_ops.argmax(x)

      inputs = array_ops.ones([10], dtype=dtypes.float32)
      hlo = f.experimental_get_compiler_ir(inputs)(stage='hlo')

      # Test that reduction occurs only once.
      self.assertGreater(hlo.count('reduce'), 1)
      self._compareTwoMethodsCompilerIROutput(f, [inputs], {})

  def testExperimentalGetCompilerIRBasic(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True)
      def inner_tf_func(x):
        return math_ops.sin(x)

      x = constant_op.constant([2.0, 3.0])
      self._compareTwoMethodsCompilerIROutput(inner_tf_func, [x], {})

  def testExperimentalGetCompilerIRAutograph(self):
    with ops.device('device:{}:0'.format(self.device)):

      @polymorphic_function.function(jit_compile=True, autograph=True)
      def f(x, y):
        if x[0] > 1:
          return y[0]
        else:
          return y[1]

      x, y = constant_op.constant([2, 3]), constant_op.constant([2, 3])
      self._compareTwoMethodsCompilerIROutput(f, [x, y], {})


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
