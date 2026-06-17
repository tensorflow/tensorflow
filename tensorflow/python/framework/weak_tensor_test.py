# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for tensorflow.python.framework.weak_tensor."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.framework.weak_tensor import EagerWeakTensor
from tensorflow.python.framework.weak_tensor import GraphWeakTensor
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.module import module
from tensorflow.python.platform import googletest
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.types import core


class WeakTensorTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_weak_tensor_basic(self):
    a = WeakTensor.from_tensor(constant_op.constant(1, dtypes.int32))
    self.assertEqual(a.dtype, dtypes.int32)
    self.assertEqual(a.shape, [])

    b = [1.0, 2.0], [3.0, 4.0]
    b_wt = WeakTensor.from_tensor(constant_op.constant(b, dtypes.float32))
    self.assertEqual(b_wt.dtype, dtypes.float32)
    self.assertEqual(b_wt.shape, [2, 2])

  @test_util.run_in_graph_and_eager_modes
  def test_weak_tensor_init(self):
    # Make sure an exception is thrown for unallowed dtypes.
    t = constant_op.constant(1, dtypes.int16)
    with self.assertRaises(TypeError):
      _ = WeakTensor.from_tensor(t)

  @test_util.run_in_graph_and_eager_modes
  def test_weak_tensor_inheritance(self):
    a = WeakTensor.from_tensor(constant_op.constant([1, 2, 3], dtypes.int32))
    self.assertIsInstance(a, WeakTensor)
    self.assertIsInstance(a, core.Tensor)
    self.assertIsInstance(a, extension_type.ExtensionType)
    if context.executing_eagerly():
      self.assertIsInstance(a, core.Value)
      self.assertIsInstance(a, EagerWeakTensor)
    else:
      self.assertIsInstance(a, core.Symbol)
      self.assertIsInstance(a, GraphWeakTensor)

  def test_weak_tensor_eager_methods(self):
    wt = WeakTensor.from_tensor(constant_op.constant(2, dtypes.int32))
    b = [1.0, 2.0], [3.0, 4.0]
    b_wt = WeakTensor.from_tensor(constant_op.constant(b, dtypes.float32))

    self.assertEqual(complex(wt), complex(2))
    self.assertEqual(int(wt), int(2))
    self.assertEqual(float(wt), float(2))
    self.assertEqual(wt.__index__(), int(2))
    self.assertEqual(wt.numpy(), 2)
    self.assertEqual(format(wt, 'b'), '10 weakly typed')
    self.assertEqual(np.array(wt), 2)
    self.assertAllEqual(np.array(b_wt), np.array(b, dtype=np.float32))

  @test_util.run_in_graph_and_eager_modes
  def test_weak_tensor_bool(self):
    # Test to make sure WeakTensor(bool) isn't used as a bool.
    with self.assertRaises(TypeError):
      if WeakTensor.from_tensor(constant_op.constant(True)):
        raise TypeError('Type error is raised because WeakTensor != bool')

  @test_util.run_in_graph_and_eager_modes
  def test_weak_tensor_getattr(self):
    wt = WeakTensor.from_tensor(constant_op.constant(1, dtypes.int32))
    wt_name = getattr(wt, '__name__', None)
    if context.executing_eagerly():
      self.assertEqual(wt_name, 'tf.EagerWeakTensor')
    else:
      self.assertEqual(wt_name, 'tf.GraphWeakTensor')

  @test_util.run_in_graph_and_eager_modes
  def test_weak_tensor_in_tf_func(self):
    @def_function.function()
    def f(x):
      return x

    t = constant_op.constant(1, dtypes.int32)
    wt = WeakTensor.from_tensor(t)
    res = f(wt)
    self.assertIsInstance(res, WeakTensor)

    _ = f(t)
    self.assertEqual(f.experimental_get_tracing_count(), 2)

  @test_util.run_in_graph_and_eager_modes
  def test_weak_tensor_in_tf_func_with_branch_error(self):
    a = constant_op.constant(1, dtypes.int32)
    b = WeakTensor.from_tensor(a)

    @def_function.function()
    def f(c, a, b):
      if c > 1:
        return a
      else:
        return b

    with self.assertRaises(TypeError):
      # if and else branch cannot return two different types in a tf.function.
      _ = f(constant_op.constant(2, dtypes.int32), a, b)

  @test_util.run_in_graph_and_eager_modes
  def test_weak_tensor_in_tf_func_with_spec(self):
    # Test weak tensor spec with matching input.
    weak_tensor_spec = WeakTensor.Spec(tensor.TensorSpec([2]))
    wt = WeakTensor.from_tensor(constant_op.constant([1.0, 2.0]))

    @def_function.function(input_signature=[weak_tensor_spec])
    def f(x):
      return x

    _ = f(wt)
    # Test weak tensor spec with mismatching input.
    wt_mismatch = WeakTensor.from_tensor(constant_op.constant([1.0, 2.0, 3.0]))
    with self.assertRaises(TypeError):
      _ = f(wt_mismatch)

  @test_util.run_in_graph_and_eager_modes
  def test_weak_tensor_gradient(self):
    x = WeakTensor.from_tensor(constant_op.constant([3.0, 4.0, 5.0]))
    with backprop.GradientTape() as g:
      g.watch(x)
      y = x
    dy_dx = g.gradient(y, x)
    self.assertAllEqual(dy_dx, [1.0, 1.0, 1.0])
    self.assertIsInstance(dy_dx, WeakTensor)

  @test_util.run_in_graph_and_eager_modes
  def test_weak_tensor_in_restored_function(self):
    class CustomModule(module.Module):

      @def_function.function
      def __call__(self, x):
        if isinstance(x, tensor.Tensor):
          raise TypeError('Weak tensor should not be tensor.Tensor type.')
        return x

    m = CustomModule()
    a = WeakTensor.from_tensor(constant_op.constant(1, dtypes.int32))
    _ = m(a)

    save(m, '/tmp/f')
    m_loaded = load('/tmp/f')
    res = m_loaded(a)
    self.assertIsInstance(res, WeakTensor)

    b = constant_op.constant(1, dtypes.int32)
    with self.assertRaisesRegex(
        ValueError, 'Could not find matching concrete function'
    ):
      m_loaded(b)

  def test_weak_tensor_format_to_string(self):
    # __str__ test in eager mode
    t = constant_op.constant([1.0, 2.0], dtypes.float32)
    wt = WeakTensor(t)
    wt_str = 'tf.Tensor([1. 2.], shape=(2,), dtype=float32, weak=True)'
    self.assertEqual(str(wt), wt_str)

    # __repr__ test in eager mode
    wt_repr = (
        '<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 2.],'
        ' dtype=float32), weak=True>'
    )
    self.assertEqual(repr(wt), wt_repr)

    @def_function.function()
    def f():
      # __str__ test in graph mode
      t = constant_op.constant([1.0, 2.0], dtypes.float32)
      wt = WeakTensor(t)
      wt_str = 'Tensor("Const:0", shape=(2,), dtype=float32, weak=True)'
      self.assertEqual(str(wt), wt_str)

      # __repr__ test in graph mode
      wt_repr = "<tf.Tensor 'Const:0' shape=(2,) dtype=float32, weak=True>"
      self.assertEqual(repr(wt), wt_repr)
      return wt

    _ = f()

  def test_weak_tensor_iter(self):
    # Test normal weakTensor iteration.
    t = constant_op.constant([0, 1, 2], dtypes.int32)
    wt = WeakTensor.from_tensor(t)
    it_weak_tensor = iter(wt)
    for i in range(len(wt)):
      self.assertAllEqual(
          next(it_weak_tensor),
          WeakTensor.from_tensor(constant_op.constant(i)),
      )

    # Test multi-dimensional weakTensor iteration.
    t_multi = constant_op.constant([[1, 2], [3, 4]], dtypes.int32)
    wt_multi = WeakTensor(t_multi)
    it_wt_multi_tensor = iter(wt_multi)
    self.assertAllEqual(
        next(it_wt_multi_tensor), WeakTensor.from_tensor(t_multi[0])
    )
    self.assertAllEqual(
        next(it_wt_multi_tensor), WeakTensor.from_tensor(t_multi[1])
    )

    # Test scalar weakTensor iteration.
    t_scalar = constant_op.constant(1, dtypes.int32)
    wt_scalar = WeakTensor.from_tensor(t_scalar)
    with self.assertRaises(TypeError):
      # Cannot iterate over a scalar tensor.
      _ = iter(wt_scalar)

  @test_util.deprecated_graph_mode_only
  def test_weak_tensor_iter_graph_mode(self):
    # Make sure iteration is not allowed in Graph mode.
    wt = WeakTensor.from_tensor(constant_op.constant([0, 1, 2], dtypes.int32))
    with self.assertRaisesRegex(
        errors.OperatorNotAllowedInGraphError,
        'Iterating over a symbolic `tf.WeakTensor` is not allowed. You can'
        ' attempt the following resolutions to the problem: If you are running'
        ' in Graph mode, use Eager execution mode or decorate this function'
        ' with @tf.function. If you are using AutoGraph, you can try decorating'
        ' this function with @tf.function. If that does not work, then you may'
        ' be using an unsupported feature or your source code may not be'
        ' visible to AutoGraph. See'
        ' https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code'
        ' for more information.',
    ):
      _ = iter(wt)


if __name__ == '__main__':
  ops.enable_eager_execution()
  googletest.main()
