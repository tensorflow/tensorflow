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
"""Tests for Keras TF utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class KerasTfUtilsConstantValueTest(test_util.TensorFlowTestCase):

  def test_pred_boolean(self):
    with ops.Graph().as_default():
      x = tf_utils.smart_cond(True,
                              lambda: constant_op.constant(5),
                              lambda: constant_op.constant(32))
      self.assertEqual(tf_utils.constant_value(x), 5)

      x = tf_utils.smart_cond(False,
                              lambda: constant_op.constant(5),
                              lambda: constant_op.constant(32))
      self.assertEqual(tf_utils.constant_value(x), 32)

  def test_pred_integer(self):
    with ops.Graph().as_default():
      x = tf_utils.smart_cond(1,
                              lambda: constant_op.constant(5),
                              lambda: constant_op.constant(32))
      self.assertEqual(tf_utils.constant_value(x), 5)

      x = tf_utils.smart_cond(0,
                              lambda: constant_op.constant(5),
                              lambda: constant_op.constant(32))
      self.assertEqual(tf_utils.constant_value(x), 32)

  def test_pred_unknown(self):
    with ops.Graph().as_default():
      pred = array_ops.placeholder_with_default(True, shape=())
      x = tf_utils.smart_cond(pred,
                              lambda: constant_op.constant(1),
                              lambda: constant_op.constant(2))
      self.assertIsNone(tf_utils.constant_value(x))

  def test_pred_type_error(self):
    with ops.Graph().as_default():
      with self.assertRaises(TypeError):
        tf_utils.constant_value(5)


class KerasTfUtilsGetReachableFromInputsTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testGetReachableFromInputs(self):

    with self.cached_session():
      pl_1 = array_ops.placeholder(shape=None, dtype='float32')
      pl_2 = array_ops.placeholder(shape=None, dtype='float32')
      pl_3 = array_ops.placeholder(shape=None, dtype='float32')
      x_1 = pl_1 + pl_2
      x_2 = pl_2 * 2
      x_3 = pl_3 + 1
      x_4 = x_1 + x_2
      x_5 = x_3 * pl_1

      self.assertEqual(
          keras.utils.tf_utils.get_reachable_from_inputs([pl_1]),
          {pl_1, x_1, x_4, x_5, x_1.op, x_4.op, x_5.op})
      self.assertEqual(
          keras.utils.tf_utils.get_reachable_from_inputs([pl_1, pl_2]),
          {pl_1, pl_2, x_1, x_2, x_4, x_5, x_1.op, x_2.op, x_4.op, x_5.op})
      self.assertEqual(
          keras.utils.tf_utils.get_reachable_from_inputs([pl_3]),
          {pl_3, x_3, x_5, x_3.op, x_5.op})
      self.assertEqual(
          keras.utils.tf_utils.get_reachable_from_inputs([x_3]),
          {x_3, x_5, x_5.op})


@test_util.run_all_in_graph_and_eager_modes
class TestIsSymbolicTensor(test.TestCase):

  def test_default_behavior(self):
    if context.executing_eagerly():
      self.assertFalse(tf_utils.is_symbolic_tensor(
          variables.Variable(name='blah', initial_value=0.)))
      self.assertFalse(tf_utils.is_symbolic_tensor(
          ops.convert_to_tensor(0.)))
      self.assertFalse(tf_utils.is_symbolic_tensor(
          sparse_tensor.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])))
    else:
      self.assertTrue(tf_utils.is_symbolic_tensor(
          variables.Variable(name='blah', initial_value=0.)))
      self.assertTrue(tf_utils.is_symbolic_tensor(
          ops.convert_to_tensor(0.)))
      self.assertTrue(tf_utils.is_symbolic_tensor(
          sparse_tensor.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])))

  def test_works_with_registered(self):

    class CustomClass(object):

      def value(self):
        return ops.convert_to_tensor(42.)

    ops.register_tensor_conversion_function(
        CustomClass, lambda value, **_: value.value())

    tf_utils.register_symbolic_tensor_type(CustomClass)

    if context.executing_eagerly():
      self.assertFalse(tf_utils.is_symbolic_tensor(
          variables.Variable(name='blah', initial_value=0.)))
      self.assertFalse(tf_utils.is_symbolic_tensor(
          ops.convert_to_tensor(0.)))
      self.assertFalse(tf_utils.is_symbolic_tensor(
          sparse_tensor.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])))
      self.assertFalse(tf_utils.is_symbolic_tensor(CustomClass()))
    else:
      self.assertTrue(tf_utils.is_symbolic_tensor(
          variables.Variable(name='blah', initial_value=0.)))
      self.assertTrue(tf_utils.is_symbolic_tensor(
          ops.convert_to_tensor(0.)))
      self.assertTrue(tf_utils.is_symbolic_tensor(
          sparse_tensor.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])))
      self.assertTrue(tf_utils.is_symbolic_tensor(CustomClass()))

  def test_enables_nontensor_plumbing(self):
    # Setup.

    class Foo(object):

      def __init__(self, input_):
        self._input = input_
        self.value = ops.convert_to_tensor(42.)

    ops.register_tensor_conversion_function(
        Foo, lambda x, *args, **kwargs: x.value)
    tf_utils.register_symbolic_tensor_type(Foo)

    class PlumbingLayer(keras.layers.Lambda):

      def __init__(self, fn, **kwargs):
        def _fn(*fargs, **fkwargs):
          d = fn(*fargs, **fkwargs)
          x = ops.convert_to_tensor(d)
          d.shape = x.shape
          d.get_shape = x.get_shape
          return d, x
        super(PlumbingLayer, self).__init__(_fn, **kwargs)
        self._enter_dunder_call = False

      def __call__(self, inputs, *args, **kwargs):
        self._enter_dunder_call = True
        d, _ = super(PlumbingLayer, self).__call__(inputs, *args, **kwargs)
        self._enter_dunder_call = False
        return d

      def call(self, inputs, *args, **kwargs):
        d, v = super(PlumbingLayer, self).call(inputs, *args, **kwargs)
        if self._enter_dunder_call:
          return d, v
        return d

    # User-land.
    model = keras.Sequential([
        keras.layers.InputLayer([]),
        PlumbingLayer(Foo),  # Makes a `Foo` object.
    ])
    # Let's ensure Keras graph history is preserved by composing the models.
    model = keras.Model(model.inputs, model(model.outputs))
    # Now we instantiate the model and verify we have a `Foo` object, not a
    # `Tensor`.
    y = model(ops.convert_to_tensor(7.))
    self.assertIsInstance(y, Foo)


class ConvertInnerNodeDataTest(test.TestCase):

  def test_convert_inner_node_data(self):
    data = tf_utils.convert_inner_node_data((tf_utils.ListWrapper(['l', 2, 3]),
                                             tf_utils.ListWrapper(['l', 5, 6])))
    self.assertEqual(data, (['l', 2, 3], ['l', 5, 6]))

    data = tf_utils.convert_inner_node_data(((['l', 2, 3], ['l', 5, 6])),
                                            wrap=True)
    self.assertTrue(all(isinstance(ele, tf_utils.ListWrapper) for ele in data))


if __name__ == '__main__':
  test.main()
