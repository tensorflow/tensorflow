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
"""Tests for RNN cell wrapper v2 implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import rnn_cell_wrapper_v2
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


class RNNCellWrapperTest(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testResidualWrapper(self):
    wrapper_type = rnn_cell_wrapper_v2.ResidualWrapper
    x = ops.convert_to_tensor(np.array([[1., 1., 1.]]), dtype="float32")
    m = ops.convert_to_tensor(np.array([[0.1, 0.1, 0.1]]), dtype="float32")
    base_cell = rnn_cell_impl.GRUCell(
        3, kernel_initializer=init_ops.constant_initializer(0.5),
        bias_initializer=init_ops.constant_initializer(0.5))
    g, m_new = base_cell(x, m)
    wrapper_object = wrapper_type(base_cell)
    (name, dep), = wrapper_object._checkpoint_dependencies
    wrapper_object.get_config()  # Should not throw an error
    self.assertIs(dep, base_cell)
    self.assertEqual("cell", name)

    g_res, m_new_res = wrapper_object(x, m)
    self.evaluate([variables_lib.global_variables_initializer()])
    res = self.evaluate([g, g_res, m_new, m_new_res])
    # Residual connections
    self.assertAllClose(res[1], res[0] + [1., 1., 1.])
    # States are left untouched
    self.assertAllClose(res[2], res[3])

  @test_util.run_in_graph_and_eager_modes
  def testResidualWrapperWithSlice(self):
    wrapper_type = rnn_cell_wrapper_v2.ResidualWrapper
    x = ops.convert_to_tensor(np.array([[1., 1., 1., 1., 1.]]), dtype="float32")
    m = ops.convert_to_tensor(np.array([[0.1, 0.1, 0.1]]), dtype="float32")
    base_cell = rnn_cell_impl.GRUCell(
        3, kernel_initializer=init_ops.constant_initializer(0.5),
        bias_initializer=init_ops.constant_initializer(0.5))
    g, m_new = base_cell(x, m)

    def residual_with_slice_fn(inp, out):
      inp_sliced = array_ops.slice(inp, [0, 0], [-1, 3])
      return inp_sliced + out

    g_res, m_new_res = wrapper_type(
        base_cell, residual_with_slice_fn)(x, m)
    self.evaluate([variables_lib.global_variables_initializer()])
    res_g, res_g_res, res_m_new, res_m_new_res = self.evaluate(
        [g, g_res, m_new, m_new_res])
    # Residual connections
    self.assertAllClose(res_g_res, res_g + [1., 1., 1.])
    # States are left untouched
    self.assertAllClose(res_m_new, res_m_new_res)

  def testDeviceWrapper(self):
    wrapper_type = rnn_cell_wrapper_v2.DeviceWrapper
    x = array_ops.zeros([1, 3])
    m = array_ops.zeros([1, 3])
    cell = rnn_cell_impl.GRUCell(3)
    wrapped_cell = wrapper_type(cell, "/cpu:0")
    (name, dep), = wrapped_cell._checkpoint_dependencies
    wrapped_cell.get_config()  # Should not throw an error
    self.assertIs(dep, cell)
    self.assertEqual("cell", name)

    outputs, _ = wrapped_cell(x, m)
    self.assertIn("cpu:0", outputs.device.lower())

  @parameterized.parameters(
      [[rnn_cell_impl.DropoutWrapper, rnn_cell_wrapper_v2.DropoutWrapper],
       [rnn_cell_impl.ResidualWrapper, rnn_cell_wrapper_v2.ResidualWrapper]])
  @test_util.run_in_graph_and_eager_modes
  def testWrapperKerasStyle(self, wrapper, wrapper_v2):
    """Tests if wrapper cell is instantiated in keras style scope."""
    wrapped_cell_v2 = wrapper_v2(rnn_cell_impl.BasicRNNCell(1))
    self.assertIsNone(getattr(wrapped_cell_v2, "_keras_style", None))

    wrapped_cell = wrapper(rnn_cell_impl.BasicRNNCell(1))
    self.assertFalse(wrapped_cell._keras_style)

  @parameterized.parameters(
      [rnn_cell_wrapper_v2.DropoutWrapper, rnn_cell_wrapper_v2.ResidualWrapper])
  @test_util.run_in_graph_and_eager_modes
  def testWrapperV2VariableNames(self, wrapper):
    """Tests that variables names do not depend on wrapper in RNN layer."""

    def _rnn_input(apply_wrapper, name):
      """Creates a RNN layer with/without wrapper and returns built rnn cell."""
      with base_layer.keras_style_scope():
        base_cell = rnn_cell_impl.MultiRNNCell(
            [rnn_cell_impl.BasicRNNCell(1, name="basic_rnn_cell")
             for _ in range(2)])
      if apply_wrapper:
        rnn_cell = wrapper(base_cell)
      else:
        rnn_cell = base_cell
      rnn_layer = layers.RNN(rnn_cell, name=name)
      inputs = ops.convert_to_tensor([[[1]]], dtype=dtypes.float32)
      _ = rnn_layer(inputs)
      return base_cell._cells[0]

    rnn_1 = _rnn_input(True, name="rnn_0")
    rnn_2 = _rnn_input(False, name="rnn_1")

    for i, cell in enumerate([rnn_1, rnn_2]):
      var_prefix = "rnn_{}/cell_0/basic_rnn_cell/".format(i)
      self.assertCountEqual([v.name for v in cell.weights],
                            (var_prefix + "kernel:0", var_prefix + "bias:0"))

  @parameterized.parameters(
      [rnn_cell_wrapper_v2.DropoutWrapper, rnn_cell_wrapper_v2.ResidualWrapper])
  @test_util.run_in_graph_and_eager_modes
  def testWrapperWeights(self, wrapper):
    """Tests that wrapper weights contain wrapped cells weights."""
    base_cell = layers.SimpleRNNCell(1, name="basic_rnn_cell")
    rnn_cell = wrapper(base_cell)
    rnn_layer = layers.RNN(rnn_cell)
    inputs = ops.convert_to_tensor([[[1]]], dtype=dtypes.float32)
    rnn_layer(inputs)

    expected_weights = ["rnn/" + var for var in
                        ("kernel:0", "recurrent_kernel:0", "bias:0")]
    self.assertLen(rnn_cell.weights, 3)
    self.assertCountEqual([v.name for v in rnn_cell.weights], expected_weights)
    self.assertCountEqual([v.name for v in rnn_cell.trainable_variables],
                          expected_weights)
    self.assertCountEqual([v.name for v in rnn_cell.non_trainable_variables],
                          [])
    self.assertCountEqual([v.name for v in rnn_cell.cell.weights],
                          expected_weights)

  @parameterized.parameters(
      [rnn_cell_wrapper_v2.DropoutWrapper, rnn_cell_wrapper_v2.ResidualWrapper])
  @test_util.run_in_graph_and_eager_modes
  def testWrapperV2Caller(self, wrapper):
    """Tests that wrapper V2 is using the LayerRNNCell's caller."""

    with base_layer.keras_style_scope():
      base_cell = rnn_cell_impl.MultiRNNCell(
          [rnn_cell_impl.BasicRNNCell(1) for _ in range(2)])
    rnn_cell = wrapper(base_cell)
    inputs = ops.convert_to_tensor([[1]], dtype=dtypes.float32)
    state = ops.convert_to_tensor([[1]], dtype=dtypes.float32)
    _ = rnn_cell(inputs, [state, state])
    weights = base_cell._cells[0].weights
    self.assertLen(weights, expected_len=2)
    self.assertTrue(all("_wrapper" in v.name for v in weights))

  @parameterized.parameters(
      [rnn_cell_wrapper_v2.DropoutWrapper, rnn_cell_wrapper_v2.ResidualWrapper])
  @test_util.run_in_graph_and_eager_modes
  def testWrapperV2Build(self, wrapper):
    cell = rnn_cell_impl.LSTMCell(10)
    wrapper = wrapper(cell)
    wrapper.build((1,))
    self.assertTrue(cell.built)

  def testDeviceWrapperSerialization(self):
    wrapper_cls = rnn_cell_wrapper_v2.DeviceWrapper
    cell = layers.LSTMCell(10)
    wrapper = wrapper_cls(cell, "/cpu:0")
    config = wrapper.get_config()

    reconstructed_wrapper = wrapper_cls.from_config(config)
    self.assertDictEqual(config, reconstructed_wrapper.get_config())
    self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

  def testResidualWrapperSerialization(self):
    wrapper_cls = rnn_cell_wrapper_v2.ResidualWrapper
    cell = layers.LSTMCell(10)
    wrapper = wrapper_cls(cell)
    config = wrapper.get_config()

    reconstructed_wrapper = wrapper_cls.from_config(config)
    self.assertDictEqual(config, reconstructed_wrapper.get_config())
    self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

    wrapper = wrapper_cls(cell, residual_fn=lambda i, o: i + i + o)
    config = wrapper.get_config()

    reconstructed_wrapper = wrapper_cls.from_config(config)
    # Assert the reconstructed function will perform the math correctly.
    self.assertEqual(reconstructed_wrapper._residual_fn(1, 2), 4)

    def residual_fn(inputs, outputs):
      return inputs * 3 + outputs

    wrapper = wrapper_cls(cell, residual_fn=residual_fn)
    config = wrapper.get_config()

    reconstructed_wrapper = wrapper_cls.from_config(config)
    # Assert the reconstructed function will perform the math correctly.
    self.assertEqual(reconstructed_wrapper._residual_fn(1, 2), 5)

  def testDropoutWrapperSerialization(self):
    wrapper_cls = rnn_cell_wrapper_v2.DropoutWrapper
    cell = layers.GRUCell(10)
    wrapper = wrapper_cls(cell)
    config = wrapper.get_config()

    reconstructed_wrapper = wrapper_cls.from_config(config)
    self.assertDictEqual(config, reconstructed_wrapper.get_config())
    self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

    wrapper = wrapper_cls(cell, dropout_state_filter_visitor=lambda s: True)
    config = wrapper.get_config()

    reconstructed_wrapper = wrapper_cls.from_config(config)
    self.assertTrue(reconstructed_wrapper._dropout_state_filter(None))

    def dropout_state_filter_visitor(unused_state):
      return False

    wrapper = wrapper_cls(
        cell, dropout_state_filter_visitor=dropout_state_filter_visitor)
    config = wrapper.get_config()

    reconstructed_wrapper = wrapper_cls.from_config(config)
    self.assertFalse(reconstructed_wrapper._dropout_state_filter(None))

  def testDroputWrapperWithKerasLSTMCell(self):
    wrapper_cls = rnn_cell_wrapper_v2.DropoutWrapper
    cell = layers.LSTMCell(10)

    with self.assertRaisesRegexp(ValueError, "does not work with "):
      wrapper_cls(cell)

    cell = layers.LSTMCellV2(10)
    with self.assertRaisesRegexp(ValueError, "does not work with "):
      wrapper_cls(cell)


if __name__ == "__main__":
  test.main()
