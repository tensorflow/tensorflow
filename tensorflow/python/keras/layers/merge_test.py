# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for merge layers."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class MergeLayersTest(keras_parameterized.TestCase):

  def test_merge_add(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 5))
    i3 = keras.layers.Input(shape=(4, 5))

    add_layer = keras.layers.Add()
    o = add_layer([i1, i2, i3])
    self.assertListEqual(o.shape.as_list(), [None, 4, 5])
    model = keras.models.Model([i1, i2, i3], o)
    model.run_eagerly = testing_utils.should_run_eagerly()

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    x3 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2, x3])
    self.assertEqual(out.shape, (2, 4, 5))
    self.assertAllClose(out, x1 + x2 + x3, atol=1e-4)

    self.assertEqual(
        add_layer.compute_mask([i1, i2, i3], [None, None, None]), None)
    self.assertTrue(
        np.all(
            backend.eval(
                add_layer.compute_mask(
                    [i1, i2], [backend.variable(x1), backend.variable(x2)]))))

    with self.assertRaisesRegex(ValueError, '`mask` should be a list.'):
      add_layer.compute_mask([i1, i2, i3], x1)
    with self.assertRaisesRegex(ValueError, '`inputs` should be a list.'):
      add_layer.compute_mask(i1, [None, None, None])
    with self.assertRaisesRegex(ValueError, ' should have the same length.'):
      add_layer.compute_mask([i1, i2, i3], [None, None])

  def test_merge_subtract(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 5))
    i3 = keras.layers.Input(shape=(4, 5))

    subtract_layer = keras.layers.Subtract()
    o = subtract_layer([i1, i2])
    self.assertListEqual(o.shape.as_list(), [None, 4, 5])
    model = keras.models.Model([i1, i2], o)
    model.run_eagerly = testing_utils.should_run_eagerly()

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2])
    self.assertEqual(out.shape, (2, 4, 5))
    self.assertAllClose(out, x1 - x2, atol=1e-4)

    self.assertEqual(subtract_layer.compute_mask([i1, i2], [None, None]), None)
    self.assertTrue(
        np.all(
            backend.eval(
                subtract_layer.compute_mask(
                    [i1, i2], [backend.variable(x1), backend.variable(x2)]))))

    with self.assertRaisesRegex(ValueError, '`mask` should be a list.'):
      subtract_layer.compute_mask([i1, i2], x1)
    with self.assertRaisesRegex(ValueError, '`inputs` should be a list.'):
      subtract_layer.compute_mask(i1, [None, None])
    with self.assertRaisesRegex(ValueError,
                                'layer should be called on exactly 2 inputs'):
      subtract_layer([i1, i2, i3])
    with self.assertRaisesRegex(ValueError,
                                'layer should be called on exactly 2 inputs'):
      subtract_layer([i1])

  def test_merge_multiply(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 5))
    i3 = keras.layers.Input(shape=(4, 5))
    o = keras.layers.multiply([i1, i2, i3])
    self.assertListEqual(o.shape.as_list(), [None, 4, 5])
    model = keras.models.Model([i1, i2, i3], o)
    model.run_eagerly = testing_utils.should_run_eagerly()

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    x3 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2, x3])
    self.assertEqual(out.shape, (2, 4, 5))
    self.assertAllClose(out, x1 * x2 * x3, atol=1e-4)

  def test_merge_average(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 5))
    o = keras.layers.average([i1, i2])
    self.assertListEqual(o.shape.as_list(), [None, 4, 5])
    model = keras.models.Model([i1, i2], o)
    model.run_eagerly = testing_utils.should_run_eagerly()

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2])
    self.assertEqual(out.shape, (2, 4, 5))
    self.assertAllClose(out, 0.5 * (x1 + x2), atol=1e-4)

  def test_merge_maximum(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 5))
    o = keras.layers.maximum([i1, i2])
    self.assertListEqual(o.shape.as_list(), [None, 4, 5])
    model = keras.models.Model([i1, i2], o)
    model.run_eagerly = testing_utils.should_run_eagerly()

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2])
    self.assertEqual(out.shape, (2, 4, 5))
    self.assertAllClose(out, np.maximum(x1, x2), atol=1e-4)

  def test_merge_minimum(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 5))
    o = keras.layers.minimum([i1, i2])
    self.assertListEqual(o.shape.as_list(), [None, 4, 5])
    model = keras.models.Model([i1, i2], o)
    model.run_eagerly = testing_utils.should_run_eagerly()

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2])
    self.assertEqual(out.shape, (2, 4, 5))
    self.assertAllClose(out, np.minimum(x1, x2), atol=1e-4)

  def test_merge_concatenate(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 5))
    concat_layer = keras.layers.Concatenate(axis=1)
    o = concat_layer([i1, i2])
    self.assertListEqual(o.shape.as_list(), [None, 8, 5])
    model = keras.models.Model([i1, i2], o)
    model.run_eagerly = testing_utils.should_run_eagerly()

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2])
    self.assertEqual(out.shape, (2, 8, 5))
    self.assertAllClose(out, np.concatenate([x1, x2], axis=1), atol=1e-4)

    self.assertEqual(concat_layer.compute_mask([i1, i2], [None, None]), None)
    self.assertTrue(
        np.all(
            backend.eval(
                concat_layer.compute_mask(
                    [i1, i2], [backend.variable(x1), backend.variable(x2)]))))

    # Should work with unit-length input.
    unit_length_o = concat_layer([i1])
    self.assertListEqual(unit_length_o.shape.as_list(), i1.shape.as_list())

    with self.assertRaisesRegex(ValueError, '`mask` should be a list.'):
      concat_layer.compute_mask([i1, i2], x1)
    with self.assertRaisesRegex(ValueError, '`inputs` should be a list.'):
      concat_layer.compute_mask(i1, [None, None])
    with self.assertRaisesRegex(ValueError, 'should have the same length'):
      concat_layer.compute_mask([i1, i2], [None])
    with self.assertRaisesRegex(ValueError,
                                'layer should be called on a list of inputs'):
      concat_layer(i1)

  def test_merge_dot(self):
    i1 = keras.layers.Input(shape=(4,))
    i2 = keras.layers.Input(shape=(4,))
    o = keras.layers.dot([i1, i2], axes=1)
    self.assertListEqual(o.shape.as_list(), [None, 1])
    model = keras.models.Model([i1, i2], o)
    model.run_eagerly = testing_utils.should_run_eagerly()
    _ = keras.layers.Dot(axes=1).get_config()

    x1 = np.random.random((2, 4))
    x2 = np.random.random((2, 4))
    out = model.predict([x1, x2])
    self.assertEqual(out.shape, (2, 1))
    expected = np.zeros((2, 1))
    expected[0, 0] = np.dot(x1[0], x2[0])
    expected[1, 0] = np.dot(x1[1], x2[1])
    self.assertAllClose(out, expected, atol=1e-4)

    # Test with negative tuple of axes.
    o = keras.layers.dot([i1, i2], axes=(-1, -1))
    self.assertListEqual(o.shape.as_list(), [None, 1])
    model = keras.models.Model([i1, i2], o)
    model.run_eagerly = testing_utils.should_run_eagerly()
    out = model.predict([x1, x2])
    self.assertEqual(out.shape, (2, 1))
    self.assertAllClose(out, expected, atol=1e-4)

    # test compute_output_shape
    layer = keras.layers.Dot(axes=-1)
    self.assertEqual(layer.compute_output_shape([(4, 5), (4, 5)]), (4, 1))

  @parameterized.named_parameters(
      *testing_utils.generate_combinations_with_testcase_name(
          layer=[keras.layers.Add, keras.layers.Subtract,
                 keras.layers.Multiply, keras.layers.Minimum,
                 keras.layers.Maximum, keras.layers.Average,
                 keras.layers.Concatenate]))
  def test_merge_with_ragged_input(self, layer):
    ragged_data = ragged_factory_ops.constant(
        [[1., 1., 1.], [1., 1.], [1., 1., 1., 1.]], ragged_rank=1)
    dense_data = ragged_data.to_tensor()
    input1 = keras.Input(shape=(None,), ragged=True)
    input2 = keras.Input(shape=(None,), ragged=True)
    out = keras.layers.Add()([input1, input2])
    model = keras.models.Model(inputs=[input1, input2], outputs=out)
    out_ragged = model.predict([ragged_data, ragged_data], steps=1)
    out_ragged = convert_ragged_tensor_value(out_ragged).to_tensor()

    input1 = keras.Input(shape=(None,))
    input2 = keras.Input(shape=(None,))
    out = keras.layers.Add()([input1, input2])
    model = keras.models.Model(inputs=[input1, input2], outputs=out)
    out_dense = model.predict([dense_data, dense_data], steps=1)

    self.assertAllEqual(out_dense, out_ragged)

  @parameterized.named_parameters(
      *testing_utils.generate_combinations_with_testcase_name(
          layer=[keras.layers.Add, keras.layers.Subtract,
                 keras.layers.Multiply, keras.layers.Minimum,
                 keras.layers.Maximum, keras.layers.Average]))
  def test_merge_with_scalar_input(self, layer):
    x1 = np.array((1))
    x2 = np.array((2))
    out = layer()([x1, x2])
    self.assertEqual(out.shape, ())


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MergeLayersTestNoExecution(test.TestCase):

  def test_merge_elementwise_errors(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 6))
    with self.assertRaises(ValueError):
      keras.layers.add([i1, i2])
    with self.assertRaises(ValueError):
      keras.layers.add([i1])
    with self.assertRaises(ValueError):
      keras.layers.add(i1)
    with self.assertRaises(ValueError):
      keras.layers.add([i1])

  def test_concatenate_errors(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(3, 5))
    with self.assertRaisesRegex(ValueError, 'inputs with matching shapes'):
      keras.layers.concatenate([i1, i2], axis=-1)
    with self.assertRaisesRegex(ValueError, 'called on a list'):
      keras.layers.concatenate(i1, axis=-1)

  def test_concatenate_with_partial_shape(self):
    i1 = keras.layers.Input(shape=(5,), batch_size=32)
    i2 = keras.layers.Input(shape=(5,))
    i3 = keras.layers.Input(shape=(4, 5), batch_size=32)
    i4 = keras.layers.Input(shape=(None,), batch_size=64)
    i5 = keras.layers.Input(shape=(7,))

    # Valid case since the i2 has a dynamic batch size.
    keras.layers.concatenate([i1, i2], axis=-1)

    # Different rank
    with self.assertRaisesRegex(ValueError, 'inputs with matching shapes'):
      keras.layers.concatenate([i1, i3], axis=-1)

    # Valid case with partial dimension information
    keras.layers.concatenate([i1, i4], axis=0)
    keras.layers.concatenate([i2, i4], axis=0)
    keras.layers.concatenate([i2, i4], axis=1)
    keras.layers.concatenate([i1, i2, i4], axis=0)
    keras.layers.concatenate([i1, i5], axis=1)

    # Mismatch in batch dimension.
    with self.assertRaisesRegex(ValueError, 'inputs with matching shapes'):
      keras.layers.concatenate([i1, i4], axis=-1)

    with self.assertRaisesRegex(ValueError, 'inputs with matching shapes'):
      keras.layers.concatenate([i1, i2, i4], axis=-1)

  def test_dot_errors(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 6))
    i3 = keras.layers.Input(shape=(4, 6))
    with self.assertRaises(ValueError):
      keras.layers.dot([i1, i2], axes=-1)
    with self.assertRaises(ValueError):
      keras.layers.dot(i1, axes=-1)
    with self.assertRaises(ValueError):
      keras.layers.dot([i1], axes=-1)
    with self.assertRaises(ValueError):
      keras.layers.dot([i1, i2, i3], axes=-1)
    with self.assertRaises(ValueError):
      dot = keras.layers.Dot(1)
      dot.compute_output_shape(1)

  def test_merge_subtract(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 5))
    y = keras.layers.subtract([i1, i2])
    self.assertEqual(y.shape.as_list(), [None, 4, 5])

    # Test invalid use cases
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(3, 5))
    with self.assertRaises(ValueError):
      keras.layers.subtract([i1, i2])
    with self.assertRaises(ValueError):
      keras.layers.subtract([i1, i1, i1])

  def test_merge_add_masking(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 5))
    m1 = keras.layers.Masking()(i1)
    layer = keras.layers.Add()
    o = layer([m1, i2])
    self.assertListEqual(o.shape.as_list(), [None, 4, 5])
    mask = layer.output_mask
    self.assertListEqual(mask.shape.as_list(), [None, 4])

  def test_merge_add_dynamic_shape(self):
    i1 = keras.Input(batch_shape=(4, None), dtype='float32')
    i2 = keras.Input(batch_shape=(4, 5), dtype='float32')
    layer = keras.layers.Add()
    o = layer([i1, i2])
    self.assertListEqual(o.shape.as_list(), [4, 5])

  def test_merge_concatenate_masking(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 5))
    m1 = keras.layers.Masking()(i1)
    layer = keras.layers.Concatenate()
    o = layer([m1, i2])
    self.assertListEqual(o.shape.as_list(), [None, 4, 10])
    mask = layer.output_mask
    self.assertListEqual(mask.shape.as_list(), [None, 4])

  def test_merge_concatenate_sparse_shape(self):
    i1 = keras.layers.Input(shape=(1,), batch_size=2, sparse=True)
    i2 = keras.layers.Input(shape=(2,), batch_size=2, sparse=True)
    layer = keras.layers.Concatenate(axis=1)
    o = layer([i1, i2])
    self.assertListEqual(o.shape.as_list(), [2, 3])

    # Make sure it also respect None as the batch size
    i1 = keras.layers.Input(shape=(1,), sparse=True)
    i2 = keras.layers.Input(shape=(2,), sparse=True)
    layer = keras.layers.Concatenate(axis=1)
    o = layer([i1, i2])
    self.assertListEqual(o.shape.as_list(), [None, 3])

  def test_user_changes_to_input_structure(self):
    a = keras.layers.Input(shape=(4, 5))
    struct = [a, a]
    concat1 = keras.layers.Concatenate(1)
    b = concat1(struct)
    struct.append(b)
    concat2 = keras.layers.Concatenate(1)
    c = concat2(struct)

    # Checks that the append to `struct` doesn't affect `concat1`s
    # node data.
    self.assertLen(concat1.inbound_nodes[0].input_tensors, 2)
    self.assertLen(concat2.inbound_nodes[0].input_tensors, 3)

    keras.Model(a, c)  # Ensure model can be built.


def convert_ragged_tensor_value(inputs):
  if isinstance(inputs, ragged_tensor_value.RaggedTensorValue):
    flat_values = ops.convert_to_tensor_v2_with_dispatch(
        value=inputs.flat_values,
        name='flat_values')
    return ragged_tensor.RaggedTensor.from_nested_row_splits(
        flat_values, inputs.nested_row_splits, validate=False)
  return inputs


if __name__ == '__main__':
  test.main()
