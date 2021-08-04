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
"""Tests for the attention layer."""

from absl.testing import parameterized

import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers import multi_head_attention
from tensorflow.python.platform import test


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class MultiHeadAttentionTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      ("key_value_same_proj", None, None, [40, 80]),
      ("key_value_different_proj", 32, 60, [40, 60]),
  )
  def test_non_masked_attention(self, value_dim, output_shape, output_dims):
    """Test that the attention layer can be created without a mask tensor."""
    test_layer = multi_head_attention.MultiHeadAttention(
        num_heads=12,
        key_dim=64,
        value_dim=value_dim,
        output_shape=output_shape)
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    value = keras.Input(shape=(20, 80))
    output = test_layer(query=query, value=value)
    self.assertEqual(output.shape.as_list(), [None] + output_dims)

  def test_non_masked_self_attention(self):
    """Test with one input (self-attenntion) and no mask tensor."""
    test_layer = multi_head_attention.MultiHeadAttention(
        num_heads=12, key_dim=64)
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    output = test_layer(query, query)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])

  def test_attention_scores(self):
    """Test attention outputs with coefficients."""
    test_layer = multi_head_attention.MultiHeadAttention(
        num_heads=12, key_dim=64)
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    output, coef = test_layer(query, query, return_attention_scores=True)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])
    self.assertEqual(coef.shape.as_list(), [None, 12, 40, 40])

  def test_attention_scores_with_values(self):
    """Test attention outputs with coefficients."""
    test_layer = multi_head_attention.MultiHeadAttention(
        num_heads=12, key_dim=64)
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    value = keras.Input(shape=(60, 80))
    output, coef = test_layer(query, value, return_attention_scores=True)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])
    self.assertEqual(coef.shape.as_list(), [None, 12, 40, 60])

  @parameterized.named_parameters(("with_bias", True), ("no_bias", False))
  def test_masked_attention(self, use_bias):
    """Test with a mask tensor."""
    test_layer = multi_head_attention.MultiHeadAttention(
        num_heads=2, key_dim=2, use_bias=use_bias)
    # Create a 3-dimensional input (the first dimension is implicit).
    batch_size = 3
    query = keras.Input(shape=(4, 8))
    value = keras.Input(shape=(2, 8))
    mask_tensor = keras.Input(shape=(4, 2))
    output = test_layer(query=query, value=value, attention_mask=mask_tensor)

    # Create a model containing the test layer.
    model = keras.Model([query, value, mask_tensor], output)

    # Generate data for the input (non-mask) tensors.
    from_data = 10 * np.random.random_sample((batch_size, 4, 8))
    to_data = 10 * np.random.random_sample((batch_size, 2, 8))

    # Invoke the data with a random set of mask data. This should mask at least
    # one element.
    mask_data = np.random.randint(2, size=(batch_size, 4, 2))
    masked_output_data = model.predict([from_data, to_data, mask_data])

    # Invoke the same data, but with a null mask (where no elements are masked).
    null_mask_data = np.ones((batch_size, 4, 2))
    unmasked_output_data = model.predict([from_data, to_data, null_mask_data])

    # Because one data is masked and one is not, the outputs should not be the
    # same.
    self.assertNotAllClose(masked_output_data, unmasked_output_data)

    # Tests the layer with three inputs: Q, K, V.
    key = keras.Input(shape=(2, 8))
    output = test_layer(query, value=value, key=key, attention_mask=mask_tensor)
    model = keras.Model([query, value, key, mask_tensor], output)

    masked_output_data = model.predict([from_data, to_data, to_data, mask_data])
    unmasked_output_data = model.predict(
        [from_data, to_data, to_data, null_mask_data])
    # Because one data is masked and one is not, the outputs should not be the
    # same.
    self.assertNotAllClose(masked_output_data, unmasked_output_data)

    if use_bias:
      self.assertLen(test_layer._query_dense.trainable_variables, 2)
      self.assertLen(test_layer._output_dense.trainable_variables, 2)
    else:
      self.assertLen(test_layer._query_dense.trainable_variables, 1)
      self.assertLen(test_layer._output_dense.trainable_variables, 1)

  def test_initializer(self):
    """Test with a specified initializer."""
    test_layer = multi_head_attention.MultiHeadAttention(
        num_heads=12,
        key_dim=64,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    output = test_layer(query, query)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])

  def test_masked_attention_with_scores(self):
    """Test with a mask tensor."""
    test_layer = multi_head_attention.MultiHeadAttention(
        num_heads=2, key_dim=2)
    # Create a 3-dimensional input (the first dimension is implicit).
    batch_size = 3
    query = keras.Input(shape=(4, 8))
    value = keras.Input(shape=(2, 8))
    mask_tensor = keras.Input(shape=(4, 2))
    output = test_layer(query=query, value=value, attention_mask=mask_tensor)

    # Create a model containing the test layer.
    model = keras.Model([query, value, mask_tensor], output)

    # Generate data for the input (non-mask) tensors.
    from_data = 10 * np.random.random_sample((batch_size, 4, 8))
    to_data = 10 * np.random.random_sample((batch_size, 2, 8))

    # Invoke the data with a random set of mask data. This should mask at least
    # one element.
    mask_data = np.random.randint(2, size=(batch_size, 4, 2))
    masked_output_data = model.predict([from_data, to_data, mask_data])

    # Invoke the same data, but with a null mask (where no elements are masked).
    null_mask_data = np.ones((batch_size, 4, 2))
    unmasked_output_data = model.predict([from_data, to_data, null_mask_data])

    # Because one data is masked and one is not, the outputs should not be the
    # same.
    self.assertNotAllClose(masked_output_data, unmasked_output_data)

    # Create a model containing attention scores.
    output, scores = test_layer(
        query=query, value=value, attention_mask=mask_tensor,
        return_attention_scores=True)
    model = keras.Model([query, value, mask_tensor], [output, scores])
    masked_output_data_score, masked_score = model.predict(
        [from_data, to_data, mask_data])
    unmasked_output_data_score, unmasked_score = model.predict(
        [from_data, to_data, null_mask_data])
    self.assertNotAllClose(masked_output_data_score, unmasked_output_data_score)
    self.assertAllClose(masked_output_data, masked_output_data_score)
    self.assertAllClose(unmasked_output_data, unmasked_output_data_score)
    self.assertNotAllClose(masked_score, unmasked_score)

  @parameterized.named_parameters(
      ("4d_inputs_1freebatch_mask2", [3, 4], [3, 2], [4, 2],
       (2,)), ("4d_inputs_1freebatch_mask3", [3, 4], [3, 2], [3, 4, 2], (2,)),
      ("4d_inputs_1freebatch_mask4", [3, 4], [3, 2], [3, 2, 4, 2],
       (2,)), ("4D_inputs_2D_attention", [3, 4], [3, 2], [3, 4, 3, 2], (1, 2)),
      ("5D_inputs_2D_attention", [5, 3, 4], [5, 3, 2], [3, 4, 3, 2], (2, 3)),
      ("5D_inputs_2D_attention_fullmask", [5, 3, 4], [5, 3, 2], [5, 3, 4, 3, 2],
       (2, 3)))
  def test_high_dim_attention(self, q_dims, v_dims, mask_dims, attention_axes):
    """Test with a mask tensor."""
    test_layer = multi_head_attention.MultiHeadAttention(
        num_heads=2, key_dim=2, attention_axes=attention_axes)
    batch_size, hidden_size = 3, 8
    # Generate data for the input (non-mask) tensors.
    query_shape = [batch_size] + q_dims + [hidden_size]
    value_shape = [batch_size] + v_dims + [hidden_size]
    mask_shape = [batch_size] + mask_dims
    query = 10 * np.random.random_sample(query_shape)
    value = 10 * np.random.random_sample(value_shape)

    # Invoke the data with a random set of mask data. This should mask at least
    # one element.
    mask_data = np.random.randint(2, size=mask_shape).astype("bool")
    # Invoke the same data, but with a null mask (where no elements are masked).
    null_mask_data = np.ones(mask_shape)
    # Because one data is masked and one is not, the outputs should not be the
    # same.
    query_tensor = keras.Input(query_shape[1:], name="query")
    value_tensor = keras.Input(value_shape[1:], name="value")
    mask_tensor = keras.Input(mask_shape[1:], name="mask")
    output = test_layer(query=query_tensor, value=value_tensor,
                        attention_mask=mask_tensor)
    model = keras.Model([query_tensor, value_tensor, mask_tensor], output)

    self.assertNotAllClose(
        model.predict([query, value, mask_data]),
        model.predict([query, value, null_mask_data]))

  def test_dropout(self):
    test_layer = multi_head_attention.MultiHeadAttention(
        num_heads=2, key_dim=2, dropout=0.5)

    # Generate data for the input (non-mask) tensors.
    from_data = keras.backend.ones(shape=(32, 4, 8))
    to_data = keras.backend.ones(shape=(32, 2, 8))
    train_out = test_layer(from_data, to_data, None, None, None, True)
    test_out = test_layer(from_data, to_data, None, None, None, False)

    # Output should be close when not in training mode,
    # and should not be close when enabling dropout in training mode.
    self.assertNotAllClose(
        keras.backend.eval(train_out),
        keras.backend.eval(test_out))


class SubclassAttention(multi_head_attention.MultiHeadAttention):

  def _build_attention(self, qkv_rank):
    pass

  def _compute_attention(self,
                         query_tensor,
                         key_tensor,
                         value_tensor,
                         attention_mask=None,
                         training=None):
    return value_tensor, None


@keras_parameterized.run_all_keras_modes
class AttentionSubclassTest(keras_parameterized.TestCase):

  def test_initializer(self):
    """Test with a specified initializer."""
    test_layer = SubclassAttention(num_heads=12, key_dim=64)
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    output = test_layer(query, query)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])


class TestModel(keras.Model):

  def __init__(self):
    super(TestModel, self).__init__()
    self.attention = multi_head_attention.MultiHeadAttention(
        num_heads=3,
        key_dim=4,
        value_dim=4,
        use_bias=True,
        dropout=0.0,
        output_shape=[12])

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    return {}

  def call(self, x, training=False):
    return self.attention(x, x, training=training)


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class KerasModelSavingTest(keras_parameterized.TestCase):

  def test_keras_saving_subclass(self):
    model = TestModel()
    query = keras.Input(shape=(40, 80))
    _ = model(query)
    model_path = self.get_temp_dir() + "/tmp_model"
    keras.models.save_model(model, model_path, save_format="tf")
    reloaded_model = keras.models.load_model(model_path)
    self.assertEqual(
        len(model.trainable_variables), len(reloaded_model.trainable_variables))
    for src_v, loaded_v in zip(model.trainable_variables,
                               reloaded_model.trainable_variables):
      self.assertAllEqual(src_v, loaded_v)

  @parameterized.parameters("h5", "tf")
  def test_keras_saving_functional(self, save_format):
    model = TestModel()
    query = keras.Input(shape=(40, 80))
    output = multi_head_attention.MultiHeadAttention(
        num_heads=3,
        key_dim=4,
        value_dim=4,
        use_bias=True,
        dropout=0.0)(query, query)
    model = keras.Model(inputs=query, outputs=output)
    model_path = self.get_temp_dir() + "/tmp_model"
    keras.models.save_model(model, model_path, save_format=save_format)
    reloaded_model = keras.models.load_model(model_path)
    self.assertEqual(
        len(model.trainable_variables), len(reloaded_model.trainable_variables))
    for src_v, loaded_v in zip(model.trainable_variables,
                               reloaded_model.trainable_variables):
      self.assertAllEqual(src_v, loaded_v)

  def test_create_without_build(self):
    not_intialized_layer = multi_head_attention.MultiHeadAttention(
        num_heads=3, key_dim=4, value_dim=4)
    multi_head_attention.MultiHeadAttention.from_config(
        not_intialized_layer.get_config())


if __name__ == "__main__":
  test.main()
