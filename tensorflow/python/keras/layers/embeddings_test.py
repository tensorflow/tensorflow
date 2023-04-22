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
"""Tests for embedding layers."""

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad


class EmbeddingTest(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_embedding(self):
    if tf_test_util.is_gpu_available():
      self.skipTest('Only test embedding on CPU.')

    testing_utils.layer_test(
        keras.layers.Embedding,
        kwargs={'output_dim': 4,
                'input_dim': 10,
                'input_length': 2},
        input_shape=(3, 2),
        input_dtype='int32',
        expected_output_dtype='float32')

    testing_utils.layer_test(
        keras.layers.Embedding,
        kwargs={'output_dim': 4,
                'input_dim': 10,
                'mask_zero': True},
        input_shape=(3, 2),
        input_dtype='int32',
        expected_output_dtype='float32')

    testing_utils.layer_test(
        keras.layers.Embedding,
        kwargs={'output_dim': 4,
                'input_dim': 10,
                'mask_zero': True},
        input_shape=(3, 4, 2),
        input_dtype='int32',
        expected_output_dtype='float32')

    testing_utils.layer_test(
        keras.layers.Embedding,
        kwargs={'output_dim': 4,
                'input_dim': 10,
                'mask_zero': True,
                'input_length': (None, 2)},
        input_shape=(3, 4, 2),
        input_dtype='int32',
        expected_output_dtype='float32')

  @keras_parameterized.run_all_keras_modes
  def test_embedding_correctness(self):
    layer = keras.layers.Embedding(output_dim=2, input_dim=2)
    model = keras.models.Sequential([layer])

    layer.set_weights([np.array([[1, 1], [2, 2]])])
    model.run_eagerly = testing_utils.should_run_eagerly()
    outputs = model.predict(np.array([[0, 1, 0]], dtype='int32'))
    self.assertAllClose(outputs, [[[1, 1], [2, 2], [1, 1]]])

  def test_embedding_incorrect_dimension(self):
    with self.assertRaises(ValueError):
      keras.layers.Embedding(input_dim=0, output_dim=1)

    with self.assertRaises(ValueError):
      keras.layers.Embedding(input_dim=1, output_dim=0)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_eager_gpu_cpu(self):
    l = keras.layers.Embedding(output_dim=2, input_dim=2)
    l.build((None, 2))
    inputs = keras.backend.constant([[0, 1, 0]], dtype='int32')
    with backprop.GradientTape() as tape:
      output = l(inputs)
    gs = tape.gradient(output, l.weights)
    opt = adagrad.AdagradOptimizer(0.1)
    opt.apply_gradients(zip(gs, l.weights))
    self.assertAllEqual(len(gs), 1)

  @keras_parameterized.run_all_keras_modes
  def test_embedding_with_ragged_input(self):
    layer = keras.layers.Embedding(
        input_dim=3,
        output_dim=2,
        weights=[np.array([[0., 0.], [1., 1.], [2., 2.]])])
    inputs = keras.layers.Input(
        shape=(None,), dtype=dtypes.float32, ragged=True)
    # pylint: disable=unnecessary-lambda
    outputs = keras.layers.Lambda(lambda args: keras.backend.identity(args))(
        inputs)
    # pylint: enable=unnecessary-lambda
    outputs = layer(outputs)

    model = keras.Model(inputs, outputs)
    model.run_eagerly = testing_utils.should_run_eagerly()
    outputs = model.predict(
        ragged_factory_ops.constant([[1., 2., 2.], [0.], [1., 2.]],
                                    ragged_rank=1))
    self.assertAllClose(
        outputs,
        ragged_factory_ops.constant(
            [[[1., 1.], [2., 2.], [2., 2.]], [[0., 0.]], [[1., 1.], [2., 2.]]],
            ragged_rank=1))

  @testing_utils.enable_v2_dtype_behavior
  def test_mixed_precision_embedding(self):
    try:
      policy.set_policy('mixed_float16')
      layer = keras.layers.Embedding(input_dim=5, output_dim=2)
      self.assertEqual(layer._dtype_policy.name, 'mixed_float16')
      outputs = layer(np.array([0, 1, 2]))
      self.assertEqual(outputs.dtype, 'float16')
    finally:
      policy.set_policy('float32')


if __name__ == '__main__':
  test.main()
