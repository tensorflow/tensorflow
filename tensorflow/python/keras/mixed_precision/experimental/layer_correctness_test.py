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
"""Tests various Layer subclasses have correct outputs with mixed precision."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.platform import test


def create_mirrored_strategy():
  if context.num_gpus() >= 1:
    return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
  else:
    return mirrored_strategy.MirroredStrategy(['cpu:0'])


@test_util.run_all_in_graph_and_eager_modes
class LayerCorrectnessTest(keras_parameterized.TestCase):

  def _create_model_from_layer(self, layer, input_shape):
    x = layers.Input(batch_input_shape=input_shape)
    y = layer(x)
    model = models.Model(x, y)
    model.compile('sgd', 'mse')
    return model

  def _test_layer(self, f32_layer, input_shape):
    """Tests a layer by comparing the float32 and mixed precision weights.

    A float32 layer, a mixed precision layer, a distributed float32 layer, and a
    distributed mixed precision layer are run. The four layers are identical
    other than their dtypes and distribution strategies. The weights after
    running fit() are asserted to be close.

    Running the distributed float32 layer does not test mixed precision but we
    still test it for debugging purposes. If the distributed mixed precision
    layer fails, it's easier to debug if you know whether the issue also occurs
    in the distributed float32 layer.

    Args:
      f32_layer: A float32 layer. The other three layers will automatically
        be created from this
      input_shape: The shape of the inputs to the layer, including the batch
        dimension.
    """
    strategy = create_mirrored_strategy()

    # Create the layers
    assert f32_layer.dtype == f32_layer._compute_dtype == 'float32'
    config = f32_layer.get_config()
    distributed_f32_layer = f32_layer.__class__.from_config(config)
    config['dtype'] = policy.Policy('mixed_float16')
    mp_layer = f32_layer.__class__.from_config(config)
    distributed_mp_layer = f32_layer.__class__.from_config(config)

    # Compute per_replica_input_shape for the distributed models
    global_batch_size = input_shape[0]
    assert global_batch_size % strategy.num_replicas_in_sync == 0
    per_replica_batch_size = (
        global_batch_size // strategy.num_replicas_in_sync)
    per_replica_input_shape = list(input_shape)
    per_replica_input_shape[0] = per_replica_batch_size

    # Create the models
    f32_model = self._create_model_from_layer(f32_layer, input_shape)
    mp_model = self._create_model_from_layer(mp_layer, input_shape)
    with strategy.scope():
      distributed_f32_model = self._create_model_from_layer(
          distributed_f32_layer, per_replica_input_shape)
      distributed_mp_model = self._create_model_from_layer(
          distributed_mp_layer, per_replica_input_shape)

    # Set all model weights to the same values
    f32_weights = f32_model.get_weights()
    for model in mp_model, distributed_f32_model, distributed_mp_model:
      model.set_weights(f32_weights)

    # Run fit() on models
    x = np.random.normal(size=input_shape)
    y = np.random.normal(size=input_shape)
    for model in (f32_model, mp_model, distributed_f32_model,
                  distributed_mp_model):
      model.fit(x, y, batch_size=global_batch_size)

    # Assert all models have close weights
    f32_weights = f32_model.get_weights()
    self.assertAllClose(
        mp_model.get_weights(), f32_weights, rtol=1e-2, atol=1e-4)
    self.assertAllClose(
        distributed_f32_model.get_weights(), f32_weights, rtol=1e-2, atol=1e-4)
    self.assertAllClose(
        distributed_mp_model.get_weights(), f32_weights, rtol=1e-2, atol=1e-4)

  # Note: There is no need to test every layer subclass here, as otherwise this
  # test would take too long. Only layers which do something special or are
  # unusual in regards to mixed precision need to be tested.

  # We test RNNs as some RNNs use the implementation_selector grappler pass,
  # which can cause issues with AutoCastVariables.
  @testing_utils.enable_v2_dtype_behavior
  def test_simple_rnn(self):
    self._test_layer(recurrent.SimpleRNN(units=4, return_sequences=True),
                     input_shape=(4, 4, 4))

  @testing_utils.enable_v2_dtype_behavior
  def test_gru(self):
    self._test_layer(recurrent_v2.GRU(units=4, return_sequences=True),
                     input_shape=(4, 4, 4))

  @testing_utils.enable_v2_dtype_behavior
  def test_lstm(self):
    self._test_layer(recurrent_v2.LSTM(units=4, return_sequences=True),
                     input_shape=(4, 4, 4))

if __name__ == '__main__':
  test.main()
