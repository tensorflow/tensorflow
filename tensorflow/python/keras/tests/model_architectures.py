# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for saving/loading function for keras Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python import keras

# Declaring namedtuple()
ModelFn = collections.namedtuple('ModelFn',
                                 ['model', 'input_shape', 'target_shape'])


def basic_sequential():
  """Basic sequential model."""
  model = keras.Sequential([
      keras.layers.Dense(3, activation='relu', input_shape=(3,)),
      keras.layers.Dense(2, activation='softmax'),
  ])
  return ModelFn(model, (None, 3), (None, 2))


def basic_sequential_deferred():
  """Sequential model with deferred input shape."""
  model = keras.Sequential([
      keras.layers.Dense(3, activation='relu'),
      keras.layers.Dense(2, activation='softmax'),
  ])
  return ModelFn(model, (None, 3), (None, 2))


def stacked_rnn():
  """Stacked RNN model."""
  inputs = keras.Input((None, 3))
  layer = keras.layers.RNN([keras.layers.LSTMCell(2) for _ in range(3)])
  x = layer(inputs)
  outputs = keras.layers.Dense(2)(x)
  model = keras.Model(inputs, outputs)
  return ModelFn(model, (None, 4, 3), (None, 2))


def lstm():
  """LSTM model."""
  inputs = keras.Input((None, 3))
  x = keras.layers.LSTM(4, return_sequences=True)(inputs)
  x = keras.layers.LSTM(3, return_sequences=True)(x)
  x = keras.layers.LSTM(2, return_sequences=False)(x)
  outputs = keras.layers.Dense(2)(x)
  model = keras.Model(inputs, outputs)
  return ModelFn(model, (None, 4, 3), (None, 2))


def multi_input_multi_output():
  """Multi-input Multi-output model."""
  body_input = keras.Input(shape=(None,), name='body')
  tags_input = keras.Input(shape=(2,), name='tags')

  x = keras.layers.Embedding(10, 4)(body_input)
  body_features = keras.layers.LSTM(5)(x)
  x = keras.layers.concatenate([body_features, tags_input])

  pred_1 = keras.layers.Dense(2, activation='sigmoid', name='priority')(x)
  pred_2 = keras.layers.Dense(3, activation='softmax', name='department')(x)

  model = keras.Model(
      inputs=[body_input, tags_input], outputs=[pred_1, pred_2])
  return ModelFn(model, [(None, 1), (None, 2)], [(None, 2), (None, 3)])


def nested_sequential_in_functional():
  """A sequential model nested in a functional model."""
  inner_model = keras.Sequential([
      keras.layers.Dense(3, activation='relu', input_shape=(3,)),
      keras.layers.Dense(2, activation='relu'),
  ])

  inputs = keras.Input(shape=(3,))
  x = inner_model(inputs)
  outputs = keras.layers.Dense(2, activation='softmax')(x)
  model = keras.Model(inputs, outputs)
  return ModelFn(model, (None, 3), (None, 2))


def seq_to_seq():
  """Sequence to sequence model."""
  num_encoder_tokens = 3
  num_decoder_tokens = 3
  latent_dim = 2
  encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
  encoder = keras.layers.LSTM(latent_dim, return_state=True)
  _, state_h, state_c = encoder(encoder_inputs)
  encoder_states = [state_h, state_c]
  decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
  decoder_lstm = keras.layers.LSTM(
      latent_dim, return_sequences=True, return_state=True)
  decoder_outputs, _, _ = decoder_lstm(
      decoder_inputs, initial_state=encoder_states)
  decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax')
  decoder_outputs = decoder_dense(decoder_outputs)
  model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
  return ModelFn(
      model, [(None, 2, num_encoder_tokens), (None, 2, num_decoder_tokens)],
      (None, 2, num_decoder_tokens))


def shared_layer_functional():
  """Shared layer in a functional model."""
  main_input = keras.Input(shape=(10,), dtype='int32', name='main_input')
  x = keras.layers.Embedding(
      output_dim=5, input_dim=4, input_length=10)(main_input)
  lstm_out = keras.layers.LSTM(3)(x)
  auxiliary_output = keras.layers.Dense(
      1, activation='sigmoid', name='aux_output')(lstm_out)
  auxiliary_input = keras.Input(shape=(5,), name='aux_input')
  x = keras.layers.concatenate([lstm_out, auxiliary_input])
  x = keras.layers.Dense(2, activation='relu')(x)
  main_output = keras.layers.Dense(
      1, activation='sigmoid', name='main_output')(x)
  model = keras.Model(
      inputs=[main_input, auxiliary_input],
      outputs=[main_output, auxiliary_output])
  return ModelFn(model, [(None, 10), (None, 5)], [(None, 1), (None, 1)])


def shared_sequential():
  """Shared sequential model in a functional model."""
  inner_model = keras.Sequential([
      keras.layers.Conv2D(2, 3, activation='relu'),
      keras.layers.Conv2D(2, 3, activation='relu'),
  ])
  inputs_1 = keras.Input((5, 5, 3))
  inputs_2 = keras.Input((5, 5, 3))
  x1 = inner_model(inputs_1)
  x2 = inner_model(inputs_2)
  x = keras.layers.concatenate([x1, x2])
  outputs = keras.layers.GlobalAveragePooling2D()(x)
  model = keras.Model([inputs_1, inputs_2], outputs)
  return ModelFn(model, [(None, 5, 5, 3), (None, 5, 5, 3)], (None, 4))


class MySubclassModel(keras.Model):
  """A subclass model."""

  def __init__(self, input_dim=3):
    super(MySubclassModel, self).__init__(name='my_subclass_model')
    self._config = {'input_dim': input_dim}
    self.dense1 = keras.layers.Dense(8, activation='relu')
    self.dense2 = keras.layers.Dense(2, activation='softmax')
    self.bn = keras.layers.BatchNormalization()
    self.dp = keras.layers.Dropout(0.5)

  def call(self, inputs, **kwargs):
    x = self.dense1(inputs)
    x = self.dp(x)
    x = self.bn(x)
    return self.dense2(x)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


def nested_subclassed_model():
  """A subclass model nested in another subclass model."""

  class NestedSubclassModel(keras.Model):
    """A nested subclass model."""

    def __init__(self):
      super(NestedSubclassModel, self).__init__()
      self.dense1 = keras.layers.Dense(4, activation='relu')
      self.dense2 = keras.layers.Dense(2, activation='relu')
      self.bn = keras.layers.BatchNormalization()
      self.inner_subclass_model = MySubclassModel()

    def call(self, inputs):
      x = self.dense1(inputs)
      x = self.bn(x)
      x = self.inner_subclass_model(x)
      return self.dense2(x)

  return ModelFn(NestedSubclassModel(), (None, 3), (None, 2))


def nested_subclassed_in_functional_model():
  """A subclass model nested in a functional model."""
  inner_subclass_model = MySubclassModel()
  inputs = keras.Input(shape=(3,))
  x = inner_subclass_model(inputs)
  x = keras.layers.BatchNormalization()(x)
  outputs = keras.layers.Dense(2, activation='softmax')(x)
  model = keras.Model(inputs, outputs)
  return ModelFn(model, (None, 3), (None, 2))


def nested_functional_in_subclassed_model():
  """A functional model nested in a subclass model."""
  def get_functional_model():
    inputs = keras.Input(shape=(4,))
    x = keras.layers.Dense(4, activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(2)(x)
    return keras.Model(inputs, outputs)

  class NestedFunctionalInSubclassModel(keras.Model):
    """A functional nested in subclass model."""

    def __init__(self):
      super(NestedFunctionalInSubclassModel, self).__init__(
          name='nested_functional_in_subclassed_model')
      self.dense1 = keras.layers.Dense(4, activation='relu')
      self.dense2 = keras.layers.Dense(2, activation='relu')
      self.inner_functional_model = get_functional_model()

    def call(self, inputs):
      x = self.dense1(inputs)
      x = self.inner_functional_model(x)
      return self.dense2(x)
  return ModelFn(NestedFunctionalInSubclassModel(), (None, 3), (None, 2))


def shared_layer_subclassed_model():
  """Shared layer in a subclass model."""

  class SharedLayerSubclassModel(keras.Model):
    """A subclass model with shared layers."""

    def __init__(self):
      super(SharedLayerSubclassModel, self).__init__(
          name='shared_layer_subclass_model')
      self.dense = keras.layers.Dense(3, activation='relu')
      self.dp = keras.layers.Dropout(0.5)
      self.bn = keras.layers.BatchNormalization()

    def call(self, inputs):
      x = self.dense(inputs)
      x = self.dp(x)
      x = self.bn(x)
      return self.dense(x)
  return ModelFn(SharedLayerSubclassModel(), (None, 3), (None, 3))


def functional_with_keyword_args():
  """A functional model with keyword args."""
  inputs = keras.Input(shape=(3,))
  x = keras.layers.Dense(4)(inputs)
  x = keras.layers.BatchNormalization()(x)
  outputs = keras.layers.Dense(2)(x)

  model = keras.Model(inputs, outputs, name='m', trainable=False)
  return ModelFn(model, (None, 3), (None, 2))


ALL_MODELS = [
    ('basic_sequential', basic_sequential),
    ('basic_sequential_deferred', basic_sequential_deferred),
    ('stacked_rnn', stacked_rnn),
    ('lstm', lstm),
    ('multi_input_multi_output', multi_input_multi_output),
    ('nested_sequential_in_functional', nested_sequential_in_functional),
    ('seq_to_seq', seq_to_seq),
    ('shared_layer_functional', shared_layer_functional),
    ('shared_sequential', shared_sequential),
    ('nested_subclassed_model', nested_subclassed_model),
    ('nested_subclassed_in_functional_model',
     nested_subclassed_in_functional_model),
    ('nested_functional_in_subclassed_model',
     nested_functional_in_subclassed_model),
    ('shared_layer_subclassed_model', shared_layer_subclassed_model),
    ('functional_with_keyword_args', functional_with_keyword_args)
]


def get_models(exclude_models=None):
  """Get all models excluding the specified ones."""
  models = [model for model in ALL_MODELS
            if model[0] not in exclude_models]
  return models
