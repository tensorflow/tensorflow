# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=protected-access
"""Wrapper layers: layers that augment the functionality of another layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import inspect

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras.python.keras.engine import Layer
from tensorflow.python.framework import tensor_shape


class Wrapper(Layer):
  """Abstract wrapper base class.

  Wrappers take another layer and augment it in various ways.
  Do not use this class as a layer, it is only an abstract base class.
  Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

  Arguments:
      layer: The layer to be wrapped.
  """

  def __init__(self, layer, **kwargs):
    self.layer = layer
    super(Wrapper, self).__init__(**kwargs)

  def build(self, input_shape=None):
    # Assumes that self.layer is already set.
    # Should be called at the end of .build() in the children classes.
    self.trainable_weights = getattr(self.layer, 'trainable_weights', [])
    self.non_trainable_weights = getattr(self.layer, 'non_trainable_weights',
                                         [])
    self.updates = getattr(self.layer, 'updates', [])
    self.losses = getattr(self.layer, 'losses', [])
    self.constraints = getattr(self.layer, 'constraints', {})
    self.built = True

  def get_weights(self):
    weights = self.layer.get_weights()
    return weights

  def set_weights(self, weights):
    self.layer.set_weights(weights)

  def get_config(self):
    config = {
        'layer': {
            'class_name': self.layer.__class__.__name__,
            'config': self.layer.get_config()
        }
    }
    base_config = super(Wrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    from tensorflow.contrib.keras.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    layer = deserialize_layer(
        config.pop('layer'), custom_objects=custom_objects)
    return cls(layer, **config)


class TimeDistributed(Wrapper):
  """This wrapper allows to apply a layer to every temporal slice of an input.

  The input should be at least 3D, and the dimension of index one
  will be considered to be the temporal dimension.

  Consider a batch of 32 samples,
  where each sample is a sequence of 10 vectors of 16 dimensions.
  The batch input shape of the layer is then `(32, 10, 16)`,
  and the `input_shape`, not including the samples dimension, is `(10, 16)`.

  You can then use `TimeDistributed` to apply a `Dense` layer
  to each of the 10 timesteps, independently:

  ```python
      # as the first layer in a model
      model = Sequential()
      model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
      # now model.output_shape == (None, 10, 8)

      # subsequent layers: no need for input_shape
      model.add(TimeDistributed(Dense(32)))
      # now model.output_shape == (None, 10, 32)
  ```

  The output will then have shape `(32, 10, 8)`.

  `TimeDistributed` can be used with arbitrary layers, not just `Dense`,
  for instance with a `Conv2D` layer:

  ```python
      model = Sequential()
      model.add(TimeDistributed(Conv2D(64, (3, 3)),
                                input_shape=(10, 299, 299, 3)))
  ```

  Arguments:
      layer: a layer instance.
  """

  def __init__(self, layer, **kwargs):
    super(TimeDistributed, self).__init__(layer, **kwargs)
    self.supports_masking = True

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    assert len(input_shape) >= 3
    self.input_spec = InputSpec(shape=input_shape)
    child_input_shape = [input_shape[0]] + input_shape[2:]
    if not self.layer.built:
      self.layer.build(child_input_shape)
      self.layer.built = True
    super(TimeDistributed, self).build()

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    child_input_shape = tensor_shape.TensorShape([input_shape[0]] + input_shape[
        2:])
    child_output_shape = self.layer._compute_output_shape(  # pylint: disable=protected-access
        child_input_shape).as_list()
    timesteps = input_shape[1]
    return tensor_shape.TensorShape([child_output_shape[0], timesteps] +
                                    child_output_shape[1:])

  def call(self, inputs, mask=None):
    input_shape = K.int_shape(inputs)
    if input_shape[0]:
      # batch size matters, use rnn-based implementation
      def step(x, _):
        output = self.layer.call(x)
        return output, []

      _, outputs, _ = K.rnn(
          step,
          inputs,
          initial_states=[],
          input_length=input_shape[1],
          unroll=False)
      y = outputs
    else:
      # No batch size specified, therefore the layer will be able
      # to process batches of any size.
      # We can go with reshape-based implementation for performance.
      input_length = input_shape[1]
      if not input_length:
        input_length = K.shape(inputs)[1]
      # Shape: (num_samples * timesteps, ...)
      inputs = K.reshape(inputs, (-1,) + input_shape[2:])
      y = self.layer.call(inputs)  # (num_samples * timesteps, ...)
      # Shape: (num_samples, timesteps, ...)
      output_shape = self._compute_output_shape(input_shape).as_list()  # pylint: disable=protected-access
      y = K.reshape(y, [-1, input_length] + output_shape[2:])

    # Apply activity regularizer if any:
    if (hasattr(self.layer, 'activity_regularizer') and
        self.layer.activity_regularizer is not None):
      regularization_loss = self.layer.activity_regularizer(y)
      self.add_loss(regularization_loss, inputs)
    return y


class Bidirectional(Wrapper):
  """Bidirectional wrapper for RNNs.

  Arguments:
      layer: `Recurrent` instance.
      merge_mode: Mode by which outputs of the
          forward and backward RNNs will be combined.
          One of {'sum', 'mul', 'concat', 'ave', None}.
          If None, the outputs will not be combined,
          they will be returned as a list.

  Raises:
      ValueError: In case of invalid `merge_mode` argument.

  Examples:

  ```python
      model = Sequential()
      model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5,
      10)))
      model.add(Bidirectional(LSTM(10)))
      model.add(Dense(5))
      model.add(Activation('softmax'))
      model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
  ```
  """

  def __init__(self, layer, merge_mode='concat', weights=None, **kwargs):
    super(Bidirectional, self).__init__(layer, **kwargs)
    if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
      raise ValueError('Invalid merge mode. '
                       'Merge mode should be one of '
                       '{"sum", "mul", "ave", "concat", None}')
    self.forward_layer = copy.copy(layer)
    config = layer.get_config()
    config['go_backwards'] = not config['go_backwards']
    self.backward_layer = layer.__class__.from_config(config)
    self.forward_layer.name = 'forward_' + self.forward_layer.name
    self.backward_layer.name = 'backward_' + self.backward_layer.name
    self.merge_mode = merge_mode
    if weights:
      nw = len(weights)
      self.forward_layer.initial_weights = weights[:nw // 2]
      self.backward_layer.initial_weights = weights[nw // 2:]
    self.stateful = layer.stateful
    self.return_sequences = layer.return_sequences
    self.supports_masking = True

  def get_weights(self):
    return self.forward_layer.get_weights() + self.backward_layer.get_weights()

  def set_weights(self, weights):
    nw = len(weights)
    self.forward_layer.set_weights(weights[:nw // 2])
    self.backward_layer.set_weights(weights[nw // 2:])

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.merge_mode in ['sum', 'ave', 'mul']:
      return self.forward_layer._compute_output_shape(input_shape)  # pylint: disable=protected-access
    elif self.merge_mode == 'concat':
      shape = self.forward_layer._compute_output_shape(input_shape).as_list()  # pylint: disable=protected-access
      shape[-1] *= 2
      return tensor_shape.TensorShape(shape)
    elif self.merge_mode is None:
      shape = self.forward_layer._compute_output_shape(input_shape)  # pylint: disable=protected-access
      return [shape, copy.copy(shape)]

  def call(self, inputs, training=None, mask=None):
    kwargs = {}
    func_args = inspect.getargspec(self.layer.call).args
    if 'training' in func_args:
      kwargs['training'] = training
    if 'mask' in func_args:
      kwargs['mask'] = mask

    y = self.forward_layer.call(inputs, **kwargs)
    y_rev = self.backward_layer.call(inputs, **kwargs)
    if self.return_sequences:
      y_rev = K.reverse(y_rev, 1)
    if self.merge_mode == 'concat':
      output = K.concatenate([y, y_rev])
    elif self.merge_mode == 'sum':
      output = y + y_rev
    elif self.merge_mode == 'ave':
      output = (y + y_rev) / 2
    elif self.merge_mode == 'mul':
      output = y * y_rev
    elif self.merge_mode is None:
      output = [y, y_rev]

    # Properly set learning phase
    if 0 < self.layer.dropout + self.layer.recurrent_dropout:
      if self.merge_mode is None:
        for out in output:
          out._uses_learning_phase = True
      else:
        output._uses_learning_phase = True
    return output

  def reset_states(self):
    self.forward_layer.reset_states()
    self.backward_layer.reset_states()

  def build(self, input_shape):
    with K.name_scope(self.forward_layer.name):
      self.forward_layer.build(input_shape)
    with K.name_scope(self.backward_layer.name):
      self.backward_layer.build(input_shape)
    self.built = True

  def compute_mask(self, inputs, mask):
    if self.return_sequences:
      if not self.merge_mode:
        return [mask, mask]
      else:
        return mask
    else:
      return None

  @property
  def trainable_weights(self):
    if hasattr(self.forward_layer, 'trainable_weights'):
      return (self.forward_layer.trainable_weights +
              self.backward_layer.trainable_weights)
    return []

  @property
  def non_trainable_weights(self):
    if hasattr(self.forward_layer, 'non_trainable_weights'):
      return (self.forward_layer.non_trainable_weights +
              self.backward_layer.non_trainable_weights)
    return []

  @property
  def updates(self):
    if hasattr(self.forward_layer, 'updates'):
      return self.forward_layer.updates + self.backward_layer.updates
    return []

  @property
  def losses(self):
    if hasattr(self.forward_layer, 'losses'):
      return self.forward_layer.losses + self.backward_layer.losses
    return []

  @property
  def constraints(self):
    constraints = {}
    if hasattr(self.forward_layer, 'constraints'):
      constraints.update(self.forward_layer.constraints)
      constraints.update(self.backward_layer.constraints)
    return constraints

  def get_config(self):
    config = {'merge_mode': self.merge_mode}
    base_config = super(Bidirectional, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
