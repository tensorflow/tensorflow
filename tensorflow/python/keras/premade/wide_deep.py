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
"""Built-in WideNDeep model classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training as keras_training
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.experimental.WideDeepModel')
class WideDeepModel(keras_training.Model):
  r"""Wide & Deep Model for regression and classification problems.

  This model jointly train a linear and a dnn model.

  Example:

  ```python
  linear_model = LinearModel()
  dnn_model = keras.Sequential([keras.layers.Dense(units=64),
                               keras.layers.Dense(units=1)])
  combined_model = WideDeepModel(dnn_model, linear_model)
  combined_model.compile(optimizer=['sgd', 'adam'], 'mse', ['mse'])
  # define dnn_inputs and linear_inputs as separate numpy arrays or
  # a single numpy array if dnn_inputs is same as linear_inputs.
  combined_model.fit([dnn_inputs, linear_inputs], y, epochs)
  # or define a single `tf.data.Dataset` that contains a single tensor or
  # separate tensors for dnn_inputs and linear_inputs.
  dataset = tf.data.Dataset.from_tensors(([dnn_inputs, linear_inputs], y))
  combined_model.fit(dataset, epochs)
  ```

  Both linear and dnn model can be pre-compiled and trained separately
  before jointly training:

  Example:
  ```python
  linear_model = LinearModel()
  linear_model.compile('adagrad', 'mse')
  linear_model.fit(linear_inputs, y, epochs)
  dnn_model = keras.Sequential([keras.layers.Dense(units=1)])
  dnn_model.compile('rmsprop', 'mse')
  dnn_model.fit(dnn_inputs, y, epochs)
  combined_model = WideDeepModel(dnn_model, linear_model)
  combined_model.compile(optimizer=['sgd', 'adam'], 'mse', ['mse'])
  combined_model.fit([dnn_inputs, linear_inputs], y, epochs)
  ```

  """

  def __init__(self, linear_model, dnn_model, activation=None, **kwargs):
    """Create a Wide & Deep Model.

    Args:
      linear_model: a premade LinearModel, its output must match the output of
        the dnn model.
      dnn_model: a `tf.keras.Model`, its output must match the output of the
        linear model.
      activation: Activation function. Set it to None to maintain a linear
        activation.
      **kwargs: The keyword arguments that are passed on to BaseLayer.__init__.
        Allowed keyword arguments include `name`.
    """
    super(WideDeepModel, self).__init__(**kwargs)
    base_layer._keras_model_gauge.get_cell('WideDeep').set(True)  # pylint: disable=protected-access
    self.linear_model = linear_model
    self.dnn_model = dnn_model
    self.activation = activations.get(activation)

  def call(self, inputs, training=None):
    if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
      linear_inputs = dnn_inputs = inputs
    else:
      linear_inputs, dnn_inputs = inputs
    linear_output = self.linear_model(linear_inputs)
    # pylint: disable=protected-access
    if self.dnn_model._expects_training_arg:
      if training is None:
        training = K.learning_phase()
      dnn_output = self.dnn_model(dnn_inputs, training=training)
    else:
      dnn_output = self.dnn_model(dnn_inputs)
    output = nest.map_structure(lambda x, y: 0.5 * (x + y), linear_output,
                                dnn_output)
    if self.activation:
      return nest.map_structure(self.activation, output)
    return output

  def _get_optimizers(self):
    if isinstance(self.optimizer, (tuple, list)):
      return (self.optimizer[0], self.optimizer[1])
    else:
      return (self.optimizer, self.optimizer)

  # This does not support gradient scaling and LossScaleOptimizer.
  def _backwards(self, tape, loss):
    linear_vars = self.linear_model.trainable_weights  # pylint: disable=protected-access
    dnn_vars = self.dnn_model.trainable_weights  # pylint: disable=protected-access
    linear_grads, dnn_grads = tape.gradient(loss, (linear_vars, dnn_vars))
    linear_optimizer, dnn_optimizer = self._get_optimizers()
    linear_optimizer.apply_gradients(zip(linear_grads, linear_vars))
    dnn_optimizer.apply_gradients(zip(dnn_grads, dnn_vars))
    return

  def _make_train_function(self):
    # TODO(tanzheny): This is a direct copy from super to make it work
    # refactor it so that common logic can be shared.
    has_recompiled = self._recompile_weights_loss_and_weighted_metrics()
    self._check_trainable_weights_consistency()
    # If we have re-compiled the loss/weighted metric sub-graphs then create
    # train function even if one exists already. This is because
    # `_feed_sample_weights` list has been updated on re-copmpile.
    if getattr(self, 'train_function', None) is None or has_recompiled:
      # Restore the compiled trainable state.
      current_trainable_state = self._get_trainable_state()
      self._set_trainable_state(self._compiled_trainable_state)

      inputs = (
          self._feed_inputs + self._feed_targets + self._feed_sample_weights)
      if not isinstance(K.symbolic_learning_phase(), int):
        inputs += [K.symbolic_learning_phase()]

      linear_optimizer, dnn_optimizer = self._get_optimizers()
      with K.get_graph().as_default():
        with K.name_scope('training'):
          # Training updates
          updates = []
          linear_updates = linear_optimizer.get_updates(
              params=self.linear_model.trainable_weights,  # pylint: disable=protected-access
              loss=self.total_loss)
          updates += linear_updates
          dnn_updates = dnn_optimizer.get_updates(
              params=self.dnn_model.trainable_weights,  # pylint: disable=protected-access
              loss=self.total_loss)
          updates += dnn_updates
          # Unconditional updates
          updates += self.get_updates_for(None)
          # Conditional updates relevant to this model
          updates += self.get_updates_for(self.inputs)

        metrics = self._get_training_eval_metrics()
        metrics_tensors = [
            m._call_result for m in metrics if hasattr(m, '_call_result')  # pylint: disable=protected-access
        ]

      with K.name_scope('training'):
        # Gets loss and metrics. Updates weights at each call.
        fn = K.function(
            inputs, [self.total_loss] + metrics_tensors,
            updates=updates,
            name='train_function',
            **self._function_kwargs)
        setattr(self, 'train_function', fn)

      # Restore the current trainable state
      self._set_trainable_state(current_trainable_state)

  def get_config(self):
    linear_config = generic_utils.serialize_keras_object(self.linear_model)
    dnn_config = generic_utils.serialize_keras_object(self.dnn_model)
    config = {
        'linear_model': linear_config,
        'dnn_model': dnn_config,
        'activation': activations.serialize(self.activation),
    }
    base_config = base_layer.Layer.get_config(self)
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    linear_config = config.pop('linear_model')
    linear_model = layer_module.deserialize(linear_config, custom_objects)
    dnn_config = config.pop('dnn_model')
    dnn_model = layer_module.deserialize(dnn_config, custom_objects)
    activation = activations.deserialize(
        config.pop('activation', None), custom_objects=custom_objects)
    return cls(
        linear_model=linear_model,
        dnn_model=dnn_model,
        activation=activation,
        **config)
