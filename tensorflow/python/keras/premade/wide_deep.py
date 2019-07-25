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

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import training
from tensorflow.python.training.tracking import base as trackable


class WideDeepModel(training.Model):
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
    self.linear_model = linear_model
    self.dnn_model = dnn_model
    self.activation = activation

  def call(self, inputs):
    if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
      linear_inputs = dnn_inputs = inputs
    else:
      linear_inputs, dnn_inputs = inputs
    linear_output = self.linear_model(linear_inputs)
    dnn_output = self.dnn_model(dnn_inputs)
    output = .5 * (linear_output + dnn_output)
    if self.activation:
      return self.activation(output)
    return output

  @trackable.no_automatic_dependency_tracking
  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              **kwargs):
    """Configures the model for training.

    Arguments:
        optimizer: A single String (name of optimizer) or optimizer instance if
          linear and dnn model share the same optimizer, or a list or tuple of 2
          optimizers if not. See `tf.keras.optimizers`.
        loss: String (name of objective function), objective function or
          `tf.losses.Loss` instance. See `tf.losses`. If the model has multiple
          outputs, you can use a different loss on each output by passing a
          dictionary or a list of losses. The loss value that will be minimized
          by the model will then be the sum of all individual losses.
        metrics: List of metrics to be evaluated by the model during training
          and testing. Typically you will use `metrics=['accuracy']`. To specify
          different metrics for different outputs of a multi-output model, you
          could also pass a dictionary, such as
            `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
              You can also pass a list (len = len(outputs)) of lists of metrics
              such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or
              `metrics=['accuracy', ['accuracy', 'mse']]`.
        loss_weights: Optional list or dictionary specifying scalar coefficients
          (Python floats) to weight the loss contributions of different model
          outputs. The loss value that will be minimized by the model will then
          be the *weighted sum* of all individual losses, weighted by the
          `loss_weights` coefficients.
            If a list, it is expected to have a 1:1 mapping to the model's
              outputs. If a tensor, it is expected to map output names (strings)
              to scalar coefficients.
        sample_weight_mode: If you need to do timestep-wise sample weighting (2D
          weights), set this to `"temporal"`. `None` defaults to sample-wise
          weights (1D). If the model has multiple outputs, you can use a
          different `sample_weight_mode` on each output by passing a dictionary
          or a list of modes.
        weighted_metrics: List of metrics to be evaluated and weighted by
          sample_weight or class_weight during training and testing.
        target_tensors: By default, Keras will create placeholders for the
          model's target, which will be fed with the target data during
          training. If instead you would like to use your own target tensors (in
          turn, Keras will not expect external Numpy data for these targets at
          training time), you can specify them via the `target_tensors`
          argument. It can be a single tensor (for a single-output model), a
          list of tensors, or a dict mapping output names to target tensors.
        **kwargs: Any additional arguments passed to Model.compile, including
          run_eagerly.

    Raises:
        ValueError: In case of invalid arguments for
            `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
    """
    if isinstance(optimizer, (tuple, list)):
      self.linear_optimizer = optimizers.get(optimizer[0])
      self.dnn_optimizer = optimizers.get(optimizer[1])
    else:
      # DNN and Linear sharing the same optimizer.
      opt = optimizers.get(optimizer)
      self.dnn_optimizer = opt
      self.linear_optimizer = opt
    # TODO(tanzheny): Make optimizer have default in compile (b/132909290)
    super(WideDeepModel, self).compile(
        optimizer=[self.linear_optimizer, self.dnn_optimizer],
        loss=loss,
        metrics=metrics,
        loss_weights=loss_weights,
        sample_weight_mode=sample_weight_mode,
        weighted_metrics=weighted_metrics,
        target_tensors=target_tensors,
        **kwargs)

  # This does not support gradient scaling and LossScaleOptimizer.
  def _backwards(self, tape, loss):
    linear_vars = self.linear_model._unique_trainable_weights  # pylint: disable=protected-access
    dnn_vars = self.dnn_model._unique_trainable_weights  # pylint: disable=protected-access
    linear_grads, dnn_grads = tape.gradient(loss, (linear_vars, dnn_vars))
    self.linear_optimizer.apply_gradients(zip(linear_grads, linear_vars))
    self.dnn_optimizer.apply_gradients(zip(dnn_grads, dnn_vars))
    return

  # TODO(tanzheny): Unify the path between train function and train_on_batch.
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

      with K.get_graph().as_default():
        with K.name_scope('training'):
          # Training updates
          updates = []
          dnn_updates = self.dnn_optimizer.get_updates(
              params=self.dnn_model.trainable_weights, loss=self.total_loss)
          updates += dnn_updates
          linear_updates = self.linear_optimizer.get_updates(
              params=self.linear_model.trainable_weights, loss=self.total_loss)
          updates += linear_updates
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
