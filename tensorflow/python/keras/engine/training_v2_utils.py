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
"""Training related logic for Keras model in TF 2.0 context.

Note that all the code under this module is under active development, please DO
NOT use it unless you are really sure what you are doing.
"""

# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.ops import composite_tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import training_eager
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


def _get_or_make_execution_function(model, mode):
  """Makes or reuses function to run one step of distributed model execution."""
  model._init_distributed_function_cache_if_not_compiled()

  # Use a key with 'v2' to distinguish from fall-back execution functions.
  key = (mode, 'v2')
  distributed_function = dist_utils.get_distributed_function(model, key)
  if distributed_function:
    return distributed_function

  distribution_function = _make_execution_function(model, mode)
  dist_utils.set_distributed_function(model, key, distribution_function)
  return distribution_function


def _make_execution_function(model, mode):
  """Creates a function to run one step of distributed model execution."""
  per_replica_function = _make_replica_execution_function(model, mode)

  def distributed_function(input_iterator):
    """A single step of the distributed execution across replicas."""
    x, y, sample_weights = _prepare_feed_values(
        model, input_iterator, mode)
    # Call `Model.{train,test,predict}_on_batch` on every replica passing
    # PerReplicas as arguments.  On every replica inside this call, each
    # PerReplica object will return the value for that replica.  The outputs
    # are PerReplicas too.
    strategy = distribution_strategy_context.get_strategy()
    outputs = strategy.experimental_run_v2(
        per_replica_function, args=(x, y, sample_weights))
    # Out of PerReplica outputs reduce or pick values to return.
    all_outputs = dist_utils.unwrap_output_dict(
        strategy, outputs, mode)
    return all_outputs

  if not model.run_eagerly:
    distributed_function = def_function.function(
        distributed_function, autograph=False)

  def execution_function(input_fn):
    # `numpy` translates Tensors to values in Eager mode.
    return nest.map_structure(_non_none_constant_value,
                              distributed_function(input_fn))

  return execution_function


def _non_none_constant_value(v):
  constant_value = tensor_util.constant_value(v)
  return constant_value if constant_value is not None else v


def _prepare_feed_values(model, inputs, mode):
  """Prepare feed values to the model execution function.

  Arguments:
    model: Model to prepare feed values for.
    inputs: An iterator of model inputs, targets, and sample_weights.
      model inputs may be lists, single values, or dicts mapping input feed
      names to values.
    mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.

  Returns:
    Feed values for the model in the given mode. This is a tuple of
    the structure (inputs, targets, sample_weights), where each of
    (tuple, targets, sample_weights) may be a python list. Single values
    for inputs will always be wrapped in lists.
  """
  inputs, targets, sample_weights = _get_input_from_iterator(inputs)

  # When the inputs are dict, then we want to flatten it in the same order as
  # the input layers, such that the data are fed into the input layers in the
  # correct order.
  if isinstance(inputs, dict):
    inputs = [inputs[key] for key in model._feed_input_names]
  else:
    inputs = training_utils.ModelInputs(inputs).as_list()

  if mode == ModeKeys.PREDICT:
    sample_weights = []
    targets = []

  ins = [inputs, targets, sample_weights]
  return tuple(ins)


def _get_input_from_iterator(iterator):
  """Get elements from the iterator and verify the input shape and type."""
  next_element = next(iterator)

  if (tensor_util.is_tensor(next_element) or
      isinstance(next_element, (dict, composite_tensor.CompositeTensor))):
    next_element = [next_element]
  if len(next_element) == 1:
    x, = next_element
    y = None
    sample_weights = None
  elif len(next_element) == 2:
    x, y = next_element
    sample_weights = None
  else:
    x, y, sample_weights = next_element

  # Validate that all the elements in x and y are of the same type and shape.
  dist_utils.validate_distributed_dataset_inputs(
      distribution_strategy_context.get_strategy(), x, y, sample_weights)
  return x, y, sample_weights


def _make_replica_execution_function(model, mode):
  """A single step of the distributed execution on a replica."""
  if mode == ModeKeys.TRAIN:
    func = functools.partial(train_on_batch, model)
  elif mode == ModeKeys.TEST:
    func = functools.partial(test_on_batch, model)
  else:
    def _predict_on_batch(x, y=None, sample_weights=None):
      del y, sample_weights
      return predict_on_batch(model, x)

    func = _predict_on_batch

  if mode != ModeKeys.PREDICT:
    # `reset_metrics` is set to False to maintain stateful metrics across
    # batch-level calls.
    func = functools.partial(func, reset_metrics=False)

  return func


def _prepare_model_with_inputs(model, dataset):
  """Use the data from the adapter to config the model.

  Model need to be properly configured before training, eg build with inputs, or
  compile with inputs for subclass model.

  Args:
    model: a Keras model object.
    dataset: a eager dataset instance where the data will be extracted.
  """
  if not model.inputs:
    inputs, target, _ = model._build_model_with_inputs(dataset, targets=None)
  else:
    inputs, target, _ = _get_input_from_iterator(iter(dataset))

  if not model._is_compiled and model.optimizer:
    model._compile_from_inputs(inputs, target, dataset, None)

  if target is not None:
    training_utils.prepare_sample_weight_modes(model._training_endpoints,
                                               model.sample_weight_mode)


def train_on_batch(
    model,
    x,
    y=None,
    sample_weight=None,
    class_weight=None,
    reset_metrics=True):
  """Runs a single gradient update on a single batch of data.

  Arguments:
      model: The model to train.
      x: Input data. It could be:
        - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
        - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
        - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
        - A `tf.data` dataset.
      y: Target data. Like the input data `x`, it could be either Numpy
        array(s) or TensorFlow tensor(s). It should be consistent with `x`
        (you cannot have Numpy inputs and tensor targets, or inversely). If
        `x` is a dataset `y` should not be specified
        (since targets will be obtained from the iterator).
      sample_weight: Optional array of the same length as x, containing
        weights to apply to the model's loss for each sample. In the case of
        temporal data, you can pass a 2D array with shape (samples,
        sequence_length), to apply a different weight to every timestep of
        every sample. In this case you should make sure to specify
        sample_weight_mode="temporal" in compile(). This argument is not
        supported when `x` is a dataset.
      class_weight: Optional dictionary mapping class indices (integers) to a
        weight (float) to apply to the model's loss for the samples from this
        class during training. This can be useful to tell the model to "pay
        more attention" to samples from an under-represented class.
      reset_metrics: If `True`, the metrics returned will be only for this
        batch. If `False`, the metrics will be statefully accumulated across
        batches.

  Returns:
      Scalar training loss
      (if the model has a single output and no metrics)
      or list of scalars (if the model has multiple outputs
      and/or metrics). The attribute `model.metrics_names` will give you
      the display labels for the scalar outputs.

  Raises:
    ValueError: In case of invalid user-provided arguments.
  """
  model._assert_compile_was_called()

  # TODO(scottzhu): Standardization should happen in the data handlers,
  ## not on a per batch basis in the *_on_batch methods
  # Validate and standardize user data.
  x, y, sample_weights = model._standardize_user_data(
      x, y, sample_weight=sample_weight, class_weight=class_weight,
      extract_tensors_from_dataset=True)
  batch_size = array_ops.shape(nest.flatten(x, expand_composites=True)[0])[0]
  # If `model._distribution_strategy` is True, then we are in a replica context
  # at this point because of the check above.  `train_on_batch` is being run
  # for each replica by `model._distribution_strategy` and the same code path
  # as Eager is expected to be taken.
  outputs = training_eager.train_on_batch(
      model,
      x,
      y,
      sample_weights=sample_weights,
      output_loss_metrics=model._output_loss_metrics)

  if reset_metrics:
    model.reset_metrics()

  outputs['batch_size'] = math_ops.cast(batch_size, dtypes.int64)
  return outputs


def test_on_batch(model, x, y=None, sample_weight=None, reset_metrics=True):
  """Test the model on a single batch of samples.

  Arguments:
      model: The model to test.
      x: Input data. It could be:
        - A Numpy array (or array-like), or a list of arrays
          (in case the model has multiple inputs).
        - A TensorFlow tensor, or a list of tensors
          (in case the model has multiple inputs).
        - A dict mapping input names to the corresponding array/tensors,
          if the model has named inputs.
        - A `tf.data` dataset.
      y: Target data. Like the input data `x`,
        it could be either Numpy array(s) or TensorFlow tensor(s).
        It should be consistent with `x` (you cannot have Numpy inputs and
        tensor targets, or inversely). If `x` is a dataset,
        `y` should not be specified
        (since targets will be obtained from the iterator).
      sample_weight: Optional array of the same length as x, containing
          weights to apply to the model's loss for each sample.
          In the case of temporal data, you can pass a 2D array
          with shape (samples, sequence_length),
          to apply a different weight to every timestep of every sample.
          In this case you should make sure to specify
          sample_weight_mode="temporal" in compile(). This argument is not
          supported when `x` is a dataset.
      reset_metrics: If `True`, the metrics returned will be only for this
        batch. If `False`, the metrics will be statefully accumulated across
        batches.

  Returns:
      Scalar test loss (if the model has a single output and no metrics)
      or list of scalars (if the model has multiple outputs
      and/or metrics). The attribute `model.metrics_names` will give you
      the display labels for the scalar outputs.

  Raises:
      ValueError: In case of invalid user-provided arguments.
  """
  model._assert_compile_was_called()

  # TODO(scottzhu): Standardization should happen in the data handlers,
  ## not on a per batch basis in the *_on_batch methods
  # Validate and standardize user data.
  x, y, sample_weights = model._standardize_user_data(
      x, y, sample_weight=sample_weight, extract_tensors_from_dataset=True)

  batch_size = array_ops.shape(nest.flatten(x, expand_composites=True)[0])[0]
  outputs = training_eager.test_on_batch(
      model,
      x,
      y,
      sample_weights=sample_weights,
      output_loss_metrics=model._output_loss_metrics)

  if reset_metrics:
    model.reset_metrics()

  outputs['batch_size'] = math_ops.cast(batch_size, dtypes.int64)
  return outputs


def predict_on_batch(model, x):
  """Returns predictions for a single batch of samples.

  Arguments:
      model: The model to predict with.
      x: Input data. It could be:
        - A Numpy array (or array-like), or a list of arrays
          (in case the model has multiple inputs).
        - A TensorFlow tensor, or a list of tensors
          (in case the model has multiple inputs).
        - A `tf.data` dataset.

  Returns:
      Numpy array(s) of predictions.

  Raises:
      ValueError: In case of mismatch between given number of inputs and
        expectations of the model.
  """
  # TODO(scottzhu): Standardization should happen in the data handlers,
  ## not on a per batch basis in the *_on_batch methods
  # Validate and standardize user data.
  inputs, _, _ = model._standardize_user_data(
      x, extract_tensors_from_dataset=True)

  # If `model._distribution_strategy` is True, then we are in a replica context
  # at this point.
  inputs = training_utils.cast_if_floating_dtype(inputs)
  if isinstance(inputs, collections.Sequence):
    # Unwrap lists with only one input, as we do when training on batch
    if len(inputs) == 1:
      inputs = inputs[0]

  with backend.eager_learning_phase_scope(0):
    return model(inputs)  # pylint: disable=not-callable
