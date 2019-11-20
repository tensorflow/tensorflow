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

import numpy as np

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.ops import composite_tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import training_eager
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest


def _get_or_make_function(model, mode, key_fn, make_fn):
  """Helper function for managing cached execution functions."""
  model._init_distributed_function_cache_if_not_compiled()
  key = key_fn(mode)

  function = dist_utils.get_distributed_function(model, key)
  if function:
    return function

  function = make_fn(model, mode)
  dist_utils.set_distributed_function(model, key, function)
  return function


def _get_or_make_execution_function(model, mode):
  """Makes or reuses function to run one step of distributed model execution."""
  return _get_or_make_function(
      model, mode,
      # Use a key with 'v2' to distinguish from fall-back execution functions.
      key_fn=lambda m: (m, 'v2'),
      make_fn=_make_execution_function)


def _make_execution_function(model, mode):
  """Creates a function to run one step of distributed model execution."""
  per_replica_function = _make_replica_execution_function(model, mode)

  def distributed_function(input_iterator):
    """A single step of the distributed execution across replicas."""
    # Call `Model.{train,test,predict}_on_batch` on every replica passing
    # PerReplicas as arguments.  On every replica inside this call, each
    # PerReplica object will return the value for that replica.  The outputs
    # are PerReplicas too.
    strategy = distribution_strategy_context.get_strategy()
    args = _prepare_feed_values(model, input_iterator, mode, strategy)
    outputs = strategy.experimental_run_v2(
        per_replica_function, args=args)
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


def _get_or_make_on_batch_function(model, mode):
  """Makes or reuses function to run one step of distributed model execution."""
  return _get_or_make_function(
      model, mode,
      # Use a key with 'v2' to distinguish from fall-back execution functions.
      key_fn=lambda m: (m, 'v2_on_batch'),
      make_fn=_make_on_batch_function)


def _make_on_batch_function(model, mode):
  """Creates a function of Model.*_on_batch methods."""
  if mode == ModeKeys.TRAIN:
    func = training_eager.train_on_batch
  elif mode == ModeKeys.TEST:
    func = training_eager.test_on_batch
  else:
    func = model

  if not model.run_eagerly:
    func = def_function.function(func)

  return func


def _non_none_constant_value(v):
  constant_value = tensor_util.constant_value(v)
  return constant_value if constant_value is not None else v


def _prepare_feed_values(model, inputs, mode, strategy):
  """Prepare feed values to the model execution function.

  Arguments:
    model: Model to prepare feed values for.
    inputs: An iterator of model inputs, targets, and sample_weights.
      model inputs may be lists, single values, or dicts mapping input feed
      names to values.
    mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.
    strategy: The current distribution strategy for the model.

  Returns:
    Feed values for the model in the given mode. This is a tuple of
    the structure (inputs, targets, sample_weights), where each of
    (tuple, targets, sample_weights) may be a python list. Single values
    for inputs will always be wrapped in lists.
  """
  # For predict, we need to extract the manually added batch_index first.
  with_batch_index = _should_add_batch_index_to_element(strategy, mode)

  inputs, targets, sample_weights, batch_index = _get_input_from_iterator(
      inputs, with_batch_index)

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
  if batch_index is not None:
    ins.append(batch_index)
  return tuple(ins)


def _get_input_from_iterator(iterator, with_batch_index=False):
  """Get elements from the iterator and verify the input shape and type."""
  next_element = next(iterator)
  if with_batch_index:
    batch_index, next_element = next_element
  else:
    batch_index = None

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
  return x, y, sample_weights, batch_index


def _make_replica_execution_function(model, mode):
  """A single step of the distributed execution on a replica."""
  if mode == ModeKeys.TRAIN:
    func = functools.partial(train_on_batch, model)
  elif mode == ModeKeys.TEST:
    func = functools.partial(test_on_batch, model)
  else:
    def _predict_on_batch(x, y=None, sample_weights=None, batch_index=None):
      del y, sample_weights
      # Note that the x and batch_index is already per-replica value.
      result = predict_on_batch(model, x)
      if batch_index is None:
        return result
      else:
        return batch_index, result

    func = _predict_on_batch

  if mode != ModeKeys.PREDICT:
    # `reset_metrics` is set to False to maintain stateful metrics across
    # batch-level calls.
    func = functools.partial(func, reset_metrics=False)

  return func


def _aggregate_predict_results(strategy, batch_outs, model):
  """Aggregate the prediction result from each replica."""
  num_replicas = strategy.num_replicas_in_sync
  num_outputs = len(model.outputs)

  if not isinstance(batch_outs, list):
    batch_outs = [batch_outs]

  with_batch_index = _should_add_batch_index_to_element(
      strategy, ModeKeys.PREDICT)

  # batch_outs is in following structure:
  # [
  #  replica_1_batch_index, replica_2_batch_index, ...., replica_x_batch_index,
  #  replica_1_output_1, replica_2_output_1, ...., replica_x_output_1,
  #  ......
  #  replica_1_output_y, replica_2_output_y, ...., replica_x_output_y,
  # ]
  # The replica_x_batch_index is optional and depended on teh strategy type.
  if with_batch_index:
    batch_index, batch_outs = (batch_outs[:num_replicas],
                               batch_outs[num_replicas:])
    batch_index = dist_utils.concat_along_batch_dimension(batch_index)
    # Reorder the batch_index for it to do proper gather. Eg, if the original
    # index is [0, 2, 4, 6, 1, 3, 5, 7], then the index for gather should be
    # [0, 4, 1, 5, 2, 6, 3, 7].
    batch_index = np.argsort(batch_index)
    # Only need to gather if the batch index is not sorted.
    need_batch_index_gather = np.any(np.diff(batch_index) < 0)
  else:
    need_batch_index_gather = False

  total_batch_outs = []
  for i in range(num_outputs):
    nested_outs = batch_outs[i * num_replicas:i * num_replicas + num_replicas]
    per_output_result = dist_utils.concat_along_batch_dimension(
        nest.flatten(nested_outs))

    if need_batch_index_gather:
      if _get_batch_size(per_output_result).numpy() == len(batch_index):
        # Skip the gather if the output has a different batch size than the
        # batch_index. There will be some error handling in upper layer.
        per_output_result = _gather_result_by_index(per_output_result,
                                                    batch_index)
    total_batch_outs.append(per_output_result)
  return total_batch_outs


def _gather_result_by_index(input_tensor, batch_index):
  """Handle the data element gather for different type of tensor."""
  if isinstance(input_tensor, sparse_tensor.SparseTensor):
    # For sparse tensor, both the index and value component should be gathered.
    return sparse_tensor.SparseTensor(
        indices=array_ops.gather_v2(input_tensor.indices, batch_index),
        values=array_ops.gather_v2(input_tensor.values, batch_index),
        dense_shape=input_tensor.dense_shape
    )
  # For both ragged tensor or eager tensor or np array, tf.gather should do the
  # correct thing.
  elif isinstance(input_tensor, ragged_tensor.RaggedTensor):
    return array_ops.gather_v2(input_tensor, batch_index)
  elif isinstance(input_tensor, (ops.EagerTensor, np.ndarray)):
    return array_ops.gather_v2(input_tensor, batch_index).numpy()
  else:
    raise ValueError('Unexpected type {} encountered when gathering '
                     'batch slices.'.format(input_tensor))


def _get_batch_size(inputs):
  first_inputs = nest.flatten(inputs)[0]
  if isinstance(first_inputs, ragged_tensor.RaggedTensor):
    return first_inputs.bounding_shape()[0]
  else:
    return array_ops.shape(first_inputs)[0]


def _add_batch_index_to_element(dataset):
  """Adding a new batch index field to the every element in the batch.

  This is need in the model.predict() when running with multi-worker
  distribution strategy. When sharding/distributing a dataset, the continuity of
  the sharded dataset can't be easily ensured without performance sacrifice. It
  is fine to train and eval with the reordered data, but not for prediction. To
  solve this issue, Keras will add a batch index to each of the element in the
  dataset, which will then pass to pre-replica execution function. The real
  execution function will remove it before feeding the input to the model, and
  pre-replica function will then zip the index with the result. Finally Keras
  will sort the batch result based on the added batch-index field, remove it and
  return the sorted result.

  Note that we didn't add single index to the per-replica batch, but to each of
  the element in the batch, since we can't ensure the data in pre-replica is
  continuous. Eg: model with 2 replica and predict with 4 elements per batch
  like [1, 2, 3, 4], it is possible to shard as [1, 2], [3, 4],
  or [1, 3], [2, 4].

  Args:
    dataset: a dataset that is created by any of the data_adapter, with the
    element structure as (x, y, sample_weights).

  Returns:
    a new dataset, with the element shape as
    (batch_index, (x, y, sample_weights)).
  """
  return dataset.map(lambda *inp: (math_ops.range(_get_batch_size(inp)), inp))


def _should_add_batch_index_to_element(strategy, mode):
  """Whether or not the batch index should be added to the input dataset.

  See docstring of _add_batch_index_to_element() for more details. So far the
  batch index is only need when using TPUStrategy with a multi-worker setting.
  We will try to avoid adding batch index for other cases since it has the
  performance implication.

  Args:
    strategy: the current distribution strategy for the model.
    mode: the current mode (Training/Eval/Predict) for the model.
  Returns:
    Boolean, whether the batch index should be added for the input data to
      preserve the ordering.
  """
  # TODO(priyag, rxsang): Come up a better way to determine when the batch index
  # should be added.
  return (mode == ModeKeys.PREDICT
          and dist_utils.is_tpu_strategy(strategy)
          and strategy.extended.num_hosts > 1)


def train_on_batch(
    model,
    x,
    y=None,
    sample_weight=None,
    class_weight=None,
    reset_metrics=True,
    standalone=False):
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
      standalone: If True, this method is not called as part of
        Model.fit/evaluate/predict and can therefore be tf.function'd.

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

  if standalone:
    train_on_batch_fn = _get_or_make_on_batch_function(model, ModeKeys.TRAIN)
  else:
    train_on_batch_fn = training_eager.train_on_batch

  outputs = train_on_batch_fn(
      model,
      x,
      y,
      sample_weights=sample_weights,
      output_loss_metrics=model._output_loss_metrics)

  if reset_metrics:
    model.reset_metrics()

  outputs['batch_size'] = math_ops.cast(batch_size, dtypes.int64)
  return outputs


def test_on_batch(model, x, y=None, sample_weight=None, reset_metrics=True,
                  standalone=False):
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
      standalone: If True, this method is not called as part of
        Model.fit/evaluate/predict and can therefore be tf.function'd.

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

  if standalone:
    test_on_batch_fn = _get_or_make_on_batch_function(model, ModeKeys.TEST)
  else:
    test_on_batch_fn = training_eager.test_on_batch

  outputs = test_on_batch_fn(
      model,
      x,
      y,
      sample_weights=sample_weights,
      output_loss_metrics=model._output_loss_metrics)

  if reset_metrics:
    model.reset_metrics()

  outputs['batch_size'] = math_ops.cast(batch_size, dtypes.int64)
  return outputs


def predict_on_batch(model, x, standalone=False):
  """Returns predictions for a single batch of samples.

  Arguments:
      model: The model to predict with.
      x: Input data. It could be:
        - A Numpy array (or array-like), or a list of arrays
          (in case the model has multiple inputs).
        - A TensorFlow tensor, or a list of tensors
          (in case the model has multiple inputs).
        - A `tf.data` dataset.
      standalone: If True, this method is not called as part of
        Model.fit/evaluate/predict and can therefore be tf.function'd.

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
  inputs = training_utils.cast_to_model_input_dtypes(inputs, model)
  if isinstance(inputs, collections.Sequence):
    # Unwrap lists with only one input, as we do when training on batch
    if len(inputs) == 1:
      inputs = inputs[0]

  if standalone:
    predict_on_batch_fn = _get_or_make_on_batch_function(
        model, ModeKeys.PREDICT)
  else:
    predict_on_batch_fn = model

  with backend.eager_learning_phase_scope(0):
    return predict_on_batch_fn(inputs)  # pylint: disable=not-callable
