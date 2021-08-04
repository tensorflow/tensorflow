# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Training-related utilities."""

import numpy as np

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest


def slice_arrays(arrays, indices, contiguous=True):
  """Slices batches out of provided arrays (workaround for eager tensors).

  Unfortunately eager tensors don't have the same slicing behavior as
  Numpy arrays (they follow the same slicing behavior as symbolic TF tensors),
  hence we cannot use `generic_utils.slice_arrays` directly
  and we have to implement this workaround based on `concat`. This has a
  performance cost.

  Args:
    arrays: Single array or list of arrays.
    indices: List of indices in the array that should be included in the output
      batch.
    contiguous: Boolean flag indicating whether the indices are contiguous.

  Returns:
    Slice of data (either single array or list of arrays).
  """
  converted_to_list = False
  if not isinstance(arrays, list):
    converted_to_list = True
    arrays = [arrays]
  if any(tensor_util.is_tf_type(x) for x in arrays):
    if not contiguous:
      entries = [[x[i:i + 1] for i in indices] for x in arrays]
      slices = [array_ops.concat(x, axis=0) for x in entries]
    else:
      slices = [x[indices[0]:indices[-1] + 1] for x in arrays]
  else:
    slices = generic_utils.slice_arrays(arrays, indices)

  if converted_to_list:
    slices = slices[0]
  return slices


def handle_partial_sample_weights(outputs, sample_weights, sample_weight_modes,
                                  check_all_flat=False):
  """Adds 1.0 as sample weights for the outputs for which there is no weight.

  Args:
    outputs: List of model outputs.
    sample_weights: List of sample weight inputs.
    sample_weight_modes: List of sample weight modes or None.
    check_all_flat: Ensure that inputs are not nested structures. This is not
      a free check, so we may not want to run it eagerly every iteration.

  Returns:
    Tuple of sample weights, one sample weight for every output, and booleans
    describing the raw sample weights.
  """
  any_sample_weight = sample_weights is not None and any(
      w is not None for w in sample_weights)
  partial_sample_weight = any_sample_weight and any(
      w is None for w in sample_weights)

  if not any_sample_weight:
    return None, any_sample_weight, partial_sample_weight

  if not partial_sample_weight:
    return sample_weights, any_sample_weight, partial_sample_weight

  if check_all_flat:
    nest.assert_same_structure(
        list_to_tuple(sample_weights),
        list_to_tuple(nest.flatten(sample_weights)))
    nest.assert_same_structure(
        list_to_tuple(outputs),
        list_to_tuple(nest.flatten(outputs)))
    if sample_weight_modes is not None:
      nest.assert_same_structure(
          sample_weight_modes, nest.flatten(sample_weight_modes))

  new_sample_weights = []
  for i, sw in enumerate(sample_weights):
    if sw is None:
      as_numpy = isinstance(outputs[i], np.ndarray)
      output = outputs[i]
      output_shape = output.shape if as_numpy else array_ops.shape(output)

      is_temporal = (
          sample_weight_modes is not None and
          sample_weight_modes[i] == 'temporal')
      sw_shape = (output_shape[0],
                  output_shape[1]) if is_temporal else (output_shape[0],)

      new_sample_weights.append(
          np.ones(sw_shape) if as_numpy else array_ops.ones(sw_shape))

    else:
      new_sample_weights.append(sw)
  return (list_to_tuple(new_sample_weights),
          any_sample_weight, partial_sample_weight)


class RespectCompiledTrainableState(object):
  """Set and restore trainable state if it has changed since compile.

  The keras API guarantees that the value of each Layer's `trainable` property
  at `Model.compile` time will be used when training that model. In order to
  respect this requirement, it may be necessary to set the trainable value of
  layers to their compile time values before beginning a training endpoint and
  restore the values before returing from said endpoint. This scope checks if
  any layer's trainable state has changed since Model compile, and performs this
  set and un-set bookkeeping.

  However, the trainable state of a layer changes quite infrequently, if ever,
  for many kinds of workflows. Moreover, updating every layer in a model is an
  expensive operation. As a result, we will only explicitly set and unset the
  trainable state of a model if a trainable value has changed since compile.
  """

  def __init__(self, model):
    self._model = model
    self._current_trainable_state = None
    self._compiled_trainable_state = None
    self._should_set_trainable = False

  def __enter__(self):
    self._current_trainable_state = self._model._get_trainable_state()  # pylint: disable=protected-access
    self._compiled_trainable_state = self._model._compiled_trainable_state  # pylint: disable=protected-access

    # Check to see if any layer's trainable state has changed since `compile`.
    for layer, trainable in self._compiled_trainable_state.items():
      if (layer in self._current_trainable_state and
          trainable != self._current_trainable_state[layer]):
        self._should_set_trainable = True
        break

    # If so, restore the model to its compiled state.
    if self._should_set_trainable:
      self._model._set_trainable_state(self._compiled_trainable_state)  # pylint: disable=protected-access

  def __exit__(self, type_arg, value_arg, traceback_arg):
    # If we set the values to their compiled state in __enter__, we need to
    # restore the original values before leaving the scope.
    if self._should_set_trainable:
      self._model._set_trainable_state(self._current_trainable_state)  # pylint: disable=protected-access
    return False  # False values do not suppress exceptions


# Allow use of methods not exposed to the user.
# pylint: disable=protected-access
def get_input_shape_and_dtype(layer):
  """Retrieves input shape and input dtype of layer if applicable.

  Args:
    layer: Layer (or model) instance.

  Returns:
    Tuple (input_shape, input_dtype). Both could be None if the layer
      does not have a defined input shape.

  Raises:
    ValueError: in case an empty Sequential or Functional model is passed.
  """

  def _is_graph_model(layer):
    return ((hasattr(layer, '_is_graph_network') and layer._is_graph_network) or
            layer.__class__.__name__ == 'Sequential')

  # In case of nested models: recover the first layer
  # of the deepest model to infer input shape and dtype.
  # Subclassed Models may not have been built so can't be checked.
  while _is_graph_model(layer):
    if not layer.layers:
      raise ValueError('An empty Model cannot be used as a Layer.')
    layer = layer.layers[0]

  if getattr(layer, '_batch_input_shape', None):
    return layer._batch_input_shape, layer.dtype
  return None, None


# pylint: enable=protected-access


def get_static_batch_size(layer):
  """Gets the static batch size of a Layer.

  Args:
    layer: a `Layer` instance.

  Returns:
    The static batch size of a Layer.
  """
  batch_input_shape, _ = get_input_shape_and_dtype(layer)
  if batch_input_shape is not None:
    return tensor_shape.Dimension(batch_input_shape[0]).value
  return None


def list_to_tuple(maybe_list):
  """Datasets will stack the list of tensor, so switch them to tuples."""
  if isinstance(maybe_list, list):
    return tuple(maybe_list)
  return maybe_list
