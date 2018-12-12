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
"""Recurrent layers and their base classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.layers.StackedRNNCells')
class StackedRNNCells(Layer):
  """Wrapper allowing a stack of RNN cells to behave as a single cell.

  Used to implement efficient stacked RNNs.

  Arguments:
      cells: List of RNN cell instances.

  Examples:

  ```python
      cells = [
          keras.layers.LSTMCell(output_dim),
          keras.layers.LSTMCell(output_dim),
          keras.layers.LSTMCell(output_dim),
      ]

      inputs = keras.Input((timesteps, input_dim))
      x = keras.layers.RNN(cells)(inputs)
  ```
  """

  def __init__(self, cells, **kwargs):
    for cell in cells:
      if not hasattr(cell, 'call'):
        raise ValueError('All cells must have a `call` method. '
                         'received cells:', cells)
      if not hasattr(cell, 'state_size'):
        raise ValueError('All cells must have a '
                         '`state_size` attribute. '
                         'received cells:', cells)
    self.cells = cells
    # reverse_state_order determines whether the state size will be in a reverse
    # order of the cells' state. User might want to set this to True to keep the
    # existing behavior. This is only useful when use RNN(return_state=True)
    # since the state will be returned as the same order of state_size.
    self.reverse_state_order = kwargs.pop('reverse_state_order', False)
    if self.reverse_state_order:
      logging.warning('reverse_state_order=True in StackedRNNCells will soon '
                      'be deprecated. Please update the code to work with the '
                      'natural order of states if you reply on the RNN states, '
                      'eg RNN(return_state=True).')
    super(StackedRNNCells, self).__init__(**kwargs)

  @property
  def state_size(self):
    return tuple(c.state_size for c in
                 (self.cells[::-1] if self.reverse_state_order else self.cells))

  @property
  def output_size(self):
    if getattr(self.cells[-1], 'output_size', None) is not None:
      return self.cells[-1].output_size
    elif _is_multiple_state(self.cells[-1].state_size):
      return self.cells[-1].state_size[0]
    else:
      return self.cells[-1].state_size

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    initial_states = []
    for cell in self.cells[::-1] if self.reverse_state_order else self.cells:
      get_initial_state_fn = getattr(cell, 'get_initial_state', None)
      if get_initial_state_fn:
        initial_states.append(get_initial_state_fn(
            inputs=inputs, batch_size=batch_size, dtype=dtype))
      else:
        initial_states.append(_generate_zero_filled_state_for_cell(
            cell, inputs, batch_size, dtype))

    return tuple(initial_states)

  def call(self, inputs, states, constants=None, **kwargs):
    # Recover per-cell states.
    state_size = (self.state_size[::-1]
                  if self.reverse_state_order else self.state_size)
    nested_states = nest.pack_sequence_as(state_size, nest.flatten(states))

    # Call the cells in order and store the returned states.
    new_nested_states = []
    for cell, states in zip(self.cells, nested_states):
      states = states if nest.is_sequence(states) else [states]
      if generic_utils.has_arg(cell.call, 'constants'):
        inputs, states = cell.call(inputs, states, constants=constants,
                                   **kwargs)
      else:
        inputs, states = cell.call(inputs, states, **kwargs)
      new_nested_states.append(states)

    return inputs, nest.pack_sequence_as(state_size,
                                         nest.flatten(new_nested_states))

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if isinstance(input_shape, list):
      constants_shape = input_shape[1:]
      input_shape = input_shape[0]
    for cell in self.cells:
      if isinstance(cell, Layer):
        if generic_utils.has_arg(cell.call, 'constants'):
          cell.build([input_shape] + constants_shape)
        else:
          cell.build(input_shape)
      if getattr(cell, 'output_size', None) is not None:
        output_dim = cell.output_size
      elif _is_multiple_state(cell.state_size):
        output_dim = cell.state_size[0]
      else:
        output_dim = cell.state_size
      input_shape = tuple([input_shape[0]] +
                          tensor_shape.as_shape(output_dim).as_list())
    self.built = True

  def get_config(self):
    cells = []
    for cell in self.cells:
      cells.append({
          'class_name': cell.__class__.__name__,
          'config': cell.get_config()
      })
    config = {'cells': cells}
    base_config = super(StackedRNNCells, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    cells = []
    for cell_config in config.pop('cells'):
      cells.append(
          deserialize_layer(cell_config, custom_objects=custom_objects))
    return cls(cells, **config)

  @property
  def trainable_weights(self):
    if not self.trainable:
      return []
    weights = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        weights += cell.trainable_weights
    return weights

  @property
  def non_trainable_weights(self):
    weights = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        weights += cell.non_trainable_weights
    if not self.trainable:
      trainable_weights = []
      for cell in self.cells:
        if isinstance(cell, Layer):
          trainable_weights += cell.trainable_weights
      return trainable_weights + weights
    return weights

  def get_weights(self):
    """Retrieves the weights of the model.

    Returns:
        A flat list of Numpy arrays.
    """
    weights = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        weights += cell.weights
    return K.batch_get_value(weights)

  def set_weights(self, weights):
    """Sets the weights of the model.

    Arguments:
        weights: A list of Numpy arrays with shapes and types matching
            the output of `model.get_weights()`.
    """
    tuples = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        num_param = len(cell.weights)
        weights = weights[:num_param]
        for sw, w in zip(cell.weights, weights):
          tuples.append((sw, w))
        weights = weights[num_param:]
    K.batch_set_value(tuples)

  @property
  def losses(self):
    losses = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        losses += cell.losses
    return losses + self._losses

  @property
  def updates(self):
    updates = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        updates += cell.updates
    return updates + self._updates


@tf_export('keras.layers.RNN')
class RNN(Layer):
  """Base class for recurrent layers.

  Arguments:
      cell: A RNN cell instance or a list of RNN cell instances.
          A RNN cell is a class that has:
          - a `call(input_at_t, states_at_t)` method, returning
              `(output_at_t, states_at_t_plus_1)`. The call method of the
              cell can also take the optional argument `constants`, see
              section "Note on passing external constants" below.
          - a `state_size` attribute. This can be a single integer
              (single state) in which case it is the size of the recurrent
              state. This can also be a list/tuple of integers (one size per
              state).
              The `state_size` can also be TensorShape or tuple/list of
              TensorShape, to represent high dimension state.
          - a `output_size` attribute. This can be a single integer or a
              TensorShape, which represent the shape of the output. For backward
              compatible reason, if this attribute is not available for the
              cell, the value will be inferred by the first element of the
              `state_size`.
          - a `get_initial_state(inputs=None, batch_size=None, dtype=None)`
              method that creates a tensor meant to be fed to `call()` as the
              initial state, if user didn't specify any initial state via other
              means. The returned initial state should be in shape of
              [batch, cell.state_size]. Cell might choose to create zero filled
              tensor, or with other values based on the cell implementations.
              `inputs` is the input tensor to the RNN layer, which should
              contain the batch size as its shape[0], and also dtype. Note that
              the shape[0] might be None during the graph construction. Either
              the `inputs` or the pair of `batch` and `dtype `are provided.
              `batch` is a scalar tensor that represent the batch size
              of the input. `dtype` is `tf.dtype` that represent the dtype of
              the input.
              For backward compatible reason, if this method is not implemented
              by the cell, RNN layer will create a zero filled tensors with the
              size of [batch, cell.state_size].
          In the case that `cell` is a list of RNN cell instances, the cells
          will be stacked on after the other in the RNN, implementing an
          efficient stacked RNN.
      return_sequences: Boolean. Whether to return the last output
          in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
          in addition to the output.
      go_backwards: Boolean (default False).
          If True, process the input sequence backwards and return the
          reversed sequence.
      stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.
      unroll: Boolean (default False).
          If True, the network will be unrolled,
          else a symbolic loop will be used.
          Unrolling can speed-up a RNN,
          although it tends to be more memory-intensive.
          Unrolling is only suitable for short sequences.
      input_dim: dimensionality of the input (integer or tuple of integers).
          This argument (or alternatively, the keyword argument `input_shape`)
          is required when using this layer as the first layer in a model.
      input_length: Length of input sequences, to be specified
          when it is constant.
          This argument is required if you are going to connect
          `Flatten` then `Dense` layers upstream
          (without it, the shape of the dense outputs cannot be computed).
          Note that if the recurrent layer is not the first layer
          in your model, you would need to specify the input length
          at the level of the first layer
          (e.g. via the `input_shape` argument)
      time_major: The shape format of the `inputs` and `outputs` tensors.
          If True, the inputs and outputs will be in shape
          `(timesteps, batch, ...)`, whereas in the False case, it will be
          `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
          efficient because it avoids transposes at the beginning and end of the
          RNN calculation. However, most TensorFlow data is batch-major, so by
          default this function accepts input and emits output in batch-major
          form.

  Input shape:
      N-D tensor with shape `(batch_size, timesteps, ...)` or
      `(timesteps, batch_size, ...)` when time_major is True.

  Output shape:
      - if `return_state`: a list of tensors. The first tensor is
          the output. The remaining tensors are the last states,
          each with shape `(batch_size, state_size)`, where `state_size` could
          be a high dimension tensor shape.
      - if `return_sequences`: N-D tensor with shape
          `(batch_size, timesteps, output_size)`, where `output_size` could
          be a high dimension tensor shape, or
          `(timesteps, batch_size, output_size)` when `time_major` is True.
      - else, N-D tensor with shape `(batch_size, output_size)`, where
          `output_size` could be a high dimension tensor shape.

  # Masking
      This layer supports masking for input data with a variable number
      of timesteps. To introduce masks to your data,
      use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
      set to `True`.

  # Note on using statefulness in RNNs
      You can set RNN layers to be 'stateful', which means that the states
      computed for the samples in one batch will be reused as initial states
      for the samples in the next batch. This assumes a one-to-one mapping
      between samples in different successive batches.

      To enable statefulness:
          - specify `stateful=True` in the layer constructor.
          - specify a fixed batch size for your model, by passing
              if sequential model:
                `batch_input_shape=(...)` to the first layer in your model.
              else for functional model with 1 or more Input layers:
                `batch_shape=(...)` to all the first layers in your model.
              This is the expected shape of your inputs
              *including the batch size*.
              It should be a tuple of integers, e.g. `(32, 10, 100)`.
          - specify `shuffle=False` when calling fit().

      To reset the states of your model, call `.reset_states()` on either
      a specific layer, or on your entire model.

  # Note on specifying the initial state of RNNs
      You can specify the initial state of RNN layers symbolically by
      calling them with the keyword argument `initial_state`. The value of
      `initial_state` should be a tensor or list of tensors representing
      the initial state of the RNN layer.

      You can specify the initial state of RNN layers numerically by
      calling `reset_states` with the keyword argument `states`. The value of
      `states` should be a numpy array or list of numpy arrays representing
      the initial state of the RNN layer.

  # Note on passing external constants to RNNs
      You can pass "external" constants to the cell using the `constants`
      keyword argument of `RNN.__call__` (as well as `RNN.call`) method. This
      requires that the `cell.call` method accepts the same keyword argument
      `constants`. Such constants can be used to condition the cell
      transformation on additional static inputs (not changing over time),
      a.k.a. an attention mechanism.

  Examples:

  ```python
      # First, let's define a RNN Cell, as a layer subclass.

      class MinimalRNNCell(keras.layers.Layer):

          def __init__(self, units, **kwargs):
              self.units = units
              self.state_size = units
              super(MinimalRNNCell, self).__init__(**kwargs)

          def build(self, input_shape):
              self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                            initializer='uniform',
                                            name='kernel')
              self.recurrent_kernel = self.add_weight(
                  shape=(self.units, self.units),
                  initializer='uniform',
                  name='recurrent_kernel')
              self.built = True

          def call(self, inputs, states):
              prev_output = states[0]
              h = K.dot(inputs, self.kernel)
              output = h + K.dot(prev_output, self.recurrent_kernel)
              return output, [output]

      # Let's use this cell in a RNN layer:

      cell = MinimalRNNCell(32)
      x = keras.Input((None, 5))
      layer = RNN(cell)
      y = layer(x)

      # Here's how to use the cell to build a stacked RNN:

      cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
      x = keras.Input((None, 5))
      layer = RNN(cells)
      y = layer(x)
  ```
  """

  def __init__(self,
               cell,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               time_major=False,
               **kwargs):
    if isinstance(cell, (list, tuple)):
      cell = StackedRNNCells(cell)
    if not hasattr(cell, 'call'):
      raise ValueError('`cell` should have a `call` method. '
                       'The RNN was passed:', cell)
    if not hasattr(cell, 'state_size'):
      raise ValueError('The RNN cell should have '
                       'an attribute `state_size` '
                       '(tuple of integers, '
                       'one integer per RNN state).')
    # If True, the output for masked timestep will be zeros, whereas in the
    # False case, output from previous timestep is returned for masked timestep.
    self.zero_output_for_mask = kwargs.pop('zero_output_for_mask', False)
    super(RNN, self).__init__(**kwargs)
    self.cell = cell
    if isinstance(cell, checkpointable.CheckpointableBase):
      self._track_checkpointable(self.cell, name='cell')
    self.return_sequences = return_sequences
    self.return_state = return_state
    self.go_backwards = go_backwards
    self.stateful = stateful
    self.unroll = unroll
    self.time_major = time_major

    self.supports_masking = True
    # The input shape is unknown yet, it could have nested tensor inputs, and
    # the input spec will be the list of specs for flattened inputs.
    self.input_spec = None
    self.state_spec = None
    self._states = None
    self.constants_spec = None
    self._num_constants = None
    self._num_inputs = None

  @property
  def states(self):
    if self._states is None:
      state = nest.map_structure(lambda _: None, self.cell.state_size)
      return state if nest.is_sequence(self.cell.state_size) else [state]
    return self._states

  @states.setter
  def states(self, states):
    self._states = states

  def compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    # Check whether the input shape contains any nested shapes. It could be
    # (tensor_shape(1, 2), tensor_shape(3, 4)) or (1, 2, 3) which is from numpy
    # inputs.
    try:
      input_shape = tensor_shape.as_shape(input_shape)
    except (ValueError, TypeError):
      # A nested tensor input
      input_shape = nest.flatten(input_shape)[0]

    batch = input_shape[0]
    time_step = input_shape[1]
    if self.time_major:
      batch, time_step = time_step, batch

    if _is_multiple_state(self.cell.state_size):
      state_size = self.cell.state_size
    else:
      state_size = [self.cell.state_size]

    def _get_output_shape(flat_output_size):
      output_dim = tensor_shape.as_shape(flat_output_size).as_list()
      if self.return_sequences:
        if self.time_major:
          output_shape = tensor_shape.as_shape([time_step, batch] + output_dim)
        else:
          output_shape = tensor_shape.as_shape([batch, time_step] + output_dim)
      else:
        output_shape = tensor_shape.as_shape([batch] + output_dim)
      return output_shape

    if getattr(self.cell, 'output_size', None) is not None:
      # cell.output_size could be nested structure.
      output_shape = nest.flatten(nest.map_structure(
          _get_output_shape, self.cell.output_size))
      output_shape = output_shape[0] if len(output_shape) == 1 else output_shape
    else:
      # Note that state_size[0] could be a tensor_shape or int.
      output_shape = _get_output_shape(state_size[0])

    if self.return_state:
      def _get_state_shape(flat_state):
        state_shape = [batch] + tensor_shape.as_shape(flat_state).as_list()
        return tensor_shape.as_shape(state_shape)
      state_shape = nest.map_structure(_get_state_shape, state_size)
      return generic_utils.to_list(output_shape) + nest.flatten(state_shape)
    else:
      return output_shape

  def compute_mask(self, inputs, mask):
    if isinstance(mask, list):
      mask = mask[0]
    output_mask = mask if self.return_sequences else None
    if self.return_state:
      state_mask = [None for _ in self.states]
      return [output_mask] + state_mask
    else:
      return output_mask

  def build(self, input_shape):
    # Note input_shape will be list of shapes of initial states and
    # constants if these are passed in __call__.
    if self._num_constants is not None:
      constants_shape = input_shape[-self._num_constants:]  # pylint: disable=invalid-unary-operand-type
      constants_shape = nest.map_structure(
          lambda s: tuple(tensor_shape.TensorShape(s).as_list()),
          constants_shape)
    else:
      constants_shape = None

    if isinstance(input_shape, list):
      input_shape = input_shape[0]
      # The input_shape here could be a nest structure.

    # do the tensor_shape to shapes here. The input could be single tensor, or a
    # nested structure of tensors.
    def get_input_spec(shape):
      if isinstance(shape, tensor_shape.TensorShape):
        input_spec_shape = shape.as_list()
      else:
        input_spec_shape = list(shape)
      batch_index, time_step_index = (1, 0) if self.time_major else (0, 1)
      if not self.stateful:
        input_spec_shape[batch_index] = None
      input_spec_shape[time_step_index] = None
      return InputSpec(shape=tuple(input_spec_shape))

    def get_step_input_shape(shape):
      if isinstance(shape, tensor_shape.TensorShape):
        shape = tuple(shape.as_list())
      # remove the timestep from the input_shape
      return shape[1:] if self.time_major else (shape[0],) + shape[2:]

    # Check whether the input shape contains any nested shapes. It could be
    # (tensor_shape(1, 2), tensor_shape(3, 4)) or (1, 2, 3) which is from numpy
    # inputs.
    try:
      input_shape = tensor_shape.as_shape(input_shape)
    except (ValueError, TypeError):
      # A nested tensor input
      pass

    if not nest.is_sequence(input_shape):
      # This indicates the there is only one input.
      if self.input_spec is not None:
        self.input_spec[0] = get_input_spec(input_shape)
      else:
        self.input_spec = [get_input_spec(input_shape)]
      step_input_shape = get_step_input_shape(input_shape)
    else:
      flat_input_shapes = nest.flatten(input_shape)
      flat_input_shapes = nest.map_structure(get_input_spec, flat_input_shapes)
      assert len(flat_input_shapes) == self._num_inputs
      if self.input_spec is not None:
        self.input_spec[:self._num_inputs] = flat_input_shapes
      else:
        self.input_spec = flat_input_shapes
      step_input_shape = nest.map_structure(get_step_input_shape, input_shape)

    # allow cell (if layer) to build before we set or validate state_spec
    if isinstance(self.cell, Layer):
      if constants_shape is not None:
        self.cell.build([step_input_shape] + constants_shape)
      else:
        self.cell.build(step_input_shape)

    # set or validate state_spec
    if _is_multiple_state(self.cell.state_size):
      state_size = list(self.cell.state_size)
    else:
      state_size = [self.cell.state_size]

    if self.state_spec is not None:
      # initial_state was passed in call, check compatibility
      self._validate_state_spec(state_size, self.state_spec)
    else:
      self.state_spec = [
          InputSpec(shape=[None] + tensor_shape.as_shape(dim).as_list())
          for dim in state_size
      ]
    if self.stateful:
      self.reset_states()
    self.built = True

  @staticmethod
  def _validate_state_spec(cell_state_sizes, init_state_specs):
    """Validate the state spec between the initial_state and the state_size.

    Args:
      cell_state_sizes: list, the `state_size` attribute from the cell.
      init_state_specs: list, the `state_spec` from the initial_state that is
        passed in call()

    Raises:
      ValueError: When initial state spec is not compatible with the state size.
    """
    validation_error = ValueError(
        'An `initial_state` was passed that is not compatible with '
        '`cell.state_size`. Received `state_spec`={}; '
        'however `cell.state_size` is '
        '{}'.format(init_state_specs, cell_state_sizes))
    if len(cell_state_sizes) == len(init_state_specs):
      for i in range(len(cell_state_sizes)):
        if not tensor_shape.TensorShape(
            # Ignore the first axis for init_state which is for batch
            init_state_specs[i].shape[1:]).is_compatible_with(
                tensor_shape.TensorShape(cell_state_sizes[i])):
          raise validation_error
    else:
      raise validation_error

  def get_initial_state(self, inputs):
    get_initial_state_fn = getattr(self.cell, 'get_initial_state', None)

    if nest.is_sequence(inputs):
      # The input are nested sequences. Use the first element in the seq to get
      # batch size and dtype.
      inputs = nest.flatten(inputs)[0]

    input_shape = array_ops.shape(inputs)
    batch_size = input_shape[1] if self.time_major else input_shape[0]
    dtype = inputs.dtype
    if get_initial_state_fn:
      init_state = get_initial_state_fn(
          inputs=None, batch_size=batch_size, dtype=dtype)
    else:
      init_state = _generate_zero_filled_state(batch_size, self.cell.state_size,
                                               dtype)
    # Keras RNN expect the states in a list, even if it's a single state tensor.
    if not nest.is_sequence(init_state):
      init_state = [init_state]
    # Force the state to be a list in case it is a namedtuple eg LSTMStateTuple.
    return list(init_state)

  def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
    inputs, initial_state, constants = _standardize_args(inputs,
                                                         initial_state,
                                                         constants,
                                                         self._num_constants,
                                                         self._num_inputs)
    # in case the real inputs is a nested structure, set the size of flatten
    # input so that we can distinguish between real inputs, initial_state and
    # constants.
    self._num_inputs = len(nest.flatten(inputs))

    if initial_state is None and constants is None:
      return super(RNN, self).__call__(inputs, **kwargs)

    # If any of `initial_state` or `constants` are specified and are Keras
    # tensors, then add them to the inputs and temporarily modify the
    # input_spec to include them.

    additional_inputs = []
    additional_specs = []
    if initial_state is not None:
      additional_inputs += initial_state
      self.state_spec = [
          InputSpec(shape=K.int_shape(state)) for state in initial_state
      ]
      additional_specs += self.state_spec
    if constants is not None:
      additional_inputs += constants
      self.constants_spec = [
          InputSpec(shape=K.int_shape(constant)) for constant in constants
      ]
      self._num_constants = len(constants)
      additional_specs += self.constants_spec
    # at this point additional_inputs cannot be empty
    is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
    for tensor in additional_inputs:
      if K.is_keras_tensor(tensor) != is_keras_tensor:
        raise ValueError('The initial state or constants of an RNN'
                         ' layer cannot be specified with a mix of'
                         ' Keras tensors and non-Keras tensors'
                         ' (a "Keras tensor" is a tensor that was'
                         ' returned by a Keras layer, or by `Input`)')

    if is_keras_tensor:
      # Compute the full input spec, including state and constants
      full_input = [inputs] + additional_inputs
      # The original input_spec is None since there could be a nested tensor
      # input. Update the input_spec to match the inputs.
      full_input_spec = [None for _ in range(len(nest.flatten(inputs)))
                        ] + additional_specs
      # Perform the call with temporarily replaced input_spec
      original_input_spec = self.input_spec
      self.input_spec = full_input_spec
      output = super(RNN, self).__call__(full_input, **kwargs)
      self.input_spec = original_input_spec
      return output
    else:
      if initial_state is not None:
        kwargs['initial_state'] = initial_state
      if constants is not None:
        kwargs['constants'] = constants
      return super(RNN, self).__call__(inputs, **kwargs)

  def call(self,
           inputs,
           mask=None,
           training=None,
           initial_state=None,
           constants=None):
    inputs, initial_state, constants = self._process_inputs(
        inputs, initial_state, constants)

    if isinstance(mask, list):
      mask = mask[0]

    if nest.is_sequence(inputs):
      # In the case of nested input, use the first element for shape check.
      input_shape = K.int_shape(nest.flatten(inputs)[0])
    else:
      input_shape = K.int_shape(inputs)
    timesteps = input_shape[0] if self.time_major else input_shape[1]
    if self.unroll and timesteps in [None, 1]:
      raise ValueError('Cannot unroll a RNN if the '
                       'time dimension is undefined or equal to 1. \n'
                       '- If using a Sequential model, '
                       'specify the time dimension by passing '
                       'an `input_shape` or `batch_input_shape` '
                       'argument to your first layer. If your '
                       'first layer is an Embedding, you can '
                       'also use the `input_length` argument.\n'
                       '- If using the functional API, specify '
                       'the time dimension by passing a `shape` '
                       'or `batch_shape` argument to your Input layer.')

    kwargs = {}
    if generic_utils.has_arg(self.cell.call, 'training'):
      kwargs['training'] = training

    # TF RNN cells expect single tensor as state instead of list wrapped tensor.
    is_tf_rnn_cell = getattr(self.cell, '_is_tf_rnn_cell', None) is not None
    if constants:
      if not generic_utils.has_arg(self.cell.call, 'constants'):
        raise ValueError('RNN cell does not support constants')

      def step(inputs, states):
        constants = states[-self._num_constants:]  # pylint: disable=invalid-unary-operand-type
        states = states[:-self._num_constants]  # pylint: disable=invalid-unary-operand-type

        states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
        output, new_states = self.cell.call(
            inputs, states, constants=constants, **kwargs)
        if not nest.is_sequence(new_states):
          new_states = [new_states]
        return output, new_states
    else:

      def step(inputs, states):
        states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
        output, new_states = self.cell.call(inputs, states, **kwargs)
        if not nest.is_sequence(new_states):
          new_states = [new_states]
        return output, new_states

    last_output, outputs, states = K.rnn(
        step,
        inputs,
        initial_state,
        constants=constants,
        go_backwards=self.go_backwards,
        mask=mask,
        unroll=self.unroll,
        input_length=timesteps,
        time_major=self.time_major,
        zero_output_for_mask=self.zero_output_for_mask)
    if self.stateful:
      updates = []
      for i in range(len(states)):
        updates.append(state_ops.assign(self.states[i], states[i]))
      self.add_update(updates, inputs)

    if self.return_sequences:
      output = outputs
    else:
      output = last_output

    if self.return_state:
      if not isinstance(states, (list, tuple)):
        states = [states]
      else:
        states = list(states)
      return generic_utils.to_list(output) + states
    else:
      return output

  def _process_inputs(self, inputs, initial_state, constants):
    # input shape: `(samples, time (padded with zeros), input_dim)`
    # note that the .build() method of subclasses MUST define
    # self.input_spec and self.state_spec with complete input shapes.
    if isinstance(inputs, list):
      # get initial_state from full input spec
      # as they could be copied to multiple GPU.
      if self._num_constants is None:
        initial_state = inputs[1:]
      else:
        initial_state = inputs[1:-self._num_constants]
        constants = inputs[-self._num_constants:]
      if len(initial_state) == 0:
        initial_state = None
      inputs = inputs[0]
    if initial_state is not None:
      pass
    elif self.stateful:
      initial_state = self.states
    else:
      initial_state = self.get_initial_state(inputs)

    if len(initial_state) != len(self.states):
      raise ValueError('Layer has ' + str(len(self.states)) +
                       ' states but was passed ' + str(len(initial_state)) +
                       ' initial states.')
    return inputs, initial_state, constants

  def reset_states(self, states=None):
    if not self.stateful:
      raise AttributeError('Layer must be stateful.')
    if self.time_major:
      batch_size = self.input_spec[0].shape[1]
    else:
      batch_size = self.input_spec[0].shape[0]
    if not batch_size:
      raise ValueError('If a RNN is stateful, it needs to know '
                       'its batch size. Specify the batch size '
                       'of your input tensors: \n'
                       '- If using a Sequential model, '
                       'specify the batch size by passing '
                       'a `batch_input_shape` '
                       'argument to your first layer.\n'
                       '- If using the functional API, specify '
                       'the batch size by passing a '
                       '`batch_shape` argument to your Input layer.')
    # initialize state if None
    if self.states[0] is None:
      if _is_multiple_state(self.cell.state_size):
        self.states = [
            K.zeros([batch_size] + tensor_shape.as_shape(dim).as_list())
            for dim in self.cell.state_size
        ]
      else:
        self.states = [
            K.zeros([batch_size] +
                    tensor_shape.as_shape(self.cell.state_size).as_list())
        ]
    elif states is None:
      if _is_multiple_state(self.cell.state_size):
        for state, dim in zip(self.states, self.cell.state_size):
          K.set_value(state,
                      np.zeros([batch_size] +
                               tensor_shape.as_shape(dim).as_list()))
      else:
        K.set_value(self.states[0], np.zeros(
            [batch_size] +
            tensor_shape.as_shape(self.cell.state_size).as_list()))
    else:
      if not isinstance(states, (list, tuple)):
        states = [states]
      if len(states) != len(self.states):
        raise ValueError('Layer ' + self.name + ' expects ' +
                         str(len(self.states)) + ' states, '
                         'but it received ' + str(len(states)) +
                         ' state values. Input received: ' + str(states))
      for index, (value, state) in enumerate(zip(states, self.states)):
        if _is_multiple_state(self.cell.state_size):
          dim = self.cell.state_size[index]
        else:
          dim = self.cell.state_size
        if value.shape != tuple([batch_size] +
                                tensor_shape.as_shape(dim).as_list()):
          raise ValueError(
              'State ' + str(index) + ' is incompatible with layer ' +
              self.name + ': expected shape=' + str(
                  (batch_size, dim)) + ', found shape=' + str(value.shape))
        # TODO(fchollet): consider batch calls to `set_value`.
        K.set_value(state, value)

  def get_config(self):
    config = {
        'return_sequences': self.return_sequences,
        'return_state': self.return_state,
        'go_backwards': self.go_backwards,
        'stateful': self.stateful,
        'unroll': self.unroll,
        'time_major': self.time_major
    }
    if self._num_constants is not None:
      config['num_constants'] = self._num_constants
    if self.zero_output_for_mask:
      config['zero_output_for_mask'] = self.zero_output_for_mask

    cell_config = self.cell.get_config()
    config['cell'] = {
        'class_name': self.cell.__class__.__name__,
        'config': cell_config
    }
    base_config = super(RNN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    cell = deserialize_layer(config.pop('cell'), custom_objects=custom_objects)
    num_constants = config.pop('num_constants', None)
    layer = cls(cell, **config)
    layer._num_constants = num_constants
    return layer

  @property
  def trainable_weights(self):
    if not self.trainable:
      return []
    if isinstance(self.cell, Layer):
      return self.cell.trainable_weights
    return []

  @property
  def non_trainable_weights(self):
    if isinstance(self.cell, Layer):
      if not self.trainable:
        return self.cell.weights
      return self.cell.non_trainable_weights
    return []

  @property
  def losses(self):
    layer_losses = super(RNN, self).losses
    if isinstance(self.cell, Layer):
      return self.cell.losses + layer_losses
    return layer_losses

  @property
  def updates(self):
    updates = []
    if isinstance(self.cell, Layer):
      updates += self.cell.updates
    return updates + self._updates


@tf_export('keras.layers.SimpleRNNCell')
class SimpleRNNCell(Layer):
  """Cell class for SimpleRNN.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          Default: hyperbolic tangent (`tanh`).
          If you pass `None`, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
          the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
          the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.
  """

  def __init__(self,
               units,
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
    super(SimpleRNNCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.state_size = self.units
    self.output_size = self.units
    self._dropout_mask = None
    self._recurrent_dropout_mask = None

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    self.kernel = self.add_weight(
        shape=(input_shape[-1], self.units),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.units,),
          name='bias',
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, states, training=None):
    prev_output = states[0]
    if 0 < self.dropout < 1 and self._dropout_mask is None:
      self._dropout_mask = _generate_dropout_mask(
          array_ops.ones_like(inputs),
          self.dropout,
          training=training)
    if (0 < self.recurrent_dropout < 1 and
        self._recurrent_dropout_mask is None):
      self._recurrent_dropout_mask = _generate_dropout_mask(
          array_ops.ones_like(prev_output),
          self.recurrent_dropout,
          training=training)

    dp_mask = self._dropout_mask
    rec_dp_mask = self._recurrent_dropout_mask

    if dp_mask is not None:
      h = K.dot(inputs * dp_mask, self.kernel)
    else:
      h = K.dot(inputs, self.kernel)
    if self.bias is not None:
      h = K.bias_add(h, self.bias)

    if rec_dp_mask is not None:
      prev_output *= rec_dp_mask
    output = h + K.dot(prev_output, self.recurrent_kernel)
    if self.activation is not None:
      output = self.activation(output)

    return output, [output]

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout
    }
    base_config = super(SimpleRNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.SimpleRNN')
class SimpleRNN(RNN):
  """Fully-connected RNN where the output is to be fed back to input.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          Default: hyperbolic tangent (`tanh`).
          If you pass None, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
          the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
          the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.
      return_sequences: Boolean. Whether to return the last output
          in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
          in addition to the output.
      go_backwards: Boolean (default False).
          If True, process the input sequence backwards and return the
          reversed sequence.
      stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.
      unroll: Boolean (default False).
          If True, the network will be unrolled,
          else a symbolic loop will be used.
          Unrolling can speed-up a RNN,
          although it tends to be more memory-intensive.
          Unrolling is only suitable for short sequences.
  """

  def __init__(self,
               units,
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if 'implementation' in kwargs:
      kwargs.pop('implementation')
      logging.warning('The `implementation` argument '
                      'in `SimpleRNN` has been deprecated. '
                      'Please remove it from your layer call.')
    cell = SimpleRNNCell(
        units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout)
    super(SimpleRNN, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self.cell._dropout_mask = None
    self.cell._recurrent_dropout_mask = None
    return super(SimpleRNN, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout
    }
    base_config = super(SimpleRNN, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config:
      config.pop('implementation')
    return cls(**config)


@tf_export('keras.layers.GRUCell')
class GRUCell(Layer):
  """Cell class for the GRU layer.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          Default: hyperbolic tangent (`tanh`).
          If you pass None, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
          for the recurrent step.
          Default: hard sigmoid (`hard_sigmoid`).
          If you pass `None`, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
          the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
          the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
          Mode 1 will structure its operations as a larger number of
          smaller dot products and additions, whereas mode 2 will
          batch them into fewer, larger operations. These modes will
          have different performance profiles on different hardware and
          for different applications.
      reset_after: GRU convention (whether to apply reset gate after or
          before matrix multiplication). False = "before" (default),
          True = "after" (CuDNN compatible).
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               reset_after=False,
               **kwargs):
    super(GRUCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.implementation = implementation
    self.reset_after = reset_after
    self.state_size = self.units
    self.output_size = self.units
    self._dropout_mask = None
    self._recurrent_dropout_mask = None

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 3),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 3),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      if not self.reset_after:
        bias_shape = (3 * self.units,)
      else:
        # separate biases for input and recurrent kernels
        # Note: the shape is intentionally different from CuDNNGRU biases
        # `(2 * 3 * self.units,)`, so that we can distinguish the classes
        # when loading and converting saved weights.
        bias_shape = (2, 3 * self.units)
      self.bias = self.add_weight(shape=bias_shape,
                                  name='bias',
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, states, training=None):
    h_tm1 = states[0]  # previous memory

    if 0 < self.dropout < 1 and self._dropout_mask is None:
      self._dropout_mask = _generate_dropout_mask(
          array_ops.ones_like(inputs),
          self.dropout,
          training=training,
          count=3)
    if (0 < self.recurrent_dropout < 1 and
        self._recurrent_dropout_mask is None):
      self._recurrent_dropout_mask = _generate_dropout_mask(
          array_ops.ones_like(h_tm1),
          self.recurrent_dropout,
          training=training,
          count=3)

    # dropout matrices for input units
    dp_mask = self._dropout_mask
    # dropout matrices for recurrent units
    rec_dp_mask = self._recurrent_dropout_mask

    if self.use_bias:
      if not self.reset_after:
        input_bias, recurrent_bias = self.bias, None
      else:
        input_bias, recurrent_bias = array_ops.unstack(self.bias)

    if self.implementation == 1:
      if 0. < self.dropout < 1.:
        inputs_z = inputs * dp_mask[0]
        inputs_r = inputs * dp_mask[1]
        inputs_h = inputs * dp_mask[2]
      else:
        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

      x_z = K.dot(inputs_z, self.kernel[:, :self.units])
      x_r = K.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
      x_h = K.dot(inputs_h, self.kernel[:, self.units * 2:])

      if self.use_bias:
        x_z = K.bias_add(x_z, input_bias[:self.units])
        x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
        x_h = K.bias_add(x_h, input_bias[self.units * 2:])

      if 0. < self.recurrent_dropout < 1.:
        h_tm1_z = h_tm1 * rec_dp_mask[0]
        h_tm1_r = h_tm1 * rec_dp_mask[1]
        h_tm1_h = h_tm1 * rec_dp_mask[2]
      else:
        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

      recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
      recurrent_r = K.dot(h_tm1_r,
                          self.recurrent_kernel[:, self.units:self.units * 2])
      if self.reset_after and self.use_bias:
        recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
        recurrent_r = K.bias_add(recurrent_r,
                                 recurrent_bias[self.units:self.units * 2])

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      # reset gate applied after/before matrix multiplication
      if self.reset_after:
        recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
        if self.use_bias:
          recurrent_h = K.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = K.dot(r * h_tm1_h,
                            self.recurrent_kernel[:, self.units * 2:])

      hh = self.activation(x_h + recurrent_h)
    else:
      if 0. < self.dropout < 1.:
        inputs *= dp_mask[0]

      # inputs projected by all gate matrices at once
      matrix_x = K.dot(inputs, self.kernel)
      if self.use_bias:
        # biases: bias_z_i, bias_r_i, bias_h_i
        matrix_x = K.bias_add(matrix_x, input_bias)

      x_z = matrix_x[:, :self.units]
      x_r = matrix_x[:, self.units: 2 * self.units]
      x_h = matrix_x[:, 2 * self.units:]

      if 0. < self.recurrent_dropout < 1.:
        h_tm1 *= rec_dp_mask[0]

      if self.reset_after:
        # hidden state projected by all gate matrices at once
        matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
          matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
      else:
        # hidden state projected separately for update/reset and new
        matrix_inner = K.dot(h_tm1, self.recurrent_kernel[:, :2 * self.units])

      recurrent_z = matrix_inner[:, :self.units]
      recurrent_r = matrix_inner[:, self.units:2 * self.units]

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      if self.reset_after:
        recurrent_h = r * matrix_inner[:, 2 * self.units:]
      else:
        recurrent_h = K.dot(r * h_tm1,
                            self.recurrent_kernel[:, 2 * self.units:])

      hh = self.activation(x_h + recurrent_h)
    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    return h, [h]

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'dropout': self.dropout,
        'recurrent_dropout': self.recurrent_dropout,
        'implementation': self.implementation,
        'reset_after': self.reset_after
    }
    base_config = super(GRUCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)


@tf_export('keras.layers.GRU')
class GRU(RNN):
  """Gated Recurrent Unit - Cho et al. 2014.

  There are two variants. The default one is based on 1406.1078v3 and
  has reset gate applied to hidden state before matrix multiplication. The
  other one is based on original 1406.1078v1 and has the order reversed.

  The second variant is compatible with CuDNNGRU (GPU-only) and allows
  inference on CPU. Thus it has separate biases for `kernel` and
  `recurrent_kernel`. Use `'reset_after'=True` and
  `recurrent_activation='sigmoid'`.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          Default: hyperbolic tangent (`tanh`).
          If you pass `None`, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
          for the recurrent step.
          Default: hard sigmoid (`hard_sigmoid`).
          If you pass `None`, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
          the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
          the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
          Mode 1 will structure its operations as a larger number of
          smaller dot products and additions, whereas mode 2 will
          batch them into fewer, larger operations. These modes will
          have different performance profiles on different hardware and
          for different applications.
      return_sequences: Boolean. Whether to return the last output
          in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
          in addition to the output.
      go_backwards: Boolean (default False).
          If True, process the input sequence backwards and return the
          reversed sequence.
      stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.
      unroll: Boolean (default False).
          If True, the network will be unrolled,
          else a symbolic loop will be used.
          Unrolling can speed-up a RNN,
          although it tends to be more memory-intensive.
          Unrolling is only suitable for short sequences.
      reset_after: GRU convention (whether to apply reset gate after or
          before matrix multiplication). False = "before" (default),
          True = "after" (CuDNN compatible).

  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               reset_after=False,
               **kwargs):
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    cell = GRUCell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        reset_after=reset_after)
    super(GRU, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self.cell._dropout_mask = None
    self.cell._recurrent_dropout_mask = None
    return super(GRU, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  @property
  def reset_after(self):
    return self.cell.reset_after

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation,
        'reset_after':
            self.reset_after
    }
    base_config = super(GRU, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)


@tf_export('keras.layers.LSTMCell')
class LSTMCell(Layer):
  """Cell class for the LSTM layer.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          Default: hyperbolic tangent (`tanh`).
          If you pass `None`, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
          for the recurrent step.
          Default: hard sigmoid (`hard_sigmoid`).
          If you pass `None`, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
          If True, add 1 to the bias of the forget gate at initialization.
          Setting it to true will also force `bias_initializer="zeros"`.
          This is recommended in [Jozefowicz et
            al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
          the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
          the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
          Mode 1 will structure its operations as a larger number of
          smaller dot products and additions, whereas mode 2 will
          batch them into fewer, larger operations. These modes will
          have different performance profiles on different hardware and
          for different applications.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               **kwargs):
    super(LSTMCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.implementation = implementation
    self.state_size = [self.units, self.units]
    self.output_size = self.units
    self._dropout_mask = None
    self._recurrent_dropout_mask = None

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.Ones()((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels."""
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    c = f * c_tm1 + i * self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
    return c, o

  def _compute_carry_and_output_fused(self, z, c_tm1):
    """Computes carry and output using fused kernels."""
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0)
    f = self.recurrent_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3)
    return c, o

  def call(self, inputs, states, training=None):
    if 0 < self.dropout < 1 and self._dropout_mask is None:
      self._dropout_mask = _generate_dropout_mask(
          array_ops.ones_like(inputs),
          self.dropout,
          training=training,
          count=4)
    if (0 < self.recurrent_dropout < 1 and
        self._recurrent_dropout_mask is None):
      self._recurrent_dropout_mask = _generate_dropout_mask(
          array_ops.ones_like(states[0]),
          self.recurrent_dropout,
          training=training,
          count=4)

    # dropout matrices for input units
    dp_mask = self._dropout_mask
    # dropout matrices for recurrent units
    rec_dp_mask = self._recurrent_dropout_mask

    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
      x_i = K.dot(inputs_i, self.kernel[:, :self.units])
      x_f = K.dot(inputs_f, self.kernel[:, self.units:self.units * 2])
      x_c = K.dot(inputs_c, self.kernel[:, self.units * 2:self.units * 3])
      x_o = K.dot(inputs_o, self.kernel[:, self.units * 3:])
      if self.use_bias:
        x_i = K.bias_add(x_i, self.bias[:self.units])
        x_f = K.bias_add(x_f, self.bias[self.units:self.units * 2])
        x_c = K.bias_add(x_c, self.bias[self.units * 2:self.units * 3])
        x_o = K.bias_add(x_o, self.bias[self.units * 3:])

      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o)
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
    else:
      if 0. < self.dropout < 1.:
        inputs *= dp_mask[0]
      z = K.dot(inputs, self.kernel)
      if 0. < self.recurrent_dropout < 1.:
        h_tm1 *= rec_dp_mask[0]
      z += K.dot(h_tm1, self.recurrent_kernel)
      if self.use_bias:
        z = K.bias_add(z, self.bias)

      z0 = z[:, :self.units]
      z1 = z[:, self.units:2 * self.units]
      z2 = z[:, 2 * self.units:3 * self.units]
      z3 = z[:, 3 * self.units:]

      z = (z0, z1, z2, z3)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)

    h = o * self.activation(c)
    return h, [h, c]

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(LSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype))


@tf_export('keras.experimental.PeepholeLSTMCell')
class PeepholeLSTMCell(LSTMCell):
  """Equivalent to LSTMCell class but adds peephole connections.

  Peephole connections allow the gates to utilize the previous internal state as
  well as the previous hidden state (which is what LSTMCell is limited to).
  This allows PeepholeLSTMCell to better learn precise timings over LSTMCell.

  From [Gers et al.](http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf):

    "We find that LSTM augmented by 'peephole connections' from its internal
    cells to its multiplicative gates can learn the fine distinction between
    sequences of spikes spaced either 50 or 49 time steps apart without the help
    of any short training exemplars."

  The peephole implementation is based on:

    https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.

  Example:

  ```python
      # Create 2 PeepholeLSTMCells
      peephole_lstm_cells = [PeepholeLSTMCell(size) for size in [128, 256]]
      # Create a layer composed sequentially of the peephole LSTM cells.
      layer = RNN(peephole_lstm_cells)
      input = keras.Input((timesteps, input_dim))
      output = layer(input)
  ```
  """

  def build(self, input_shape):
    super(PeepholeLSTMCell, self).build(input_shape)
    # The following are the weight matrices for the peephole connections. These
    # are multiplied with the previous internal state during the computation of
    # carry and output.
    self.input_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='input_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.forget_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='forget_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.output_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='output_gate_peephole_weights',
        initializer=self.kernel_initializer)

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) +
        self.input_gate_peephole_weights * c_tm1)
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]) +
                                  self.forget_gate_peephole_weights * c_tm1)
    c = f * c_tm1 + i * self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]) +
        self.output_gate_peephole_weights * c)
    return c, o

  def _compute_carry_and_output_fused(self, z, c_tm1):
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0 +
                                  self.input_gate_peephole_weights * c_tm1)
    f = self.recurrent_activation(z1 +
                                  self.forget_gate_peephole_weights * c_tm1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3 + self.output_gate_peephole_weights * c)
    return c, o


@tf_export(v1=['keras.layers.LSTM'])
class LSTM(RNN):
  """Long Short-Term Memory layer - Hochreiter 1997.

   Note that this cell is not optimized for performance on GPU. Please use
  `tf.keras.layers.CuDNNLSTM` for better performance on GPU.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          Default: hyperbolic tangent (`tanh`).
          If you pass `None`, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
          for the recurrent step.
          Default: hard sigmoid (`hard_sigmoid`).
          If you pass `None`, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs..
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state..
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
          If True, add 1 to the bias of the forget gate at initialization.
          Setting it to true will also force `bias_initializer="zeros"`.
          This is recommended in [Jozefowicz et
            al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
          the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
          the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
          Mode 1 will structure its operations as a larger number of
          smaller dot products and additions, whereas mode 2 will
          batch them into fewer, larger operations. These modes will
          have different performance profiles on different hardware and
          for different applications.
      return_sequences: Boolean. Whether to return the last output.
          in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
          in addition to the output.
      go_backwards: Boolean (default False).
          If True, process the input sequence backwards and return the
          reversed sequence.
      stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.
      unroll: Boolean (default False).
          If True, the network will be unrolled,
          else a symbolic loop will be used.
          Unrolling can speed-up a RNN,
          although it tends to be more memory-intensive.
          Unrolling is only suitable for short sequences.

  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn('%s: Note that this layer is not optimized for performance. '
                   'Please use tf.keras.layers.CuDNNLSTM for better '
                   'performance on GPU.', self)
    cell = LSTMCell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation)
    super(LSTM, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self.cell._dropout_mask = None
    self.cell._recurrent_dropout_mask = None
    return super(LSTM, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def unit_forget_bias(self):
    return self.cell.unit_forget_bias

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(LSTM, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)


@tf_export('keras.layers.LSTM', v1=[])
class UnifiedLSTM(LSTM):
  """Long Short-Term Memory layer - Hochreiter 1997.

  `UnifiedLSTM` unifies the implementations between standard `LSTM` layer and
  `CuDNNLSTM` layer. Based on available runtime hardware and constrains,
  `UnifiedLSTM` will choose different implementations to maximize the
  performance. For instance, if GPU is available and all the parameters meet the
  requirement of CuDNN kernel, `UnifiedLSTM` will use CuDNN kernel for the
  calculation.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
      is applied (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
      applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix, used for
      the linear transformation of the inputs..
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state..
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at
      initialization. Setting it to true will also force
      `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation")..
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2. Mode 1 will structure
      its operations as a larger number of smaller dot products and additions,
      whereas mode 2 will batch them into fewer, larger operations. These modes
      will have different performance profiles on different hardware and for
      different applications.
    return_sequences: Boolean. Whether to return the last output. in the output
      sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state in addition to the
      output.
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    stateful: Boolean (default False). If True, the last state for each sample
      at index i in a batch will be used as initial state for the sample of
      index i in the following batch.
    unroll: Boolean (default False). If True, the network will be unrolled, else
      a symbolic loop will be used. Unrolling can speed-up a RNN, although it
      tends to be more memory-intensive. Unrolling is only suitable for short
      sequences.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               time_major=False,
               unroll=False,
               **kwargs):
    # return_runtime is a flag for testing, which shows the real backend
    # implementation chosen by grappler in graph mode.
    self.return_runtime = kwargs.pop('return_runtime', False)

    super(UnifiedLSTM, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        time_major=time_major,
        unroll=unroll,
        **kwargs)

    self.state_spec = [
        InputSpec(shape=(None, dim)) for dim in (self.units, self.units)
    ]
    self._dropout_mask = None
    self.could_use_cudnn = (
        activation == 'tanh' and recurrent_activation == 'sigmoid' and
        recurrent_dropout == 0 and not unroll and use_bias)

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # LSTM does not support constants. Ignore it during process.
    inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)

    if isinstance(mask, list):
      mask = mask[0]

    input_shape = K.int_shape(inputs)
    timesteps = input_shape[0] if self.time_major else input_shape[1]

    if mask is not None or not self.could_use_cudnn:
      # CuDNN does not support masking, fall back to use the normal LSTM.
      kwargs = {'training': training}

      def step(inputs, states):
        return self.cell.call(inputs, states, **kwargs)

      last_output, outputs, states = K.rnn(
          step,
          inputs,
          initial_state,
          constants=None,
          go_backwards=self.go_backwards,
          mask=mask,
          unroll=self.unroll,
          input_length=timesteps,
          time_major=self.time_major,
          zero_output_for_mask=self.zero_output_for_mask)
      runtime = constant_op.constant(
          'unknown', dtype=dtypes.string, name='runtime')
    else:
      # Use the new defun approach for backend implementation swap.
      # Note that different implementations need to have same function
      # signature, eg, the tensor parameters need to have same shape and dtypes.
      # Since the CuDNN has an extra set of bias, those bias will be passed to
      # both normal and CuDNN implementations.
      if self.go_backwards:
        # Reverse time axis.
        inputs = K.reverse(inputs, 0 if self.time_major else 1)

      if 0 < self.dropout < 1:
        if self._dropout_mask is None:
          self._dropout_mask = _generate_dropout_mask(
              array_ops.ones_like(inputs),
              self.dropout,
              training=training,
              count=4)

        inputs *= self._dropout_mask[0]

      # Each time a defun function is called, we will give a unique identifiable
      # API name, so that the grappler won't get confused when it sees multiple
      # LSTM layer added into same graph, and it will be able to pair up the
      # different implementations across them.
      experimental_api_name = 'lstm_' + str(uuid.uuid4())
      standard_lstm_attributes = {
          'experimental_api_implements': experimental_api_name,
          'experimental_api_preferred_device': 'CPU',
      }
      cudnn_lstm_attributes = {
          'experimental_api_implements': experimental_api_name,
          'experimental_api_preferred_device': 'GPU',
      }
      defun_standard_lstm = function.defun_with_attributes(
          standard_lstm, attributes=standard_lstm_attributes)
      defun_cudnn_lstm = function.defun_with_attributes(
          cudnn_lstm, attributes=cudnn_lstm_attributes)

      if ops.executing_eagerly_outside_functions():
        # Under eager context, the device placement is already known. Prefer the
        # GPU implementation here.
        if context.num_gpus() > 0:
          last_output, outputs, new_h, new_c, runtime = defun_cudnn_lstm(
              inputs, initial_state[0], initial_state[1], self.cell.kernel,
              self.cell.recurrent_kernel, self.cell.bias, self.time_major)
        else:
          last_output, outputs, new_h, new_c, runtime = defun_standard_lstm(
              inputs, initial_state[0], initial_state[1], self.cell.kernel,
              self.cell.recurrent_kernel, self.cell.bias, self.activation,
              self.recurrent_activation, self.time_major)
      else:
        # Call the normal LSTM impl and register the CuDNN impl function. The
        # grappler will kick in during session execution to optimize the graph.
        last_output, outputs, new_h, new_c, runtime = defun_standard_lstm(
            inputs, initial_state[0], initial_state[1], self.cell.kernel,
            self.cell.recurrent_kernel, self.cell.bias, self.activation,
            self.recurrent_activation, self.time_major)

        function.register(defun_cudnn_lstm, inputs, initial_state[0],
                          initial_state[1], self.cell.kernel,
                          self.cell.recurrent_kernel, self.cell.bias,
                          self.time_major)
      states = [new_h, new_c]

    if self.stateful:
      updates = []
      for i in range(len(states)):
        updates.append(state_ops.assign(self.states[i], states[i]))
      self.add_update(updates, inputs)

    if self.return_sequences:
      output = outputs
    else:
      output = last_output

    if self.return_state:
      return [output] + states
    elif self.return_runtime:
      return output, runtime
    else:
      return output


def _canonical_to_params(weights, biases, shape, transpose_weights=False):
  """Utility function convert variable to CuDNN compatible parameter.

  Note that Keras weights for kernels are different from the CuDNN format. Eg.:

  ```
    Keras                 CuDNN
    [[0, 1, 2],  <--->  [[0, 2, 4],
     [3, 4, 5]]          [1, 3, 5]]
  ```

  If the input weights need to be in a unified format, then set
  `transpose_weights=True` to convert the weights.

  Args:
    weights: list of weights for the individual kernels and recurrent kernels.
    biases: list of biases for individual gate.
    shape: the shape for the converted variables that will be feed to CuDNN.
    transpose_weights: boolean, whether to transpose the weights.

  Returns:
    The converted weights that can be feed to CuDNN ops as param.
  """
  def convert(w):
    return array_ops.transpose(w) if transpose_weights else w

  weights = [array_ops.reshape(convert(x), shape) for x in weights]
  biases = [array_ops.reshape(x, shape) for x in biases]
  return array_ops.concat(weights + biases, axis=0)


def standard_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias,
                  activation, recurrent_activation, time_major):
  """LSTM with standard kernel implementation.

  This implementation can be run on all types for hardware.

  This implementation lifts out all the layer weights and make them function
  parameters. It has same number of tensor input params as the CuDNN
  counterpart. The RNN step logic has been simplified, eg dropout and mask is
  removed since CuDNN implementation does not support that.

  Note that the first half of the bias tensor should be ignored by this impl.
  The CuDNN impl need an extra set of input gate bias. In order to make the both
  function take same shape of parameter, that extra set of bias is also feed
  here.

  Args:
    inputs: input tensor of LSTM layer.
    init_h: initial state tensor for the cell output.
    init_c: initial state tensor for the cell hidden state.
    kernel: weights for cell kernel.
    recurrent_kernel: weights for cell recurrent kernel.
    bias: weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    activation: Activation function to use for output.
    recurrent_activation: Activation function to use for hidden recurrent state.
    time_major: boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].

  Returns:
    last_output: output tensor for the last timestep, which has shape
      [batch, units].
    outputs: output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: the cell output, which has same shape as init_h.
    state_1: the cell hidden state, which has same shape as init_c.
    runtime: constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should be used by user.
  """
  input_shape = K.int_shape(inputs)
  timesteps = input_shape[0] if time_major else input_shape[1]

  def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]  # previous memory state
    c_tm1 = cell_states[1]  # previous carry state

    z = K.dot(cell_inputs, kernel)
    z += K.dot(h_tm1, recurrent_kernel)
    z = K.bias_add(z, bias)

    z0, z1, z2, z3 = array_ops.split(z, 4, axis=1)

    i = recurrent_activation(z0)
    f = recurrent_activation(z1)
    c = f * c_tm1 + i * activation(z2)
    o = recurrent_activation(z3)

    h = o * activation(c)
    return h, [h, c]

  last_output, outputs, new_states = K.rnn(
      step,
      inputs, [init_h, init_c],
      constants=None,
      unroll=False,
      time_major=time_major,
      input_length=timesteps)
  return last_output, outputs, new_states[0], new_states[
      1], constant_op.constant('cpu', dtype=dtypes.string, name='runtime')


def cudnn_lstm(inputs, input_h, input_c, kernel, recurrent_kernel, bias,
               time_major):
  """LSTM with CuDNN implementation which is only available for GPU."""
  if not time_major:
    inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
  input_h = array_ops.expand_dims(input_h, axis=0)
  input_c = array_ops.expand_dims(input_c, axis=0)

  weights = array_ops.split(kernel, 4, axis=1)
  weights += array_ops.split(recurrent_kernel, 4, axis=1)
  # CuDNN has an extra set of bias for inputs, we disable them (setting to 0),
  # so that mathematically it is same as the canonical LSTM implementation.
  full_bias = array_ops.concat((array_ops.zeros_like(bias), bias), 0)

  params = _canonical_to_params(
      weights=weights,
      biases=array_ops.split(full_bias, 8),
      shape=constant_op.constant([-1]),
      transpose_weights=True)

  outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
      inputs, input_h=input_h, input_c=input_c, params=params, is_training=True)
  last_output = outputs[-1]
  if not time_major:
    outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
  h = h[0]
  c = c[0]

  return last_output, outputs, h, c, constant_op.constant(
      'cudnn', dtype=dtypes.string, name='runtime')


def _generate_dropout_mask(ones, rate, training=None, count=1):
  def dropped_inputs():
    return K.dropout(ones, rate)

  if count > 1:
    return [
        K.in_train_phase(dropped_inputs, ones, training=training)
        for _ in range(count)
    ]
  return K.in_train_phase(dropped_inputs, ones, training=training)


def _standardize_args(
    inputs, initial_state, constants, num_constants, num_inputs=1):
  """Standardizes `__call__` to a single list of tensor inputs.

  When running a model loaded from a file, the input tensors
  `initial_state` and `constants` can be passed to `RNN.__call__()` as part
  of `inputs` instead of by the dedicated keyword arguments. This method
  makes sure the arguments are separated and that `initial_state` and
  `constants` are lists of tensors (or None).

  Arguments:
      inputs: Tensor or list/tuple of tensors. which may include constants
        and initial states. In that case `num_constant` must be specified.
      initial_state: Tensor or list of tensors or None, initial states.
      constants: Tensor or list of tensors or None, constant tensors.
      num_constants: Expected number of constants (if constants are passed as
        part of the `inputs` list.
      num_inputs: Expected number of real input tensors (exclude initial_states
        and constants).

  Returns:
      inputs: Single tensor or tuple of tensors.
      initial_state: List of tensors or None.
      constants: List of tensors or None.
  """
  if isinstance(inputs, list):
    # There are several situations here:
    # In the graph mode, __call__ will be only called once. The initial_state
    # and constants could be in inputs (from file loading).
    # In the eager mode, __call__ will be called twice, once during
    # rnn_layer(inputs=input_t, constants=c_t, ...), and second time will be
    # model.fit/train_on_batch/predict with real np data. In the second case,
    # the inputs will contain initial_state and constants, and more importantly,
    # the real inputs will be in a flat list, instead of nested tuple.
    #
    # For either case, we will use num_inputs to split the input list, and
    # restructure the real input into tuple.
    assert initial_state is None and constants is None
    if num_constants is not None:
      constants = inputs[-num_constants:]
      inputs = inputs[:-num_constants]
    if num_inputs is None:
      num_inputs = 1
    if len(inputs) > num_inputs:
      initial_state = inputs[num_inputs:]
      inputs = inputs[:num_inputs]

    if len(inputs) > 1:
      inputs = tuple(inputs)
    else:
      inputs = inputs[0]

  def to_list_or_none(x):
    if x is None or isinstance(x, list):
      return x
    if isinstance(x, tuple):
      return list(x)
    return [x]

  initial_state = to_list_or_none(initial_state)
  constants = to_list_or_none(constants)

  return inputs, initial_state, constants


def _is_multiple_state(state_size):
  """Check whether the state_size contains multiple states."""
  return (hasattr(state_size, '__len__') and
          not isinstance(state_size, tensor_shape.TensorShape))


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  if inputs is not None:
    batch_size = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
  """Generate a zero filled tensor with shape [batch_size, state_size]."""
  if batch_size_tensor is None or dtype is None:
    raise ValueError(
        'batch_size and dtype cannot be None while constructing initial state: '
        'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

  def create_zeros(unnested_state_size):
    flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return array_ops.zeros(init_state_size, dtype=dtype)

  if nest.is_sequence(state_size):
    return nest.map_structure(create_zeros, state_size)
  else:
    return create_zeros(state_size)
