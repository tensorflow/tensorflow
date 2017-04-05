# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A powerful dynamic attention wrapper object.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib.rnn import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


__all__ = [
    "AttentionWrapper",
    "AttentionWrapperState",
    "LuongAttention",
    "BahdanauAttention",
    "hardmax",
]


_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


class AttentionMechanism(object):
  pass


def _prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
  """Convert to tensor and possibly mask `memory`.

  Args:
    memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
    memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
    check_inner_dims_defined: Python boolean.  If `True`, the `memory`
      argument's shape is checked to ensure all but the two outermost
      dimensions are fully defined.

  Returns:
    A (possibly masked), checked, new `memory`.

  Raises:
    ValueError: If `check_inner_dims_defined` is `True` and not
      `memory.shape[2:].is_fully_defined()`.
  """
  memory = nest.map_structure(
      lambda m: ops.convert_to_tensor(m, name="memory"), memory)
  if check_inner_dims_defined:
    def _check_dims(m):
      if not m.get_shape()[2:].is_fully_defined():
        raise ValueError("Expected memory %s to have fully defined inner dims, "
                         "but saw shape: %s" % (m.name, m.get_shape()))
    nest.map_structure(_check_dims, memory)
  if memory_sequence_length is None:
    seq_len_mask = None
  else:
    seq_len_mask = array_ops.sequence_mask(
        memory_sequence_length,
        maxlen=array_ops.shape(nest.flatten(memory)[0])[1],
        dtype=nest.flatten(memory)[0].dtype)
  def _maybe_mask(m, seq_len_mask):
    rank = m.get_shape().ndims
    rank = rank if rank is not None else array_ops.rank(m)
    extra_ones = array_ops.ones(rank - 2, dtype=dtypes.int32)
    if memory_sequence_length is not None:
      seq_len_mask = array_ops.reshape(
          seq_len_mask,
          array_ops.concat((array_ops.shape(seq_len_mask), extra_ones), 0))
      return m * seq_len_mask
    else:
      return m
  return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


class _BaseAttentionMechanism(AttentionMechanism):
  """A base AttentionMechanism class providing common functionality.

  Common functionality includes:
    1. Storing the query and memory layers.
    2. Preprocessing and storing the memory.
  """

  def __init__(self, query_layer, memory, memory_sequence_length=None,
               memory_layer=None, check_inner_dims_defined=True, name=None):
    """Construct base AttentionMechanism class.

    Args:
      query_layer: Callable.  Instance of `tf.layers.Layer`.  The layer's depth
        must match the depth of `memory_layer`.  If `query_layer` is not
        provided, the shape of `query` must match that of `memory_layer`.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      memory_layer: Instance of `tf.layers.Layer` (may be None).  The layer's
        depth must match the depth of `query_layer`.
        If `memory_layer` is not provided, the shape of `memory` must match
        that of `query_layer`.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.
      name: Name to use when creating ops.
    """
    if (query_layer is not None
        and not isinstance(query_layer, layers_base._Layer)):  # pylint: disable=protected-access
      raise TypeError(
          "query_layer is not a Layer: %s" % type(query_layer).__name__)
    if (memory_layer is not None
        and not isinstance(memory_layer, layers_base._Layer)):  # pylint: disable=protected-access
      raise TypeError(
          "memory_layer is not a Layer: %s" % type(memory_layer).__name__)
    self._query_layer = query_layer
    self._memory_layer = memory_layer
    with ops.name_scope(
        name, "BaseAttentionMechanismInit", nest.flatten(memory)):
      self._values = _prepare_memory(
          memory, memory_sequence_length,
          check_inner_dims_defined=check_inner_dims_defined)
      self._keys = (
          self.memory_layer(self._values) if self.memory_layer  # pylint: disable=not-callable
          else self._values)

  @property
  def memory_layer(self):
    return self._memory_layer

  @property
  def query_layer(self):
    return self._query_layer

  @property
  def values(self):
    return self._values

  @property
  def keys(self):
    return self._keys


class LuongAttention(_BaseAttentionMechanism):
  """Implements Luong-style (multiplicative) attention scoring.

  This attention has two forms.  The first is standard Luong attention,
  as described in:

  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025

  The second is the scaled form inspired partly by the normalized form of
  Bahdanau attention.

  To enable the second form, construct the object with parameter
  `scale=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               scale=False,
               name="LuongAttention"):
    """Construct the AttentionMechanism mechanism.

    Args:
      num_units: The depth of the attention mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      scale: Python boolean.  Whether to scale the energy term.
      name: Name to use when creating ops.
    """
    # For LuongAttention, we only transform the memory layer; thus
    # num_units **must** match expected the query depth.
    super(LuongAttention, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False),
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._name = name

  def __call__(self, query):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.

    Returns:
      score: Tensor of dtype matching `self.values` and shape
        `[batch_size, max_time]` (`max_time` is memory's `max_time`).

    Raises:
      ValueError: If `key` and `query` depths do not match.
    """
    depth = query.get_shape()[-1]
    key_units = self.keys.get_shape()[-1]
    if depth != key_units:
      raise ValueError(
          "Incompatible or unknown inner dimensions between query and keys.  "
          "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
          "Perhaps you need to set num_units to the the keys' dimension (%s)?"
          % (query, depth, self.keys, key_units, key_units))
    dtype = query.dtype

    with variable_scope.variable_scope(None, "luong_attention", [query]):
      # Reshape from [batch_size, depth] to [batch_size, 1, depth]
      # for matmul.
      query = array_ops.expand_dims(query, 1)

      # Inner product along the query units dimension.
      # matmul shapes: query is [batch_size, 1, depth] and
      #                keys is [batch_size, max_time, depth].
      # the inner product is asked to **transpose keys' inner shape** to get a
      # batched matmul on:
      #   [batch_size, 1, depth] . [batch_size, depth, max_time]
      # resulting in an output shape of:
      #   [batch_time, 1, max_time].
      # we then squeee out the center singleton dimension.
      score = math_ops.matmul(query, self.keys, transpose_b=True)
      score = array_ops.squeeze(score, [1])

      if self._scale:
        # Scalar used in weight scaling
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype, initializer=1.)
        score = g * score

    return score


class BahdanauAttention(_BaseAttentionMechanism):
  """Implements Bhadanau-style (additive) attention.

  This attention has two forms.  The first is Bhandanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form.  This form is inspired by the
  weight normalization article:

  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868

  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               name="BahdanauAttention"):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      name: Name to use when creating ops.
    """
    super(BahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False),
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name

  def __call__(self, query):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.

    Returns:
      score: Tensor of dtype matching `self.values` and shape
        `[batch_size, max_time]` (`max_time` is memory's `max_time`).
    """
    with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      dtype = processed_query.dtype
      # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
      processed_query = array_ops.expand_dims(processed_query, 1)
      v = variable_scope.get_variable(
          "attention_v", [self._num_units], dtype=dtype)
      if self._normalize:
        # Scalar used in weight normalization
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype,
            initializer=math.sqrt((1. / self._num_units)))
        # Bias added prior to the nonlinearity
        b = variable_scope.get_variable(
            "attention_b", [self._num_units], dtype=dtype,
            initializer=init_ops.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * math_ops.rsqrt(
            math_ops.reduce_sum(math_ops.square(v)))
        score = math_ops.reduce_sum(
            normed_v * math_ops.tanh(self.keys + processed_query + b), [2])
      else:
        score = math_ops.reduce_sum(
            v * math_ops.tanh(self.keys + processed_query), [2])

    return score


class AttentionWrapperState(
    collections.namedtuple(
        "AttentionWrapperState", (
            "cell_state", "attention", "time", "attention_history"))):
  """`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell`.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `attention_history`: (if enabled) a `TensorArray` containing attention
       matrices from all time steps.  Call `stack()` to convert to a `Tensor`.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.

    Example:

    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```

    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `AttentionWrapperState`.

    Returns:
      A new `AttentionWrapperState` whose properties are the same as
      this one, except any overriden properties as provided in `kwargs`.
    """
    return super(AttentionWrapperState, self)._replace(**kwargs)


def hardmax(logits, name=None):
  """Returns batched one-hot vectors.

  The depth index containing the `1` is that of the maximum logit value.

  Args:
    logits: A batch tensor of logit values.
    name: Name to use when creating ops.
  Returns:
    A batched one-hot tensor.
  """
  with ops.name_scope(name, "Hardmax", [logits]):
    logits = ops.convert_to_tensor(logits, name="logits")
    if logits.get_shape()[-1].value is not None:
      depth = logits.get_shape()[-1].value
    else:
      depth = array_ops.shape(logits)[-1]
    return array_ops.one_hot(
        math_ops.argmax(logits, -1), depth, dtype=logits.dtype)


class AttentionWrapper(core_rnn_cell.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               attention_size,
               attention_history=False,
               cell_input_fn=None,
               probability_fn=None,
               output_attention=True,
               name=None):
    """Construct the `AttentionWrapper`.

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: An instance of `AttentionMechanism`.
      attention_size: Python integer, the depth of the attention (output)
        tensor.
      attention_history: Python boolean, whether to store attention history
        from all time steps in the final output state (currently stored as a
        time major `TensorArray` on which you must call `stack()`).
      cell_input_fn: (optional) A `callable`.  The default is:
        `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
      output_attention: Python bool.  If `True` (default), the output at each
        time step is the attention value.  This is the behavior of Luong-style
        attention mechanisms.  If `False`, the output at each time step is
        the output of `cell`.  This is the beahvior of Bhadanau-style
        attention mechanisms.  In both cases, the `attention` tensor is
        propagated to the next time step via the state and is used there.
        This flag only controls whether the attention mechanism is propagated
        up to the next cell in an RNN stack or to the top RNN output.
      name: Name to use when creating ops.
    """
    if not isinstance(cell, core_rnn_cell.RNNCell):
      raise TypeError(
          "cell must be an RNNCell, saw type: %s" % type(cell).__name__)
    if not isinstance(attention_mechanism, AttentionMechanism):
      raise TypeError(
          "attention_mechanism must be a AttentionMechanism, saw type: %s"
          % type(attention_mechanism).__name__)
    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "probability_fn must be callable, saw type: %s"
            % type(probability_fn).__name__)
    self._cell = cell
    self._attention_mechanism = attention_mechanism
    self._attention_size = attention_size
    self._attention_layer = layers_core.Dense(
        attention_size, name="attention_layer", use_bias=False)
    self._cell_input_fn = cell_input_fn
    self._probability_fn = probability_fn
    self._output_attention = output_attention
    self._attention_history = attention_history

  @property
  def output_size(self):
    if self._output_attention:
      return self._attention_size
    else:
      return self._cell.output_size

  @property
  def state_size(self):
    return AttentionWrapperState(
        cell_state=self._cell.state_size,
        time=tensor_shape.TensorShape([]),
        attention=self._attention_size,
        attention_history=())  # attention_history is sometimes a TensorArray

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._attention_history:
        attention_history = tensor_array_ops.TensorArray(
            dtype=dtype, size=0, dynamic_size=True)
      else:
        attention_history = ()
      return AttentionWrapperState(
          cell_state=self._cell.zero_state(batch_size, dtype),
          time=array_ops.zeros([], dtype=dtypes.int32),
          attention=_zero_state_tensors(
              self._attention_size, batch_size, dtype),
          attention_history=attention_history)

  def __call__(self, inputs, state, scope=None):
    """Perform a step of attention-wrapped RNN.

    - Step 1: Mix the `inputs` and previous step's `attention` output via
      `cell_input_fn`.
    - Step 2: Call the wrapped `cell` with this input and its previous state.
    - Step 3: Score the cell's output with `attention_mechanism`.
    - Step 4: Calculate the alignments by passing the score through the
      `normalizer`.
    - Step 5: Calculate the context vector as the inner product between the
      alignments and the attention_mechanism's values (memory).
    - Step 6: Calculate the attention output by concatenating the cell output
      and context through the attention layer (a linear layer with
      `attention_size` outputs).

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `AttentionWrapperState` containing
        tensors from the previous time step.
      scope: Must be `None`.

    Returns:
      A tuple `(attention_or_cell_output, next_state)`, where:

      - `attention_or_cell_output` depending on `output_attention`.
      - `next_state` is an instance of `DynamicAttentionWrapperState`
         containing the state calculated at this time step.

    Raises:
      NotImplementedError: if `scope` is not `None`.
    """
    if scope is not None:
      raise NotImplementedError("scope not None is not supported")

    with variable_scope.variable_scope("attention"):
      # Step 1: Calculate the true inputs to the cell based on the
      # previous attention value.
      cell_inputs = self._cell_input_fn(inputs, state.attention)
      cell_state = state.cell_state

      cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

      score = self._attention_mechanism(cell_output)
      alignments = self._probability_fn(score)

      # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
      alignments = array_ops.expand_dims(alignments, 1)
      # Context is the inner product of alignments and values along the
      # memory time dimension.
      # alignments shape is
      #   [batch_size, 1, memory_time]
      # attention_mechanism.values shape is
      #   [batch_size, memory_time, attention_mechanism.num_units]
      # the batched matmul is over memory_time, so the output shape is
      #   [batch_size, 1, attention_mechanism.num_units].
      # we then squeeze out the singleton dim.
      context = math_ops.matmul(alignments, self._attention_mechanism.values)
      context = array_ops.squeeze(context, [1])

      attention = self._attention_layer(
          array_ops.concat([cell_output, context], 1))

      if self._attention_history:
        attention_history = state.attention_history.write(state.time, attention)
      else:
        attention_history = ()

      next_state = AttentionWrapperState(
          time=state.time + 1,
          cell_state=next_cell_state,
          attention=attention,
          attention_history=attention_history)

    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state
