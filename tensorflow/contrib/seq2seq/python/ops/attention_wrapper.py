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
"""A powerful dynamic attention wrapper object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


__all__ = [
    "AttentionMechanism",
    "AttentionWrapper",
    "AttentionWrapperState",
    "LuongAttention",
    "BahdanauAttention",
    "hardmax",
    "safe_cumprod",
    "monotonic_attention",
    "BahdanauMonotonicAttention",
    "LuongMonotonicAttention",
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
  if memory_sequence_length is not None:
    memory_sequence_length = ops.convert_to_tensor(
        memory_sequence_length, name="memory_sequence_length")
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
    seq_len_batch_size = (
        memory_sequence_length.shape[0].value
        or array_ops.shape(memory_sequence_length)[0])
  def _maybe_mask(m, seq_len_mask):
    rank = m.get_shape().ndims
    rank = rank if rank is not None else array_ops.rank(m)
    extra_ones = array_ops.ones(rank - 2, dtype=dtypes.int32)
    m_batch_size = m.shape[0].value or array_ops.shape(m)[0]
    if memory_sequence_length is not None:
      message = ("memory_sequence_length and memory tensor batch sizes do not "
                 "match.")
      with ops.control_dependencies([
          check_ops.assert_equal(
              seq_len_batch_size, m_batch_size, message=message)]):
        seq_len_mask = array_ops.reshape(
            seq_len_mask,
            array_ops.concat((array_ops.shape(seq_len_mask), extra_ones), 0))
        return m * seq_len_mask
    else:
      return m
  return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
  if memory_sequence_length is None:
    return score
  message = ("All values in memory_sequence_length must greater than zero.")
  with ops.control_dependencies(
      [check_ops.assert_positive(memory_sequence_length, message=message)]):
    score_mask = array_ops.sequence_mask(
        memory_sequence_length, maxlen=array_ops.shape(score)[1])
    score_mask_values = score_mask_value * array_ops.ones_like(score)
    return array_ops.where(score_mask, score, score_mask_values)


class _BaseAttentionMechanism(AttentionMechanism):
  """A base AttentionMechanism class providing common functionality.

  Common functionality includes:
    1. Storing the query and memory layers.
    2. Preprocessing and storing the memory.
  """

  def __init__(self,
               query_layer,
               memory,
               probability_fn,
               memory_sequence_length=None,
               memory_layer=None,
               check_inner_dims_defined=True,
               score_mask_value=float("-inf"),
               name=None):
    """Construct base AttentionMechanism class.

    Args:
      query_layer: Callable.  Instance of `tf.layers.Layer`.  The layer's depth
        must match the depth of `memory_layer`.  If `query_layer` is not
        provided, the shape of `query` must match that of `memory_layer`.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      probability_fn: A `callable`.  Converts the score and previous alignments
        to probabilities. Its signature should be:
        `probabilities = probability_fn(score, previous_alignments)`.
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
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    if (query_layer is not None
        and not isinstance(query_layer, layers_base.Layer)):
      raise TypeError(
          "query_layer is not a Layer: %s" % type(query_layer).__name__)
    if (memory_layer is not None
        and not isinstance(memory_layer, layers_base.Layer)):
      raise TypeError(
          "memory_layer is not a Layer: %s" % type(memory_layer).__name__)
    self._query_layer = query_layer
    self._memory_layer = memory_layer
    if not callable(probability_fn):
      raise TypeError("probability_fn must be callable, saw type: %s" %
                      type(probability_fn).__name__)
    self._probability_fn = lambda score, prev: (  # pylint:disable=g-long-lambda
        probability_fn(
            _maybe_mask_score(score, memory_sequence_length, score_mask_value),
            prev))
    with ops.name_scope(
        name, "BaseAttentionMechanismInit", nest.flatten(memory)):
      self._values = _prepare_memory(
          memory, memory_sequence_length,
          check_inner_dims_defined=check_inner_dims_defined)
      self._keys = (
          self.memory_layer(self._values) if self.memory_layer  # pylint: disable=not-callable
          else self._values)
      self._batch_size = (
          self._keys.shape[0].value or array_ops.shape(self._keys)[0])
      self._alignments_size = (self._keys.shape[1].value or
                               array_ops.shape(self._keys)[1])

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

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def alignments_size(self):
    return self._alignments_size

  def initial_alignments(self, batch_size, dtype):
    """Creates the initial alignment values for the `AttentionWrapper` class.

    This is important for AttentionMechanisms that use the previous alignment
    to calculate the alignment at the next time step (e.g. monotonic attention).

    The default behavior is to return a tensor of all zeros.

    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.

    Returns:
      A `dtype` tensor shaped `[batch_size, alignments_size]`
      (`alignments_size` is the values' `max_time`).
    """
    max_time = self._alignments_size
    return _zero_state_tensors(max_time, batch_size, dtype)


def _luong_score(query, keys, scale):
  """Implements Luong-style (multiplicative) scoring function.

  This attention has two forms.  The first is standard Luong attention,
  as described in:

  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025

  The second is the scaled form inspired partly by the normalized form of
  Bahdanau attention.

  To enable the second form, call this function with `scale=True`.

  Args:
    query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    scale: Whether to apply a scale to the score function.

  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.

  Raises:
    ValueError: If `key` and `query` depths do not match.
  """
  depth = query.get_shape()[-1]
  key_units = keys.get_shape()[-1]
  if depth != key_units:
    raise ValueError(
        "Incompatible or unknown inner dimensions between query and keys.  "
        "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
        "Perhaps you need to set num_units to the keys' dimension (%s)?"
        % (query, depth, keys, key_units, key_units))
  dtype = query.dtype

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
  # we then squeeze out the center singleton dimension.
  score = math_ops.matmul(query, keys, transpose_b=True)
  score = array_ops.squeeze(score, [1])

  if scale:
    # Scalar used in weight scaling
    g = variable_scope.get_variable(
        "attention_g", dtype=dtype, initializer=1.)
    score = g * score
  return score


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
               probability_fn=None,
               score_mask_value=float("-inf"),
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
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    # For LuongAttention, we only transform the memory layer; thus
    # num_units **must** match expected the query depth.
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(LuongAttention, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._name = name

  def __call__(self, query, previous_alignments):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      previous_alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(None, "luong_attention", [query]):
      score = _luong_score(query, self._keys, self._scale)
    alignments = self._probability_fn(score, previous_alignments)
    return alignments


def _bahdanau_score(processed_query, keys, normalize):
  """Implements Bahdanau-style (additive) scoring function.

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

  To enable the second form, set `normalize=True`.

  Args:
    processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    normalize: Whether to normalize the score function.

  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.
  """
  dtype = processed_query.dtype
  # Get the number of hidden units from the trailing dimension of keys
  num_units = keys.shape[2].value or array_ops.shape(keys)[2]
  # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
  processed_query = array_ops.expand_dims(processed_query, 1)
  v = variable_scope.get_variable(
      "attention_v", [num_units], dtype=dtype)
  if normalize:
    # Scalar used in weight normalization
    g = variable_scope.get_variable(
        "attention_g", dtype=dtype,
        initializer=math.sqrt((1. / num_units)))
    # Bias added prior to the nonlinearity
    b = variable_scope.get_variable(
        "attention_b", [num_units], dtype=dtype,
        initializer=init_ops.zeros_initializer())
    # normed_v = g * v / ||v||
    normed_v = g * v * math_ops.rsqrt(
        math_ops.reduce_sum(math_ops.square(v)))
    return math_ops.reduce_sum(
        normed_v * math_ops.tanh(keys + processed_query + b), [2])
  else:
    return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query), [2])


class BahdanauAttention(_BaseAttentionMechanism):
  """Implements Bahdanau-style (additive) attention.

  This attention has two forms.  The first is Bahdanau attention,
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
               probability_fn=None,
               score_mask_value=float("-inf"),
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
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(BahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name

  def __call__(self, query, previous_alignments):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      previous_alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      score = _bahdanau_score(processed_query, self._keys, self._normalize)
    alignments = self._probability_fn(score, previous_alignments)
    return alignments


def safe_cumprod(x, *args, **kwargs):
  """Computes cumprod of x in logspace using cumsum to avoid underflow.

  The cumprod function and its gradient can result in numerical instabilities
  when its argument has very small and/or zero values.  As long as the argument
  is all positive, we can instead compute the cumulative product as
  exp(cumsum(log(x))).  This function can be called identically to tf.cumprod.

  Args:
    x: Tensor to take the cumulative product of.
    *args: Passed on to cumsum; these are identical to those in cumprod.
    **kwargs: Passed on to cumsum; these are identical to those in cumprod.
  Returns:
    Cumulative product of x.
  """
  with ops.name_scope(None, "SafeCumprod", [x]):
    x = ops.convert_to_tensor(x, name="x")
    tiny = np.finfo(x.dtype.as_numpy_dtype).tiny
    return math_ops.exp(math_ops.cumsum(
        math_ops.log(clip_ops.clip_by_value(x, tiny, 1)), *args, **kwargs))


def monotonic_attention(p_choose_i, previous_attention, mode):
  """Compute monotonic attention distribution from choosing probabilities.

  Monotonic attention implies that the input sequence is processed in an
  explicitly left-to-right manner when generating the output sequence.  In
  addition, once an input sequence element is attended to at a given output
  timestep, elements occurring before it cannot be attended to at subsequent
  output timesteps.  This function generates attention distributions according
  to these assumptions.  For more information, see ``Online and Linear-Time
  Attention by Enforcing Monotonic Alignments''.

  Args:
    p_choose_i: Probability of choosing input sequence/memory element i.  Should
      be of shape (batch_size, input_sequence_length), and should all be in the
      range [0, 1].
    previous_attention: The attention distribution from the previous output
      timestep.  Should be of shape (batch_size, input_sequence_length).  For
      the first output timestep, preevious_attention[n] should be [1, 0, 0, ...,
      0] for all n in [0, ... batch_size - 1].
    mode: How to compute the attention distribution.  Must be one of
      'recursive', 'parallel', or 'hard'.
        * 'recursive' uses tf.scan to recursively compute the distribution.
          This is slowest but is exact, general, and does not suffer from
          numerical instabilities.
        * 'parallel' uses parallelized cumulative-sum and cumulative-product
          operations to compute a closed-form solution to the recurrence
          relation defining the attention distribution.  This makes it more
          efficient than 'recursive', but it requires numerical checks which
          make the distribution non-exact.  This can be a problem in particular
          when input_sequence_length is long and/or p_choose_i has entries very
          close to 0 or 1.
        * 'hard' requires that the probabilities in p_choose_i are all either 0
          or 1, and subsequently uses a more efficient and exact solution.

  Returns:
    A tensor of shape (batch_size, input_sequence_length) representing the
    attention distributions for each sequence in the batch.

  Raises:
    ValueError: mode is not one of 'recursive', 'parallel', 'hard'.
  """
  # Force things to be tensors
  p_choose_i = ops.convert_to_tensor(p_choose_i, name="p_choose_i")
  previous_attention = ops.convert_to_tensor(
      previous_attention, name="previous_attention")
  if mode == "recursive":
    # Use .shape[0].value when it's not None, or fall back on symbolic shape
    batch_size = p_choose_i.shape[0].value or array_ops.shape(p_choose_i)[0]
    # Compute [1, 1 - p_choose_i[0], 1 - p_choose_i[1], ..., 1 - p_choose_i[-2]]
    shifted_1mp_choose_i = array_ops.concat(
        [array_ops.ones((batch_size, 1)), 1 - p_choose_i[:, :-1]], 1)
    # Compute attention distribution recursively as
    # q[i] = (1 - p_choose_i[i])*q[i - 1] + previous_attention[i]
    # attention[i] = p_choose_i[i]*q[i]
    attention = p_choose_i*array_ops.transpose(functional_ops.scan(
        # Need to use reshape to remind TF of the shape between loop iterations
        lambda x, yz: array_ops.reshape(yz[0]*x + yz[1], (batch_size,)),
        # Loop variables yz[0] and yz[1]
        [array_ops.transpose(shifted_1mp_choose_i),
         array_ops.transpose(previous_attention)],
        # Initial value of x is just zeros
        array_ops.zeros((batch_size,))))
  elif mode == "parallel":
    # safe_cumprod computes cumprod in logspace with numeric checks
    cumprod_1mp_choose_i = safe_cumprod(1 - p_choose_i, axis=1, exclusive=True)
    # Compute recurrence relation solution
    attention = p_choose_i*cumprod_1mp_choose_i*math_ops.cumsum(
        previous_attention /
        # Clip cumprod_1mp to avoid divide-by-zero
        clip_ops.clip_by_value(cumprod_1mp_choose_i, 1e-10, 1.), axis=1)
  elif mode == "hard":
    # Remove any probabilities before the index chosen last time step
    p_choose_i *= math_ops.cumsum(previous_attention, axis=1)
    # Now, use exclusive cumprod to remove probabilities after the first
    # chosen index, like so:
    # p_choose_i = [0, 0, 0, 1, 1, 0, 1, 1]
    # cumprod(1 - p_choose_i, exclusive=True) = [1, 1, 1, 1, 0, 0, 0, 0]
    # Product of above: [0, 0, 0, 1, 0, 0, 0, 0]
    attention = p_choose_i*math_ops.cumprod(
        1 - p_choose_i, axis=1, exclusive=True)
  else:
    raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")
  return attention


def _monotonic_probability_fn(score, previous_alignments, sigmoid_noise, mode,
                              seed=None):
  """Attention probability function for monotonic attention.

  Takes in unnormalized attention scores, adds pre-sigmoid noise to encourage
  the model to make discrete attention decisions, passes them through a sigmoid
  to obtain "choosing" probabilities, and then calls monotonic_attention to
  obtain the attention distribution.  For more information, see

  Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
  "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
  ICML 2017.  https://arxiv.org/abs/1704.00784

  Args:
    score: Unnormalized attention scores, shape `[batch_size, alignments_size]`
    previous_alignments: Previous attention distribution, shape
      `[batch_size, alignments_size]`
    sigmoid_noise: Standard deviation of pre-sigmoid noise.  Setting this larger
      than 0 will encourage the model to produce large attention scores,
      effectively making the choosing probabilities discrete and the resulting
      attention distribution one-hot.  It should be set to 0 at test-time, and
      when hard attention is not desired.
    mode: How to compute the attention distribution.  Must be one of
      'recursive', 'parallel', or 'hard'.  See the docstring for
      `tf.contrib.seq2seq.monotonic_attention` for more information.
    seed: (optional) Random seed for pre-sigmoid noise.

  Returns:
    A `[batch_size, alignments_size]`-shape tensor corresponding to the
    resulting attention distribution.
  """
  # Optionally add pre-sigmoid noise to the scores
  if sigmoid_noise > 0:
    noise = random_ops.random_normal(array_ops.shape(score), dtype=score.dtype,
                                     seed=seed)
    score += sigmoid_noise*noise
  # Compute "choosing" probabilities from the attention scores
  p_choose_i = math_ops.sigmoid(score)
  # Convert from choosing probabilities to attention distribution
  return monotonic_attention(p_choose_i, previous_alignments, mode)


class _BaseMonotonicAttentionMechanism(_BaseAttentionMechanism):
  """Base attention mechanism for monotonic attention.

  Simply overrides the initial_alignments function to provide a dirac
  distribution,which is needed in order for the monotonic attention
  distributions to have the correct behavior.
  """

  def initial_alignments(self, batch_size, dtype):
    """Creates the initial alignment values for the monotonic attentions.

    Initializes to dirac distributions, i.e. [1, 0, 0, ...memory length..., 0]
    for all entries in the batch.

    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.

    Returns:
      A `dtype` tensor shaped `[batch_size, alignments_size]`
      (`alignments_size` is the values' `max_time`).
    """
    max_time = self._alignments_size
    return array_ops.one_hot(
        array_ops.zeros((batch_size,), dtype=dtypes.int32), max_time,
        dtype=dtype)


class BahdanauMonotonicAttention(_BaseMonotonicAttentionMechanism):
  """Monotonic attention mechanism with Bahadanau-style energy function.

  This type of attention encorces a monotonic constraint on the attention
  distributions; that is once the model attends to a given point in the memory
  it can't attend to any prior points at subsequence output timesteps.  It
  achieves this by using the _monotonic_probability_fn instead of softmax to
  construct its attention distributions.  Since the attention scores are passed
  through a sigmoid, a learnable scalar bias parameter is applied after the
  score function and before the sigmoid.  Otherwise, it is equivalent to
  BahdanauAttention.  This approach is proposed in

  Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
  "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
  ICML 2017.  https://arxiv.org/abs/1704.00784
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               score_mask_value=float("-inf"),
               sigmoid_noise=0.,
               sigmoid_noise_seed=None,
               score_bias_init=0.,
               mode="parallel",
               name="BahdanauMonotonicAttention"):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      sigmoid_noise: Standard deviation of pre-sigmoid noise.  See the docstring
        for `_monotonic_probability_fn` for more information.
      sigmoid_noise_seed: (optional) Random seed for pre-sigmoid noise.
      score_bias_init: Initial value for score bias scalar.  It's recommended to
        initialize this to a negative value when the length of the memory is
        large.
      mode: How to compute the attention distribution.  Must be one of
        'recursive', 'parallel', or 'hard'.  See the docstring for
        `tf.contrib.seq2seq.monotonic_attention` for more information.
      name: Name to use when creating ops.
    """
    # Set up the monotonic probability fn with supplied parameters
    wrapped_probability_fn = functools.partial(
        _monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
        seed=sigmoid_noise_seed)
    super(BahdanauMonotonicAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name
    self._score_bias_init = score_bias_init

  def __call__(self, query, previous_alignments):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      previous_alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(
        None, "bahdanau_monotonic_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      score = _bahdanau_score(processed_query, self._keys, self._normalize)
      score_bias = variable_scope.get_variable(
          "attention_score_bias", dtype=processed_query.dtype,
          initializer=self._score_bias_init)
      score += score_bias
    alignments = self._probability_fn(score, previous_alignments)
    return alignments


class LuongMonotonicAttention(_BaseMonotonicAttentionMechanism):
  """Monotonic attention mechanism with Luong-style energy function.

  This type of attention encorces a monotonic constraint on the attention
  distributions; that is once the model attends to a given point in the memory
  it can't attend to any prior points at subsequence output timesteps.  It
  achieves this by using the _monotonic_probability_fn instead of softmax to
  construct its attention distributions.  Otherwise, it is equivalent to
  LuongAttention.  This approach is proposed in

  Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
  "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
  ICML 2017.  https://arxiv.org/abs/1704.00784
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               scale=False,
               score_mask_value=float("-inf"),
               sigmoid_noise=0.,
               sigmoid_noise_seed=None,
               score_bias_init=0.,
               mode="parallel",
               name="LuongMonotonicAttention"):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      scale: Python boolean.  Whether to scale the energy term.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      sigmoid_noise: Standard deviation of pre-sigmoid noise.  See the docstring
        for `_monotonic_probability_fn` for more information.
      sigmoid_noise_seed: (optional) Random seed for pre-sigmoid noise.
      score_bias_init: Initial value for score bias scalar.  It's recommended to
        initialize this to a negative value when the length of the memory is
        large.
      mode: How to compute the attention distribution.  Must be one of
        'recursive', 'parallel', or 'hard'.  See the docstring for
        `tf.contrib.seq2seq.monotonic_attention` for more information.
      name: Name to use when creating ops.
    """
    # Set up the monotonic probability fn with supplied parameters
    wrapped_probability_fn = functools.partial(
        _monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
        seed=sigmoid_noise_seed)
    super(LuongMonotonicAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._score_bias_init = score_bias_init
    self._name = name

  def __call__(self, query, previous_alignments):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      previous_alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(None, "luong_monotonic_attention",
                                       [query]):
      score = _luong_score(query, self._keys, self._scale)
      score_bias = variable_scope.get_variable(
          "attention_score_bias", dtype=query.dtype,
          initializer=self._score_bias_init)
      score += score_bias
    alignments = self._probability_fn(score, previous_alignments)
    return alignments


class AttentionWrapperState(
    collections.namedtuple("AttentionWrapperState",
                           ("cell_state", "attention", "time", "alignments",
                            "alignment_history"))):
  """`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
       emitted at the previous time step for each attention mechanism.
    - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
       containing alignment matrices from all time steps for each attention
       mechanism. Call `stack()` on each to convert to a `Tensor`.
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
      this one, except any overridden properties as provided in `kwargs`.
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


def _compute_attention(attention_mechanism, cell_output, previous_alignments,
                       attention_layer):
  """Computes the attention and alignments for a given attention_mechanism."""
  alignments = attention_mechanism(
      cell_output, previous_alignments=previous_alignments)

  # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
  expanded_alignments = array_ops.expand_dims(alignments, 1)
  # Context is the inner product of alignments and values along the
  # memory time dimension.
  # alignments shape is
  #   [batch_size, 1, memory_time]
  # attention_mechanism.values shape is
  #   [batch_size, memory_time, memory_size]
  # the batched matmul is over memory_time, so the output shape is
  #   [batch_size, 1, memory_size].
  # we then squeeze out the singleton dim.
  context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
  context = array_ops.squeeze(context, [1])

  if attention_layer is not None:
    attention = attention_layer(array_ops.concat([cell_output, context], 1))
  else:
    attention = context

  return attention, alignments


class AttentionWrapper(rnn_cell_impl.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               attention_layer_size=None,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               name=None):
    """Construct the `AttentionWrapper`.

    **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
    `AttentionWrapper`, then you must ensure that:

    - The encoder output has been tiled to `beam_width` via
      @{tf.contrib.seq2seq.tile_batch} (NOT `tf.tile`).
    - The `batch_size` argument passed to the `zero_state` method of this
      wrapper is equal to `true_batch_size * beam_width`.
    - The initial state created with `zero_state` above contains a
      `cell_state` value containing properly tiled final state from the
      encoder.

    An example:

    ```
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
        encoder_outputs, multiplier=beam_width)
    tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=beam_width)
    attention_mechanism = MyFavoriteAttentionMechanism(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
    decoder_initial_state = attention_cell.zero_state(
        dtype, batch_size=true_batch_size * beam_width)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=tiled_encoder_final_state)
    ```

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: A list of `AttentionMechanism` instances or a single
        instance.
      attention_layer_size: A list of Python integers or a single Python
        integer, the depth of the attention (output) layer(s). If None
        (default), use the context as attention at each time step. Otherwise,
        feed the context and cell output into the attention layer to generate
        attention at each time step. If attention_mechanism is a list,
        attention_layer_size must be a list of the same length.
      alignment_history: Python boolean, whether to store alignment history
        from all time steps in the final output state (currently stored as a
        time major `TensorArray` on which you must call `stack()`).
      cell_input_fn: (optional) A `callable`.  The default is:
        `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
      output_attention: Python bool.  If `True` (default), the output at each
        time step is the attention value.  This is the behavior of Luong-style
        attention mechanisms.  If `False`, the output at each time step is
        the output of `cell`.  This is the beahvior of Bhadanau-style
        attention mechanisms.  In both cases, the `attention` tensor is
        propagated to the next time step via the state and is used there.
        This flag only controls whether the attention mechanism is propagated
        up to the next cell in an RNN stack or to the top RNN output.
      initial_cell_state: The initial state value to use for the cell when
        the user calls `zero_state()`.  Note that if this value is provided
        now, and the user uses a `batch_size` argument of `zero_state` which
        does not match the batch size of `initial_cell_state`, proper
        behavior is not guaranteed.
      name: Name to use when creating ops.

    Raises:
      TypeError: `attention_layer_size` is not None and (`attention_mechanism`
        is a list but `attention_layer_size` is not; or vice versa).
      ValueError: if `attention_layer_size` is not None, `attention_mechanism`
        is a list, and its length does not match that of `attention_layer_size`.
    """
    super(AttentionWrapper, self).__init__(name=name)
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError(
          "cell must be an RNNCell, saw type: %s" % type(cell).__name__)
    if isinstance(attention_mechanism, (list, tuple)):
      self._is_multi = True
      attention_mechanisms = attention_mechanism
      for attention_mechanism in attention_mechanisms:
        if not isinstance(attention_mechanism, AttentionMechanism):
          raise TypeError(
              "attention_mechanism must contain only instances of "
              "AttentionMechanism, saw type: %s"
              % type(attention_mechanism).__name__)
    else:
      self._is_multi = False
      if not isinstance(attention_mechanism, AttentionMechanism):
        raise TypeError(
            "attention_mechanism must be an AttentionMechanism or list of "
            "multiple AttentionMechanism instances, saw type: %s"
            % type(attention_mechanism).__name__)
      attention_mechanisms = (attention_mechanism,)

    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)

    if attention_layer_size is not None:
      attention_layer_sizes = tuple(
          attention_layer_size
          if isinstance(attention_layer_size, (list, tuple))
          else (attention_layer_size,))
      if len(attention_layer_sizes) != len(attention_mechanisms):
        raise ValueError(
            "If provided, attention_layer_size must contain exactly one "
            "integer per attention_mechanism, saw: %d vs %d"
            % (len(attention_layer_sizes), len(attention_mechanisms)))
      self._attention_layers = tuple(
          layers_core.Dense(
              attention_layer_size, name="attention_layer", use_bias=False)
          for attention_layer_size in attention_layer_sizes)
      self._attention_layer_size = sum(attention_layer_sizes)
    else:
      self._attention_layers = None
      self._attention_layer_size = sum(
          attention_mechanism.values.get_shape()[-1].value
          for attention_mechanism in attention_mechanisms)

    self._cell = cell
    self._attention_mechanisms = attention_mechanisms
    self._cell_input_fn = cell_input_fn
    self._output_attention = output_attention
    self._alignment_history = alignment_history
    with ops.name_scope(name, "AttentionWrapperInit"):
      if initial_cell_state is None:
        self._initial_cell_state = None
      else:
        final_state_tensor = nest.flatten(initial_cell_state)[-1]
        state_batch_size = (
            final_state_tensor.shape[0].value
            or array_ops.shape(final_state_tensor)[0])
        error_message = (
            "When constructing AttentionWrapper %s: " % self._base_name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and initial_cell_state.  Are you using "
            "the BeamSearchDecoder?  You may need to tile your initial state "
            "via the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            self._batch_size_checks(state_batch_size, error_message)):
          self._initial_cell_state = nest.map_structure(
              lambda s: array_ops.identity(s, name="check_initial_cell_state"),
              initial_cell_state)

  def _batch_size_checks(self, batch_size, error_message):
    return [check_ops.assert_equal(batch_size,
                                   attention_mechanism.batch_size,
                                   message=error_message)
            for attention_mechanism in self._attention_mechanisms]

  def _item_or_tuple(self, seq):
    """Returns `seq` as tuple or the singular element.

    Which is returned is determined by how the AttentionMechanism(s) were passed
    to the constructor.

    Args:
      seq: A non-empty sequence of items or generator.

    Returns:
       Either the values in the sequence as a tuple if AttentionMechanism(s)
       were passed to the constructor as a sequence or the singular element.
    """
    t = tuple(seq)
    if self._is_multi:
      return t
    else:
      return t[0]

  @property
  def output_size(self):
    if self._output_attention:
      return self._attention_layer_size
    else:
      return self._cell.output_size

  @property
  def state_size(self):
    """The `state_size` property of `AttentionWrapper`.

    Returns:
      An `AttentionWrapperState` tuple containing shapes used by this object.
    """
    return AttentionWrapperState(
        cell_state=self._cell.state_size,
        time=tensor_shape.TensorShape([]),
        attention=self._attention_layer_size,
        alignments=self._item_or_tuple(
            a.alignments_size for a in self._attention_mechanisms),
        alignment_history=self._item_or_tuple(
            () for _ in self._attention_mechanisms))  # sometimes a TensorArray

  def zero_state(self, batch_size, dtype):
    """Return an initial (zero) state tuple for this `AttentionWrapper`.

    **NOTE** Please see the initializer documentation for details of how
    to call `zero_state` if using an `AttentionWrapper` with a
    `BeamSearchDecoder`.

    Args:
      batch_size: `0D` integer tensor: the batch size.
      dtype: The internal state data type.

    Returns:
      An `AttentionWrapperState` tuple containing zeroed out tensors and,
      possibly, empty `TensorArray` objects.

    Raises:
      ValueError: (or, possibly at runtime, InvalidArgument), if
        `batch_size` does not match the output size of the encoder passed
        to the wrapper object at initialization time.
    """
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._initial_cell_state is not None:
        cell_state = self._initial_cell_state
      else:
        cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
          "When calling zero_state of AttentionWrapper %s: " % self._base_name +
          "Non-matching batch sizes between the memory "
          "(encoder output) and the requested batch size.  Are you using "
          "the BeamSearchDecoder?  If so, make sure your encoder output has "
          "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
          "the batch_size= argument passed to zero_state is "
          "batch_size * beam_width.")
      with ops.control_dependencies(
          self._batch_size_checks(batch_size, error_message)):
        cell_state = nest.map_structure(
            lambda s: array_ops.identity(s, name="checked_cell_state"),
            cell_state)
      return AttentionWrapperState(
          cell_state=cell_state,
          time=array_ops.zeros([], dtype=dtypes.int32),
          attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                        dtype),
          alignments=self._item_or_tuple(
              attention_mechanism.initial_alignments(batch_size, dtype)
              for attention_mechanism in self._attention_mechanisms),
          alignment_history=self._item_or_tuple(
              tensor_array_ops.TensorArray(dtype=dtype, size=0,
                                           dynamic_size=True)
              if self._alignment_history else ()
              for _ in self._attention_mechanisms))

  def call(self, inputs, state):
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
      `attention_layer_size` outputs).

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `AttentionWrapperState` containing
        tensors from the previous time step.

    Returns:
      A tuple `(attention_or_cell_output, next_state)`, where:

      - `attention_or_cell_output` depending on `output_attention`.
      - `next_state` is an instance of `AttentionWrapperState`
         containing the state calculated at this time step.

    Raises:
      TypeError: If `state` is not an instance of `AttentionWrapperState`.
    """
    if not isinstance(state, AttentionWrapperState):
      raise TypeError("Expected state to be instance of AttentionWrapperState. "
                      "Received type %s instead."  % type(state))

    # Step 1: Calculate the true inputs to the cell based on the
    # previous attention value.
    cell_inputs = self._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

    cell_batch_size = (
        cell_output.shape[0].value or array_ops.shape(cell_output)[0])
    error_message = (
        "When applying AttentionWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    with ops.control_dependencies(
        self._batch_size_checks(cell_batch_size, error_message)):
      cell_output = array_ops.identity(
          cell_output, name="checked_cell_output")

    if self._is_multi:
      previous_alignments = state.alignments
      previous_alignment_history = state.alignment_history
    else:
      previous_alignments = [state.alignments]
      previous_alignment_history = [state.alignment_history]

    all_alignments = []
    all_attentions = []
    all_histories = []
    for i, attention_mechanism in enumerate(self._attention_mechanisms):
      attention, alignments = _compute_attention(
          attention_mechanism, cell_output, previous_alignments[i],
          self._attention_layers[i] if self._attention_layers else None)
      alignment_history = previous_alignment_history[i].write(
          state.time, alignments) if self._alignment_history else ()

      all_alignments.append(alignments)
      all_histories.append(alignment_history)
      all_attentions.append(attention)

    attention = array_ops.concat(all_attentions, 1)
    next_state = AttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        alignments=self._item_or_tuple(all_alignments),
        alignment_history=self._item_or_tuple(all_histories))

    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state
