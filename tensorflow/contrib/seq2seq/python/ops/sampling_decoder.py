# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import six

from tensorflow.contrib.rnn import core_rnn_cell
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest

__all__ = [
    "Sampler", "SamplingDecoderOutput", "BasicSamplingDecoder",
    "BasicTrainingSampler", "GreedyEmbeddingSampler", "CustomSampler",
]

_transpose_batch_time = decoder._transpose_batch_time  # pylint: disable=protected-access


@six.add_metaclass(abc.ABCMeta)
class Sampler(object):
  """Sampler interface.  Sampler instances are used by BasicSamplingDecoder."""

  @abc.abstractproperty
  def batch_size(self):
    """Returns a scalar int32 tensor."""
    raise NotImplementedError("batch_size has not been implemented")

  @abc.abstractmethod
  def initialize(self, name=None):
    """Returns `(initial_finished, initial_inputs)`."""
    pass

  @abc.abstractmethod
  def sample(self, time, outputs, state, name=None):
    """Returns `sample_ids`."""
    pass

  @abc.abstractmethod
  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """Returns `(finished, next_inputs, next_state)`."""
    pass


class SamplingDecoderOutput(
    collections.namedtuple("SamplingDecoderOutput",
                           ("rnn_output", "sample_id"))):
  pass


class BasicSamplingDecoder(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(self, cell, sampler, initial_state):
    """Initialize BasicSamplingDecoder.

    Args:
      cell: An `RNNCell` instance.
      sampler: A `Sampler` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.

    Raises:
      TypeError: if `cell` is not an instance of `RNNCell` or `sampler`
        is not an instance of `Sampler`.
    """
    if not isinstance(cell, core_rnn_cell.RNNCell):
      raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    if not isinstance(sampler, Sampler):
      raise TypeError("sampler must be a Sampler, received: %s" %
                      type(sampler))
    self._cell = cell
    self._sampler = sampler
    self._initial_state = initial_state

  @property
  def batch_size(self):
    return self._sampler.batch_size

  @property
  def output_size(self):
    # Return the cell output and the id
    return SamplingDecoderOutput(
        rnn_output=self._cell.output_size,
        sample_id=tensor_shape.TensorShape([]))

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = nest.flatten(self._initial_state)[0].dtype
    return SamplingDecoderOutput(
        nest.map_structure(lambda _: dtype, self._cell.output_size),
        dtypes.int32)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self._sampler.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(
        name, "BasicSamplingDecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      sample_ids = self._sampler.sample(
          time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self._sampler.next_inputs(
          time=time, outputs=cell_outputs, state=cell_state,
          sample_ids=sample_ids)
    outputs = SamplingDecoderOutput(cell_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)


class CustomSampler(Sampler):
  """Base abstract class that allows the user to customize sampling."""

  def __init__(self, initialize_fn, sample_fn, next_inputs_fn):
    """Initializer.

    Args:
      initialize_fn: callable that returns `(finished, next_inputs)`
        for the first iteration.
      sample_fn: callable that takes `(time, outputs, state)`
        and emits tensor `sample_ids`.
      next_inputs_fn: callable that takes `(time, outputs, state, sample_ids)`
        and emits `(finished, next_inputs, next_state)`.
    """
    self._initialize_fn = initialize_fn
    self._sample_fn = sample_fn
    self._next_inputs_fn = next_inputs_fn
    self._batch_size = None

  @property
  def batch_size(self):
    if self._batch_size is None:
      raise ValueError("batch_size accessed before initialize was called")
    return self._batch_size

  def initialize(self, name=None):
    with ops.name_scope(name, "%sInitialize" % type(self).__name__):
      (finished, next_inputs) = self._initialize_fn()
      if self._batch_size is None:
        self._batch_size = array_ops.size(finished)
    return (finished, next_inputs)

  def sample(self, time, outputs, state, name=None):
    with ops.name_scope(
        name, "%sSample" % type(self).__name__, (time, outputs, state)):
      return self._sample_fn(time=time, outputs=outputs, state=state)

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    with ops.name_scope(
        name, "%sNextInputs" % type(self).__name__, (time, outputs, state)):
      return self._next_inputs_fn(
          time=time, outputs=outputs, state=state, sample_ids=sample_ids)


class BasicTrainingSampler(Sampler):
  """A (non-)sampler for use during training.  Only reads inputs.

  Returned sample_ids are the argmax of the RNN output logits.
  """

  def __init__(self, inputs, sequence_length, time_major=False):
    """Initializer.

    Args:
      inputs: A (structure of) input tensors.
      sequence_length: An int32 vector tensor.
      time_major: Python bool.

    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    inputs = ops.convert_to_tensor(inputs, name="inputs")
    if not time_major:
      inputs = nest.map_structure(_transpose_batch_time, inputs)

    def _unstack_ta(inp):
      return tensor_array_ops.TensorArray(
          dtype=inp.dtype, size=array_ops.shape(inp)[0],
          element_shape=inp.get_shape()[1:]).unstack(inp)

    self._input_tas = nest.map_structure(_unstack_ta, inputs)
    self._sequence_length = ops.convert_to_tensor(
        sequence_length, name="sequence_length")
    if self._sequence_length.get_shape().ndims != 1:
      raise ValueError(
          "Expected sequence_length to be a vector, but received shape: %s" %
          self._sequence_length.get_shape())

    self._zero_inputs = nest.map_structure(
        lambda inp: array_ops.zeros_like(inp[0, :]), inputs)

    self._batch_size = array_ops.size(sequence_length)

  @property
  def batch_size(self):
    return self._batch_size

  def initialize(self, name=None):
    finished = math_ops.equal(0, self._sequence_length)
    all_finished = math_ops.reduce_all(finished)
    next_inputs = control_flow_ops.cond(
        all_finished, lambda: self._zero_inputs,
        lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
    return (finished, next_inputs)

  def sample(self, time, outputs, name=None, **unused_kwargs):
    del time  # unused by sample_fn
    sample_ids = math_ops.cast(
        math_ops.argmax(outputs, axis=-1), dtypes.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    """next_inputs_fn for BasicTrainingSampler."""
    del outputs  # unused by next_inputs_fn
    next_time = time + 1
    finished = (next_time >= self._sequence_length)
    all_finished = math_ops.reduce_all(finished)
    def read_from_ta(inp):
      return inp.read(next_time)
    next_inputs = control_flow_ops.cond(
        all_finished, lambda: self._zero_inputs,
        lambda: nest.map_structure(read_from_ta, self._input_tas))
    return (finished, next_inputs, state)


class GreedyEmbeddingSampler(Sampler):
  """A (non-)sampler for use during inference.

  Uses the argmax of the output (treated as logits) and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, start_tokens, end_token):
    """Initializer.

    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.

    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    if callable(embedding):
      self._embedding_fn = embedding
    else:
      self._embedding_fn = (
          lambda ids: embedding_ops.embedding_lookup(embedding, ids))

    self._start_tokens = ops.convert_to_tensor(
        start_tokens, dtype=dtypes.int32, name="start_tokens")
    self._end_token = ops.convert_to_tensor(
        end_token, dtype=dtypes.int32, name="end_token")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._batch_size = array_ops.size(start_tokens)
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")
    self._start_inputs = self._embedding_fn(self._start_tokens)

  @property
  def batch_size(self):
    return self._batch_size

  def initialize(self, name=None):
    finished = array_ops.tile([False], [self._batch_size])
    return (finished, self._start_inputs)

  def sample(self, time, outputs, state, name=None):
    """sample for GreedyEmbeddingSampler."""
    del time, state  # unused by sample_fn
    # Outputs are logits, use argmax to get the most probable id
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      outputs)
    sample_ids = math_ops.cast(
        math_ops.argmax(outputs, axis=-1), dtypes.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """next_inputs_fn for GreedyEmbeddingSampler."""
    del time, outputs  # unused by next_inputs_fn
    finished = math_ops.equal(sample_ids, self._end_token)
    all_finished = math_ops.reduce_all(finished)
    next_inputs = control_flow_ops.cond(
        all_finished,
        # If we're finished, the next_inputs value doesn't matter
        lambda: self._start_inputs,
        lambda: self._embedding_fn(sample_ids))
    return (finished, next_inputs, state)
