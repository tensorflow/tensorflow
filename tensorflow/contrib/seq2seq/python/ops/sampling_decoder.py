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
from tensorflow.python.framework import constant_op
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
    "BasicTrainingSampler"
]

_transpose_batch_time = decoder._transpose_batch_time  # pylint: disable=protected-access


@six.add_metaclass(abc.ABCMeta)
class Sampler(object):

  @property
  def batch_size(self):
    pass

  @abc.abstractmethod
  def initialize(self):
    pass

  @abc.abstractmethod
  def sample(self, time, outputs, state):
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
    return self._sampler.initialize() + (self._initial_state,)

  def step(self, time, inputs, state):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    cell_outputs, next_state = self._cell(inputs, state)
    (sample_id, finished, next_inputs) = self._sampler.sample(
        time=time, outputs=cell_outputs, state=next_state)
    outputs = SamplingDecoderOutput(cell_outputs, sample_id)
    return (outputs, next_state, next_inputs, finished)


class BasicTrainingSampler(Sampler):
  """A (non-)sampler for use during training.  Only reads inputs."""

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
    sequence_length = ops.convert_to_tensor(
        sequence_length, name="sequence_length")
    if sequence_length.get_shape().ndims != 1:
      raise ValueError(
          "Expected sequence_length to be a vector, but received shape: %s" %
          sequence_length.get_shape())
    self._sequence_length = sequence_length
    self._zero_inputs = nest.map_structure(
        lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
    self._batch_size = array_ops.size(sequence_length)

  @property
  def batch_size(self):
    return self._batch_size

  def initialize(self):
    finished = math_ops.equal(0, self._sequence_length)
    all_finished = math_ops.reduce_all(finished)
    next_inputs = control_flow_ops.cond(
        all_finished, lambda: self._zero_inputs,
        lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
    return (finished, next_inputs)

  def sample(self, time, **unused_kwargs):
    next_time = time + 1
    finished = (next_time >= self._sequence_length)
    all_finished = math_ops.reduce_all(finished)
    sample_id = array_ops.tile([constant_op.constant(-1)], [self._batch_size])
    def read_from_ta(inp):
      return inp.read(next_time)
    next_inputs = control_flow_ops.cond(
        all_finished, lambda: self._zero_inputs,
        lambda: nest.map_structure(read_from_ta, self._input_tas))
    return (sample_id, finished, next_inputs)


class ArgmaxEmbeddingInferenceSampler(Sampler):
  """A (non-)sampler for use during inference.

  Uses the argmax of the output (treated as logits) and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, start_tokens, end_token, max_time=None):
    """Initializer.

    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      max_time: `int32` scalar, maximum allowed number of decoding steps.
        Default is `None` (decode until `end_token` is seen).

    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    if callable(embedding):
      self._embedding_fn = embedding
    else:

      def embedding_fn(ids):
        return embedding_ops.embedding_lookup(embedding, ids)

      self._embedding_fn = embedding_fn

    self._start_tokens = ops.convert_to_tensor(
        start_tokens, dtype=dtypes.int32, name="start_tokens")
    self._end_token = ops.convert_to_tensor(
        end_token, dtype=dtypes.int32, name="end_token")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._batch_size = array_ops.size(self._start_tokens)
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")
    if max_time is not None:
      self._max_time = ops.convert_to_tensor(
          max_time, dtype=dtypes.int32, name="max_time")
      if self._max_time.get_shape().ndims != 0:
        raise ValueError("max_time must be a scalar")
    else:
      self._max_time = None
    self._start_inputs = self._embedding_fn(self._start_tokens)

  @property
  def batch_size(self):
    return self._batch_size

  def initialize(self):
    if self._max_time is not None:
      finished = array_ops.tile([math_ops.equal(self._max_time, 0)],
                                [self._batch_size])
    else:
      finished = array_ops.tile([False], [self._batch_size])
    return (finished, self._start_inputs)

  def sample(self, time, outputs, **unused_kwargs):
    # Outputs are logits, use argmax to get the most probable id
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      outputs)
    sample_ids = math_ops.cast(math_ops.argmax(outputs, axis=-1), dtypes.int32)
    finished = math_ops.equal(sample_ids, self._end_token)
    if self._max_time is not None:
      finished = math_ops.logical_or(finished, time + 1 >= self._max_time)
    all_finished = math_ops.reduce_all(finished)

    next_inputs = control_flow_ops.cond(
        all_finished,
        # If we're finished, the next_inputs value doesn't matter
        lambda: self._start_inputs,
        lambda: self._embedding_fn(sample_ids))
    return (sample_ids, finished, next_inputs)
