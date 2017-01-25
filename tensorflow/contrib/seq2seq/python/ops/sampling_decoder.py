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
    next_inputs = control_flow_ops.cond(
        all_finished, lambda: self._zero_inputs,
        lambda: nest.map_structure(lambda inp: inp.read(next_time), self._input_tas))
    return (sample_id, finished, next_inputs)
