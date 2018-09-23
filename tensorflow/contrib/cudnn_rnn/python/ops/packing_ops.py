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
"""Cudnn RNN operators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import gen_packing_ops

def packed_sequence_alignment(
    sequence_lengths,
    name=None):
  """Docstring goes here

  Args:
    sequence_lengths: the sequence lengths.
    name: name of the operation.
  Returns:
    tuple of alignments and batch sizes
  Raises:
    ValueError: if rnn_mode or direction is invalid.
  """
  ret = gen_packing_ops.packed_sequence_alignment(
      sequence_lengths=sequence_lengths,
      name=name)
  return ret[0], ret[1]
  
def sequence_gather_scatter_indices(
    total_length,
    sequence_lengths,
    batch_order,
    name=None):
  """Docstring goes here

  Args:
    sequence_lengths: the sequence lengths.
    name: name of the operation.
  Returns:
    tuple of alignments and batch sizes
  Raises:
    ValueError: if rnn_mode or direction is invalid.
  """
  ret = gen_packing_ops.sequence_gather_scatter_indices(
      total_length=total_length,
      sequence_lengths=sequence_lengths,
      batch_order=batch_order,
      name=name)
  return ret

def pack_sequence(
    sequence,
    alignments,
    batch_sizes,
    name=None):
  """Docstring goes here

  Args:
    sequence_lengths: the sequence lengths.
    name: name of the operation.
  Returns:
    tuple of alignments and batch sizes
  Raises:
    ValueError: if rnn_mode or direction is invalid.
  """
  ret = gen_packing_ops.pack_sequence(
      sequence=sequence,
      alignments=alignments,
      batch_sizes=batch_sizes,
      name=name)
  return ret
  
def unpack_sequence(
    packed,
    alignments,
    batch_sizes,
    name=None):
  """Docstring goes here

  Args:
    sequence_lengths: the sequence lengths.
    name: name of the operation.
  Returns:
    tuple of alignments and batch sizes
  Raises:
    ValueError: if rnn_mode or direction is invalid.
  """
  ret = gen_packing_ops.unpack_sequence(
      packed=packed,
      alignments=alignments,
      batch_sizes=batch_sizes,
      name=name)
  return ret
  