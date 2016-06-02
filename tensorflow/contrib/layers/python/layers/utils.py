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
"""Common util functions used by layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


__all__ = ['collect_named_outputs',
           'get_variable_collections',
           'two_element_tuple',
           'last_dimension',
           'first_dimension']


def collect_named_outputs(collections, name, outputs):
  """Add tuple (name, outputs) to collections.

  It is useful to collect end-points or tags for summaries. Example of usage:

  logits = collect_named_outputs('end_points', 'inception_v3/logits', logits)

  Args:
    collections: A collection or list of collections. If None skip collection.
    name: String, name to represent the outputs, ex. 'inception_v3/conv1'
    outputs: Tensor, an output tensor to collect

  Returns:
    The outputs Tensor to allow inline call.
  """
  if collections:
    # Remove ending '/' if present.
    if name[-1] == '/':
      name = name[:-1]
    ops.add_to_collections(collections, (name, outputs))
  return outputs


def get_variable_collections(variables_collections, name):
  if isinstance(variables_collections, dict):
    variable_collections = variables_collections[name]
  else:
    variable_collections = variables_collections
  return variable_collections


def first_dimension(shape, min_rank=1):
  """Returns the first dimension of shape while checking it has min_rank.

  Args:
    shape: A `TensorShape`.
    min_rank: Integer, minimum rank of shape.

  Returns:
    The value of the first dimension.

  Raises:
    ValueError: if inputs don't have at least min_rank dimensions, or if the
      first dimension value is not defined.
  """
  dims = shape.dims
  if dims is None:
    raise ValueError('dims of shape must be known but is None')
  if len(dims) < min_rank:
    raise ValueError('rank of shape must be at least %d not: %d' % (min_rank,
                                                                    len(dims)))
  value = dims[0].value
  if value is None:
    raise ValueError('first dimension shape must be known but is None')
  return value


def last_dimension(shape, min_rank=1):
  """Returns the last dimension of shape while checking it has min_rank.

  Args:
    shape: A `TensorShape`.
    min_rank: Integer, minimum rank of shape.

  Returns:
    The value of the last dimension.

  Raises:
    ValueError: if inputs don't have at least min_rank dimensions, or if the
      last dimension value is not defined.
  """
  dims = shape.dims
  if dims is None:
    raise ValueError('dims of shape must be known but is None')
  if len(dims) < min_rank:
    raise ValueError('rank of shape must be at least %d not: %d' % (min_rank,
                                                                    len(dims)))
  value = dims[-1].value
  if value is None:
    raise ValueError('last dimension shape must be known but is None')
  return value


def two_element_tuple(int_or_tuple):
  """Converts `int_or_tuple` to height, width.

  Several of the functions that follow accept arguments as either
  a tuple of 2 integers or a single integer.  A single integer
  indicates that the 2 values of the tuple are the same.

  This functions normalizes the input value by always returning a tuple.

  Args:
    int_or_tuple: A list of 2 ints, a single int or a `TensorShape`.

  Returns:
    A tuple with 2 values.

  Raises:
    ValueError: If `int_or_tuple` it not well formed.
  """
  if isinstance(int_or_tuple, (list, tuple)):
    if len(int_or_tuple) != 2:
      raise ValueError('Must be a list with 2 elements: %s' % int_or_tuple)
    return int(int_or_tuple[0]), int(int_or_tuple[1])
  if isinstance(int_or_tuple, int):
    return int(int_or_tuple), int(int_or_tuple)
  if isinstance(int_or_tuple, tensor_shape.TensorShape):
    if len(int_or_tuple) == 2:
      return int_or_tuple[0], int_or_tuple[1]
  raise ValueError('Must be an int, a list with 2 elements or a TensorShape of '
                   'length 2')
