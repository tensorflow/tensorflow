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
"""Contains helper functions for creating summaries.

This module contains various helper functions for quickly and easily adding
tensorflow summaries. These allow users to print summary values
automatically as they are computed and add prefixes to collections of summaries.

Example usage:

  import tensorflow as tf
  slim = tf.contrib.slim

  slim.summaries.add_histogram_summaries(slim.variables.get_model_variables())
  slim.summaries.add_scalar_summary(total_loss, 'Total Loss')
  slim.summaries.add_scalar_summary(learning_rate, 'Learning Rate')
  slim.summaries.add_histogram_summaries(my_tensors)
  slim.summaries.add_zero_fraction_summaries(my_tensors)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import nn_impl as nn
from tensorflow.python.summary import summary


def _get_summary_name(tensor, name=None, prefix=None, postfix=None):
  """Produces the summary name given.

  Args:
    tensor: A variable or op `Tensor`.
    name: The optional name for the summary.
    prefix: An optional prefix for the summary name.
    postfix: An optional postfix for the summary name.

  Returns:
    a summary name.
  """
  if not name:
    name = tensor.op.name
  if prefix:
    name = prefix + '/' + name
  if postfix:
    name = name + '/' + postfix
  return name


def add_histogram_summary(tensor, name=None, prefix=None):
  """Adds a histogram summary for the given tensor.

  Args:
    tensor: A variable or op tensor.
    name: The optional name for the summary.
    prefix: An optional prefix for the summary names.

  Returns:
    A scalar `Tensor` of type `string` whose contents are the serialized
    `Summary` protocol buffer.
  """
  return summary.histogram(
      _get_summary_name(tensor, name, prefix), tensor)


def add_image_summary(tensor, name=None, prefix=None, print_summary=False):
  """Adds an image summary for the given tensor.

  Args:
    tensor: a variable or op tensor with shape [batch,height,width,channels]
    name: the optional name for the summary.
    prefix: An optional prefix for the summary names.
    print_summary: If `True`, the summary is printed to stdout when the summary
      is computed.

  Returns:
    An image `Tensor` of type `string` whose contents are the serialized
    `Summary` protocol buffer.
  """
  summary_name = _get_summary_name(tensor, name, prefix)
  # If print_summary, then we need to make sure that this call doesn't add the
  # non-printing op to the collection. We'll add it to the collection later.
  collections = [] if print_summary else None
  op = summary.image(
      name=summary_name, tensor=tensor, collections=collections)
  if print_summary:
    op = logging_ops.Print(op, [tensor], summary_name)
    ops.add_to_collection(ops.GraphKeys.SUMMARIES, op)
  return op


def add_scalar_summary(tensor, name=None, prefix=None, print_summary=False):
  """Adds a scalar summary for the given tensor.

  Args:
    tensor: a variable or op tensor.
    name: the optional name for the summary.
    prefix: An optional prefix for the summary names.
    print_summary: If `True`, the summary is printed to stdout when the summary
      is computed.

  Returns:
    A scalar `Tensor` of type `string` whose contents are the serialized
    `Summary` protocol buffer.
  """
  collections = [] if print_summary else None
  summary_name = _get_summary_name(tensor, name, prefix)

  # If print_summary, then we need to make sure that this call doesn't add the
  # non-printing op to the collection. We'll add it to the collection later.
  op = summary.scalar(
      name=summary_name, tensor=tensor, collections=collections)
  if print_summary:
    op = logging_ops.Print(op, [tensor], summary_name)
    ops.add_to_collection(ops.GraphKeys.SUMMARIES, op)
  return op


def add_zero_fraction_summary(tensor, name=None, prefix=None,
                              print_summary=False):
  """Adds a summary for the percentage of zero values in the given tensor.

  Args:
    tensor: a variable or op tensor.
    name: the optional name for the summary.
    prefix: An optional prefix for the summary names.
    print_summary: If `True`, the summary is printed to stdout when the summary
      is computed.

  Returns:
    A scalar `Tensor` of type `string` whose contents are the serialized
    `Summary` protocol buffer.
  """
  name = _get_summary_name(tensor, name, prefix, 'Fraction of Zero Values')
  tensor = nn.zero_fraction(tensor)
  return add_scalar_summary(tensor, name, print_summary=print_summary)


def add_histogram_summaries(tensors, prefix=None):
  """Adds a histogram summary for each of the given tensors.

  Args:
    tensors: A list of variable or op tensors.
    prefix: An optional prefix for the summary names.

  Returns:
    A list of scalar `Tensors` of type `string` whose contents are the
    serialized `Summary` protocol buffer.
  """
  summary_ops = []
  for tensor in tensors:
    summary_ops.append(add_histogram_summary(tensor, prefix=prefix))
  return summary_ops


def add_image_summaries(tensors, prefix=None):
  """Adds an image summary for each of the given tensors.

  Args:
    tensors: A list of variable or op tensors.
    prefix: An optional prefix for the summary names.

  Returns:
    A list of scalar `Tensors` of type `string` whose contents are the
    serialized `Summary` protocol buffer.
  """
  summary_ops = []
  for tensor in tensors:
    summary_ops.append(add_image_summary(tensor, prefix=prefix))
  return summary_ops


def add_scalar_summaries(tensors, prefix=None, print_summary=False):
  """Adds a scalar summary for each of the given tensors.

  Args:
    tensors: a list of variable or op tensors.
    prefix: An optional prefix for the summary names.
    print_summary: If `True`, the summary is printed to stdout when the summary
      is computed.

  Returns:
    A list of scalar `Tensors` of type `string` whose contents are the
    serialized `Summary` protocol buffer.
  """
  summary_ops = []
  for tensor in tensors:
    summary_ops.append(add_scalar_summary(tensor, prefix=prefix,
                                          print_summary=print_summary))
  return summary_ops


def add_zero_fraction_summaries(tensors, prefix=None):
  """Adds a scalar zero-fraction summary for each of the given tensors.

  Args:
    tensors: a list of variable or op tensors.
    prefix: An optional prefix for the summary names.

  Returns:
    A list of scalar `Tensors` of type `string` whose contents are the
    serialized `Summary` protocol buffer.
  """
  summary_ops = []
  for tensor in tensors:
    summary_ops.append(add_zero_fraction_summary(tensor, prefix=prefix))
  return summary_ops
