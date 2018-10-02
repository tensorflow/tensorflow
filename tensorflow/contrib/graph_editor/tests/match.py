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
"""Simple graph matching functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six import string_types

from tensorflow.contrib.graph_editor import select
from tensorflow.python.framework import ops as tf_ops

__all__ = [
    "op_type",
    "OpMatcher",
]


def _make_graph_match(graph_match):
  """Convert to a OpMatcher instance."""
  if graph_match is None:
    return None
  if not isinstance(graph_match, OpMatcher):
    graph_match = OpMatcher(graph_match)
  return graph_match


def op_type(op_types, op=None):
  """Check if an op is of the given type.

  Args:
    op_types: tuple of strings containing the types to check against.
      For instance: ("Add", "Const")
    op: the operation to check (or None).
  Returns:
    if op is not None, return True if the op is of the correct type.
    if op is None, return a lambda function which does the type checking.
  """
  if isinstance(op_types, string_types):
    op_types = (op_types)
  if op is None:
    return lambda op: op.node_def.op in op_types
  else:
    return op.node_def.op in op_types


class OpMatcher(object):
  """Graph match class."""

  def __init__(self, positive_filter):
    """Graph match constructor."""
    self.positive_filters = []
    self.input_op_matches = None
    self.control_input_op_matches = None
    self.output_op_matches = None
    positive_filter = self._finalize_positive_filter(positive_filter)
    self.positive_filters.append(positive_filter)

  def _finalize_positive_filter(self, elem):
    """Convert to a filter function."""
    if select.can_be_regex(elem):
      regex_ = select.make_regex(elem)
      return lambda op, regex=regex_: regex.search(op.name) is not None
    elif isinstance(elem, tf_ops.Operation):
      return lambda op, match_op=elem: op is match_op
    elif callable(elem):
      return elem
    elif elem is True:
      return lambda op: True
    else:
      raise ValueError("Cannot finalize the positive filter: {}".format(elem))

  def __call__(self, op):
    """Evaluate if the op matches or not."""
    if not isinstance(op, tf_ops.Operation):
      raise TypeError("Expect tf.Operation, got: {}".format(type(op)))
    for positive_filter in self.positive_filters:
      if not positive_filter(op):
        return False
    if self.input_op_matches is not None:
      if len(op.inputs) != len(self.input_op_matches):
        return False
      for input_t, input_op_match in zip(op.inputs, self.input_op_matches):
        if input_op_match is None:
          continue
        if not input_op_match(input_t.op):
          return False
    if self.control_input_op_matches is not None:
      if len(op.control_inputs) != len(self.control_input_op_matches):
        return False
      for cinput_op, cinput_op_match in zip(op.control_inputs,
                                            self.control_input_op_matches):
        if cinput_op_match is None:
          continue
        if not cinput_op_match(cinput_op):
          return False
    if self.output_op_matches is not None:
      if len(op.outputs) != len(self.output_op_matches):
        return False
      for output_t, output_op_matches in zip(op.outputs,
                                             self.output_op_matches):
        if output_op_matches is None:
          continue
        if len(output_t.consumers()) != len(output_op_matches):
          return False
        for consumer_op, consumer_op_match in zip(output_t.consumers(),
                                                  output_op_matches):
          if consumer_op_match is None:
            continue
          if not consumer_op_match(consumer_op):
            return False
    return True

  def input_ops(self, *args):
    """Add input matches."""
    if self.input_op_matches is not None:
      raise ValueError("input_op_matches is already set.")
    self.input_op_matches = []
    for input_match in args:
      self.input_op_matches.append(_make_graph_match(input_match))
    return self

  def control_input_ops(self, *args):
    """Add input matches."""
    if self.control_input_op_matches is not None:
      raise ValueError("control_input_op_matches is already set.")
    self.control_input_op_matches = []
    for input_match in args:
      self.control_input_op_matches.append(_make_graph_match(input_match))
    return self

  def output_ops(self, *args):
    """Add output matches."""
    if self.output_op_matches is not None:
      raise ValueError("output_op_matches is already set.")
    self.output_op_matches = []
    for consumer_op_matches in args:
      if consumer_op_matches is None:
        self.output_op_matches.append(None)
      if not isinstance(consumer_op_matches, list):
        consumer_op_matches = [consumer_op_matches]
      consumer_op_matches = [_make_graph_match(consumer_op_match)
                             for consumer_op_match in consumer_op_matches]
      self.output_op_matches.append(consumer_op_matches)
    return self
