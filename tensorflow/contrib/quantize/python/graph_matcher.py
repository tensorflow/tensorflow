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
"""Utilities that match patterns in a tf.Graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Pattern(object):
  """The parent class of all patterns (e.g. OpTypePattern and OneofPattern)."""

  @abc.abstractmethod
  def match(self, op, tensor):
    """Returns the result of matching op/tensor against this pattern."""
    raise NotImplementedError('Method "match" not implemented.')


class OpTypePattern(Pattern):
  """A tree pattern that matches TF expressions with certain op types."""

  def __init__(self, op_type, name=None, inputs=None):
    """Initializes an OpTypePattern.

    Args:
      op_type: string that specifies the allowed types of the root. It can be
        (1) an op type, e.g. 'Conv2D',
        (2) '*', i.e. wildcard, or
        (3) multiple op types separated by '|', e.g., 'Relu|Relu6'.
        We could use regex strings, which might be worthwhile when we have many
        similar TF op types.
      name: Optional string. The name of the pattern that can be looked up in
        MatchResult.
      inputs: Optional list of `Pattern`s or strings that specify the
        patterns for the inputs of a matching op. If None, this pattern accepts
        any inputs of a matching op.
    """
    self._op_type = op_type
    self._name = name
    if inputs is None:
      inputs = []
    self._inputs = [
        input_pattern
        if isinstance(input_pattern, Pattern) else OpTypePattern(input_pattern)
        for input_pattern in inputs
    ]

  @property
  def name(self):
    return self._name

  def match(self, op, tensor):
    if self._op_type != '*':
      if op.type not in self._op_type.split('|'):
        return None

    match_result = MatchResult()
    match_result.add(self, op, tensor)

    if not self._inputs:
      # If pattern.inputs is empty, skips the rest and accepts all the inputs.
      return match_result

    if len(op.inputs) != len(self._inputs):
      return None

    for input_tensor, input_pattern in zip(op.inputs, self._inputs):
      input_match_result = input_pattern.match(input_tensor.op, input_tensor)
      if input_match_result is None:
        return None
      match_result.merge_from(input_match_result)
    return match_result


class OneofPattern(Pattern):
  """Matches one of the given sub-patterns."""

  def __init__(self, sub_patterns):
    self._sub_patterns = sub_patterns

  def match(self, op, tensor):
    for sub_pattern in self._sub_patterns:
      match_result = sub_pattern.match(op, tensor)
      if match_result is not None:
        return match_result
    return None


class MatchResult(object):
  r"""Encapsulates the result of a match done by GraphMatcher.

  MatchResult contains a map from OpTypePattern to the matching op and tensor.
  When the matching op has multiple output tensors, the matching tensor is the
  output tensor used by the matching op of the parent pattern. E.g., when we
  match graph

      -         +
     / \y0   y1/ \
    x    split    z
          |
          y         (nodes are ops; edges are going up)

  against add_pattern defined as

    y1_pattern = OpTypePattern('*')
    z_pattern = OpTypePattern('*')
    add_pattern = OpTypePattern('+', inputs=[y1_pattern, z_pattern])

  the matching op of `y1_pattern` is `split`, and the matching tensor of
  `y1_pattern`
  is `y1` not `y0`.
  """

  def __init__(self):
    self._pattern_to_op_tensor = {}
    self._name_to_pattern = {}

  def add(self, pattern, op, tensor):
    self._pattern_to_op_tensor[pattern] = op, tensor
    if pattern.name is not None:
      if pattern.name in self._name_to_pattern:
        raise ValueError(
            'Name %s is already bound to another pattern' % pattern.name)
      self._name_to_pattern[pattern.name] = pattern

  def _to_pattern(self, pattern_or_name):
    if isinstance(pattern_or_name, OpTypePattern):
      return pattern_or_name

    if isinstance(pattern_or_name, str):
      if pattern_or_name not in self._name_to_pattern:
        return None
      return self._name_to_pattern[pattern_or_name]

    raise ValueError('pattern_or_name has type %s. Expect OpTypePattern or str.'
                     % type(pattern_or_name))

  def _get_op_tensor(self, pattern_or_name):
    pattern = self._to_pattern(pattern_or_name)
    if pattern is None:
      return None

    if pattern not in self._pattern_to_op_tensor:
      return None

    return self._pattern_to_op_tensor[pattern]

  def get_op(self, pattern_or_name):
    op_tensor = self._get_op_tensor(pattern_or_name)
    return op_tensor[0] if op_tensor else None

  def get_tensor(self, pattern_or_name):
    op_tensor = self._get_op_tensor(pattern_or_name)
    return op_tensor[1] if op_tensor else None

  def merge_from(self, other_match_result):
    # pylint: disable=protected-access
    self._pattern_to_op_tensor.update(other_match_result._pattern_to_op_tensor)
    self._name_to_pattern.update(other_match_result._name_to_pattern)
    # pylint: enable=protected-access


class GraphMatcher(object):
  """Checks if a particular subgraph matches a given pattern."""

  def __init__(self, pattern):
    """Initializes a GraphMatcher.

    Args:
      pattern: The `Pattern` against which `GraphMatcher` matches
        subgraphs.
    """
    self._pattern = pattern

  def _match_pattern(self, pattern, op, tensor):
    """Returns whether an TF expression rooted at `op` matches `pattern`.

    If there is a match, adds to `self._match_result` the matching op and tensor
    with key `pattern`.

    Args:
      pattern: An `Pattern`.
      op: A `tf.Operation` to match against the pattern.
      tensor: the output `tf.Tensor` of `op` that is used by the matching op of
        `pattern`'s parent. Can be None if `pattern` is already the root of the
        pattern tree.

    Returns:
      True if an TF expression rooted at `op` matches `pattern`.
    """
    match_result = pattern.match(op, tensor)
    if match_result is None:
      return False
    self._match_result.merge_from(match_result)
    return True

  def match_op(self, op):
    """Matches `op` against `self._pattern`.

    Args:
      op: `tf.Operation` to match against the pattern.

    Returns:
      Returns a `MatchResult` if `op` matches the pattern; otherwise, returns
      None.
    """
    self._match_result = MatchResult()
    if not self._match_pattern(self._pattern, op, tensor=None):
      return None
    return self._match_result

  def match_ops(self, ops):
    """Matches each operation in `ops` against `self._pattern`.

    Args:
      ops: collection of `tf.Operation` to match against the pattern.

    Yields:
      `MatchResult` for each `tf.Operation` that matches the pattern.
    """
    for op in ops:
      match_result = self.match_op(op)
      if match_result:
        yield match_result

  def match_graph(self, graph):
    """Matches each operation in `graph` against `self._pattern`.

    Args:
      graph: `tf.Graph` containing operations to match.

    Yields:
      `MatchResult` for each `tf.Operation` in `graph` that matches the pattern.
    """
    # Python 3.3.2+ implements `yield from`, but for now:
    for match_result in self.match_ops(graph.get_operations()):
      yield match_result
