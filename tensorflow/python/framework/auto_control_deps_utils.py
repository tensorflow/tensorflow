# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for AutomaticControlDependencies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes

READ_ONLY_RESOURCE_INPUTS_ATTR = "_read_only_resource_inputs"
RESOURCE_READ_OPS = set()


def register_read_only_resource_op(op_type):
  """Declares that `op_type` does not update its touched resource."""
  RESOURCE_READ_OPS.add(op_type)


def resource_has_writes(handle):
  """Returns whether any of the consumers of handle write to it.

  Args:
    handle: Tensor of type DT_RESOURCE.

  Returns:
    Returns True if at least one consumer of `handle` writes to it.
    Returns False if all consumers of `handle` do not write to it or if the
    `handle` has no consumers.
  """
  assert handle.dtype == dtypes.resource
  return any(op_writes_to_resource(handle, op) for op in handle.consumers())


def op_writes_to_resource(handle, op):
  """Returns whether op writes to resource handle.

  Args:
    handle: Resource handle. Must be an input of `op`.
    op: Operation.

  Returns:
    Returns False if op is a read-only op registered using
    `register_read_only_resource_op` or if `handle` is an input at one of
    the indices in the `READ_ONLY_RESOURCE_INPUTS_ATTR` attr of the op, True
    otherwise.

  Raises:
    ValueError: if `handle` is not an input of `op`.
  """
  if op.type in RESOURCE_READ_OPS:
    return False
  input_index = _input_index(op, handle)
  try:
    read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
  except ValueError:
    # Attr was not set. Conservatively assume that the resource is written to.
    return True
  return input_index not in read_only_input_indices


def _input_index(op, handle):
  """Returns the index of `handle` in `op.inputs`.

  Args:
    op: Operation.
    handle: Resource handle.

  Returns:
    Index in `op.inputs` receiving the resource `handle`.

  Raises:
    ValueError: If handle and its replicated input are both not found in
    `op.inputs`.
  """
  for i, t in enumerate(op.inputs):
    if handle is t:
      return i
  raise ValueError("%s not in list" % str(handle))
