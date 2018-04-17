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
"""Operators specific to data structures: list append, subscripts, etc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import tensor_array_ops

# TODO(mdan): Add support for TensorList once functional.
# TODO(mdan): Add primitives for empty list, list with elements.


def append(target, element):
  """The list append function.

  Note: it is unspecified where target will be mutated or not. If target is
  a TensorFlow entity, it will not be typically mutated. If target is a plain
  list, it will be. In general, if the target is mutated then the return value
  should point to the original entity.

  Args:
    target: An entity that supports append semantics.
    element: The element to append.

  Returns:
    Same as target, after the append was performed.
  """
  if isinstance(target, tensor_array_ops.TensorArray):
    return _tf_tensorarray_append(target, element)
  else:
    return _py_append(target, element)


def _tf_tensorarray_append(target, element):
  """Overload of append that stages a TensorArray write at the last position."""
  return target.write(target.size(), element)


def _py_append(target, element):
  """Overload of append that executes a Python list append."""
  target.append(element)
  return target
