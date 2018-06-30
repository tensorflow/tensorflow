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
"""Miscellaneous utilities that don't fit anywhere else."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def alias_tensors(*args):
  """Wrap any Tensor arguments with an identity op.

  Any other argument, including Variables, is returned unchanged.

  Args:
    *args: Any arguments. Must contain at least one element.

  Returns:
    Same as *args, with Tensor instances replaced as described.

  Raises:
    ValueError: If args doesn't meet the requirements.
  """

  def alias_if_tensor(a):
    return array_ops.identity(a) if isinstance(a, ops.Tensor) else a

  # TODO(mdan): Recurse into containers?
  # TODO(mdan): Anything we can do about variables? Fake a scope reuse?
  if len(args) > 1:
    return (alias_if_tensor(a) for a in args)
  elif len(args) == 1:
    return alias_if_tensor(args[0])

  raise ValueError('at least one argument required')
