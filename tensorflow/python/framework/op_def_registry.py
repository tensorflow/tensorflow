# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Global registry for OpDefs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import op_def_pb2


_registered_ops = {}


def register_op_list(op_list):
  """Register all the ops in an op_def_pb2.OpList."""
  if not isinstance(op_list, op_def_pb2.OpList):
    raise TypeError("%s is %s, not an op_def_pb2.OpList" %
                    (op_list, type(op_list)))
  for op_def in op_list.op:
    if op_def.name in _registered_ops:
      assert _registered_ops[op_def.name] == op_def
    else:
      _registered_ops[op_def.name] = op_def


def get_registered_ops():
  """Returns a dictionary mapping names to OpDefs."""
  return _registered_ops
