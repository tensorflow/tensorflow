# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops

_XLA_SHARDING = '_XlaSharding'


def has_attr(o, attr_name):
  for i in o.node_def.attr.items():
    if i[0] == attr_name:
      return True
  return False


def dependencies(roots):
  working = roots
  op_set = set()
  while len(working) > 0:
    o = working[0]
    if type(o) is ops.Tensor:
      o = o.op
    working = working[1:]
    op_set.add(o)
    working += [t.op for t in o.inputs if t.op not in op_set]
    working += [c for c in o.control_inputs if c not in op_set]
  return op_set


def get_shard_from_colocation(op):
  g = op.graph
  for c in op.colocation_groups():
    try:
      coloc_op = g.get_operation_by_name(c.decode('utf-8')[5:])
      if has_attr(coloc_op, _XLA_SHARDING):
        attr = coloc_op.get_attr(_XLA_SHARDING)
        return attr
    except KeyError:
      continue
  for c in op.inputs:
    coloc_op = c.op
    if has_attr(coloc_op, _XLA_SHARDING):
      attr = coloc_op.get_attr(_XLA_SHARDING)
      return attr
  return None


def propagate_sharding(g):
  changed = True
  while changed:
    changed = False

    op_list = g.get_operations()
    op_list = filter(lambda o: has_attr(o, '_class'), op_list)
    op_list = filter(lambda o: not has_attr(o, '_XlaSharding'), op_list)
    for o in op_list:
      attr = get_shard_from_colocation(o)
      if attr is not None:
        o._set_attr(_XLA_SHARDING, attr_value_pb2.AttrValue(s=attr))
        changed = True
        break
