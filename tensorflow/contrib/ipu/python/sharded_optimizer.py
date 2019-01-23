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

"""An optimizer wrapper for replicating sharding information from the forward
   pass to the backward pass."""

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.training import optimizer
from tensorflow.python.framework import ops

_XLA_SHARDING = '_XlaSharding'
def has_attr(o, attr_name):
  for i in o.node_def.attr.items():
    if i[0] == attr_name:
      return True
  return False

def propagate_sharding(g):
  changed = True;
  while changed == True:
    changed = False

    op_list = g.get_operations()
    op_list = filter(lambda o : has_attr(o, '_class'), op_list)
    op_list = filter(lambda o : not has_attr(o, '_XlaSharding'), op_list)
    for o in op_list:
      for c in o.colocation_groups():
        coloc_op = g.get_operation_by_name(c.decode('utf-8')[5:])
        if has_attr(coloc_op, _XLA_SHARDING):
          attr = coloc_op.get_attr(_XLA_SHARDING);
          o._set_attr(_XLA_SHARDING, attr_value_pb2.AttrValue(s=attr))
          changed = True
          break

class ShardedOptimizer(optimizer.Optimizer):

  def __init__(self,
               optimizer):
    """Construct a new sharded optimizer.

    Args:
      optimizer: The optimizer to wrap.
    """

    super(ShardedOptimizer, self).__init__(False, name="ShardedOptimizer")
    self._optimizer = optimizer


  def compute_gradients(self, loss, var_list=None, **kwargs):
    kwargs['colocate_gradients_with_ops'] = True
    ret = self._optimizer.compute_gradients(loss, var_list=var_list,
                                            **kwargs)

    propagate_sharding(ops.get_default_graph())
    return ret

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    ret = self._optimizer.apply_gradients(grads_and_vars, global_step, name)
    propagate_sharding(ops.get_default_graph())
    return ret

  def get_slot_names(self, *args, **kwargs):
    return self._optimizer.get_slot_names(*args, **kwargs)

  def variables(self):
    return self._optimizer.variables()
