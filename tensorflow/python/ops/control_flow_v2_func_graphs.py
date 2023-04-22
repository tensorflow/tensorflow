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
# ==============================================================================
"""FuncGraphs for V2 control flow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops


class ControlFlowFuncGraph(func_graph.FuncGraph):
  """Contains control flow-specific FuncGraph logic."""

  def __init__(self, *args, **kwargs):
    super(ControlFlowFuncGraph, self).__init__(*args, **kwargs)
    outer_graph = self.outer_graph
    # Unlike tf.function, control flow FuncGraphs are generally created one per
    # op. This means hard-coding any outer device scopes in the body (rather
    # than inspecting the call-time placement of the control flow op) makes
    # sense.
    self._device_function_stack = outer_graph._device_function_stack.copy()  # pylint: disable=protected-access
    self.is_control_flow_graph = True
    if ops.executing_eagerly_outside_functions():
      func_graph.override_func_graph_name_scope(
          self, self.outer_graph.get_name_scope())


class CondBranchFuncGraph(ControlFlowFuncGraph):
  """FuncGraph for branches of tf.cond().

  This is used to distinguish cond branches from other functions.
  """


class WhileCondFuncGraph(ControlFlowFuncGraph):
  """FuncGraph for the condition of tf.while_loop().

  This is used to distinguish while conditions from other functions.
  """


class WhileBodyFuncGraph(ControlFlowFuncGraph):
  """FuncGraph for the body of tf.while_loop().

  This is used to distinguish while bodies from other functions.
  """
