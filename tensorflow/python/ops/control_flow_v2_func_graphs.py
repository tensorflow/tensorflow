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

from tensorflow.python.framework.func_graph import FuncGraph


class CondBranchFuncGraph(FuncGraph):
  """FuncGraph for branches of tf.cond().

  This is used to distinguish cond branches from other functions.
  """
  pass


class WhileCondFuncGraph(FuncGraph):
  """FuncGraph for the condition of tf.while_loop().

  This is used to distinguish while conditions from other functions.
  """
  pass


class WhileBodyFuncGraph(FuncGraph):
  """FuncGraph for the body of tf.while_loop().

  This is used to distinguish while bodies from other functions.
  """
  pass
