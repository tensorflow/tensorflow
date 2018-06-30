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
"""SavedModel main op implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.tf_export import tf_export


@tf_export('saved_model.main_op.main_op')
def main_op():
  """Returns a main op to init variables and tables.

  Returns the main op including the group of ops that initializes all
  variables, initializes local variables and initialize all tables.

  Returns:
    The set of ops to be run as part of the main op upon the load operation.
  """
  init = variables.global_variables_initializer()
  init_local = variables.local_variables_initializer()
  init_tables = lookup_ops.tables_initializer()
  return control_flow_ops.group(init, init_local, init_tables)


# TODO(sukritiramesh): Integrate with Saver for complete restore functionality.
@tf_export('saved_model.main_op.main_op_with_restore')
def main_op_with_restore(restore_op_name):
  """Returns a main op to init variables, tables and restore the graph.

  Returns the main op including the group of ops that initializes all
  variables, initialize local variables, initialize all tables and the restore
  op name.

  Args:
    restore_op_name: Name of the op to use to restore the graph.

  Returns:
    The set of ops to be run as part of the main op upon the load operation.
  """
  with ops.control_dependencies([main_op()]):
    main_op_with_restore = control_flow_ops.group(restore_op_name)
  return main_op_with_restore
