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
"""TensorFlow Eager execution prototype.

EXPERIMENTAL: APIs here are unstable and likely to change without notice.

To use, at program startup, call `tf.enable_eager_execution()`.

@@metrics

@@list_devices
@@num_gpus

@@py_func
@@defun
@@function
@@make_template
@@implicit_gradients
@@implicit_value_and_gradients
@@gradients_function
@@value_and_gradients_function
@@GradientTape

@@run
@@enable_eager_execution
@@enable_remote_eager_execution

@@custom_gradient

@@add_execution_callback
@@clear_execution_callbacks
@@errstate
@@ExecutionCallback
@@inf_callback
@@inf_nan_callback
@@nan_callback
@@seterr

@@Iterator
@@Saver
@@restore_variables_on_create
@@Variable
@@get_optimizer_variables
@@EagerVariableStore

@@Network
@@Sequential
@@save_network_checkpoint
@@restore_network_checkpoint

@@Checkpoint
@@Checkpointable
@@CheckpointableSaver

@@executing_eagerly
@@in_eager_mode
@@set_execution_mode
@@execution_mode
@@async_wait
@@async_clear_error
@@set_server_def

@@run_test_in_graph_and_eager_modes
@@run_all_tests_in_graph_and_eager_modes

@@TensorSpec

@@connect_to_remote_host

@@DEVICE_PLACEMENT_EXPLICIT
@@DEVICE_PLACEMENT_WARN
@@DEVICE_PLACEMENT_SILENT
@@SYNC
@@ASYNC
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint:disable=g-bad-import-order,g-import-not-at-top,unused-import
#
from tensorflow.contrib.eager.python import metrics
from tensorflow.contrib.eager.python.datasets import Iterator
from tensorflow.contrib.eager.python.network import Network
from tensorflow.contrib.eager.python.network import Sequential
from tensorflow.contrib.eager.python.network import save_network_checkpoint
from tensorflow.contrib.eager.python.network import restore_network_checkpoint
from tensorflow.contrib.eager.python.remote import connect_to_remote_host
from tensorflow.contrib.eager.python.saver import get_optimizer_variables
from tensorflow.contrib.eager.python.saver import restore_variables_on_create
from tensorflow.contrib.eager.python.saver import Saver
from tensorflow.python.eager import backprop
from tensorflow.python.eager import function as _function_lib
from tensorflow.python.eager.context import DEVICE_PLACEMENT_EXPLICIT
from tensorflow.python.eager.context import DEVICE_PLACEMENT_WARN
from tensorflow.python.eager.context import DEVICE_PLACEMENT_SILENT
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.python.eager.context import list_devices
from tensorflow.python.eager.context import set_execution_mode
from tensorflow.python.eager.context import execution_mode
from tensorflow.python.eager.context import async_wait
from tensorflow.python.eager.context import async_clear_error
from tensorflow.python.eager.context import SYNC
from tensorflow.python.eager.context import ASYNC
from tensorflow.python.eager.context import num_gpus
from tensorflow.python.eager.context import set_server_def
from tensorflow.python.eager.def_function import function
from tensorflow.python.eager.execution_callbacks import add_execution_callback
from tensorflow.python.eager.execution_callbacks import clear_execution_callbacks
from tensorflow.python.eager.execution_callbacks import errstate
from tensorflow.python.eager.execution_callbacks import ExecutionCallback
from tensorflow.python.eager.execution_callbacks import inf_callback
from tensorflow.python.eager.execution_callbacks import inf_nan_callback
from tensorflow.python.eager.execution_callbacks import nan_callback
from tensorflow.python.eager.execution_callbacks import seterr
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.framework.ops import enable_eager_execution
from tensorflow.python.framework.ops import enable_eager_execution_internal as enable_remote_eager_execution
from tensorflow.python.framework.ops import eager_run as run
from tensorflow.python.framework.test_util import run_in_graph_and_eager_modes as run_test_in_graph_and_eager_modes
from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes as run_all_tests_in_graph_and_eager_modes
from tensorflow.python.ops.custom_gradient import custom_gradient
from tensorflow.python.ops.resource_variable_ops import ResourceVariable as Variable
from tensorflow.python.ops.variable_scope import EagerVariableStore
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import template
from tensorflow.python.training.checkpointable.tracking import Checkpointable
from tensorflow.python.training.checkpointable.util import CheckpointableSaver
from tensorflow.python.training.checkpointable.util import Checkpoint
from tensorflow.python.util.all_util import remove_undocumented

py_func = script_ops.eager_py_func
defun = _function_lib.defun
make_template = template.make_template_internal
implicit_gradients = backprop.implicit_grad
implicit_value_and_gradients = backprop.implicit_val_and_grad
gradients_function = backprop.gradients_function
value_and_gradients_function = backprop.val_and_grad_function
GradientTape = backprop.GradientTape  # pylint: disable=invalid-name
in_eager_mode = executing_eagerly

remove_undocumented(__name__)
