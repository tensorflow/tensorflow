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

To use, at program startup, call `tfe.enable_eager_execution()`.

@@device
@@list_devices
@@num_gpus

@@defun
@@implicit_gradients
@@implicit_value_and_gradients
@@gradients_function
@@value_and_gradients_function

@@enable_tracing
@@flush_trace

@@run
@@enable_eager_execution

@@custom_gradient

@@add_execution_callback
@@clear_execution_callbacks
@@inf_callback
@@inf_nan_callback
@@nan_callback
@@seterr

@@Iterator
@@Saver
@@Variable
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint:disable=g-bad-import-order,g-import-not-at-top,unused-import
#
from tensorflow.contrib.eager.python.datasets import Iterator
from tensorflow.contrib.eager.python.saver import Saver
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.eager import backprop
from tensorflow.python.eager.custom_gradient import custom_gradient
from tensorflow.python.eager import function
from tensorflow.python.eager.context import device
from tensorflow.python.eager.context import enable_eager_execution
from tensorflow.python.eager.context import list_devices
from tensorflow.python.eager.context import num_gpus
from tensorflow.python.eager.context import run
from tensorflow.python.eager.core import enable_tracing
from tensorflow.python.eager.execution_callbacks import add_execution_callback
from tensorflow.python.eager.execution_callbacks import clear_execution_callbacks
from tensorflow.python.eager.execution_callbacks import inf_callback
from tensorflow.python.eager.execution_callbacks import inf_nan_callback
from tensorflow.python.eager.execution_callbacks import nan_callback
from tensorflow.python.eager.execution_callbacks import seterr
from tensorflow.python.ops.resource_variable_ops import ResourceVariable as Variable

defun = function.defun
implicit_gradients = backprop.implicit_grad
implicit_value_and_gradients = backprop.implicit_val_and_grad
gradients_function = backprop.gradients_function
value_and_gradients_function = backprop.val_and_grad_function

remove_undocumented(__name__)
