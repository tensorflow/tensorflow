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
"""Public Python API of TensorFlow Debugger (tfdbg).

## Functions for adding debug watches

These functions help you modify `RunOptions` to specify which `Tensor`s are to
be watched when the TensorFlow graph is executed at runtime.

@@add_debug_tensor_watch
@@watch_graph
@@watch_graph_with_blacklists


## Classes for debug-dump data and directories

These classes allow you to load and inspect tensor values dumped from
TensorFlow graphs during runtime.

@@DebugTensorDatum
@@DebugDumpDir


## Functions for loading debug-dump data

@@load_tensor_from_event_file


## Tensor-value predicates

Built-in tensor-filter predicates to support conditional breakpoint between
runs. See `DebugDumpDir.find()` for more details.

@@has_inf_or_nan


## Session wrapper class and `SessionRunHook` implementations

These classes allow you to

* wrap aroundTensorFlow `Session` objects to debug  plain TensorFlow models
  (see `LocalCLIDebugWrapperSession`), or
* generate `SessionRunHook` objects to debug `tf.contrib.learn` models (see
  `LocalCLIDebugHook`).

@@LocalCLIDebugHook
@@LocalCLIDebugWrapperSession

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-imports
from tensorflow.python.debug.debug_data import DebugDumpDir
from tensorflow.python.debug.debug_data import DebugTensorDatum
from tensorflow.python.debug.debug_data import has_inf_or_nan
from tensorflow.python.debug.debug_data import load_tensor_from_event_file

from tensorflow.python.debug.debug_utils import add_debug_tensor_watch
from tensorflow.python.debug.debug_utils import watch_graph
from tensorflow.python.debug.debug_utils import watch_graph_with_blacklists

from tensorflow.python.debug.wrappers.hooks import LocalCLIDebugHook
from tensorflow.python.debug.wrappers.local_cli_wrapper import LocalCLIDebugWrapperSession
