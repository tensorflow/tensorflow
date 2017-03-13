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

See the @{$python/tfdbg} guide.

@@add_debug_tensor_watch
@@watch_graph
@@watch_graph_with_blacklists
@@DebugTensorDatum
@@DebugDumpDir
@@load_tensor_from_event_file
@@has_inf_or_nan
@@DumpingDebugHook
@@DumpingDebugWrapperSession
@@LocalCLIDebugHook
@@LocalCLIDebugWrapperSession
@@WatchOptions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-imports
from tensorflow.python.debug.lib.debug_data import DebugDumpDir
from tensorflow.python.debug.lib.debug_data import DebugTensorDatum
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
from tensorflow.python.debug.lib.debug_data import load_tensor_from_event_file

from tensorflow.python.debug.lib.debug_utils import add_debug_tensor_watch
from tensorflow.python.debug.lib.debug_utils import watch_graph
from tensorflow.python.debug.lib.debug_utils import watch_graph_with_blacklists

from tensorflow.python.debug.wrappers.dumping_wrapper import DumpingDebugWrapperSession
from tensorflow.python.debug.wrappers.framework import WatchOptions
from tensorflow.python.debug.wrappers.hooks import DumpingDebugHook
from tensorflow.python.debug.wrappers.hooks import LocalCLIDebugHook
from tensorflow.python.debug.wrappers.local_cli_wrapper import LocalCLIDebugWrapperSession

from tensorflow.python.util import all_util as _all_util


_all_util.remove_undocumented(__name__)
