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
# =============================================================================

"""Classes for TPU trace events."""

# pylint: disable=wildcard-import,unused-import
from tensorflow.core.profiler.protobuf.trace_events_pb2 import *
from tensorflow.core.profiler.profiler_analysis_pb2 import *
# pylint: enable=wildcard-import,unused-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = ['Trace', 'Resource', 'Device', 'TraceEvent']

remove_undocumented(__name__, _allowed_symbols)
