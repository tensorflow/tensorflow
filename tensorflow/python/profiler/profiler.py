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
"""profiler python module provides APIs to profile TensorFlow models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.core.profiler.tfprof_log_pb2 import OpLogProto
from tensorflow.core.profiler.tfprof_output_pb2 import AdviceProto
from tensorflow.core.profiler.tfprof_output_pb2 import GraphNodeProto
from tensorflow.core.profiler.tfprof_output_pb2 import MultiGraphNodeProto

from tensorflow.python.profiler.model_analyzer import advise
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.model_analyzer import Profiler
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.python.profiler.tfprof_logger import write_op_log

from tensorflow.python.util.tf_export import tf_export


_allowed_symbols = [
    'Profiler',
    'profile',
    'ProfileOptionBuilder',
    'advise',
    'write_op_log',
]

_allowed_symbols.extend([
    'GraphNodeProto',
    'MultiGraphNodeProto',
    'AdviceProto',
    'OpLogProto',
])

# Export protos
tf_export('profiler.GraphNodeProto')(GraphNodeProto)
tf_export('profiler.MultiGraphNodeProto')(MultiGraphNodeProto)
tf_export('profiler.AdviceProto')(AdviceProto)
tf_export('profiler.OpLogProto')(OpLogProto)
