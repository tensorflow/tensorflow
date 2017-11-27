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
"""Logging tensorflow::tfprof::OpLogProto.

OpLogProto is used to add extra model information for offline analysis.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.profiler.tfprof_logger import write_op_log as _write_op_log
from tensorflow.python.util.deprecation import deprecated


@deprecated("2018-01-01", "Use `tf.profiler.write_op_log. go/tfprof`")
def write_op_log(graph, log_dir, op_log=None, run_meta=None, add_trace=True):
  _write_op_log(graph, log_dir, op_log, run_meta, add_trace)
