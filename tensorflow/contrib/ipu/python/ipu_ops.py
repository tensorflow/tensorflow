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

"""Ops related to the Graphcore IPU."""

from tensorflow.core.framework import summary_pb2
from tensorflow.python.ops.summary_ops import tensor_summary
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

def ipu_compile_summary(name, collections=None):

  report = gen_ipu_ops.ipu_summary()

  summary_metadata = summary_pb2.SummaryMetadata(
    plugin_data=summary_pb2.SummaryMetadata.PluginData(
      plugin_name="text"))
  t_summary = tensor_summary(
    name=name,
    tensor=report,
    summary_metadata=summary_metadata,
    collections=collections)

  return t_summary
