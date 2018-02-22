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
from tensorflow.python.util.tf_export import tf_export

@tf_export("summary.ipu_text")
def ipu_compile_summary(name, tensor, collections=None):

  summary_metadata = summary_pb2.SummaryMetadata(
    plugin_data=summary_pb2.SummaryMetadata.PluginData(
      plugin_name="text"))
  t_summary = tensor_summary(
    name=name,
    tensor=tensor,
    summary_metadata=summary_metadata,
    collections=collections)
  return t_summary
