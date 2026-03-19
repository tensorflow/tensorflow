# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Registers protos with tf_export that should be public."""

from xla.tsl.protobuf import histogram_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.util import tf_export

AttrValue = tf_export.tf_export(v1=['AttrValue'])(attr_value_pb2.AttrValue)
ConfigProto = tf_export.tf_export(v1=['ConfigProto'])(config_pb2.ConfigProto)
Event = tf_export.tf_export(v1=['Event', 'summary.Event'])(event_pb2.Event)
GPUOptions = tf_export.tf_export(v1=['GPUOptions'])(config_pb2.GPUOptions)
GraphOptions = tf_export.tf_export(v1=['GraphOptions'])(config_pb2.GraphOptions)
HistogramProto = tf_export.tf_export(v1=['HistogramProto'])(
    histogram_pb2.HistogramProto
)
LogMessage = tf_export.tf_export(v1=['LogMessage'])(event_pb2.LogMessage)
MetaGraphDef = tf_export.tf_export(v1=['MetaGraphDef'])(
    meta_graph_pb2.MetaGraphDef
)
NameAttrList = tf_export.tf_export(v1=['NameAttrList'])(
    attr_value_pb2.NameAttrList
)
NodeDef = tf_export.tf_export(v1=['NodeDef'])(node_def_pb2.NodeDef)
OptimizerOptions = tf_export.tf_export(v1=['OptimizerOptions'])(
    config_pb2.OptimizerOptions
)
RunMetadata = tf_export.tf_export(v1=['RunMetadata'])(config_pb2.RunMetadata)
RunOptions = tf_export.tf_export(v1=['RunOptions'])(config_pb2.RunOptions)
SessionLog = tf_export.tf_export(v1=['SessionLog', 'summary.SessionLog'])(
    event_pb2.SessionLog
)
Summary = tf_export.tf_export(v1=['Summary', 'summary.Summary'])(
    summary_pb2.Summary
)
SummaryDescription = tf_export.tf_export(v1=['summary.SummaryDescription'])(
    summary_pb2.SummaryDescription
)
SummaryMetadata = tf_export.tf_export(v1=['SummaryMetadata'])(
    summary_pb2.SummaryMetadata
)
TaggedRunMetadata = tf_export.tf_export(v1=['summary.TaggedRunMetadata'])(
    event_pb2.TaggedRunMetadata
)
TensorInfo = tf_export.tf_export(v1=['TensorInfo'])(meta_graph_pb2.TensorInfo)
