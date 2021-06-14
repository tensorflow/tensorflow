/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/metric_util.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

using NodeType = string;
using NamePatterns = std::vector<string>;

static auto* kMetricTriggers = new absl::flat_hash_map<NodeType, NamePatterns>(
    {{kTpuExecuteStagingOp, {kTpuExecuteStagingNodeName}}});

bool NameSubstrMatch(const NamePatterns& list, const string& name) {
  return std::find_if(list.begin(), list.end(), [&](const string& s) {
           return absl::StrContains(name, s);
         }) != list.end();
}

}  // namespace

bool ShouldLogLatencyMetrics(const NodeDef& ndef) {
  const auto& it = kMetricTriggers->find(ndef.op());
  return (it != kMetricTriggers->end()) &&
         NameSubstrMatch(it->second, ndef.name());
}

void LogLatencyMetrics(const NodeDef& ndef, const int64 cur_time_usecs,
                       const int64 start_time_usecs) {
  // Execution of TPUExecute staging op signals that all args have been
  // transferred and system is ready for TPUExecute invocation.
  if (ndef.op() == kTpuExecuteStagingOp &&
      absl::StrContains(ndef.name(), kTpuExecuteStagingNodeName)) {
    metrics::UpdateTpuVariableDistributionTime(cur_time_usecs -
                                               start_time_usecs);
  }
}

}  // namespace tensorflow
