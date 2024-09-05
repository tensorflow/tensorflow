/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/common/metrics.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "tsl/lib/monitoring/sampler.h"

namespace tensorflow {
namespace tfrt_metrics {

tsl::monitoring::SamplerCell* GetTfrtGraphExecutorLatencySampler(
    const std::string& model_name, int64_t model_version,
    const std::string& graph_name) {
  static auto* cell = tsl::monitoring::Sampler<3>::New(
      {"/tfrt/graph_executor/latency",
       "Tracks the latency of GraphExecutor (in microseconds) of a graph.",
       "model_name", "model_version", "graph_name"},
      tsl::monitoring::Buckets::Exponential(10, 1.5, 33));
  return cell->GetCell(model_name, absl::StrCat(model_version), graph_name);
}

tsl::monitoring::SamplerCell* GetTfrtDeviceExecutionLatency(
    const std::string& model_name, int64_t model_version) {
  static auto* cell = tsl::monitoring::Sampler<2>::New(
      {"/tfrt/device_execution/latency",
       "Tracks the latency of device execution (in microseconds).",
       "model_name", "model_version"},
      tsl::monitoring::Buckets::Exponential(10, 1.5, 33));
  return cell->GetCell(model_name, absl::StrCat(model_version));
}

}  // namespace tfrt_metrics
}  // namespace tensorflow
