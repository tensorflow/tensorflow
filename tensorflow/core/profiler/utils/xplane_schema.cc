/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/xplane_schema.h"

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {

const absl::string_view kHostThreads = "Host Threads";
const absl::string_view kGpuPlanePrefix = "GPU:";

constexpr int kNumHostEventTypes =
    HostEventType::kLastHostEventType - HostEventType::kFirstHostEventType + 1;

constexpr int kNumStatTypes =
    StatType::kLastStatType - StatType::kFirstStatType + 1;

static const absl::string_view kHostEventTypeMetadataMap[] = {
    "UnknownHostEventType",
    "TraceContext",
    "SessionRun",
    "FunctionRun",
    "RunGraph",
    "EagerKernelExecute",
    "ExecutorState::Process",
    "ExecutorDoneCallback",
    // tf data captured function events.
    "InstantiatedCapturedFunction::Run",
    "InstantiatedCapturedFunction::RunWithBorrowedArgs",
    "InstantiatedCapturedFunction::RunInstantiated",
    "InstantiatedCapturedFunction::RunAsync",
    // Functional ops.
    "CallOp",
    "ParallelForOp",
    "ForeverOp",
    "NumericalGradientOp-EvalRight",
    "NumericalGradientOp-EvalLeft",
    "SymbolicGradientOp",
    "RemoteCallOp",
    "IfOp",
    "CaseOp",
    "WhileOp-EvalCond",
    "WhileOp-StartBody",
    "ForOp",
    "PartitionedCallOp",
};

static_assert(sizeof(kHostEventTypeMetadataMap) / sizeof(absl::string_view) ==
                  kNumHostEventTypes,
              "Mismatch between enum and string map.");

static const absl::string_view kStatTypeStrMap[] = {
    "UnknownStatType", "id",
    "parent_step_id",  "function_step_id",
    "device_ordinal",  "chip_ordinal",
    "node_ordinal",    "model_id",
    "queue_addr",      "request_id",
    "run_id",          "graph_type",
    "step_num",        "iter_num",
    "index_on_host",   "bytes_reserved",
    "bytes_allocated", "bytes_available",
    "fragmentation",   "device_id",
    "context_id",      "correlation_id",
    "memcpy_details",  "memalloc_details",
    "kernel_details",  "group_id",
    "step_name",       "level 0",
    "tf_op",           "hlo_op",
    "hlo_module",
};

static_assert(sizeof(kStatTypeStrMap) / sizeof(absl::string_view) ==
                  kNumStatTypes,
              "Mismatch between enum and string map.");

absl::Span<const absl::string_view> GetHostEventTypeStrMap() {
  return absl::MakeConstSpan(kHostEventTypeMetadataMap, kNumHostEventTypes);
}

absl::Span<const absl::string_view> GetStatTypeStrMap() {
  return absl::MakeConstSpan(kStatTypeStrMap, kNumStatTypes);
}

}  // namespace profiler
}  // namespace tensorflow
