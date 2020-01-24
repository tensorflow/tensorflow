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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/gtl/map_util.h"

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
    "MemoryAllocation",
    "MemoryDeallocation",
    // Performance counter related.
    "RemotePerfCounter",
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
    // TraceMe arguments.
    "UnknownStatType",
    "id",
    "parent_step_id",
    "function_step_id",
    "device_ordinal",
    "chip_ordinal",
    "node_ordinal",
    "model_id",
    "queue_addr",
    "request_id",
    "run_id",
    "graph_type",
    "step_num",
    "iter_num",
    "index_on_host",
    "allocator_name",
    "bytes_reserved",
    "bytes_allocated",
    "bytes_available",
    "fragmentation",
    "peak_bytes_in_use",
    // Device trace arguments.
    "device_id",
    "context_id",
    "correlation_id",
    "memcpy_details",
    "memalloc_details",
    "kernel_details",
    "stream",
    // Stats added when processing traces.
    "group_id",
    "step_name",
    "level 0",
    "tf_op",
    "hlo_op",
    "hlo_module",
    // Performance counter related.
    "Raw Value",
    "Scaled Value",
    "Thread Id",
    // XLA metadata map related.
    "SELF_DURATION_PS",
    "MIN_DURATION_PS",
    // Device capability related.
    "clock_rate",
    "core_count",
    "memory_bandwidth",
    "memory_size",
    "compute_cap_major",
    "compute_cap_minor",
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

const absl::flat_hash_map<absl::string_view, HostEventType>&
GetHostEventTypeMap() {
  static auto* host_event_type_map =
      new absl::flat_hash_map<absl::string_view, HostEventType>({
          {"UnknownHostEventType", kUnknownHostEventType},
          {"TraceContext", kTraceContext},
          {"SessionRun", kSessionRun},
          {"FunctionRun", kFunctionRun},
          {"RunGraph", kRunGraph},
          {"EagerKernelExecute", kEagerKernelExecute},
          {"ExecutorState::Process", kExecutorStateProcess},
          {"ExecutorDoneCallback", kExecutorDoneCallback},
          {"MemoryAllocation", kMemoryAllocation},
          {"MemoryDeallocation", kMemoryDeallocation},
          // Performance counter related.
          {"RemotePerfCounter", kRemotePerf},
          // tf data captured function events.
          {"InstantiatedCapturedFunction::Run", kTfDataCapturedFunctionRun},
          {"InstantiatedCapturedFunction::RunWithBorrowedArgs",
           kTfDataCapturedFunctionRunWithBorrowedArgs},
          {"InstantiatedCapturedFunction::RunInstantiated",
           kTfDataCapturedFunctionRunInstantiated},
          {"InstantiatedCapturedFunction::RunAsync",
           kTfDataCapturedFunctionRunAsync},
          // Functional ops.
          {"CallOp", kCallOp},
          {"ParallelForOp", kParallelForOp},
          {"ForeverOp", kForeverOp},
          {"NumericalGradientOp-EvalRight", kNumericalGradientOpEvalRight},
          {"NumericalGradientOp-EvalLeft", kNumericalGradientOpEvalLeft},
          {"SymbolicGradientOp", kSymbolicGradientOp},
          {"RemoteCallOp", kRemoteCallOp},
          {"IfOp", kIfOp},
          {"CaseOp", kCaseOp},
          {"WhileOp-EvalCond", kWhileOpEvalCond},
          {"WhileOp-StartBody", kWhileOpStartBody},
          {"ForOp", kForOp},
          {"PartitionedCallOp", kPartitionedCallOp},
      });
  return *host_event_type_map;
}

const absl::flat_hash_map<absl::string_view, StatType>& GetStatTypeMap() {
  static auto* stats_type_map =
      new absl::flat_hash_map<absl::string_view, StatType>({
          {"UnknownStatType", kUnknownStatType},
          // TraceMe arguments.
          {"id", kStepId},
          {"parent_step_id", kParentStepId},
          {"function_step_id", kFunctionStepId},
          {"device_ordinal", kDeviceOrdinal},
          {"chip_ordinal", kChipOrdinal},
          {"node_ordinal", kNodeOrdinal},
          {"model_id", kModelId},
          {"queue_addr", kQueueAddr},
          {"request_id", kRequestId},
          {"run_id", kRunId},
          {"graph_type", kGraphType},
          {"step_num", kStepNum},
          {"iter_num", kIterNum},
          {"index_on_host", kIndexOnHost},
          {"allocator_name", kAllocatorName},
          {"bytes_reserved", kBytesReserved},
          {"bytes_allocated", kBytesAllocated},
          {"bytes_available", kBytesAvailable},
          {"fragmentation", kFragmentation},
          {"peak_bytes_in_use", kPeakBytesInUse},
          // Device trace arguments.
          {"device_id", kDeviceId},
          {"context_id", kContextId},
          {"correlation_id", kCorrelationId},
          {"memcpy_details", kMemcpyDetails},
          {"memalloc_details", kMemallocDetails},
          {"kernel_details", kKernelDetails},
          {"stream", kStream},
          // Stats added when processing traces.
          {"group_id", kGroupId},
          {"step_name", kStepName},
          {"level 0", kLevel0},
          {"tf_op", kTfOp},
          {"hlo_op", kHloOp},
          {"hlo_module", kHloModule},
          // Performance counter related.
          {"Raw Value", kRawValue},
          {"Scaled Value", kScaledValue},
          {"Thread Id", kThreadId},
          // XLA metadata map related.
          {"SELF_DURATION_PS", kSelfDurationPs},
          {"MIN_DURATION_PS", kMinDurationPs},
          // Device capability related.
          {"clock_rate", kDevCapClockRateKHz},
          {"core_count", kDevCapCoreCount},
          {"memory_bandwidth", kDevCapMemoryBandwidth},
          {"memory_size", kDevCapMemorySize},
          {"compute_cap_major", kDevCapComputeCapMajor},
          {"compute_cap_minor", kDevCapComputeCapMinor},
      });
  return *stats_type_map;
}

absl::optional<int64> FindHostEventType(absl::string_view event_name) {
  if (auto event_type = gtl::FindOrNull(GetHostEventTypeMap(), event_name)) {
    return *event_type;
  }
  return absl::nullopt;
}

absl::optional<int64> FindStatType(absl::string_view stat_name) {
  if (auto stat_type = gtl::FindOrNull(GetStatTypeMap(), stat_name)) {
    return *stat_type;
  }
  return absl::nullopt;
}

}  // namespace profiler
}  // namespace tensorflow
