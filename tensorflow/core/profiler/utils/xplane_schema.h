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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace profiler {

// Name of XPlane that contains TraceMe events.
ABSL_CONST_INIT extern const absl::string_view kHostThreads;

// Interesting event types (i.e., TraceMe names).
enum HostEventType {
  kFirstHostEventType = 0,
  kUnknownHostEventType = kFirstHostEventType,
  kTraceContext,
  kSessionRun,
  kFunctionRun,
  kRunGraph,
  kEagerKernelExecute,
  kExecutorStateProcess,
  kExecutorDoneCallback,
  // tf.data captured function events.
  kTfDataCapturedFunctionRun,
  kTfDataCapturedFunctionRunWithBorrowedArgs,
  kTfDataCapturedFunctionRunInstantiated,
  kTfDataCapturedFunctionRunAsync,
  // Functional ops.
  kCallOp,
  kParallelForOp,
  kForeverOp,
  kNumericalGradientOpEvalRight,
  kNumericalGradientOpEvalLeft,
  kSymbolicGradientOp,
  kRemoteCallOp,
  kIfOp,
  kCaseOp,
  kWhileOpEvalCond,
  kWhileOpStartBody,
  kForOp,
  kPartitionedCallOp,
  kLastHostEventType = kPartitionedCallOp,
};

enum StatType {
  kFirstStatType = 0,
  kUnknownStatType = kFirstStatType,
  // TraceMe arguments.
  kStepId,
  kParentStepId,
  kFunctionStepId,
  kDeviceOrdinal,
  kChipOrdinal,
  kNodeOrdinal,
  kModelId,
  kQueueAddr,
  kRequestId,
  kRunId,
  kGraphType,
  kStepNum,
  kIterNum,
  kIndexOnHost,
  kBytesReserved,
  kBytesAllocated,
  kBytesAvailable,
  kFragmentation,
  // Device trace arguments.
  kDeviceId,
  kContextId,
  kCorrelationId,
  kMemcpyDetails,
  kMemallocDetails,
  kKernelDetails,
  // Stats added when processing traces.
  kGroupId,
  kStepName,
  kLevel0,
  kTfOp,
  kHloOp,
  kHloModule,
  kLastStatType = kHloModule,
};

absl::Span<const absl::string_view> GetHostEventTypeStrMap();

inline absl::string_view GetHostEventTypeStr(HostEventType event_type) {
  return GetHostEventTypeStrMap()[event_type];
}

inline bool IsHostEventType(HostEventType event_type,
                            absl::string_view event_name) {
  return GetHostEventTypeStrMap()[event_type] == event_name;
}

absl::Span<const absl::string_view> GetStatTypeStrMap();

inline absl::string_view GetStatTypeStr(StatType stat_type) {
  return GetStatTypeStrMap()[stat_type];
}

inline bool IsStatType(StatType stat_type, absl::string_view stat_name) {
  return GetStatTypeStr(stat_type) == stat_name;
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_
