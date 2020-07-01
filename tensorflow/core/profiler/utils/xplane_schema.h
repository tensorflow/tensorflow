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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {

// Name of XPlane that contains TraceMe events.
ABSL_CONST_INIT extern const absl::string_view kHostThreadsPlaneName;
// Name prefix of XPlane that contains GPU events.
ABSL_CONST_INIT extern const absl::string_view kGpuPlanePrefix;
// Name of XPlane that contains CUPTI driver API generated events.
ABSL_CONST_INIT extern const absl::string_view kCuptiDriverApiPlaneName;
// Name of XPlane that contains profile metadata such as XLA debug info.
ABSL_CONST_INIT extern const absl::string_view kMetadataPlaneName;
// Name of XPlane that contains kpi related metrics.
ABSL_CONST_INIT extern const absl::string_view kTFStreamzPlaneName;

// Names of XLines that contain ML-level events.
ABSL_CONST_INIT extern const absl::string_view kStepLineName;
ABSL_CONST_INIT extern const absl::string_view kTensorFlowNameScopeLineName;
ABSL_CONST_INIT extern const absl::string_view kTensorFlowOpLineName;
ABSL_CONST_INIT extern const absl::string_view kXlaModuleLineName;
ABSL_CONST_INIT extern const absl::string_view kXlaOpLineName;
ABSL_CONST_INIT extern const absl::string_view kKernelLaunchLineName;

// Interesting event types (i.e., TraceMe names).
enum HostEventType {
  kFirstHostEventType = 0,
  kUnknownHostEventType = kFirstHostEventType,
  kTraceContext,
  kSessionRun,
  kFunctionRun,
  kRunGraph,
  kRunGraphDone,
  kTfOpRun,
  kEagerKernelExecute,
  kExecutorStateProcess,
  kExecutorDoneCallback,
  kMemoryAllocation,
  kMemoryDeallocation,
  // Performance counter related.
  kRemotePerf,
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
  // tf.data related.
  kIteratorGetNextOp,
  kIteratorGetNextAsOptionalOp,
  // GPU related.
  kKernelLaunch,
  kKernelExecute,
  kLastHostEventType = kKernelExecute,
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
  kAllocatorName,
  kBytesReserved,
  kBytesAllocated,
  kBytesAvailable,
  kFragmentation,
  kPeakBytesInUse,
  kRequestedBytes,
  kAllocationBytes,
  kAddress,
  kRegionType,
  kDataType,
  kTensorShapes,
  kKpiName,
  kKpiValue,
  // XPlane semantics related.
  kProducerType,
  kConsumerType,
  kProducerId,
  kConsumerId,
  kIsRoot,
  kIsAsync,
  // Device trace arguments.
  kDeviceId,
  kContextId,
  kCorrelationId,
  kMemcpyDetails,
  kMemallocDetails,
  kKernelAnnotation,
  kKernelDetails,
  kStream,
  // Stats added when processing traces.
  kGroupId,
  kFlow,
  kStepName,
  kLevel0,
  kTfOp,
  kHloOp,
  kHloModule,
  kEquation,
  kIsEager,
  kTfFunctionCall,
  kTfFunctionTracingCount,
  kFlops,
  kBytesAccessed,
  // Performance counter related.
  kRawValue,
  kScaledValue,
  kThreadId,
  // XLA metadata map related.
  kSelfDurationPs,
  kMinDurationPs,
  kHloProto,
  // Device capability related.
  kDevCapClockRateKHz,
  kDevCapCoreCount,
  kDevCapMemoryBandwidth,
  kDevCapMemorySize,
  kDevCapComputeCapMajor,
  kDevCapComputeCapMinor,
  kLastStatType = kDevCapComputeCapMinor,
};

inline std::string GpuPlaneName(int32 device_ordinal) {
  return absl::StrCat(kGpuPlanePrefix, device_ordinal);
}

inline bool IsGpuPlaneName(absl::string_view plane_name) {
  return absl::StartsWith(plane_name, kGpuPlanePrefix);
}

absl::string_view GetHostEventTypeStr(HostEventType event_type);

bool IsHostEventType(HostEventType event_type, absl::string_view event_name);

inline bool IsHostEventType(HostEventType event_type,
                            absl::string_view event_name) {
  return GetHostEventTypeStr(event_type) == event_name;
}

absl::optional<int64> FindHostEventType(absl::string_view event_name);

absl::string_view GetStatTypeStr(StatType stat_type);

bool IsStatType(StatType stat_type, absl::string_view stat_name);

inline bool IsStatType(StatType stat_type, absl::string_view stat_name) {
  return GetStatTypeStr(stat_type) == stat_name;
}

absl::optional<int64> FindStatType(absl::string_view stat_name);

// Returns true if the given stat shouldn't be shown in the trace viewer.
bool IsInternalStat(absl::optional<int64> stat_type);

// Support for flow events:
// This class enables encoding/decoding the flow id and direction, stored as
// XStat value.
class XFlow {
 public:
  enum FlowDirection {
    kFlowUnspecified = 0x0,
    kFlowIn = 0x1,
    kFlowOut = 0x2,
    kFlowInOut = 0x3,
  };

  XFlow(uint64 flow_id, FlowDirection direction)
      : encoded_((flow_id << 2) | (direction & 0x3)) {
    DCHECK_NE(Direction(), kFlowUnspecified);
  }

  // Encoding
  uint64 ToStatValue() const { return encoded_; }

  // Decoding
  static XFlow FromStatValue(uint64 encoded) { return XFlow(encoded); }

  uint64 Id() const { return (encoded_ >> 2); }
  FlowDirection Direction() const { return FlowDirection(encoded_ & 0x3); }

 private:
  explicit XFlow(uint64 encoded) : encoded_(encoded) {}

  uint64 encoded_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_
