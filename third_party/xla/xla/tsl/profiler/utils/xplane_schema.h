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

#ifndef XLA_TSL_PROFILER_UTILS_XPLANE_SCHEMA_H_
#define XLA_TSL_PROFILER_UTILS_XPLANE_SCHEMA_H_

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/lib/context_types.h"

namespace tsl {
namespace profiler {

// Name of XPlane that contains TraceMe events.
TF_CONST_INIT extern const absl::string_view kHostThreadsPlaneName;
// Name prefix of XPlane that contains GPU events.
TF_CONST_INIT extern const absl::string_view kGpuPlanePrefix;
// Name prefix of XPlane that contains TPU events.
TF_CONST_INIT extern const absl::string_view kTpuPlanePrefix;
// Regex for XPlanes that contain TensorCore planes.
TF_CONST_INIT extern const char kTpuPlaneRegex[];
// Regex for XPlanes that contain TPU Core planes.
TF_CONST_INIT extern const char kSparseCorePlaneRegex[];
// Name prefix of XPlane that contains custom device events.
TF_CONST_INIT extern const absl::string_view kCustomPlanePrefix;
// Name prefix of XPlane that contains TPU non-core events such as HBM, ICI etc.
TF_CONST_INIT extern const absl::string_view kTpuNonCorePlaneNamePrefix;
// Name prefix of XPlane that contains TPU runtime events.
TF_CONST_INIT extern const absl::string_view kTpuRuntimePlaneName;
// Name of XPlane that contains CUPTI driver API generated events.
TF_CONST_INIT extern const absl::string_view kCuptiDriverApiPlaneName;
// Name of XPlane that contains Roctracer API generated events.
TF_CONST_INIT extern const absl::string_view kRoctracerApiPlaneName;
// Name of XPlane that contains profile metadata such as XLA debug info.
TF_CONST_INIT extern const absl::string_view kMetadataPlaneName;
// Name of XPlane that contains kpi related metrics.
TF_CONST_INIT extern const absl::string_view kTFStreamzPlaneName;
// Name of XPlane that contains events from python tracer.
TF_CONST_INIT extern const absl::string_view kPythonTracerPlaneName;
// Name of XPlane that contains kTrace thread-switch events
TF_CONST_INIT extern const absl::string_view kHostCpusPlaneName;
// Name of XPlane that contains kTrace system calls.
TF_CONST_INIT extern const absl::string_view kSyscallsPlaneName;
// Name of XPlane that contains namescope stack tree.
TF_CONST_INIT extern const absl::string_view kScopeRangeIdTreePlaneName;

// Names of XLines that contain ML-level events.
TF_CONST_INIT extern const absl::string_view kStepLineName;
TF_CONST_INIT extern const absl::string_view kTensorFlowNameScopeLineName;
TF_CONST_INIT extern const absl::string_view kTensorFlowOpLineName;
TF_CONST_INIT extern const absl::string_view kXlaModuleLineName;
TF_CONST_INIT extern const absl::string_view kXlaOpLineName;
TF_CONST_INIT extern const absl::string_view kSparseCoreStepLineName;
TF_CONST_INIT extern const absl::string_view kXlaAsyncOpLineName;
TF_CONST_INIT extern const absl::string_view kKernelLaunchLineName;
TF_CONST_INIT extern const absl::string_view kSourceLineName;
TF_CONST_INIT extern const absl::string_view kCounterEventsLineName;
TF_CONST_INIT extern const absl::string_view kHostOffloadOpLineName;

// GPU device vendors.
TF_CONST_INIT extern const absl::string_view kDeviceVendorNvidia;
TF_CONST_INIT extern const absl::string_view kDeviceVendorAMD;

// Name of Xplane that contains environment information
TF_CONST_INIT extern const absl::string_view kTaskEnvPlaneName;

// Max collectives to display per TPU.
// Since in most cases there will be more than 9 collectives, the last line
// contains all collectives that did not qualify to get their own line.
static constexpr uint32_t kMaxCollectivesToDisplay = 9;

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
  // Loop ops.
  kParallelForOp,
  kForeverOp,
  kWhileOpEvalCond,
  kWhileOpStartBody,
  kForOp,
  // tf.data related.
  kIteratorGetNextOp,
  kIteratorGetNextAsOptionalOp,
  kIterator,
  kDeviceInputPipelineSecondIterator,
  kPrefetchProduce,
  kPrefetchConsume,
  kParallelInterleaveProduce,
  kParallelInterleaveConsume,
  kParallelInterleaveInitializedInput,
  kParallelMapProduce,
  kParallelMapConsume,
  kMapAndBatchProduce,
  kMapAndBatchConsume,
  kParseExampleProduce,
  kParseExampleConsume,
  kParallelBatchProduce,
  kParallelBatchConsume,
  // Batching related.
  kBatchingSessionRun,
  kProcessBatch,
  kBrainSessionRun,
  kConcatInputTensors,
  kMergeInputTensors,
  kScheduleWithoutSplit,
  kScheduleWithSplit,
  kScheduleWithEagerSplit,
  kASBSQueueSchedule,
  // TFRT related.
  kTfrtModelRun,
  // Serving related.
  kServingModelRun,
  // GPU related.
  kKernelLaunch,
  kKernelExecute,
  // TPU related
  kEnqueueRequestLocked,
  kRunProgramRequest,
  kHostCallbackRequest,
  kTransferH2DRequest,
  kTransferPreprocessedH2DRequest,
  kTransferD2HRequest,
  kOnDeviceSendRequest,
  kOnDeviceRecvRequest,
  kOnDeviceSendRecvLocalRequest,
  kCustomWait,
  kOnDeviceSendRequestMulti,
  kOnDeviceRecvRequestMulti,
  kPjrtAsyncWait,
  kDoEnqueueProgram,
  kDoEnqueueContinuationProgram,
  kWriteHbm,
  kReadHbm,
  kTpuExecuteOp,
  kCompleteCallbacks,
  kTransferToDeviceIssueEvent,
  kTransferToDeviceDone,
  kTransferFromDeviceIssueEvent,
  kTransferFromDeviceDone,
  kTpuSystemExecute,
  kTpuPartitionedCallOpInitializeVarOnTpu,
  kTpuPartitionedCallOpExecuteRemote,
  kTpuPartitionedCallOpExecuteLocal,
  kLinearize,
  kDelinearize,
  kTransferBufferFromDeviceFastPath,
  kLastHostEventType = kTransferBufferFromDeviceFastPath,
};

enum StatType {
  kFirstStatType = 0,
  kUnknownStatType = kFirstStatType,
  // TraceMe arguments.
  kStepId,
  kDeviceOrdinal,
  kChipOrdinal,
  kNodeOrdinal,
  kModelId,
  kQueueId,
  kQueueAddr,
  kRequestId,
  kRunId,
  kReplicaId,
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
  kTensorLayout,
  kKpiName,
  kKpiValue,
  kElementId,
  kParentId,
  kCoreType,
  // XPlane semantics related.
  kProducerType,
  kConsumerType,
  kProducerId,
  kConsumerId,
  kIsRoot,
  kIsAsync,
  // Device trace arguments.
  kDeviceId,
  kDeviceTypeString,
  kContextId,
  kCorrelationId,
  // TODO(b/176137043): These "details" should differentiate between activity
  // and API event sources.
  kMemcpyDetails,
  kMemallocDetails,
  kMemFreeDetails,
  kMemsetDetails,
  kMemoryResidencyDetails,
  kNVTXRange,
  kKernelDetails,
  kStream,
  // Stats added when processing traces.
  kGroupId,
  kFlow,
  kStepName,
  kTfOp,
  kHloOp,
  kDeduplicatedName,
  kHloCategory,
  kHloModule,
  kProgramId,
  kEquation,
  kIsEager,
  kIsFunc,
  kTfFunctionCall,
  kTfFunctionTracingCount,
  kFlops,
  kModelFlops,
  kBytesAccessed,
  kRawBytesAccessed,
  kMemoryAccessBreakdown,
  kShapeWithLayout,
  kSourceInfo,
  kModelName,
  kModelVersion,
  kBytesTransferred,
  kDmaQueue,
  kDcnCollectiveInfo,
  // Performance counter related.
  kRawValue,
  kScaledValue,
  kThreadId,
  kMatrixUnitUtilizationPercent,
  // XLA metadata map related.
  kHloProto,
  // Device capability related.
  kDevCapClockRateKHz,
  // For GPU, this is the number of SMs.
  kDevCapCoreCount,
  kDevCapMemoryBandwidth,
  kDevCapMemorySize,
  kDevCapComputeCapMajor,
  kDevCapComputeCapMinor,
  kDevCapPeakTeraflopsPerSecond,
  kDevCapPeakHbmBwGigabytesPerSecond,
  kDevCapPeakCmemRdBwGigabytesPerSecond,
  kDevCapPeakCmemWrBwGigabytesPerSecond,
  kDevCapPeakVmemRdBwGigabytesPerSecond,
  kDevCapPeakVmemWrBwGigabytesPerSecond,
  kDevCapPeakSramRdBwGigabytesPerSecond,
  kDevCapPeakSramWrBwGigabytesPerSecond,
  kDevVendor,
  kDevHasMegacore,
  kDevHasMergedVmem,
  // Batching related.
  kBatchSizeAfterPadding,
  kPaddingAmount,
  kBatchingInputTaskSize,
  // GPU occupancy metrics
  kTheoreticalOccupancyPct,
  kOccupancyMinGridSize,
  kOccupancySuggestedBlockSize,
  // Aggregated Stats
  kSelfDurationPs,
  kMinDurationPs,
  kTotalProfileDurationPs,
  kMaxIterationNum,
  kDeviceType,
  kUsesMegaCore,
  kSymbolId,
  kTfOpName,
  kDmaStallDurationPs,
  kKey,
  kPayloadSizeBytes,
  kDuration,
  kBufferSize,
  kTransfers,
  // Dcn message Stats
  kDcnLabel,
  kDcnSourceSliceId,
  kDcnSourcePerSliceDeviceId,
  kDcnDestinationSliceId,
  kDcnDestinationPerSliceDeviceId,
  kDcnChunk,
  kDcnLoopIndex,
  kEdgeTpuModelInfo,
  kEdgeTpuModelProfileInfo,
  kEdgeTpuMlir,
  kDroppedTraces,
  kCudaGraphId,
  // Many events have kCudaGraphId, such as graph sub events when tracing is in
  // node level. Yet kCudaGraphExecId is used only for CudaGraphExecution events
  // on the GPU device when tracing is in graph level.
  kCudaGraphExecId,
  kCudaGraphOrigId,
  kStepIdleTimePs,
  kGpuDeviceName,
  kSourceStack,
  kDeviceOffsetPs,
  kDeviceDurationPs,
  kScopeRangeId,
  kLastStatType = kScopeRangeId,
};

enum MegaScaleStatType : uint8_t {
  kMegaScaleGraphKey,
  kFirstMegaScaleStatType = kMegaScaleGraphKey,
  kMegaScaleLocalDeviceId,
  kMegaScaleNumActions,
  kMegaScaleCollectiveType,
  kMegaScaleInputSize,
  kMegaScaleSlackUs,
  kMegaScaleActionType,
  kMegaScaleStartEndType,
  kMegaScaleActionIndex,
  kMegaScaleActionDurationNs,
  kMegaScaleActionInputs,
  kMegaScaleTransferSource,
  kMegaScaleTransferDestinations,
  kMegaScaleTransferDcnTopologyLevel,
  kMegaScaleBufferSizes,
  kMegaScaleComputeOperation,
  kMegaScaleChunk,
  kMegaScaleLaunchId,
  kMegaScaleLoopIteration,
  kMegaScaleGraphProtos,
  kMegaScaleNetworkTransportLatency,
  kMegaScaleTransmissionBudgetUs,
  kMegaScaleDelayBudgetUs,
  kMegaScaleHloModule,
  kMegaScaleMultiSliceTopology,
  kLastMegaScaleStatType = kMegaScaleMultiSliceTopology,
};

enum TaskEnvStatType {
  kFirstTaskEnvStatType = 1,
  kEnvProfileStartTime = kFirstTaskEnvStatType,
  kEnvProfileStopTime,
  kLastTaskEnvStatType = kEnvProfileStopTime,
};

static constexpr uint32_t kLineIdOffset = 10000;

enum LineIdType {
  kFirstLineIdType = kLineIdOffset,
  kUnknownLineIdType = kFirstLineIdType,
  // DCN Traffic
  kDcnHostTraffic,
  kDcnCollectiveTraffic,
  // kDcnCollectiveTrafficMax reserves id's from kDcnCollectiveTraffic to
  // (kDcnCollectiveTraffic + kMaxCollectivesToDisplay) for DcnCollective lines.
  kDcnCollectiveTrafficMax = kDcnCollectiveTraffic + kMaxCollectivesToDisplay,
  kLastLineIdType = kDcnCollectiveTrafficMax,
};

inline std::string TpuPlaneName(int32_t device_ordinal) {
  return absl::StrCat(kTpuPlanePrefix, device_ordinal);
}

inline std::string GpuPlaneName(int32_t device_ordinal) {
  return absl::StrCat(kGpuPlanePrefix, device_ordinal);
}

absl::string_view GetHostEventTypeStr(HostEventType event_type);

bool IsHostEventType(HostEventType event_type, absl::string_view event_name);

inline bool IsHostEventType(HostEventType event_type,
                            absl::string_view event_name) {
  return GetHostEventTypeStr(event_type) == event_name;
}

std::optional<int64_t> FindHostEventType(absl::string_view event_name);

std::optional<int64_t> FindTfOpEventType(absl::string_view event_name);

absl::string_view GetStatTypeStr(StatType stat_type);

bool IsStatType(StatType stat_type, absl::string_view stat_name);

inline bool IsStatType(StatType stat_type, absl::string_view stat_name) {
  return GetStatTypeStr(stat_type) == stat_name;
}

std::optional<int64_t> FindStatType(absl::string_view stat_name);

absl::string_view GetMegaScaleStatTypeStr(MegaScaleStatType stat_type);

inline bool IsMegaScaleStatType(MegaScaleStatType stat_type,
                                absl::string_view stat_name) {
  return GetMegaScaleStatTypeStr(stat_type) == stat_name;
}

std::optional<int64_t> FindMegaScaleStatType(absl::string_view stat_name);

// Returns true if the given event shouldn't be shown in the trace viewer.
bool IsInternalEvent(std::optional<int64_t> event_type);

// Returns true if the given stat shouldn't be shown in the trace viewer.
bool IsInternalStat(std::optional<int64_t> stat_type);

absl::string_view GetTaskEnvStatTypeStr(TaskEnvStatType stat_type);

std::optional<int64_t> FindTaskEnvStatType(absl::string_view stat_name);

// Support for flow events:
// This class enables encoding/decoding the flow id and direction, stored as
// XStat value. The flow id are limited to 56 bits.
class XFlow {
 public:
  enum FlowDirection {
    kFlowUnspecified = 0x0,
    kFlowIn = 0x1,
    kFlowOut = 0x2,
    kFlowInOut = 0x3,
  };

  XFlow(uint64_t flow_id, FlowDirection direction,
        ContextType category = ContextType::kGeneric) {
    DCHECK_NE(direction, kFlowUnspecified);
    encoded_.parts.direction = direction;
    encoded_.parts.flow_id = flow_id;
    encoded_.parts.category = static_cast<uint64_t>(category);
  }

  // Encoding
  uint64 ToStatValue() const { return encoded_.whole; }

  // Decoding
  static XFlow FromStatValue(uint64_t encoded) { return XFlow(encoded); }

  /* NOTE: absl::HashOf is not consistent across processes (some process level
   * salt is added), even different executions of the same program.
   * However we are not tracking cross-host flows, i.e. A single flow's
   * participating events are from the same XSpace. On the other hand,
   * events from the same XSpace is always processed in the same profiler
   * process. Flows from different hosts are unlikely to collide because of
   * 2^56 hash space. Therefore, we can consider this is good for now. We should
   * revisit the hash function when cross-hosts flows became more popular.
   */
  template <typename... Args>
  static uint64_t GetFlowId(Args&&... args) {
    return absl::HashOf(std::forward<Args>(args)...) & kFlowMask;
  }

  uint64_t Id() const { return encoded_.parts.flow_id; }
  ContextType Category() const {
    return GetSafeContextType(encoded_.parts.category);
  }
  FlowDirection Direction() const {
    return FlowDirection(encoded_.parts.direction);
  }

  static uint64_t GetUniqueId() {  // unique in current process.
    return next_flow_id_.fetch_add(1);
  }

 private:
  explicit XFlow(uint64_t encoded) { encoded_.whole = encoded; }
  static constexpr uint64_t kFlowMask = (1ULL << 56) - 1;
  static std::atomic<uint64_t> next_flow_id_;

  union {
    // Encoded representation.
    uint64_t whole;
    struct {
      uint64_t direction : 2;
      uint64_t flow_id : 56;
      uint64_t category : 6;
    } parts;
  } encoded_ ABSL_ATTRIBUTE_PACKED;

  static_assert(sizeof(encoded_) == sizeof(uint64_t), "Must be 64 bits.");
};

// String constants for XProf TraceMes for DCN Messages.
TF_CONST_INIT extern const absl::string_view kMegaScaleDcnReceive;
TF_CONST_INIT extern const absl::string_view kMegaScaleDcnSend;
TF_CONST_INIT extern const absl::string_view kMegaScaleDcnSendFinished;
TF_CONST_INIT extern const absl::string_view kMegaScaleDcnMemAllocate;
TF_CONST_INIT extern const absl::string_view kMegaScaleDcnMemCopy;
TF_CONST_INIT extern const absl::string_view kMegaScaleTopologyDiscovery;
TF_CONST_INIT extern const absl::string_view kMegaScaleBarrier;
TF_CONST_INIT extern const absl::string_view kMegaScaleHostCommand;
TF_CONST_INIT extern const absl::string_view kMegaScaleD2HTransferStart;
TF_CONST_INIT extern const absl::string_view kMegaScaleD2HTransferFinished;
TF_CONST_INIT extern const absl::string_view kMegaScaleH2DTransferStart;
TF_CONST_INIT extern const absl::string_view kMegaScaleH2DTransferFinished;
TF_CONST_INIT extern const absl::string_view kMegaScaleReductionStart;
TF_CONST_INIT extern const absl::string_view kMegaScaleReductionFinished;
TF_CONST_INIT extern const absl::string_view kMegaScaleCompressionStart;
TF_CONST_INIT extern const absl::string_view kMegaScaleCompressionFinished;
TF_CONST_INIT extern const absl::string_view kMegaScaleDecompressionStart;
TF_CONST_INIT extern const absl::string_view kMegaScaleDecompressionFinished;
TF_CONST_INIT extern const char kXProfMetadataKey[];
TF_CONST_INIT extern const char kXProfMetadataFlow[];
TF_CONST_INIT extern const char kXProfMetadataTransfers[];
TF_CONST_INIT extern const char kXProfMetadataBufferSize[];

// String constants for threadpool_listener events
TF_CONST_INIT extern const absl::string_view kThreadpoolListenerRecord;
TF_CONST_INIT extern const absl::string_view kThreadpoolListenerStartRegion;
TF_CONST_INIT extern const absl::string_view kThreadpoolListenerStopRegion;
TF_CONST_INIT extern const absl::string_view kThreadpoolListenerRegion;

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_XPLANE_SCHEMA_H_
