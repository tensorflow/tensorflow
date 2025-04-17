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

#include "xla/tsl/profiler/utils/xplane_schema.h"

#include <atomic>
#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"

namespace tsl {
namespace profiler {

const absl::string_view kHostThreadsPlaneName = "/host:CPU";
const absl::string_view kGpuPlanePrefix = "/device:GPU:";
const absl::string_view kTpuPlanePrefix = "/device:TPU:";
const absl::string_view kTpuNonCorePlaneNamePrefix = "#Chip";
const char kTpuPlaneRegex[] = {"/device:TPU:([0-9]*)$"};
const char kSparseCorePlaneRegex[] = {
    "/device:TPU:[0-9]+ SparseCore ([0-9]+)$"};
// TODO(b/195582092): change it to /device:custom once all literals are
// migrated.
const absl::string_view kCustomPlanePrefix = "/device:CUSTOM:";

const absl::string_view kScopeRangeIdTreePlaneName =
    "/host:__ScopeRangeCallStack__";
const absl::string_view kTpuRuntimePlaneName = "/host:TPU-runtime";
const absl::string_view kCuptiDriverApiPlaneName = "/host:CUPTI";
const absl::string_view kRoctracerApiPlaneName = "/host:ROCTRACER";
const absl::string_view kMetadataPlaneName = "/host:metadata";
const absl::string_view kTFStreamzPlaneName = "/host:tfstreamz";
const absl::string_view kPythonTracerPlaneName = "/host:python-tracer";
const absl::string_view kHostCpusPlaneName = "Host CPUs";
const absl::string_view kSyscallsPlaneName = "Syscalls";

const absl::string_view kStepLineName = "Steps";
const absl::string_view kSparseCoreStepLineName = "Sparse Core Steps";
const absl::string_view kSparseCoreModuleLineName = "Sparse Core Modules";
const absl::string_view kSparseCoreOpLineName = "Sparse Core Ops";
const absl::string_view kTensorFlowNameScopeLineName = "Framework Name Scope";
const absl::string_view kTensorFlowOpLineName = "Framework Ops";
const absl::string_view kXlaModuleLineName = "XLA Modules";
const absl::string_view kXlaOpLineName = "XLA Ops";
const absl::string_view kXlaAsyncOpLineName = "Async XLA Ops";
const absl::string_view kKernelLaunchLineName = "Launch Stats";
const absl::string_view kSourceLineName = "Source code";
const absl::string_view kHostOffloadOpLineName = "Host Offload Ops";
const absl::string_view kCounterEventsLineName = "_counters_";
const absl::string_view kTensorCoreSyncFlagLineName = "Tensor Core Sync Flag";
const absl::string_view kSparseCoreSyncsLineName = "Sparse Core Syncs";

const absl::string_view kDeviceVendorNvidia = "Nvidia";
const absl::string_view kDeviceVendorAMD = "AMD";

const absl::string_view kTaskEnvPlaneName = "Task Environment";

namespace {

constexpr int kNumHostEventTypes =
    HostEventType::kLastHostEventType - HostEventType::kFirstHostEventType + 1;

constexpr int kNumStatTypes =
    StatType::kLastStatType - StatType::kFirstStatType + 1;

constexpr int kNumMegaScaleStatTypes =
    MegaScaleStatType::kLastMegaScaleStatType -
    MegaScaleStatType::kFirstMegaScaleStatType + 1;

constexpr int kNumLineIdTypes =
    LineIdType::kLastLineIdType - LineIdType::kFirstLineIdType + 1;

using HostEventTypeMap = absl::flat_hash_map<absl::string_view, HostEventType>;
using HostEventTypeStrMap =
    absl::flat_hash_map<HostEventType, absl::string_view>;
using StatTypeMap = absl::flat_hash_map<absl::string_view, StatType>;
using StatTypeStrMap = absl::flat_hash_map<StatType, absl::string_view>;
using MegaScaleStatTypeMap =
    absl::flat_hash_map<absl::string_view, MegaScaleStatType>;
using MegaScaleStatTypeStrMap =
    absl::flat_hash_map<MegaScaleStatType, absl::string_view>;
using LineIdTypeMap = absl::flat_hash_map<absl::string_view, LineIdType>;
using LineIdTypeStrMap = absl::flat_hash_map<LineIdType, absl::string_view>;

const HostEventTypeMap& GetHostEventTypeMap() {
  static auto* const host_event_type_map = new HostEventTypeMap({
      {"UnknownHostEventType", kUnknownHostEventType},
      {"TraceContext", kTraceContext},
      {"SessionRun", kSessionRun},
      {"FunctionRun", kFunctionRun},
      {"RunGraph", kRunGraph},
      {"RunGraphDone", kRunGraphDone},
      {"TfOpRun", kTfOpRun},
      {"EagerExecute", kEagerKernelExecute},
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
      // Loop ops.
      {"ParallelForOp", kParallelForOp},
      {"ForeverOp", kForeverOp},
      {"WhileOp-EvalCond", kWhileOpEvalCond},
      {"WhileOp-StartBody", kWhileOpStartBody},
      {"ForOp", kForOp},
      // tf.data related.
      {"IteratorGetNextOp::DoCompute", kIteratorGetNextOp},
      {"IteratorGetNextAsOptionalOp::DoCompute", kIteratorGetNextAsOptionalOp},
      {"Iterator", kIterator},
      {"Iterator::Prefetch::Generator", kDeviceInputPipelineSecondIterator},
      {"PrefetchProduce", kPrefetchProduce},
      {"PrefetchConsume", kPrefetchConsume},
      {"ParallelInterleaveProduce", kParallelInterleaveProduce},
      {"ParallelInterleaveConsume", kParallelInterleaveConsume},
      {"ParallelInterleaveInitializeInput",
       kParallelInterleaveInitializedInput},
      {"ParallelMapProduce", kParallelMapProduce},
      {"ParallelMapConsume", kParallelMapConsume},
      {"MapAndBatchProduce", kMapAndBatchProduce},
      {"MapAndBatchConsume", kMapAndBatchConsume},
      {"ParseExampleProduce", kParseExampleProduce},
      {"ParseExampleConsume", kParseExampleConsume},
      {"ParallelBatchProduce", kParallelBatchProduce},
      {"ParallelBatchConsume", kParallelBatchConsume},
      // Batching related.
      {"BatchingSessionRun", kBatchingSessionRun},
      {"ProcessBatch", kProcessBatch},
      {"BrainSessionRun", kBrainSessionRun},
      {"ConcatInputTensors", kConcatInputTensors},
      {"MergeInputTensors", kMergeInputTensors},
      {"ScheduleWithoutSplit", kScheduleWithoutSplit},
      {"ScheduleWithSplit", kScheduleWithSplit},
      {"ScheduleWithEagerSplit", kScheduleWithEagerSplit},
      {"ASBSQueue::Schedule", kASBSQueueSchedule},
      // TFRT related.
      {"TfrtModelRun", kTfrtModelRun},
      // Serving related.
      {"ServingModelRun", kServingModelRun},
      // GPU related.
      {"KernelLaunch", kKernelLaunch},
      {"KernelExecute", kKernelExecute},
      // TPU related.
      {"EnqueueRequestLocked", kEnqueueRequestLocked},
      {"RunProgramRequest", kRunProgramRequest},
      {"HostCallbackRequest", kHostCallbackRequest},
      {"TransferH2DRequest", kTransferH2DRequest},
      {"TransferPreprocessedH2DRequest", kTransferPreprocessedH2DRequest},
      {"TransferD2HRequest", kTransferD2HRequest},
      {"OnDeviceSendRequest", kOnDeviceSendRequest},
      {"OnDeviceRecvRequest", kOnDeviceRecvRequest},
      {"OnDeviceSendRecvLocalRequest", kOnDeviceSendRecvLocalRequest},
      {"CustomWait", kCustomWait},
      {"OnDeviceSendRequestMulti", kOnDeviceSendRequestMulti},
      {"OnDeviceRecvRequestMulti", kOnDeviceRecvRequestMulti},
      {"PjrtAsyncWait", kPjrtAsyncWait},
      {"DoEnqueueProgram", kDoEnqueueProgram},
      {"DoEnqueueContinuationProgram", kDoEnqueueContinuationProgram},
      {"WriteHbm", kWriteHbm},
      {"ReadHbm", kReadHbm},
      {"TpuExecuteOp", kTpuExecuteOp},
      {"CompleteCallbacks", kCompleteCallbacks},
      {"TPUPartitionedCallOp-InitializeVarOnTPU",
       kTpuPartitionedCallOpInitializeVarOnTpu},
      {"TPUPartitionedCallOp-ExecuteRemote",
       kTpuPartitionedCallOpExecuteRemote},
      {"TPUPartitionedCallOp-ExecuteLocal", kTpuPartitionedCallOpExecuteLocal},
      {"Linearize", kLinearize},
      {"Delinearize", kDelinearize},
      {"TransferBufferFromDevice-FastPath", kTransferBufferFromDeviceFastPath},
      {"tpu::System::TransferToDevice=>IssueEvent",
       kTransferToDeviceIssueEvent},
      {"tpu::System::TransferToDevice=>IssueEvent=>Done",
       kTransferToDeviceDone},
      {"tpu::System::TransferFromDevice=>IssueEvent",
       kTransferFromDeviceIssueEvent},
      {"tpu::System::TransferFromDevice=>IssueEvent=>Done",
       kTransferFromDeviceDone},
      {"tpu::System::Execute", kTpuSystemExecute},
  });
  DCHECK_EQ(host_event_type_map->size(), kNumHostEventTypes);
  return *host_event_type_map;
}

const StatTypeMap& GetStatTypeMap() {
  static auto* const stat_type_map = new StatTypeMap(
      {{"UnknownStatType", kUnknownStatType},
       // TraceMe arguments.
       {"id", kStepId},
       {"device_ordinal", kDeviceOrdinal},
       {"chip_ordinal", kChipOrdinal},
       {"node_ordinal", kNodeOrdinal},
       {"model_id", kModelId},
       {"queue_addr", kQueueAddr},
       {"queue_id", kQueueId},
       {"request_id", kRequestId},
       {"run_id", kRunId},
       {"replica_id", kReplicaId},
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
       {"requested_bytes", kRequestedBytes},
       {"allocation_bytes", kAllocationBytes},
       {"addr", kAddress},
       {"region_type", kRegionType},
       {"data_type", kDataType},
       {"shape", kTensorShapes},
       {"layout", kTensorLayout},
       {"kpi_name", kKpiName},
       {"kpi_value", kKpiValue},
       {"element_id", kElementId},
       {"parent_id", kParentId},
       {"core_type", kCoreType},
       // XPlane semantics related.
       {"_pt", kProducerType},
       {"_ct", kConsumerType},
       {"_p", kProducerId},
       {"_c", kConsumerId},
       {"_r", kIsRoot},
       {"_a", kIsAsync},
       // Device trace arguments.
       {"device_id", kDeviceId},
       {"device_type_string", kDeviceTypeString},
       {"context_id", kContextId},
       {"correlation_id", kCorrelationId},
       {"memcpy_details", kMemcpyDetails},
       {"memalloc_details", kMemallocDetails},
       {"MemFree_details", kMemFreeDetails},
       {"Memset_details", kMemsetDetails},
       {"MemoryResidency_details", kMemoryResidencyDetails},
       {"kernel_details", kKernelDetails},
       {"nvtx_range", kNVTXRange},
       {"stream", kStream},
       // Stats added when processing traces.
       {"group_id", kGroupId},
       {"flow", kFlow},
       {"step_name", kStepName},
       {"tf_op", kTfOp},
       {"hlo_op", kHloOp},
       {"deduplicated_name", kDeduplicatedName},
       {"hlo_category", kHloCategory},
       {"hlo_module", kHloModule},
       {"program_id", kProgramId},
       {"equation", kEquation},
       {"is_eager", kIsEager},
       {"is_func", kIsFunc},
       {"tf_function_call", kTfFunctionCall},
       {"tracing_count", kTfFunctionTracingCount},
       {"flops", kFlops},
       {"model_flops", kModelFlops},
       {"bytes_accessed", kBytesAccessed},
       {"raw_bytes_accessed", kRawBytesAccessed},
       {"memory_access_breakdown", kMemoryAccessBreakdown},
       {"shape_with_layout", kShapeWithLayout},
       {"source", kSourceInfo},
       {"model_name", kModelName},
       {"model_version", kModelVersion},
       {"bytes_transferred", kBytesTransferred},
       {"queue", kDmaQueue},
       {"dcn_collective_info", kDcnCollectiveInfo},
       {"all_reduce_id", kAllReduceId},
       {"all_reduce_unique_id", kAllReduceUniqueId},
       // Performance counter related.
       {"Raw Value", kRawValue},
       {"Scaled Value", kScaledValue},
       {"Thread Id", kThreadId},
       {"matrix_unit_utilization_percent", kMatrixUnitUtilizationPercent},
       // XLA metadata map related.
       {"Hlo Proto", kHloProto},
       {"EdgeTPU Model information", kEdgeTpuModelInfo},
       {"EdgeTPU Model Profile information", kEdgeTpuModelProfileInfo},
       {"EdgeTPU MLIR", kEdgeTpuMlir},
       // Device capability related.
       {"clock_rate", kDevCapClockRateKHz},
       {"core_count", kDevCapCoreCount},
       {"memory_bandwidth", kDevCapMemoryBandwidth},
       {"memory_size", kDevCapMemorySize},
       {"compute_cap_major", kDevCapComputeCapMajor},
       {"compute_cap_minor", kDevCapComputeCapMinor},
       {"peak_teraflops_per_second", kDevCapPeakTeraflopsPerSecond},
       {"peak_hbm_bw_gigabytes_per_second", kDevCapPeakHbmBwGigabytesPerSecond},
       {"peak_cmem_rd_bw_gigabytes_per_second",
        kDevCapPeakCmemRdBwGigabytesPerSecond},
       {"peak_cmem_wr_bw_gigabytes_per_second",
        kDevCapPeakCmemWrBwGigabytesPerSecond},
       {"peak_vmem_rd_bw_gigabytes_per_second",
        kDevCapPeakVmemRdBwGigabytesPerSecond},
       {"peak_vmem_wr_bw_gigabytes_per_second",
        kDevCapPeakVmemWrBwGigabytesPerSecond},
       {"peak_sram_rd_bw_gigabytes_per_second",
        kDevCapPeakSramRdBwGigabytesPerSecond},
       {"peak_sram_wr_bw_gigabytes_per_second",
        kDevCapPeakSramWrBwGigabytesPerSecond},
       {"device_vendor", kDevVendor},
       {"has_megacore", kDevHasMegacore},
       {"has_merged_vmem", kDevHasMergedVmem},
       // Batching related.
       {"batch_size_after_padding", kBatchSizeAfterPadding},
       {"padding_amount", kPaddingAmount},
       {"batching_input_task_size", kBatchingInputTaskSize},
       // GPU related metrics.
       {"theoretical_occupancy_pct", kTheoreticalOccupancyPct},
       {"occupancy_min_grid_size", kOccupancyMinGridSize},
       {"occupancy_suggested_block_size", kOccupancySuggestedBlockSize},
       // Aggregated Stat
       {"self_duration_ps", kSelfDurationPs},
       {"min_duration_ps", kMinDurationPs},
       {"total_profile_duration_ps", kTotalProfileDurationPs},
       {"max_iteration_num", kMaxIterationNum},
       {"device_type", kDeviceType},
       {"uses_megacore", kUsesMegaCore},
       {"symbol_id", kSymbolId},
       {"hlo_category", kHloCategory},
       {"tf_op_name", kTfOpName},
       {"dma_stall_duration_ps", kDmaStallDurationPs},
       {"key", kKey},
       {"payload_size_bytes", kPayloadSizeBytes},
       {"duration_us", kDuration},
       {"buffer_size", kBufferSize},
       {"transfers", kTransfers},
       // Dcn message Stats
       {"dcn_label", kDcnLabel},
       {"dcn_source_slice_id", kDcnSourceSliceId},
       {"dcn_source_per_slice_device_id", kDcnSourcePerSliceDeviceId},
       {"dcn_destination_slice_id", kDcnDestinationSliceId},
       {"dcn_destination_per_slice_device_id", kDcnDestinationPerSliceDeviceId},
       {"dcn_chunk", kDcnChunk},
       {"dcn_loop_index", kDcnLoopIndex},
       {"dropped_traces", kDroppedTraces},
       {"cuda_graph_id", kCudaGraphId},
       {"cuda_graph_exec_id", kCudaGraphExecId},
       {"cuda_graph_orig_id", kCudaGraphOrigId},
       {"step_idle_time_ps", kStepIdleTimePs},
       {"gpu_device_name", kGpuDeviceName},
       {"source_stack", kSourceStack},
       {"device_offset_ps", kDeviceOffsetPs},
       {"device_duration_ps", kDeviceDurationPs},
       {"scope_range_id", kScopeRangeId},
       {"core_details", kCoreDetails}});
  DCHECK_EQ(stat_type_map->size(), kNumStatTypes);
  return *stat_type_map;
}

const MegaScaleStatTypeMap& GetMegaScaleStatTypeMap() {
  static auto* const stat_type_map = new MegaScaleStatTypeMap(
      {{"graph_key", kMegaScaleGraphKey},
       {"local_device_id", kMegaScaleLocalDeviceId},
       {"num_actions", kMegaScaleNumActions},
       {"collective_type", kMegaScaleCollectiveType},
       {"input_size", kMegaScaleInputSize},
       {"send_channel_id", kMegaScaleSendChannelId},
       {"recv_channel_id", kMegaScaleRecvChannelId},
       {"slack_us", kMegaScaleSlackUs},
       {"action_type", kMegaScaleActionType},
       {"start_end_type", kMegaScaleStartEndType},
       {"action_index", kMegaScaleActionIndex},
       {"action_duration_ns", kMegaScaleActionDurationNs},
       {"action_inputs", kMegaScaleActionInputs},
       {"transfer_source", kMegaScaleTransferSource},
       {"transfer_destinations", kMegaScaleTransferDestinations},
       {"dcn_topology_level", kMegaScaleTransferDcnTopologyLevel},
       {"buffer_sizes", kMegaScaleBufferSizes},
       {"compute_operation", kMegaScaleComputeOperation},
       {"chunk", kMegaScaleChunk},
       {"launch_id", kMegaScaleLaunchId},
       {"loop_iteration", kMegaScaleLoopIteration},
       {"transmission_budget_us", kMegaScaleTransmissionBudgetUs},
       {"delay_budget_us", kMegaScaleDelayBudgetUs},
       {"graph_protos", kMegaScaleGraphProtos},
       {"network_transport_latency_us", kMegaScaleNetworkTransportLatency},
       {"hlo_module", kMegaScaleHloModule},
       {"multi_slice_topology", kMegaScaleMultiSliceTopology}});
  DCHECK_EQ(stat_type_map->size(), kNumMegaScaleStatTypes);
  return *stat_type_map;
}

const LineIdTypeMap& GetLineIdTypeMap() {
  static auto* const line_id_type_map = new LineIdTypeMap({
      {"UnknownLineIdType", kUnknownLineIdType},
      {"DcnHostTraffic", kDcnHostTraffic},
      {"DcnCollectiveTraffic", kDcnCollectiveTraffic},
  });
  DCHECK_EQ(line_id_type_map->size(), kNumLineIdTypes);
  return *line_id_type_map;
}

const HostEventTypeStrMap& GetHostEventTypeStrMap() {
  static auto* const host_event_type_str_map = new HostEventTypeStrMap(
      gtl::ReverseMap<HostEventTypeStrMap>(GetHostEventTypeMap()));
  return *host_event_type_str_map;
}

const StatTypeStrMap& GetStatTypeStrMap() {
  static auto* const stat_type_str_map =
      new StatTypeStrMap(gtl::ReverseMap<StatTypeStrMap>(GetStatTypeMap()));
  return *stat_type_str_map;
}

const MegaScaleStatTypeStrMap& GetMegaScaleStatTypeStrMap() {
  static auto* const stat_type_str_map = new MegaScaleStatTypeStrMap(
      gtl::ReverseMap<MegaScaleStatTypeStrMap>(GetMegaScaleStatTypeMap()));
  return *stat_type_str_map;
}

const LineIdTypeStrMap& GetLineIdTypeStrMap() {
  static auto* const line_id_type_str_map = new LineIdTypeStrMap(
      gtl::ReverseMap<LineIdTypeStrMap>(GetLineIdTypeMap()));
  return *line_id_type_str_map;
}

using TaskEnvStatTypeMap =
    absl::flat_hash_map<absl::string_view, TaskEnvStatType>;
using TaskEnvStatTypeStrMap =
    absl::flat_hash_map<TaskEnvStatType, absl::string_view>;

constexpr int kNumTaskEnvStatTypes = TaskEnvStatType::kLastTaskEnvStatType -
                                     TaskEnvStatType::kFirstTaskEnvStatType + 1;

const TaskEnvStatTypeMap& GetTaskEnvStatTypeMap() {
  static auto* const task_env_stat_type_map = new TaskEnvStatTypeMap({
      {"profile_start_time", kEnvProfileStartTime},
      {"profile_stop_time", kEnvProfileStopTime},
  });
  DCHECK_EQ(task_env_stat_type_map->size(), kNumTaskEnvStatTypes);
  return *task_env_stat_type_map;
}

const TaskEnvStatTypeStrMap& GetTaskEnvStatTypeStrMap() {
  static auto* const task_env_stat_type_str_map = new TaskEnvStatTypeStrMap(
      gtl::ReverseMap<TaskEnvStatTypeStrMap>(GetTaskEnvStatTypeMap()));
  return *task_env_stat_type_str_map;
}

}  // namespace

absl::string_view GetHostEventTypeStr(HostEventType event_type) {
  return GetHostEventTypeStrMap().at(event_type);
}

std::optional<int64_t> FindHostEventType(absl::string_view event_name) {
  if (auto event_type = gtl::FindOrNull(GetHostEventTypeMap(), event_name)) {
    return *event_type;
  }
  return std::nullopt;
}

std::optional<int64_t> FindTfOpEventType(absl::string_view event_name) {
  // TF op names.
  Category category = ParseTfOpFullname(event_name).category;
  switch (category) {
    case Category::kTensorFlow:
      return HostEventType::kTfOpRun;
    case Category::kTfData:
      return HostEventType::kIterator;
    default:
      return std::nullopt;
  }
}

absl::string_view GetStatTypeStr(StatType stat_type) {
  return GetStatTypeStrMap().at(stat_type);
}

std::optional<int64_t> FindStatType(absl::string_view stat_name) {
  if (auto stat_type = gtl::FindOrNull(GetStatTypeMap(), stat_name)) {
    return *stat_type;
  }
  return std::nullopt;
}

absl::string_view GetMegaScaleStatTypeStr(MegaScaleStatType stat_type) {
  return GetMegaScaleStatTypeStrMap().at(stat_type);
}

std::optional<int64_t> FindMegaScaleStatType(absl::string_view stat_name) {
  if (auto stat_type = gtl::FindOrNull(GetMegaScaleStatTypeMap(), stat_name)) {
    return *stat_type;
  }
  return std::nullopt;
}

absl::string_view GetTaskEnvStatTypeStr(TaskEnvStatType stat_type) {
  return GetTaskEnvStatTypeStrMap().at(stat_type);
}

std::optional<int64_t> FindTaskEnvStatType(absl::string_view stat_name) {
  if (auto stat_type = gtl::FindOrNull(GetTaskEnvStatTypeMap(), stat_name)) {
    return *stat_type;
  }
  return std::nullopt;
}

absl::string_view GetLineIdTypeStr(LineIdType line_id_type) {
  return GetLineIdTypeStrMap().at(line_id_type);
}

bool IsInternalEvent(std::optional<int64_t> event_type) {
  // TODO(b/162102421): Introduce a prefix for internal event names.
  if (!event_type.has_value()) return false;
  switch (*event_type) {
    case HostEventType::kMemoryAllocation:
    case HostEventType::kMemoryDeallocation:
    case HostEventType::kPrefetchProduce:
    case HostEventType::kPrefetchConsume:
    case HostEventType::kParallelInterleaveProduce:
    case HostEventType::kParallelInterleaveConsume:
    case HostEventType::kParallelInterleaveInitializedInput:
    case HostEventType::kParallelMapProduce:
    case HostEventType::kParallelMapConsume:
    case HostEventType::kMapAndBatchProduce:
    case HostEventType::kMapAndBatchConsume:
    case HostEventType::kParseExampleProduce:
    case HostEventType::kParseExampleConsume:
      return true;
    default:
      return false;
  }
}

bool IsInternalStat(std::optional<int64_t> stat_type) {
  if (!stat_type.has_value()) return false;
  switch (*stat_type) {
    case StatType::kKernelDetails:
    case StatType::kProducerType:
    case StatType::kProducerId:
    case StatType::kConsumerType:
    case StatType::kConsumerId:
    case StatType::kIsRoot:
    case StatType::kFlops:
    case StatType::kProgramId:
    case StatType::kSymbolId:
      return true;
    default:
      return false;
  }
}

/*static*/ std::atomic<uint64_t> XFlow::next_flow_id_(0);

// String constants for XProf TraceMes.
const absl::string_view kMegaScaleDcnReceive =
    "MegaScale: Communication Transport Receive";
const absl::string_view kMegaScaleDcnSend =
    "MegaScale: Communication Transport Send";
const absl::string_view kMegaScaleDcnSendFinished = "MegaScale: Send Finished";
const absl::string_view kMegaScaleDcnMemAllocate = "MegaScale: Memory Allocate";
const absl::string_view kMegaScaleDcnMemCopy = "MegaScale: Memory Copy";
const absl::string_view kMegaScaleTopologyDiscovery =
    "MegaScale: Communication Topology Discovery.";
const absl::string_view kMegaScaleBarrier = "MegaScale: Barrier.";
const absl::string_view kMegaScaleHostCommand = "MegaScale: HostCommandHandle";
const absl::string_view kMegaScaleD2HTransferStart =
    "MegaScale: Device to Host Action";
const absl::string_view kMegaScaleD2HTransferFinished =
    "MegaScale: Device to Host Transfer Finished";
const absl::string_view kMegaScaleH2DTransferStart =
    "MegaScale: Host to Device Action";
const absl::string_view kMegaScaleH2DTransferFinished =
    "MegaScale: Host to Device Transfer Finished";
const absl::string_view kMegaScaleReductionStart = "MegaScale: Reduction";
const absl::string_view kMegaScaleReductionFinished =
    "MegaScale: Reduction Finished";
const absl::string_view kMegaScaleCompressionStart = "MegaScale: Compression";
const absl::string_view kMegaScaleCompressionFinished =
    "MegaScale: Compression Finished";
const absl::string_view kMegaScaleDecompressionStart =
    "MegaScale: Decompression";
const absl::string_view kMegaScaleDecompressionFinished =
    "MegaScale: Decompression Finished";
const char kXProfMetadataKey[] = "key";
const char kXProfMetadataFlow[] = "flow";
const char kXProfMetadataTransfers[] = "transfers";
const char kXProfMetadataBufferSize[] = "buffer_size";

// String constants for threadpool_listener
const absl::string_view kThreadpoolListenerRecord =
    "ThreadpoolListener::Record";
const absl::string_view kThreadpoolListenerStartRegion =
    "ThreadpoolListener::StartRegion";
const absl::string_view kThreadpoolListenerStopRegion =
    "ThreadpoolListener::StopRegion";
const absl::string_view kThreadpoolListenerRegion =
    "ThreadpoolListener::Region";
}  // namespace profiler
}  // namespace tsl
