/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/thunk.h"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Thunk::ExecuteParams
//===----------------------------------------------------------------------===//

Thunk::ExecuteParams Thunk::ExecuteParams::Create(
    const ServiceExecutableRunOptions& run_options,
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    se::Stream* command_buffer_trace_stream,
    CollectiveParams* collective_params, CollectiveCliques* collective_cliques,
    ExecutionStreamIdMap additional_compute_streams) {
  return ExecuteParams(&buffer_allocations, stream, command_buffer_trace_stream,
                       collective_params, collective_cliques,
                       run_options.run_options().device_to_host_stream(),
                       run_options.run_options().host_to_device_stream(),
                       run_options.run_options().send_device_memory_function(),
                       run_options.run_options().recv_device_memory_function(),
                       run_options.run_options().ffi_execution_context(),
                       additional_compute_streams,
                       run_options.run_options().gpu_executable_run_options()
                           ? run_options.run_options()
                                 .gpu_executable_run_options()
                                 ->enable_mock_collectives()
                           : false,
                       run_options.run_options().run_id().ToInt());
}

Thunk::ExecuteParams Thunk::ExecuteParams::CloneWithNewAllocations(
    const Thunk::ExecuteParams& params,
    const BufferAllocations& buffer_allocations) {
  return ExecuteParams(
      &buffer_allocations, params.stream, params.command_buffer_trace_stream,
      params.collective_params, params.collective_cliques,
      params.device_to_host_stream, params.host_to_device_stream,
      params.send_device_memory_function, params.recv_device_memory_function,
      params.ffi_execution_context, params.additional_compute_streams);
}

Thunk::ExecuteParams::ExecuteParams(
    const BufferAllocations* buffer_allocations, se::Stream* stream,
    se::Stream* command_buffer_trace_stream,
    CollectiveParams* collective_params, CollectiveCliques* collective_cliques,
    se::Stream* device_to_host_stream, se::Stream* host_to_device_stream,
    SendDeviceMemoryFunction* send_device_memory_function,
    RecvDeviceMemoryFunction* recv_device_memory_function,
    const ffi::ExecutionContext* ffi_execution_context,
    ExecutionStreamIdMap additional_compute_streams, bool mock_collectives,
    int64_t execution_id)
    : buffer_allocations(buffer_allocations),
      stream(stream),
      command_buffer_trace_stream(command_buffer_trace_stream),
      collective_params(collective_params),
      collective_cliques(collective_cliques),
      device_to_host_stream(device_to_host_stream),
      host_to_device_stream(host_to_device_stream),
      send_device_memory_function(send_device_memory_function),
      recv_device_memory_function(recv_device_memory_function),
      ffi_execution_context(ffi_execution_context),
      additional_compute_streams(additional_compute_streams),
      mock_collectives(mock_collectives),
      execution_id(execution_id) {}

//===----------------------------------------------------------------------===//

/*static*/ absl::string_view Thunk::KindToString(Thunk::Kind kind) {
#define CASE(x)  \
  case Thunk::x: \
    return #x
  switch (kind) {
    // # go/keep-sorted start
    CASE(kAllGather);
    CASE(kAllGatherDone);
    CASE(kAllGatherStart);
    CASE(kAllReduce);
    CASE(kAllReduceDone);
    CASE(kAllReduceStart);
    CASE(kAllToAll);
    CASE(kAllToAllDone);
    CASE(kAllToAllStart);
    CASE(kBuffersDebugChecksum);
    CASE(kBuffersDebugFloatCheck);
    CASE(kCollectiveBroadcast);
    CASE(kCollectiveBroadcastDone);
    CASE(kCollectiveBroadcastStart);
    CASE(kCollectiveKernel);
    CASE(kCollectiveMetadata);
    CASE(kCollectivePermute);
    CASE(kCollectivePermuteDone);
    CASE(kCollectivePermuteStart);
    CASE(kCommandBuffer);
    CASE(kConditional);
    CASE(kConvolution);
    CASE(kConvolutionReorder);
    CASE(kCopy);
    CASE(kCopyDone);
    CASE(kCuDnn);
    CASE(kCubSort);
    CASE(kCublasLtMatmul);
    CASE(kCustomCall);
    CASE(kCustomKernel);
    CASE(kDynamicSlice);
    CASE(kFft);
    CASE(kGemm);
    CASE(kGroupDone);
    CASE(kGroupStart);
    CASE(kHostExecuteDone);
    CASE(kHostExecuteStart);
    CASE(kHostRecv);
    CASE(kHostRecvDone);
    CASE(kHostSend);
    CASE(kHostSendDone);
    CASE(kInfeed);
    CASE(kKernel);
    CASE(kMemset32BitValue);
    CASE(kMemzero);
    CASE(kNorm);
    CASE(kNvshmemAllReduceDone);
    CASE(kNvshmemAllReduceStart);
    CASE(kNvshmemCollectivePermute);
    CASE(kNvshmemCollectivePermuteDone);
    CASE(kNvshmemCollectivePermuteStart);
    CASE(kNvshmemRecv);
    CASE(kNvshmemRecvDone);
    CASE(kNvshmemSend);
    CASE(kNvshmemSendDone);
    CASE(kOutfeed);
    CASE(kPartitionId);
    CASE(kRaggedAllToAll);
    CASE(kRaggedAllToAllDone);
    CASE(kRaggedAllToAllStart);
    CASE(kRecv);
    CASE(kRecvDone);
    CASE(kReduceScatter);
    CASE(kReduceScatterDone);
    CASE(kReduceScatterStart);
    CASE(kReplicaId);
    CASE(kSelectK);
    CASE(kSend);
    CASE(kSendDone);
    CASE(kSequential);
    CASE(kTriangularSolve);
    CASE(kWaitForStreams);
    CASE(kWhile);
    // # go/keep-sorted end
  }
}

/*static*/
absl::StatusOr<se::Stream*> Thunk::GetStreamForExecution(
    ExecutionStreamId stream_id, const ExecuteParams& params) {
  if (stream_id == kDefaultExecutionStreamId) {
    return params.stream;
  }
  auto iter = params.additional_compute_streams.find(stream_id);
  if (iter == params.additional_compute_streams.end()) {
    return absl::InvalidArgumentError("Invalid execution stream id.");
  }
  return iter->second;
}

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind) {
  return os << Thunk::KindToString(kind);
}

bool IsReductionCollective(Thunk::Kind kind) {
  return kind == Thunk::kAllReduce || kind == Thunk::kAllReduceStart ||
         kind == Thunk::kReduceScatter || kind == Thunk::kReduceScatterStart ||
         kind == Thunk::kNvshmemAllReduceStart;
  ;
}

absl::StatusOr<Thunk::ThunkInfo> Thunk::ThunkInfo::FromProto(
    const ThunkInfoProto& proto) {
  TF_RET_CHECK(proto.execution_stream_id() >= 0)
      << "The thunk execution stream ID must be non-negative, but got "
      << proto.execution_stream_id() << ".";
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.profile_annotation();
  thunk_info.execution_stream_id = proto.execution_stream_id();
  thunk_info.thunk_id = ThunkId(proto.thunk_id());
  return thunk_info;
}

Thunk::ThunkInfo Thunk::ThunkInfo::WithProfileAnnotation(
    const HloInstruction* instr, ThunkId thunk_id) {
  ThunkInfo thunk_info;
  thunk_info.profile_annotation = instr->name();
  thunk_info.thunk_id = thunk_id;
  auto gpu_backend_config = instr->backend_config<GpuBackendConfig>();
  if (gpu_backend_config.ok()) {
    thunk_info.execution_stream_id =
        std::max<uint64_t>(kDefaultExecutionStreamId.value(),
                           gpu_backend_config->operation_queue_id());
  }
  return thunk_info;
}

bool Thunk::IsCollective() const {
  switch (kind()) {
    // go/keep-sorted start
    case kAllGather:
    case kAllGatherDone:
    case kAllGatherStart:
    case kAllReduce:
    case kAllReduceDone:
    case kAllReduceStart:
    case kAllToAll:
    case kAllToAllDone:
    case kAllToAllStart:
    case kCollectiveBroadcast:
    case kCollectiveBroadcastDone:
    case kCollectiveBroadcastStart:
    case kCollectivePermute:
    case kCollectivePermuteDone:
    case kCollectivePermuteStart:
    case kGroupDone:
    case kGroupStart:
    case kRaggedAllToAll:
    case kRaggedAllToAllDone:
    case kRaggedAllToAllStart:
    case kRecv:
    case kRecvDone:
    case kReduceScatter:
    case kReduceScatterDone:
    case kReduceScatterStart:
    case kSend:
    case kSendDone:
      // go/keep-sorted end
      return true;
    default:
      return false;
  }
}

void Thunk::ForAllThunks(absl::FunctionRef<void(const Thunk*)> fn) const {
  fn(this);
}

void Thunk::ForAllThunksMutable(absl::FunctionRef<void(Thunk*)> fn) {
  fn(this);
}

absl::StatusOr<ThunkProto> Thunk::ToProto() const {
  return absl::UnimplementedError(absl::StrFormat(
      "Proto serialization for thunk of type %s is not implemented",
      typeid(*this).name()));
}

ThunkMetadataProto Thunk::ToMetadataProto() const {
  ThunkMetadataProto metadata_proto;
  *metadata_proto.mutable_thunk_info() = thunk_info_.ToProto();
  metadata_proto.set_thunk_kind(KindToString(kind_));
  return metadata_proto;
}

ThunkMetadataListProto GetMetadataListProtoFromThunkGraph(
    const Thunk& root_thunk) {
  ThunkMetadataListProto metadata_list_proto;
  root_thunk.ForAllThunks([&metadata_list_proto](const Thunk* thunk) {
    *metadata_list_proto.add_thunk_metadata() = thunk->ToMetadataProto();
  });
  return metadata_list_proto;
}

absl::StatusOr<GpuCollectives* absl_nonnull> Thunk::GetGpuCollectives(
    const CollectiveParams& params) {
  if (params.collectives == nullptr) {
    return Internal("Collectives API is not provided");
  }
  return params.collectives;
}

ThunkInfoProto Thunk::ThunkInfo::ToProto() const {
  ThunkInfoProto proto;
  proto.set_profile_annotation(profile_annotation);
  proto.set_execution_stream_id(execution_stream_id.value());
  proto.set_thunk_id(thunk_id.value());
  return proto;
}

}  // namespace xla::gpu
