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

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/runtime/thunk_kind.pb.h"
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
    CollectiveMemory* collective_memory,
    ExecutionStreamIdMap additional_compute_streams,
    ExecutionScopedState* execution_scoped_state) {
  return ExecuteParams(&buffer_allocations, stream, command_buffer_trace_stream,
                       collective_params, collective_cliques, collective_memory,
                       run_options.run_options().device_to_host_stream(),
                       run_options.run_options().host_to_device_stream(),
                       run_options.run_options().send_device_memory_function(),
                       run_options.run_options().recv_device_memory_function(),
                       run_options.run_options().ffi_execution_context(),
                       additional_compute_streams, execution_scoped_state,
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
      params.collective_memory, params.device_to_host_stream,
      params.host_to_device_stream, params.send_device_memory_function,
      params.recv_device_memory_function, params.ffi_execution_context,
      params.additional_compute_streams);
}

Thunk::ExecuteParams::ExecuteParams(
    const BufferAllocations* buffer_allocations, se::Stream* stream,
    se::Stream* command_buffer_trace_stream,
    CollectiveParams* collective_params, CollectiveCliques* collective_cliques,
    CollectiveMemory* collective_memory, se::Stream* device_to_host_stream,
    se::Stream* host_to_device_stream,
    SendDeviceMemoryFunction* send_device_memory_function,
    RecvDeviceMemoryFunction* recv_device_memory_function,
    const ffi::ExecutionContext* ffi_execution_context,
    ExecutionStreamIdMap additional_compute_streams,
    ExecutionScopedState* execution_scoped_state, bool mock_collectives,
    int64_t execution_id)
    : buffer_allocations(buffer_allocations),
      stream(stream),
      command_buffer_trace_stream(command_buffer_trace_stream),
      collective_params(collective_params),
      collective_cliques(collective_cliques),
      collective_memory(collective_memory),
      device_to_host_stream(device_to_host_stream),
      host_to_device_stream(host_to_device_stream),
      send_device_memory_function(send_device_memory_function),
      recv_device_memory_function(recv_device_memory_function),
      ffi_execution_context(ffi_execution_context),
      additional_compute_streams(additional_compute_streams),
      execution_scoped_state(execution_scoped_state),
      mock_collectives(mock_collectives),
      execution_id(execution_id) {}

//===----------------------------------------------------------------------===//

ThunkKindProto Thunk::KindToProto(Kind kind) {
  switch (kind) {
    case kAllGather:
      return THUNK_KIND_ALL_GATHER;
    case kAllGatherDone:
      return THUNK_KIND_ALL_GATHER_DONE;
    case kAllGatherStart:
      return THUNK_KIND_ALL_GATHER_START;
    case kAllReduce:
      return THUNK_KIND_ALL_REDUCE;
    case kAllReduceDone:
      return THUNK_KIND_ALL_REDUCE_DONE;
    case kAllReduceStart:
      return THUNK_KIND_ALL_REDUCE_START;
    case kAllToAll:
      return THUNK_KIND_ALL_TO_ALL;
    case kAllToAllDone:
      return THUNK_KIND_ALL_TO_ALL_DONE;
    case kAllToAllStart:
      return THUNK_KIND_ALL_TO_ALL_START;
    case kBuffersDebugChecksum:
      return THUNK_KIND_BUFFERS_DEBUG_CHECKSUM;
    case kBuffersDebugFloatCheck:
      return THUNK_KIND_BUFFERS_DEBUG_FLOAT_CHECK;
    case kCollectiveBroadcast:
      return THUNK_KIND_COLLECTIVE_BROADCAST;
    case kCollectiveBroadcastDone:
      return THUNK_KIND_COLLECTIVE_BROADCAST_DONE;
    case kCollectiveBroadcastStart:
      return THUNK_KIND_COLLECTIVE_BROADCAST_START;
    case kCollectiveKernel:
      return THUNK_KIND_COLLECTIVE_KERNEL;
    case kCollectiveMetadata:
      return THUNK_KIND_COLLECTIVE_METADATA;
    case kCollectivePermute:
      return THUNK_KIND_COLLECTIVE_PERMUTE;
    case kCollectivePermuteDone:
      return THUNK_KIND_COLLECTIVE_PERMUTE_DONE;
    case kCollectivePermuteStart:
      return THUNK_KIND_COLLECTIVE_PERMUTE_START;
    case kCommandBuffer:
      return THUNK_KIND_COMMAND_BUFFER;
    case kConditional:
      return THUNK_KIND_CONDITIONAL;
    case kConvolution:
      return THUNK_KIND_CONVOLUTION;
    case kConvolutionReorder:
      return THUNK_KIND_CONVOLUTION_REORDER;
    case kCopy:
      return THUNK_KIND_COPY;
    case kCopyDone:
      return THUNK_KIND_COPY_DONE;
    case kCuDnn:
      return THUNK_KIND_CU_DNN;
    case kCubSort:
      return THUNK_KIND_CUB_SORT;
    case kCublasLtMatmul:
      return THUNK_KIND_CUBLAS_LT_MATMUL;
    case kCustomCall:
      return THUNK_KIND_CUSTOM_CALL;
    case kCustomKernel:
      return THUNK_KIND_CUSTOM_KERNEL;
    case kDynamicSlice:
      return THUNK_KIND_DYNAMIC_SLICE;
    case kFft:
      return THUNK_KIND_FFT;
    case kGemm:
      return THUNK_KIND_GEMM;
    case kGroupDone:
      return THUNK_KIND_GROUP_DONE;
    case kGroupStart:
      return THUNK_KIND_GROUP_START;
    case kHostExecuteDone:
      return THUNK_KIND_HOST_EXECUTE_DONE;
    case kHostExecuteStart:
      return THUNK_KIND_HOST_EXECUTE_START;
    case kHostRecv:
      return THUNK_KIND_HOST_RECV;
    case kHostRecvDone:
      return THUNK_KIND_HOST_RECV_DONE;
    case kHostSend:
      return THUNK_KIND_HOST_SEND;
    case kHostSendDone:
      return THUNK_KIND_HOST_SEND_DONE;
    case kInfeed:
      return THUNK_KIND_INFEED;
    case kKernel:
      return THUNK_KIND_KERNEL;
    case kMemset32BitValue:
      return THUNK_KIND_MEMSET32_BIT_VALUE;
    case kMemzero:
      return THUNK_KIND_MEMZERO;
    case kNorm:
      return THUNK_KIND_NORM;
    case kNvshmemAllReduceDone:
      return THUNK_KIND_NVSHMEM_ALL_REDUCE_DONE;
    case kNvshmemAllReduceStart:
      return THUNK_KIND_NVSHMEM_ALL_REDUCE_START;
    case kNvshmemCollectivePermute:
      return THUNK_KIND_NVSHMEM_COLLECTIVE_PERMUTE;
    case kNvshmemCollectivePermuteDone:
      return THUNK_KIND_NVSHMEM_COLLECTIVE_PERMUTE_DONE;
    case kNvshmemCollectivePermuteStart:
      return THUNK_KIND_NVSHMEM_COLLECTIVE_PERMUTE_START;
    case kNvshmemRecv:
      return THUNK_KIND_NVSHMEM_RECV;
    case kNvshmemRecvDone:
      return THUNK_KIND_NVSHMEM_RECV_DONE;
    case kNvshmemSend:
      return THUNK_KIND_NVSHMEM_SEND;
    case kNvshmemSendDone:
      return THUNK_KIND_NVSHMEM_SEND_DONE;
    case kOutfeed:
      return THUNK_KIND_OUTFEED;
    case kPartitionId:
      return THUNK_KIND_PARTITION_ID;
    case kRaggedAllToAll:
      return THUNK_KIND_RAGGED_ALL_TO_ALL;
    case kRaggedAllToAllDone:
      return THUNK_KIND_RAGGED_ALL_TO_ALL_DONE;
    case kRaggedAllToAllStart:
      return THUNK_KIND_RAGGED_ALL_TO_ALL_START;
    case kRecv:
      return THUNK_KIND_RECV;
    case kRecvDone:
      return THUNK_KIND_RECV_DONE;
    case kReduceScatter:
      return THUNK_KIND_REDUCE_SCATTER;
    case kReduceScatterDone:
      return THUNK_KIND_REDUCE_SCATTER_DONE;
    case kReduceScatterStart:
      return THUNK_KIND_REDUCE_SCATTER_START;
    case kReplicaId:
      return THUNK_KIND_REPLICA_ID;
    case kSelectK:
      return THUNK_KIND_SELECT_K;
    case kSend:
      return THUNK_KIND_SEND;
    case kSendDone:
      return THUNK_KIND_SEND_DONE;
    case kSequential:
      return THUNK_KIND_SEQUENTIAL;
    case kTriangularSolve:
      return THUNK_KIND_TRIANGULAR_SOLVE;
    case kWaitForStreams:
      return THUNK_KIND_WAIT_FOR_STREAMS;
    case kWhile:
      return THUNK_KIND_WHILE;
  };
}

absl::StatusOr<Thunk::Kind> Thunk::KindFromProto(ThunkKindProto kind) {
  switch (kind) {
    case THUNK_KIND_ALL_GATHER:
      return kAllGather;
    case THUNK_KIND_ALL_GATHER_DONE:
      return kAllGatherDone;
    case THUNK_KIND_ALL_GATHER_START:
      return kAllGatherStart;
    case THUNK_KIND_ALL_REDUCE:
      return kAllReduce;
    case THUNK_KIND_ALL_REDUCE_DONE:
      return kAllReduceDone;
    case THUNK_KIND_ALL_REDUCE_START:
      return kAllReduceStart;
    case THUNK_KIND_ALL_TO_ALL:
      return kAllToAll;
    case THUNK_KIND_ALL_TO_ALL_DONE:
      return kAllToAllDone;
    case THUNK_KIND_ALL_TO_ALL_START:
      return kAllToAllStart;
    case THUNK_KIND_BUFFERS_DEBUG_CHECKSUM:
      return kBuffersDebugChecksum;
    case THUNK_KIND_BUFFERS_DEBUG_FLOAT_CHECK:
      return kBuffersDebugFloatCheck;
    case THUNK_KIND_COLLECTIVE_BROADCAST:
      return kCollectiveBroadcast;
    case THUNK_KIND_COLLECTIVE_BROADCAST_DONE:
      return kCollectiveBroadcastDone;
    case THUNK_KIND_COLLECTIVE_BROADCAST_START:
      return kCollectiveBroadcastStart;
    case THUNK_KIND_COLLECTIVE_KERNEL:
      return kCollectiveKernel;
    case THUNK_KIND_COLLECTIVE_METADATA:
      return kCollectiveMetadata;
    case THUNK_KIND_COLLECTIVE_PERMUTE:
      return kCollectivePermute;
    case THUNK_KIND_COLLECTIVE_PERMUTE_DONE:
      return kCollectivePermuteDone;
    case THUNK_KIND_COLLECTIVE_PERMUTE_START:
      return kCollectivePermuteStart;
    case THUNK_KIND_COMMAND_BUFFER:
      return kCommandBuffer;
    case THUNK_KIND_CONDITIONAL:
      return kConditional;
    case THUNK_KIND_CONVOLUTION:
      return kConvolution;
    case THUNK_KIND_CONVOLUTION_REORDER:
      return kConvolutionReorder;
    case THUNK_KIND_COPY:
      return kCopy;
    case THUNK_KIND_COPY_DONE:
      return kCopyDone;
    case THUNK_KIND_CU_DNN:
      return kCuDnn;
    case THUNK_KIND_CUB_SORT:
      return kCubSort;
    case THUNK_KIND_CUBLAS_LT_MATMUL:
      return kCublasLtMatmul;
    case THUNK_KIND_CUSTOM_CALL:
      return kCustomCall;
    case THUNK_KIND_CUSTOM_KERNEL:
      return kCustomKernel;
    case THUNK_KIND_DYNAMIC_SLICE:
      return kDynamicSlice;
    case THUNK_KIND_FFT:
      return kFft;
    case THUNK_KIND_GEMM:
      return kGemm;
    case THUNK_KIND_GROUP_DONE:
      return kGroupDone;
    case THUNK_KIND_GROUP_START:
      return kGroupStart;
    case THUNK_KIND_HOST_EXECUTE_DONE:
      return kHostExecuteDone;
    case THUNK_KIND_HOST_EXECUTE_START:
      return kHostExecuteStart;
    case THUNK_KIND_HOST_RECV:
      return kHostRecv;
    case THUNK_KIND_HOST_RECV_DONE:
      return kHostRecvDone;
    case THUNK_KIND_HOST_SEND:
      return kHostSend;
    case THUNK_KIND_HOST_SEND_DONE:
      return kHostSendDone;
    case THUNK_KIND_INFEED:
      return kInfeed;
    case THUNK_KIND_KERNEL:
      return kKernel;
    case THUNK_KIND_MEMSET32_BIT_VALUE:
      return kMemset32BitValue;
    case THUNK_KIND_MEMZERO:
      return kMemzero;
    case THUNK_KIND_NORM:
      return kNorm;
    case THUNK_KIND_NVSHMEM_ALL_REDUCE_DONE:
      return kNvshmemAllReduceDone;
    case THUNK_KIND_NVSHMEM_ALL_REDUCE_START:
      return kNvshmemAllReduceStart;
    case THUNK_KIND_NVSHMEM_COLLECTIVE_PERMUTE:
      return kNvshmemCollectivePermute;
    case THUNK_KIND_NVSHMEM_COLLECTIVE_PERMUTE_DONE:
      return kNvshmemCollectivePermuteDone;
    case THUNK_KIND_NVSHMEM_COLLECTIVE_PERMUTE_START:
      return kNvshmemCollectivePermuteStart;
    case THUNK_KIND_NVSHMEM_RECV:
      return kNvshmemRecv;
    case THUNK_KIND_NVSHMEM_RECV_DONE:
      return kNvshmemRecvDone;
    case THUNK_KIND_NVSHMEM_SEND:
      return kNvshmemSend;
    case THUNK_KIND_NVSHMEM_SEND_DONE:
      return kNvshmemSendDone;
    case THUNK_KIND_OUTFEED:
      return kOutfeed;
    case THUNK_KIND_PARTITION_ID:
      return kPartitionId;
    case THUNK_KIND_RAGGED_ALL_TO_ALL:
      return kRaggedAllToAll;
    case THUNK_KIND_RAGGED_ALL_TO_ALL_DONE:
      return kRaggedAllToAllDone;
    case THUNK_KIND_RAGGED_ALL_TO_ALL_START:
      return kRaggedAllToAllStart;
    case THUNK_KIND_RECV:
      return kRecv;
    case THUNK_KIND_RECV_DONE:
      return kRecvDone;
    case THUNK_KIND_REDUCE_SCATTER:
      return kReduceScatter;
    case THUNK_KIND_REDUCE_SCATTER_DONE:
      return kReduceScatterDone;
    case THUNK_KIND_REDUCE_SCATTER_START:
      return kReduceScatterStart;
    case THUNK_KIND_REPLICA_ID:
      return kReplicaId;
    case THUNK_KIND_SELECT_K:
      return kSelectK;
    case THUNK_KIND_SEND:
      return kSend;
    case THUNK_KIND_SEND_DONE:
      return kSendDone;
    case THUNK_KIND_SEQUENTIAL:
      return kSequential;
    case THUNK_KIND_TRIANGULAR_SOLVE:
      return kTriangularSolve;
    case THUNK_KIND_WAIT_FOR_STREAMS:
      return kWaitForStreams;
    case THUNK_KIND_WHILE:
      return kWhile;
    default:
      return absl::InternalError(absl::StrCat("Unknown ThunkKindProto:", kind));
  };
}

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

absl::StatusOr<se::Stream*> Thunk::GetStreamForExecution(
    ExecutionStreamId stream_id, const ExecuteParams& params) {
  if (stream_id == kDefaultExecutionStreamId) {
    return params.stream;
  }
  auto iter = params.additional_compute_streams.find(stream_id);
  if (iter == params.additional_compute_streams.end()) {
    return InvalidArgument(
        "Invalid execution stream id: %v; available streams: [%s]", stream_id,
        absl::StrJoin(params.additional_compute_streams, ",",
                      [](std::string* out, auto pair) {
                        absl::StrAppendFormat(out, "%v", pair.first);
                      }));
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
  root_thunk.Walk([&metadata_list_proto](const Thunk* thunk) {
    *metadata_list_proto.add_thunk_metadata() = thunk->ToMetadataProto();
  });
  return metadata_list_proto;
}

ThunkInfoProto Thunk::ThunkInfo::ToProto() const {
  ThunkInfoProto proto;
  proto.set_profile_annotation(profile_annotation);
  proto.set_execution_stream_id(execution_stream_id.value());
  proto.set_thunk_id(thunk_id.value());
  return proto;
}

}  // namespace xla::gpu
