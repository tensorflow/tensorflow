/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_group_thunk.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/convolution_reorder_thunk.h"
#include "xla/backends/gpu/runtime/convolution_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/cub_sort_thunk.h"
#include "xla/backends/gpu/runtime/cudnn_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/device_to_host_copy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_memcpy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/fft_thunk.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/host_execute_thunk.h"
#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"
#include "xla/backends/gpu/runtime/host_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/infeed_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/norm_thunk.h"
#include "xla/backends/gpu/runtime/outfeed_thunk.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/triangular_solve_thunk.h"
#include "xla/backends/gpu/runtime/wait_for_streams_thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

static std::optional<absl::string_view> GetStoredThunkTypeName(
    const ThunkProto& proto) {
  const tsl::protobuf::Descriptor* descriptor = proto.GetDescriptor();
  const tsl::protobuf::Reflection* reflection = proto.GetReflection();
  const tsl::protobuf::OneofDescriptor* impl_descriptor =
      descriptor->FindOneofByName("impl");
  const tsl::protobuf::FieldDescriptor* field_descriptor =
      reflection->GetOneofFieldDescriptor(proto, impl_descriptor);
  CHECK(impl_descriptor != nullptr);

  if (field_descriptor == nullptr) {
    // This happens when `proto` has no thunk set at all
    return std::nullopt;
  }

  return field_descriptor->name();
}

absl::StatusOr<std::unique_ptr<Thunk>> DeserializeThunkProtoImpl(
    const ThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const HloModule* absl_nullable hlo_module, absl::string_view platform_name,
    HostExecuteAsyncEventsMap& host_executable_async_events_map,
    HostSendRecvAsyncEventsMap& host_send_recv_async_events_map,
    CollectiveThunk::AsyncEventsMap& collective_async_events_map,
    const std::optional<stream_executor::KernelLoaderSpec::SymbolResolver>&
        symbol_resolver) {
  TF_ASSIGN_OR_RETURN(Thunk::ThunkInfo thunk_info,
                      Thunk::ThunkInfo::FromProto(thunk_proto.thunk_info()));
  auto deserializer = [&](const ThunkProto& thunk_proto) {
    return DeserializeThunkProtoImpl(
        thunk_proto, buffer_allocations, hlo_module, platform_name,
        host_executable_async_events_map, host_send_recv_async_events_map,
        collective_async_events_map, symbol_resolver);
  };

  switch (thunk_proto.impl_case()) {
    case ThunkProto::kSequentialThunk: {
      return SequentialThunk::FromProto(
          std::move(thunk_info), thunk_proto.sequential_thunk(), deserializer);
    }
    case ThunkProto::kCopyThunk:
      return CopyThunk::FromProto(std::move(thunk_info),
                                  thunk_proto.copy_thunk(), buffer_allocations);
    case ThunkProto::kDeviceToHostCopyThunk:
      return DeviceToHostCopyThunk::FromProto(
          std::move(thunk_info), thunk_proto.device_to_host_copy_thunk(),
          buffer_allocations);
    case ThunkProto::kHostToDeviceCopyThunk:
      return HostToDeviceCopyThunk::FromProto(
          std::move(thunk_info), thunk_proto.host_to_device_copy_thunk(),
          buffer_allocations);
    case ThunkProto::kDeviceToDeviceCopyThunk:
      return DeviceToDeviceCopyThunk::FromProto(
          std::move(thunk_info), thunk_proto.device_to_device_copy_thunk(),
          buffer_allocations);
    case ThunkProto::kWhileThunk:
      return WhileThunk::FromProto(std::move(thunk_info),
                                   thunk_proto.while_thunk(),
                                   buffer_allocations, deserializer);
    case ThunkProto::kConditionalThunk:
      return ConditionalThunk::FromProto(std::move(thunk_info),
                                         thunk_proto.conditional_thunk(),
                                         buffer_allocations, deserializer);
    case ThunkProto::kGemmThunk:
      return GemmThunk::FromProto(std::move(thunk_info),
                                  thunk_proto.gemm_thunk(), buffer_allocations);
    case ThunkProto::kWaitForStreamsThunk:
      return WaitForStreamsThunk::FromProto(
          std::move(thunk_info), thunk_proto.wait_for_streams_thunk());
    case ThunkProto::kTriangularSolveThunk:
      return TriangularSolveThunk::FromProto(
          std::move(thunk_info), thunk_proto.triangular_solve_thunk(),
          buffer_allocations);
    case ThunkProto::kKernelThunk:
      return KernelThunk::FromProto(std::move(thunk_info),
                                    thunk_proto.kernel_thunk(),
                                    buffer_allocations);
    case ThunkProto::kReplicaIdThunk:
      return ReplicaIdThunk::FromProto(std::move(thunk_info),
                                       thunk_proto.replica_id_thunk(),
                                       buffer_allocations);
    case ThunkProto::kPartitionIdThunk:
      return PartitionIdThunk::FromProto(std::move(thunk_info),
                                         thunk_proto.partition_id_thunk(),
                                         buffer_allocations);
    case ThunkProto::kCudnnThunk:
      return CuDnnThunk::FromProto(
          std::move(thunk_info), thunk_proto.cudnn_thunk(), buffer_allocations);
    case ThunkProto::kMemzeroThunk:
      return MemzeroThunk::FromProto(std::move(thunk_info),
                                     thunk_proto.memzero_thunk(),
                                     buffer_allocations);
    case ThunkProto::kInfeedThunk:
      return InfeedThunk::FromProto(std::move(thunk_info),
                                    thunk_proto.infeed_thunk(),
                                    buffer_allocations);
    case ThunkProto::kCublasLtMatmulThunk:
      return CublasLtMatmulThunk::FromProto(
          std::move(thunk_info), thunk_proto.cublas_lt_matmul_thunk(),
          buffer_allocations);
    case ThunkProto::kNormThunk:
      return NormThunk::FromProto(std::move(thunk_info),
                                  thunk_proto.norm_thunk(), buffer_allocations);
    case ThunkProto::kConvolutionThunk:
      return ConvolutionThunk::FromProto(std::move(thunk_info),
                                         thunk_proto.convolution_thunk(),
                                         buffer_allocations);
    case ThunkProto::kConvolutionReorderThunk: {
      return ConvolutionReorderThunk::FromProto(
          std::move(thunk_info), thunk_proto.convolution_reorder_thunk(),
          buffer_allocations);
    }
    case ThunkProto::kFftThunk:
      return FftThunk::FromProto(std::move(thunk_info), thunk_proto.fft_thunk(),
                                 buffer_allocations);
    case ThunkProto::kMemset32BitValueThunk:
      return Memset32BitValueThunk::FromProto(
          std::move(thunk_info), thunk_proto.memset32bit_value_thunk(),
          buffer_allocations);
    case ThunkProto::kDynamicSliceThunk: {
      auto deserializer =
          [&](const ThunkProto& thunk_proto,
              absl::Span<const BufferAllocation> custom_allocations) {
            return DeserializeThunkProtoImpl(
                thunk_proto, custom_allocations, hlo_module, platform_name,
                host_executable_async_events_map,
                host_send_recv_async_events_map, collective_async_events_map,
                symbol_resolver);
          };
      return DynamicSliceThunk::FromProto(std::move(thunk_info),
                                          thunk_proto.dynamic_slice_thunk(),
                                          buffer_allocations, deserializer);
    }
    case ThunkProto::kCustomCallThunk:
      return CustomCallThunk::FromProto(
          std::move(thunk_info), thunk_proto.custom_call_thunk(),
          buffer_allocations, hlo_module, platform_name);
    case ThunkProto::kCubSortThunk:
      return CubSortThunk::FromProto(std::move(thunk_info),
                                     thunk_proto.cub_sort_thunk(),
                                     buffer_allocations, platform_name);
    case ThunkProto::kHostExecuteStartThunk:
      return HostExecuteStartThunk::FromProto(
          std::move(thunk_info), thunk_proto.host_execute_start_thunk(),
          buffer_allocations, host_executable_async_events_map);
    case ThunkProto::kHostExecuteDoneThunk:
      return HostExecuteDoneThunk::FromProto(
          std::move(thunk_info), thunk_proto.host_execute_done_thunk(),
          buffer_allocations, host_executable_async_events_map);
    case ThunkProto::kHostSendThunk:
      return HostSendThunk::FromProto(
          std::move(thunk_info), thunk_proto.host_send_thunk(),
          buffer_allocations, host_send_recv_async_events_map);
    case ThunkProto::kHostSendDoneThunk:
      return HostSendDoneThunk::FromProto(
          std::move(thunk_info), thunk_proto.host_send_done_thunk(),
          buffer_allocations, host_send_recv_async_events_map);
    case ThunkProto::kHostRecvThunk:
      return HostRecvThunk::FromProto(
          std::move(thunk_info), thunk_proto.host_recv_thunk(),
          buffer_allocations, host_send_recv_async_events_map);
    case ThunkProto::kHostRecvDoneThunk:
      return HostRecvDoneThunk::FromProto(
          std::move(thunk_info), thunk_proto.host_recv_done_thunk(),
          buffer_allocations, host_send_recv_async_events_map);
    case ThunkProto::kOutfeedThunk:
      return OutfeedThunk::FromProto(std::move(thunk_info),
                                     thunk_proto.outfeed_thunk(),
                                     buffer_allocations);
    case ThunkProto::kCustomKernelThunk:
      return CustomKernelThunk::FromProto(std::move(thunk_info),
                                          thunk_proto.custom_kernel_thunk(),
                                          buffer_allocations, symbol_resolver);
    case ThunkProto::kCollectiveDoneThunk:
      return CollectiveDoneThunk::FromProto(std::move(thunk_info),
                                            thunk_proto.collective_done_thunk(),
                                            collective_async_events_map);
    case ThunkProto::kAllGatherStartThunk:
      return AllGatherStartThunk::FromProto(
          std::move(thunk_info), thunk_proto.all_gather_start_thunk(),
          buffer_allocations, collective_async_events_map);
    case ThunkProto::kAllReduceStartThunk:
      return AllReduceStartThunk::FromProto(
          std::move(thunk_info), thunk_proto.all_reduce_start_thunk(),
          buffer_allocations, collective_async_events_map);
    case ThunkProto::kAllToAllStartThunk:
      return AllToAllStartThunk::FromProto(
          std::move(thunk_info), thunk_proto.all_to_all_start_thunk(),
          buffer_allocations, collective_async_events_map);
    case ThunkProto::kRaggedAllToAllStartThunk:
      return RaggedAllToAllStartThunk::FromProto(
          std::move(thunk_info), thunk_proto.ragged_all_to_all_start_thunk(),
          buffer_allocations, collective_async_events_map);
    case ThunkProto::kCollectivePermuteStartThunk:
      return CollectivePermuteStartThunk::FromProto(
          std::move(thunk_info), thunk_proto.collective_permute_start_thunk(),
          buffer_allocations, collective_async_events_map);
    case ThunkProto::kSendThunk:
      return SendThunk::FromProto(std::move(thunk_info),
                                  thunk_proto.send_thunk(), buffer_allocations,
                                  collective_async_events_map);
    case ThunkProto::kRecvThunk:
      return RecvThunk::FromProto(std::move(thunk_info),
                                  thunk_proto.recv_thunk(), buffer_allocations,
                                  collective_async_events_map);
    case ThunkProto::kCollectiveBroadcastStartThunk:
      return CollectiveBroadcastStartThunk::FromProto(
          std::move(thunk_info), thunk_proto.collective_broadcast_start_thunk(),
          buffer_allocations, collective_async_events_map);
    case ThunkProto::kDynamicMemcpyThunk:
      return DynamicMemcpyThunk::FromProto(std::move(thunk_info),
                                           thunk_proto.dynamic_memcpy_thunk(),
                                           buffer_allocations);
    case ThunkProto::kCollectiveGroupThunk:
      return CollectiveGroupThunk::FromProto(
          std::move(thunk_info), thunk_proto.collective_group_thunk(),
          buffer_allocations, collective_async_events_map, deserializer);
    default:
      std::optional<absl::string_view> unsupported_thunk_type =
          GetStoredThunkTypeName(thunk_proto);

      if (!unsupported_thunk_type.has_value()) {
        return absl::InvalidArgumentError(
            "Encountered ThunkProto without an embedded thunk. This indicates "
            "that the loaded executable contains a thunk type that is not "
            "supported by this version of XLA.");
      }

      return absl::InvalidArgumentError(absl::StrFormat(
          "Thunk deserialization of thunks of type %s is not yet supported.",
          GetStoredThunkTypeName(thunk_proto).value()));
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<Thunk>> DeserializeThunkProto(
    const ThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const HloModule* absl_nullable hlo_module, absl::string_view platform_name,
    const std::optional<stream_executor::KernelLoaderSpec::SymbolResolver>&
        symbol_resolver) {
  HostExecuteAsyncEventsMap host_executable_async_events_map;
  HostSendRecvAsyncEventsMap host_send_recv_async_events_map;
  CollectiveThunk::AsyncEventsMap collective_async_events_map;
  return DeserializeThunkProtoImpl(
      thunk_proto, buffer_allocations, hlo_module, platform_name,
      host_executable_async_events_map, host_send_recv_async_events_map,
      collective_async_events_map, symbol_resolver);
}

}  // namespace xla::gpu
