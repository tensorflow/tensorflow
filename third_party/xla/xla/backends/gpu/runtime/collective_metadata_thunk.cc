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

#include "xla/backends/gpu/runtime/collective_metadata_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/rendezvous.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

// TODO(460077850): Support global device ids and channel id.
CollectiveConfig CollectiveMetadataThunk::GetCollectiveConfig(
    const HloInstruction& hlo) {
  CollectiveConfig config;
  config.operand_count = hlo.operands().size();
  config.operand_element_type.reserve(config.operand_count);
  for (int i = 0; i < config.operand_count; i++) {
    config.operand_element_type.push_back(
        hlo.operand(i)->shape().element_type());
  }

  config.collective_op_kind = RendezvousKey::kCrossReplica;
  config.op_id = static_cast<int64_t>(hlo.GetModule()->unique_id());
  if (hlo.has_backend_config()) {
    xla::gpu::GpuBackendConfig backend_config =
        hlo.backend_config<GpuBackendConfig>().value_or(GpuBackendConfig());
    if (backend_config.has_collective_metadata_backend_config()) {
      ::google::protobuf::RepeatedPtrField<ReplicaGroup> replica_groups =
          backend_config.collective_metadata_backend_config()
              .collective_devices()
              .replica_groups();
      config.replica_groups = std::vector<ReplicaGroup>(replica_groups.begin(),
                                                        replica_groups.end());
    }
  }

  config.group_mode =
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;

  return config;
}

struct CollectiveMetadataRendezvousValue {
  RankId rank;
  std::vector<se::DeviceMemoryBase> parameters;

  bool operator<(const CollectiveMetadataRendezvousValue& other) const {
    return rank < other.rank;
  }
};

absl::Status CollectiveMetadataThunk::ConstructCollectiveMetadata(
    std::vector<se::DeviceMemoryBase> parameters, se::Stream* stream,
    const GpuCliqueKey& clique_key, void* multimem_address_space,
    int device_ordinal, se::DeviceMemoryBase destination) {
  auto rendezvous_fn =
      [](absl::Span<const CollectiveMetadataRendezvousValue* const> values) {
        std::vector<CollectiveMetadataRendezvousValue> values_copy;
        for (const auto& value : values) {
          values_copy.push_back(*value);
        }
        // Sort to make sure that values are in the same order as the
        // devices are ordered in the communicator.
        absl::c_sort(values_copy);
        return values_copy;
      };

  std::string start_rendezvous_key =
      absl::StrFormat("[%d] Initializing collective metadata for clique %s",
                      device_ordinal, clique_key.ToString());

  CollectiveMetadataRendezvousValue rendezvous_value;
  rendezvous_value.rank = device_ordinal;
  rendezvous_value.parameters = std::move(parameters);

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<std::vector<CollectiveMetadataRendezvousValue>>
          rendezvous_values,
      Rendezvous<std::vector<CollectiveMetadataRendezvousValue>>(
          /*name=*/start_rendezvous_key, /*key=*/clique_key,
          /*value=*/rendezvous_value, /*num_threads=*/clique_key.num_devices(),
          rendezvous_fn));

  CollectiveKernelMetadata metadata;
  metadata.rank = clique_key.rank(GlobalDeviceId(device_ordinal))
                      .value_or(RankId(-1))
                      .value();
  if (metadata.rank == -1) {
    return absl::InternalError(
        absl::StrFormat("Device %d not found in clique %s", device_ordinal,
                        clique_key.ToString()));
  }
  metadata.multicast_buffer_ptr = multimem_address_space;
  TF_RET_CHECK(rendezvous_values->size() > 0)
      << "Not enough devices in the clique.";
  const size_t num_parameters = (*rendezvous_values)[0].parameters.size();
  for (const auto& value : *rendezvous_values) {
    TF_RET_CHECK(value.parameters.size() == num_parameters);
  }

  std::vector<void*> param_to_peers_ptrs;
  param_to_peers_ptrs.reserve(rendezvous_values->size() * num_parameters);
  for (int param = 0; param < num_parameters; ++param) {
    for (int peer = 0; peer < clique_key.num_devices(); ++peer) {
      param_to_peers_ptrs.push_back(
          (*rendezvous_values)[peer].parameters[param].opaque());
    }
  }

  const int param_to_peers_ptrs_size =
      param_to_peers_ptrs.size() * sizeof(void*);
  se::DeviceMemoryBase param_to_peers_ptrs_buffer = destination.GetByteSlice(
      sizeof(CollectiveKernelMetadata), param_to_peers_ptrs_size);

  metadata.param_to_peers =
      reinterpret_cast<void**>(param_to_peers_ptrs_buffer.opaque());

  TF_RETURN_IF_ERROR(stream->Memcpy(&destination, &metadata,
                                    sizeof(CollectiveKernelMetadata)));
  TF_RETURN_IF_ERROR(stream->Memcpy(&param_to_peers_ptrs_buffer,
                                    param_to_peers_ptrs.data(),
                                    param_to_peers_ptrs_size));
  return stream->BlockHostUntilDone();
}

absl::Status CollectiveMetadataThunk::Initialize(
    const InitializeParams& params) {
  TF_ASSIGN_OR_RETURN(
      const GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_,
                                /*use_nccl=*/false));
  const int64_t num_ranks = clique_key.num_devices();
  TF_RET_CHECK(result_.size() ==
               sizeof(CollectiveKernelMetadata) +
                   num_ranks * parameters_.size() * sizeof(uint64_t));

  std::vector<se::DeviceMemoryBase> parameters;
  parameters.reserve(parameters_.size());
  for (const CollectiveMetadataThunk::Buffer& parameter : parameters_) {
    parameters.push_back(
        params.buffer_allocations->GetDeviceAddress(parameter.slice));
  }
  se::DeviceMemoryBase result_ptr =
      params.buffer_allocations->GetDeviceAddress(result_);

  TF_ASSIGN_OR_RETURN(void* multimem_address_space,
                      SetupMultimem(clique_key, params));
  return ConstructCollectiveMetadata(
      std::move(parameters), params.stream, clique_key, multimem_address_space,
      params.executor->device_ordinal(), result_ptr);
}

absl::Status CollectiveMetadataThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  return absl::OkStatus();
}

absl::StatusOr<void*> CollectiveMetadataThunk::SetupMultimem(
    const GpuCliqueKey& clique_key, const InitializeParams& params) {
  se::DeviceMemoryBase memory_range;
  for (const CollectiveMetadataThunk::Buffer& parameter : parameters_) {
    if (parameter.memory_space == xla::Layout::kGenericFastMemorySpace) {
      TF_ASSIGN_OR_RETURN(
          memory_range,
          params.executor->GetMemoryRange(
              params.buffer_allocations->GetDeviceAddress(parameter.slice)));
      break;
    }
  }

  // Since there is no parameter in the collective memory space, we don't need
  // to set up the multicast memory.
  if (memory_range.is_null()) {
    return nullptr;
  }
  return address_space_provider_.SetupMultimemAddressSpace(
      clique_key, params.executor, memory_range);
}

absl::Status Barrier(int device_number, const GpuCliqueKey& clique_key) {
  std::string start_rendezvous_key = absl::StrFormat(
      "Barrier for device %d, "
      "clique %s",
      device_number, clique_key.ToString());
  return Rendezvous(
      /*name=*/
      start_rendezvous_key, /*key=*/clique_key,
      /*num_threads=*/clique_key.num_local_participants());
}

absl::StatusOr<void*> CollectiveMetadataThunk::MultimemAddressSpaceProvider::
    SetupMultimemAddressSpace(const GpuCliqueKey& clique_key,
                              const se::StreamExecutor* stream_executor,
                              se::DeviceMemoryBase mapped_memory) {
  const auto* gpu_executor =
      dynamic_cast<const stream_executor::gpu::GpuExecutor*>(stream_executor);
  if (gpu_executor == nullptr) {
    return absl::UnimplementedError("Multicast is not supported on device.");
  }
  int device_number = gpu_executor->device_ordinal();
  TF_RET_CHECK(clique_key.num_local_participants() > 0)
      << "Number of local participants must be greater than 0.";
  int64_t first_device = clique_key.devices()[0].value();

  if (device_number == first_device) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<stream_executor::gpu::GpuExecutor::MulticastMemory>
            multicast_memory,
        gpu_executor->CreateMulticastMemory(
            mapped_memory.size(), clique_key.num_local_participants()));
    first_device_to_multicast_memory_.emplace(device_number,
                                              std::move(multicast_memory));
  }

  // Wait for all devices to create the multicast object.
  TF_RETURN_IF_ERROR(Barrier(device_number, clique_key));

  TF_RET_CHECK(first_device_to_multicast_memory_.contains(first_device))
      << "Multicast memory is not created for device " << first_device;
  // Add current devices to the multicast object.
  TF_RETURN_IF_ERROR(
      first_device_to_multicast_memory_[first_device]->SubscribeDevice(
          device_number));

  // Wait for all devices to register the multicast object.
  TF_RETURN_IF_ERROR(Barrier(device_number, clique_key));

  return first_device_to_multicast_memory_[first_device]->MapMemory(
      mapped_memory, gpu_executor);
};

}  // namespace gpu
}  // namespace xla
