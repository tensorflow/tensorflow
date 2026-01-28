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

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_clique_rendezvous.h"
#include "xla/backends/gpu/runtime/collective_multimem.h"
#include "xla/backends/gpu/runtime/collective_multimem_registry.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

// TODO(460077850): Support global device ids and channel id.
CollectiveConfig CollectiveMetadataThunk::GetCollectiveConfig(
    const HloInstruction& hlo) {
  CollectiveConfig config;
  config.operand_element_type.reserve(hlo.operands().size());
  for (const HloInstruction* operand : hlo.operands()) {
    config.operand_element_type.push_back(operand->shape().element_type());
  }

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

  config.group_mode = CollectiveOpGroupMode::
      COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION;

  return config;
}

absl::StatusOr<CollectiveKernelMetadata>
CollectiveMetadataThunk::ConstructAndReturnCollectiveMetadata(
    const GpuCliqueKey& clique_key, RankId rank, se::Stream* stream,
    std::vector<se::DeviceAddressBase> parameters,
    std::shared_ptr<CollectiveMultimem> multimem,
    se::DeviceAddressBase destination) {
  size_t num_parameters = parameters.size();

  using DeviceParameters = std::vector<se::DeviceAddressBase>;

  // Exchange device parameters with all ranks in the clique.
  TF_ASSIGN_OR_RETURN(
      auto device_parameters,
      GpuCliqueRendezvous::Join(clique_key, rank, std::move(parameters)));

  // Collect pointers to device buffers from all participating ranks.
  std::vector<void*> param_to_peers_ptrs;
  param_to_peers_ptrs.reserve(num_parameters * clique_key.num_devices());

  absl::flat_hash_map<int, std::vector<se::DeviceAddressBase>>
      peer_to_parameters(clique_key.num_devices());

  for (auto peer = RankId(0); peer < RankId(clique_key.num_devices()); ++peer) {
    TF_ASSIGN_OR_RETURN(const DeviceParameters& peer_parameters,
                        device_parameters->at<DeviceParameters>(peer));
    peer_to_parameters[peer.value()] = std::move(peer_parameters);
  }

  for (int parameter = 0; parameter < num_parameters; ++parameter) {
    for (int peer = 0; peer < clique_key.num_devices(); ++peer) {
      param_to_peers_ptrs.push_back(
          peer_to_parameters[peer][parameter].opaque());
    }
  }

  // Check that all participants have the same number of parameters.
  TF_RET_CHECK(param_to_peers_ptrs.size() ==
               num_parameters * clique_key.num_local_participants());

  const int64_t param_to_peers_ptrs_size =
      param_to_peers_ptrs.size() * sizeof(void*);
  const size_t metadata_on_device_size =
      offsetof(CollectiveKernelMetadata, param_to_peers_host);
  se::DeviceAddressBase param_to_peers_ptrs_buffer = destination.GetByteSlice(
      metadata_on_device_size, param_to_peers_ptrs_size);

  CollectiveKernelMetadata metadata;
  metadata.rank = rank.value();
  metadata.multicast_buffer_ptr =
      multimem ? multimem->mapped_ptr(rank) : nullptr;
  metadata.param_to_peers =
      reinterpret_cast<void**>(param_to_peers_ptrs_buffer.opaque());

  TF_RETURN_IF_ERROR(
      stream->Memcpy(&destination, &metadata, metadata_on_device_size));
  TF_RETURN_IF_ERROR(stream->Memcpy(&param_to_peers_ptrs_buffer,
                                    param_to_peers_ptrs.data(),
                                    param_to_peers_ptrs_size));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  metadata.param_to_peers_host = std::move(param_to_peers_ptrs);
  return metadata;
}

absl::Status CollectiveMetadataThunk::ConstructCollectiveMetadata(
    const GpuCliqueKey& clique_key, RankId rank, se::Stream* stream,
    std::vector<se::DeviceAddressBase> parameters,
    std::shared_ptr<CollectiveMultimem> multimem,
    se::DeviceAddressBase destination) {
  TF_ASSIGN_OR_RETURN(CollectiveKernelMetadata _,
                      ConstructAndReturnCollectiveMetadata(
                          clique_key, rank, stream, std::move(parameters),
                          multimem, destination));
  return absl::OkStatus();
}

/* static */ absl::StatusOr<se::DeviceAddressBase>
CollectiveMetadataThunk::GetParameterDeviceMemoryBase(
    const se::DeviceAddressBase metadata, const int64_t num_parameters,
    const int64_t num_devices, const int64_t parameter_index) {
  TF_RET_CHECK(parameter_index >= 0 && parameter_index < num_parameters)
      << "Parameter index " << parameter_index << " is out of bounds [0, "
      << num_parameters << ")";
  // The pointer table is a flattened array laid out in parameter major order.
  // P0R0 P0R1 ... P0Rn P1R0
  // P1R1 ... P1Rn ... PnRn
  // Where Pn is the parameter index and Rn is the rank.
  se::DeviceAddressBase ptr_table_base = metadata.GetByteSlice(
      sizeof(CollectiveKernelMetadata),
      /*size_bytes=*/num_parameters * num_devices * sizeof(void*));
  return ptr_table_base.GetByteSlice(
      (parameter_index * num_devices) * sizeof(void*),
      /*size_bytes=*/num_devices * sizeof(void*));
}

absl::Status CollectiveMetadataThunk::Prepare(const PrepareParams& params) {
  // We currently support only a single memory space for multimem parameters.
  // So we just pick the first one here.
  auto fast_memory_parameter =
      absl::c_find_if(parameters_, [](const Buffer& parameter) {
        return parameter.memory_space == xla::Layout::kGenericFastMemorySpace;
      });
  if (fast_memory_parameter == parameters_.end()) {
    return absl::OkStatus();
  }

  se::DeviceAddressBase memory_range;
  TF_ASSIGN_OR_RETURN(memory_range,
                      params.executor->GetMemoryRange(
                          params.buffer_allocations->GetDeviceAddress(
                              fast_memory_parameter->slice)));

  // Since there is no parameter in the collective memory space, we don't need
  // to set up the collective multimem.
  if (memory_range.is_null()) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(
      const GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_,
                                /*include_participant_groups=*/false));
  params.multimem_registry->Register({clique_key, /*map_to=*/memory_range});
  return absl::OkStatus();
}

absl::Status CollectiveMetadataThunk::Initialize(
    const InitializeParams& params) {
  TF_ASSIGN_OR_RETURN(
      const GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_,
                                /*include_participant_groups=*/false));
  const int64_t num_ranks = clique_key.num_devices();
  TF_RET_CHECK(result_.size() ==
               sizeof(CollectiveKernelMetadata) +
                   num_ranks * parameters_.size() * sizeof(uint64_t));

  std::vector<se::DeviceAddressBase> parameters;
  parameters.reserve(parameters_.size());
  for (const Buffer& parameter : parameters_) {
    parameters.push_back(
        params.buffer_allocations->GetDeviceAddress(parameter.slice));
  }
  se::DeviceAddressBase result_ptr =
      params.buffer_allocations->GetDeviceAddress(result_);

  GlobalDeviceId global_device_id = params.collective_params->global_device_id;

  TF_ASSIGN_OR_RETURN(auto multimem, GetCollectiveMultimem(clique_key, params));

  std::optional<RankId> rank = clique_key.rank(global_device_id);
  TF_RET_CHECK(rank.has_value());
  return ConstructCollectiveMetadata(clique_key, *rank, params.stream,
                                     std::move(parameters), std::move(multimem),
                                     result_ptr);
}

absl::Status CollectiveMetadataThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<CollectiveMultimem>>
CollectiveMetadataThunk::GetCollectiveMultimem(const GpuCliqueKey& clique_key,
                                               const InitializeParams& params) {
  se::DeviceAddressBase memory_range;
  for (const Buffer& parameter : parameters_) {
    if (parameter.memory_space == xla::Layout::kGenericFastMemorySpace) {
      TF_ASSIGN_OR_RETURN(
          memory_range,
          params.executor->GetMemoryRange(
              params.buffer_allocations->GetDeviceAddress(parameter.slice)));
      break;
    }
  }

  // Since there is no parameter in the collective memory space, we don't need
  // to set up the collective multimem.
  if (memory_range.is_null()) {
    return nullptr;
  }

  const MultimemRequest request{clique_key, memory_range};
  TF_ASSIGN_OR_RETURN(std::shared_ptr<CollectiveMultimem> collective_multimem,
                      params.multicast_memory_registry->Get(request));
  absl::MutexLock lock(mutex_);
  return (collective_multimem_[params.executor] =
              std::move(collective_multimem));
}

}  // namespace gpu
}  // namespace xla
