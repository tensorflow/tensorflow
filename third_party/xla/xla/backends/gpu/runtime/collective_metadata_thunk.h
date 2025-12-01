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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_METADATA_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_METADATA_THUNK_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {
class CollectiveMetadataThunk : public Thunk {
 public:
  struct Buffer {
    BufferAllocation::Slice slice;
    int64_t memory_space;
  };

  class MultimemAddressSpaceProvider {
   public:
    // Initializes and multimem memory. Each thunk participant should call this
    // method once. Multimem should be setup before usage when multimem strategy
    // is selected.
    absl::StatusOr<void*> SetupMultimemAddressSpace(
        const GpuCliqueKey& clique_key,
        const se::StreamExecutor* stream_executor,
        se::DeviceMemoryBase mapped_memory);

   private:
    absl::flat_hash_map<
        int,
        std::unique_ptr<stream_executor::gpu::GpuExecutor::MulticastMemory>>
        first_device_to_multicast_memory_;
  };

  explicit CollectiveMetadataThunk(ThunkInfo thunk_info,
                                   CollectiveConfig collective_config,
                                   std::vector<Buffer> parameters,
                                   BufferAllocation::Slice result)
      : Thunk(Thunk::Kind::kCollectiveMetadata, thunk_info),
        collective_config_(std::move(collective_config)),
        parameters_(std::move(parameters)),
        result_(result) {}
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  static CollectiveConfig GetCollectiveConfig(const HloInstruction& hlo);

  // Constructs and places the collective metadata on the device.
  // All participants should call this method to construct their local
  // metadata.
  static absl::Status ConstructCollectiveMetadata(
      std::vector<se::DeviceMemoryBase> parameters, se::Stream* stream,
      const GpuCliqueKey& clique_key, void* multimem_address_space,
      int device_ordinal, se::DeviceMemoryBase destination);

  // Calculate the device memory base for the given parameter index.
  // The size of the returned memory is num_devices pointers.
  static absl::StatusOr<se::DeviceMemoryBase> GetParameterDeviceMemoryBase(
      se::DeviceMemoryBase metadata, int64_t num_parameters,
      int64_t num_devices, int64_t parameter_index);

  absl::StatusOr<void*> SetupMultimem(const GpuCliqueKey& clique_key,
                                      const InitializeParams& params);

 private:
  const CollectiveConfig collective_config_;
  std::vector<Buffer> parameters_;
  MultimemAddressSpaceProvider address_space_provider_;
  BufferAllocation::Slice result_;
};
}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_METADATA_THUNK_H_
