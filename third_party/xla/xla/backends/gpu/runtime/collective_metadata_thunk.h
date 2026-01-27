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

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_multimem.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

class CollectiveMetadataThunk : public Thunk {
 public:
  struct Buffer {
    BufferAllocation::Slice slice;
    int64_t memory_space;
  };

  CollectiveMetadataThunk(ThunkInfo thunk_info,
                          CollectiveConfig collective_config,
                          std::vector<Buffer> parameters,
                          BufferAllocation::Slice result)
      : Thunk(Thunk::Kind::kCollectiveMetadata, thunk_info),
        collective_config_(std::move(collective_config)),
        parameters_(std::move(parameters)),
        result_(result) {}

  absl::Status Prepare(const PrepareParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  static CollectiveConfig GetCollectiveConfig(const HloInstruction& hlo);

  // Constructs and places the collective metadata on the device.
  // All participants should call this method to construct their local
  // metadata.
  static absl::Status ConstructCollectiveMetadata(
      const GpuCliqueKey& clique_key, RankId rank, se::Stream* stream,
      std::vector<se::DeviceAddressBase> parameters,
      std::shared_ptr<CollectiveMultimem> multimem,
      se::DeviceAddressBase destination);

  // Same as above, but returns the collective metadata so it can be used on
  // CPU.
  static absl::StatusOr<CollectiveKernelMetadata>
  ConstructAndReturnCollectiveMetadata(
      const GpuCliqueKey& clique_key, RankId rank, se::Stream* stream,
      std::vector<se::DeviceAddressBase> parameters,
      std::shared_ptr<CollectiveMultimem> multimem,
      se::DeviceAddressBase destination);

  // Calculate the device memory base for the given parameter index.
  // The size of the returned memory is num_devices pointers.
  static absl::StatusOr<se::DeviceAddressBase> GetParameterDeviceMemoryBase(
      se::DeviceAddressBase metadata, int64_t num_parameters,
      int64_t num_devices, int64_t parameter_index);

 private:
  absl::StatusOr<std::shared_ptr<CollectiveMultimem>> GetCollectiveMultimem(
      const GpuCliqueKey& clique_key, const InitializeParams& params);

  const CollectiveConfig collective_config_;
  std::vector<Buffer> parameters_;
  BufferAllocation::Slice result_;

  // This is a collective multi-mem per stream executor allocated for the thunk
  // execution in the initialize stage. In theory multiple XLA executions can
  // run concurrently, and this map would lead to a data race, however XLA
  // programs with collective operations rely on locking cliques before the
  // execution starts, and we never get concurrent executions when collective
  // operations are present in the program.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::shared_ptr<CollectiveMultimem>>
      collective_multimem_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_METADATA_THUNK_H_
