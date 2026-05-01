/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct RaggedAllToAllConfig {
  CollectiveConfig config;
  int64_t num_total_updates = 1;
  int64_t num_input_rows = 1;
  int64_t num_row_elements = 1;

  // Whether the one-shot kernel is enabled. If true, the thunk will use the
  // one-shot kernel when possible.
  bool one_shot_kernel_enabled = false;

  // If true, the thunk will use the MultiGpuBarrierWithNcclKernel in the
  // one-shot kernel for multi-host synchronization. Works when devices are on
  // multiple hosts connected via a fast interconnect (e.g., MNNVL).
  bool use_multi_gpu_barrier_with_nccl_in_one_shot_kernel = false;

  // If set, the will be used to determine if optimized kernels that assume a
  // fast interconnect can be used.
  std::optional<int64_t> fast_interconnect_slice_size_override = std::nullopt;
};

// Contains the values that are passed between host threads with rendezvous.
struct RaggedAllToAllRendezvousValue {
  RankId rank;
  se::DeviceAddressBase output_buffer;

  // Exchange the address of the SIGNAL BUFFER array.
  // Peers will write to their_rank's-th cell in the signals array.
  se::DeviceAddressBase barrier_signal_buffer;

  bool operator<(const RaggedAllToAllRendezvousValue& other) const {
    return rank < other.rank;
  }
};

struct RaggedAllToAllStreamState {
  int device_ordinal;
  RankId rank;
  std::optional<int64_t> lsa_size;
  GpuCliqueKey clique_key;

  // Host memory allocations for ragged metadata.
  absl::InlinedVector<std::unique_ptr<se::MemoryAllocation>, 8>
      host_buffer_allocs;

  // Device memory buffer for output offsets.
  se::DeviceAddressHandle output_offsets_device_buffer;

  // MultiGpuBarrier: Device memory buffer for signal values (one per peer).
  // Peers write specific slots in this array to signal this device.
  std::unique_ptr<se::MemoryAllocation> barrier_signal_buffer;

  // Reference to the symmetric memory handler for the barrier signal buffer.
  tsl::TiedRef<xla::SymmetricMemory> barrier_signal_symmetric_memory;

  // MultiGpuBarrier: Device memory for the current local step counter.
  // This value is incremented locally by the kernel after every barrier.
  std::unique_ptr<se::MemoryAllocation> barrier_signal_value;

  // Device memory buffer to store the output buffer pointers.
  std::unique_ptr<se::MemoryAllocation> output_buffer_ptr_storage;

  // Reference to the symmetric memory handler for the pointer storage.
  tsl::TiedRef<xla::SymmetricMemory> output_buffer_ptr_storage_symmetric_memory;

  // Reference to the symmetric memory for the output buffers.
  std::shared_ptr<xla::SymmetricMemory> output_temporary_symmetric_memory;

  // Contains the output buffer pointers and barrier signal buffers for all
  // peers.
  std::shared_ptr<std::vector<RaggedAllToAllRendezvousValue>> participants;

  RaggedAllToAllStreamState(int device_ordinal, RankId rank,
                            GpuCliqueKey clique_key)
      : device_ordinal(device_ordinal),
        rank(rank),
        clique_key(std::move(clique_key)) {}
};

// Returns true if all replicas in every replica group of the collective
// are located on the same host (node).
bool IsAllReplicasLocal(int64_t device_count, const CollectiveConfig& config);

// Thunk that performs a NCCL-based Ragged-All-to-All among CUDA GPU-based
// replicas.
class RaggedAllToAllThunk : public CollectiveThunk {
 public:
  RaggedAllToAllThunk(ThunkInfo thunk_info,
                      const HloRaggedAllToAllInstruction* instr,
                      std::vector<Buffer> buffers, bool p2p_memcpy_enabled);
  RaggedAllToAllThunk(ThunkInfo thunk_info, const RaggedAllToAllConfig& config,
                      std::vector<CollectiveThunk::Buffer> buffers);

  // Returns whether the given instruction can be lowered to a nccl
  // ragged-all-to-all call.
  static absl::Status CheckImplementable(
      const HloRaggedAllToAllInstruction* instr, int64_t replica_count,
      int64_t partition_count);

  CollectiveCliqueRequests::CliqueRequirements GetCliqueRequirements(
      const GpuCliqueKey& clique_key) override;

  absl::Status Initialize(const InitializeParams& params) override;

  static absl::string_view GetHloOpName() { return "ragged-all-to-all-start"; }

  static CollectiveOpGroupMode GetGroupMode(
      const HloRaggedAllToAllInstruction* instr);

  const CollectiveConfig& config() const override { return config_.config; }

  const RaggedAllToAllConfig& ragged_all_to_all_config() const {
    return config_;
  }

  bool is_one_shot_kernel_enabled() const {
    return config_.one_shot_kernel_enabled;
  }

  bool use_multi_gpu_barrier_with_nccl_in_one_shot_kernel() const {
    return config_.use_multi_gpu_barrier_with_nccl_in_one_shot_kernel;
  }

  // Returns true if one shot kernel is supported
  bool IsOneShotKernelSupported() const;

  static absl::StatusOr<std::unique_ptr<RaggedAllToAllThunk>> FromProto(
      ThunkInfo thunk_info, const RaggedAllToAllThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

 protected:
  // No rendezvous needed when using one-shot kernel in local mode instead of
  // NCCL.
  bool RequiresRendezvous() const override {
    return !is_one_shot_kernel_enabled();
  }

  absl::Status PrepareCollective(const PrepareParams& params,
                                 const GpuCliqueKey& clique_key) override;

  absl::Status RunCollective(const ExecuteParams& params,
                             const GpuCliqueKey& clique_key, se::Stream& stream,
                             Communicator& comm) override;

 private:
  bool is_local(int device_count) const {
    return IsAllReplicasLocal(device_count, config_.config);
  }

  const RaggedAllToAllConfig config_;

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<RaggedAllToAllStreamState>>
      per_stream_states_ ABSL_GUARDED_BY(mutex_);

  absl::StatusOr<RaggedAllToAllStreamState*> InitializeOnce(
      const InitializeParams& params);
};

// Executes the rendezvous to exchange buffer addresses and barrier signal
// buffers.
absl::StatusOr<std::shared_ptr<std::vector<RaggedAllToAllRendezvousValue>>>
RendezvousResources(int device_ordinal, RankId rank,
                    const GpuCliqueKey& clique_key,
                    const se::DeviceAddressBase& output_buffer,
                    const se::DeviceAddressBase& barrier_signal_buffer);

// Executes a generic Ragged All-to-All collective operation using the provided
// communicator (e.g., NCCL).
//
// This function handles the "multi-step" coordination required for ragged
// data:
// 1. Exchanges metadata (data sizes) between ranks using the provided host
//    buffers (`ragged_metadata_allocs`).
// 2. Calculates the necessary output offsets based on the exchanged sizes.
// 3. Populates `output_offsets_device_buffer` on the device.
// 4. Performs the actual data transfer into the destination buffers.
//
// Arguments:
//  - ragged_metadata_allocs: Host-side pointers used to exchange row sizes
//    between ranks before the main data transfer.
//  - output_offsets_device_buffer: Device buffer where the calculated
//    destination offsets will be written.
absl::Status RunRaggedAllToAll(
    int64_t ragged_row_element_size, int64_t num_total_updates,
    const std::vector<DeviceBufferPair>& original_buffers, se::Stream& stream,
    Communicator& comm, absl::Span<int64_t* const> ragged_metadata_allocs,
    const se::DeviceAddressBase& output_offsets_device_buffer,
    bool use_symmetric_buffer);

// Executes an optimized "One-Shot" Ragged All-to-All collective.
//
// Unlike the standard implementation, this approach consolidates the
// coordination and data movement into a single execution path (typically a
// custom kernel or specialized P2P sequence) to reduce host-device
// synchronization overhead.
//
// It utilizes `MultiGpuBarrierKernel` to enforce device-side synchronization.
// This ensures input/output buffers are safe to access without requiring
// Event-based coordination, enabling compatibility with CUDA Graphs.
absl::Status RunOneShotRaggedAllToAll(
    const GpuCliqueKey& clique_key, se::Stream& stream, RankId rank,
    const se::DeviceAddressBase& barrier_signal_buffer,
    const se::DeviceAddressBase& barrier_signal_value,
    int64_t num_total_updates, int64_t num_input_rows, int64_t num_row_elements,
    absl::Span<DeviceBufferPair const> buffers,
    const std::vector<RaggedAllToAllRendezvousValue>& participants);

// It utilizes `MultiGpuBarrierWithNcclKernel` to enforce device-side
// synchronization. This ensures input/output buffers are safe to access without
// requiring Event-based coordination, enabling compatibility with CUDA Graphs.
absl::Status RunOneShotRaggedAllToAllWithNccl(
    const GpuCliqueKey& clique_key, se::Stream& stream, RankId rank,
    std::shared_ptr<xla::SymmetricMemory> barrier_signal_symmetric_memory,
    const se::DeviceAddressBase& barrier_signal_value,
    std::shared_ptr<xla::SymmetricMemory> output_temporary_symmetric_memory,
    int64_t num_total_updates, int64_t num_input_rows, int64_t num_row_elements,
    absl::Span<DeviceBufferPair const> buffers);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_THUNK_H_
