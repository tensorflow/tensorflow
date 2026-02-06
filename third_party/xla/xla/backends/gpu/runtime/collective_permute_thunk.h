#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/xla_data.pb.h"
/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {
namespace gpu {

using tsl::AsyncValueRef;

// Thunk that performs a collective permute.
class CollectivePermuteStartThunk : public CollectiveThunk {
 public:
  class RecvPtrMap {
   public:
    bool IsInitialized(int64_t current_id) const {
      absl::MutexLock lock(mutex_);
      return recv_ptrs_.find(current_id) != recv_ptrs_.end();
    }

    absl::Status InitializeId(int64_t current_id) {
      absl::MutexLock lock(mutex_);
      recv_ptrs_[current_id] =
          tsl::MakeUnconstructedAsyncValueRef<std::vector<void*>>();
      return absl::OkStatus();
    }

    absl::Status PutRecvPtr(int64_t current_id,
                            const std::vector<void*>& ptrs) {
      if (!IsInitialized(current_id)) {
        return absl::InternalError(absl::StrCat("Current ID ", current_id,
                                                " has not been initialized!"));
      }
      absl::MutexLock lock(mutex_);
      if (recv_ptrs_.at(current_id).IsUnavailable()) {
        VLOG(3) << "Putting pointers to current_id " << current_id;
        recv_ptrs_.at(current_id).emplace(ptrs);
      }
      return absl::OkStatus();
    }

    absl::StatusOr<AsyncValueRef<std::vector<void*>>> GetRecvPtr(
        int64_t target_id) const {
      if (!IsInitialized(target_id)) {
        return absl::InternalError(absl::StrCat("Target ID ", target_id,
                                                " has not been initialized!"));
      }
      absl::MutexLock lock(mutex_);
      return recv_ptrs_.at(target_id);
    }

   private:
    mutable absl::Mutex mutex_;
    absl::node_hash_map<int64_t, AsyncValueRef<std::vector<void*>>> recv_ptrs_
        ABSL_GUARDED_BY(mutex_);
  };

  CollectivePermuteStartThunk(ThunkInfo thunk_info,
                              const HloCollectivePermuteInstruction* instr,
                              int64_t replica_count, int64_t partition_count,
                              const std::vector<Buffer>& buffers,
                              bool p2p_memcpy_enabled);
  CollectivePermuteStartThunk(ThunkInfo thunk_info, const P2PConfig& config,
                              std::shared_ptr<AsyncEvents> async_events,
                              const std::vector<Buffer>& buffers,
                              bool p2p_memcpy_enabled);

  static P2PConfig GetP2PConfig(const HloCollectivePermuteInstruction* instr,
                                int64_t replica_count, int64_t partition_count);

  static bool IsDegenerate(const HloCollectivePermuteInstruction* instr,
                           int64_t replica_count, int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloCollectivePermuteInstruction* instr);

  absl::Status Initialize(const InitializeParams& params) override;

  static absl::string_view GetHloOpName() { return "collective-permute-start"; }

  const CollectiveConfig& config() const override { return config_.config; }

  absl::Span<const Buffer> buffers() const { return buffers_; }

  const P2PConfig& p2p_config() const { return config_; }

  static absl::StatusOr<std::unique_ptr<CollectivePermuteStartThunk>> FromProto(
      ThunkInfo thunk_info, const CollectivePermuteStartThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      CollectiveThunk::AsyncEventsMap& async_events_map);

  absl::StatusOr<ThunkProto> ToProto() const override;

  BufferUses buffer_uses() const override {
    BufferUses uses;
    uses.reserve(buffers_.size() * 2);
    for (const Buffer& buffer : buffers_) {
      uses.push_back(BufferUse::Read(buffer.source_buffer.slice,
                                     buffer.source_buffer.shape));
      uses.push_back(BufferUse::Write(buffer.destination_buffer.slice,
                                      buffer.destination_buffer.shape));
    }
    return uses;
  }

 protected:
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     const GpuCliqueKey& clique_key,
                                     se::Stream& stream,
                                     Communicator& comm) override;

 private:
  const P2PConfig config_;
  std::vector<Buffer> buffers_;
  RecvPtrMap recv_ptr_map_;
  absl::Mutex barrier_mutex_;
  absl::flat_hash_map<int64_t, std::unique_ptr<se::Event>>
      receiver_barrier_events_;
  absl::flat_hash_map<int64_t, std::unique_ptr<se::Event>>
      sender_barrier_events_;
  bool p2p_memcpy_enabled_ = false;
  int64_t device_count_ = 0;
};

absl::Status RunCollectivePermute(
    P2PConfig::SourceTargetMapEntry source_target,
    const std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
    Communicator& comm, absl::string_view device_string, int64_t current_id,
    bool use_memcpy = false,
    const CollectivePermuteStartThunk::RecvPtrMap* recv_ptr_map = nullptr,
    bool use_symmetric_buffer = false);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_
