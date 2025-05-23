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
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
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
    bool IsInitialized(int64_t current_id) {
      absl::MutexLock lock(&mutex_);
      return recv_ptrs_.find(current_id) != recv_ptrs_.end();
    }

    absl::Status InitializeId(int64_t current_id) {
      absl::MutexLock lock(&mutex_);
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
      absl::MutexLock lock(&mutex_);
      if (recv_ptrs_.at(current_id).IsUnavailable()) {
        VLOG(3) << "Putting pointers to current_id " << current_id;
        recv_ptrs_.at(current_id).emplace(ptrs);
      }
      return absl::OkStatus();
    }

    absl::StatusOr<AsyncValueRef<std::vector<void*>>> GetRecvPtr(
        int64_t target_id) {
      if (!IsInitialized(target_id)) {
        return absl::InternalError(absl::StrCat("Target ID ", target_id,
                                                " has not been initialized!"));
      }
      absl::MutexLock lock(&mutex_);
      return recv_ptrs_[target_id];
    }

   private:
    absl::Mutex mutex_;
    absl::node_hash_map<int64_t, AsyncValueRef<std::vector<void*>>> recv_ptrs_
        ABSL_GUARDED_BY(mutex_);
  };

  static P2PConfig GetP2PConfig(const HloCollectivePermuteInstruction* instr,
                                int64_t replica_count, int64_t partition_count);

  static bool IsDegenerate(const HloCollectivePermuteInstruction* instr,
                           int64_t replica_count, int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloCollectivePermuteInstruction* instr);

  CollectivePermuteStartThunk(ThunkInfo thunk_info,
                              const HloCollectivePermuteInstruction* instr,
                              int64_t replica_count, int64_t partition_count,
                              const std::vector<Buffer>& buffers,
                              bool p2p_memcpy_enabled,
                              AsyncStreamKind stream_kind);
  absl::Status Initialize(const InitializeParams& params) override;

  static const char* GetHloOpName() { return "collective-permute-start"; }

 protected:
  const CollectiveConfig& config() const override { return config_.config; }
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     se::Stream& stream,
                                     CommunicatorHandle comm_handle) override;

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
  int64_t device_count_;
};

absl::Status RunCollectivePermute(
    GpuCollectives* collectives, P2PConfig::SourceTargetMapEntry source_target,
    std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
    Communicator* comm, absl::string_view device_string, int64_t current_id,
    bool use_memcpy, CollectivePermuteStartThunk::RecvPtrMap& recv_ptr_map);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_
