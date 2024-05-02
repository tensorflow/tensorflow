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

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_COLLECTIVE_PERMUTE_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_COLLECTIVE_PERMUTE_THUNK_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/nccl_p2p_thunk_common.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {
namespace gpu {

using tsl::AsyncValueRef;

// Thunk that performs a NCCL-based collective permute.
class NcclCollectivePermuteStartThunk : public NcclCollectiveThunk {
 public:
  class RecvPtrMap {
   public:
    bool IsInitialized(int64_t current_id) {
      absl::MutexLock lock(&mutex_);
      return recv_ptrs_.find(current_id) != recv_ptrs_.end();
    }

    absl::Status InitializeId(int64_t current_id) {
      absl::MutexLock lock(&mutex_);
      if (recv_ptrs_.find(current_id) == recv_ptrs_.end()) {
        recv_ptrs_[current_id] = tsl::MakeUnconstructedAsyncValueRef<void*>();
      }
      return OkStatus();
    }

    absl::Status PutRecvPtr(int64_t current_id, void* ptr) {
      if (!IsInitialized(current_id)) {
        return absl::InternalError(absl::StrCat("Current ID ", current_id,
                                                " has not been initialized!"));
      }
      absl::MutexLock lock(&mutex_);
      if (recv_ptrs_.at(current_id).IsUnavailable()) {
        VLOG(3) << "Putting pointer: " << ptr << " current_id " << current_id;
        recv_ptrs_.at(current_id).emplace(ptr);
      }
      return OkStatus();
    }

    absl::StatusOr<AsyncValueRef<void*>> GetRecvPtr(int64_t target_id) {
      if (!IsInitialized(target_id)) {
        return absl::InternalError(absl::StrCat("Target ID ", target_id,
                                                " has not been initialized!"));
      }
      absl::MutexLock lock(&mutex_);
      return recv_ptrs_[target_id];
    }

   private:
    absl::Mutex mutex_;
    absl::node_hash_map<int64_t, AsyncValueRef<void*>> recv_ptrs_
        ABSL_GUARDED_BY(mutex_);
  };

  static NcclP2PConfig GetNcclP2PConfig(
      const HloCollectivePermuteInstruction* instr, int64_t replica_count,
      int64_t partition_count);

  static bool IsDegenerate(const HloCollectivePermuteInstruction* instr,
                           int64_t replica_count, int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloCollectivePermuteInstruction* instr);

  NcclCollectivePermuteStartThunk(ThunkInfo thunk_info, NcclApi* nccl_api,
                                  const HloCollectivePermuteInstruction* instr,
                                  int64_t replica_count,
                                  int64_t partition_count, const Buffer& buffer,
                                  bool p2p_memcpy_enabled);
  absl::Status Initialize(const InitializeParams& params) override;

  static const char* GetHloOpName() { return "collective-permute-start"; }

 protected:
  const NcclCollectiveConfig& config() const override { return config_.config; }
  absl::Status RunNcclCollective(const ExecuteParams& params,
                                 se::Stream& stream,
                                 NcclCommHandleWrapper comm_wrapper) override;

 private:
  const NcclP2PConfig config_;
  const Buffer buffer_;
  RecvPtrMap recv_ptr_map_;
  bool p2p_memcpy_enabled_ = false;
};

absl::Status RunCollectivePermute(
    NcclApi* nccl_api, NcclP2PConfig::SourceTargetMapEntry source_target,
    DeviceBufferPair& buffer, se::Stream& stream, NcclApi::NcclCommHandle comm,
    absl::string_view device_string, int64_t current_id, bool use_memcpy,
    NcclCollectivePermuteStartThunk::RecvPtrMap& recv_ptr_map);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_COLLECTIVE_PERMUTE_THUNK_H_
