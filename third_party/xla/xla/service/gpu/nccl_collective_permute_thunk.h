/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_NCCL_COLLECTIVE_PERMUTE_THUNK_H_
#define XLA_SERVICE_GPU_NCCL_COLLECTIVE_PERMUTE_THUNK_H_

#include <cstdint>

#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/nccl_p2p_thunk_common.h"

namespace xla {
namespace gpu {

// Thunk that performs a NCCL-based collective permute.
class NcclCollectivePermuteStartThunk : public NcclCollectiveThunk {
 public:
  static NcclP2PConfig GetNcclP2PConfig(
      mlir::lmhlo_gpu::CollectivePermuteStartOp op, int64_t replica_count,
      int64_t partition_count);

  static Status CheckImplementable(mlir::lmhlo_gpu::CollectivePermuteStartOp op,
                                   int64_t replica_count,
                                   int64_t partition_count);

  static bool IsDegenerate(mlir::lmhlo_gpu::CollectivePermuteStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::CollectivePermuteStartOp op);

  static const char* GetHloOpName() { return "collective-permute-start"; }

  NcclCollectivePermuteStartThunk(ThunkInfo thunk_info,
                                  mlir::lmhlo_gpu::CollectivePermuteStartOp op,
                                  int64_t replica_count,
                                  int64_t partition_count,
                                  const Buffer& buffer);

 protected:
  const NcclCollectiveConfig& config() const override { return config_.config; }
  Status RunNcclCollective(const ExecuteParams& params, se::Stream& stream,
                           ncclComm_t comm) override;

 private:
  const NcclP2PConfig config_;
  const Buffer buffer_;
};

Status RunCollectivePermute(NcclP2PConfig::SourceTargetMapEntry source_target,
                            DeviceBufferPair& buffer, se::Stream& stream,
                            ncclComm_t comm, absl::string_view device_string,
                            int64_t current_id);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_NCCL_COLLECTIVE_PERMUTE_THUNK_H_
