/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_TO_ALL_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_TO_ALL_THUNK_H_

#include <vector>

#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"

namespace xla {
namespace gpu {

struct NcclAllToAllConfig {
  NcclCollectiveConfig config;
  bool has_split_dimension;
};

// Base class for thunks that performs a NCCL-based All-to-All among CUDA
// GPU-based replicas.
class NcclAllToAllThunkBase : public NcclCollectiveThunk {
 public:
  NcclAllToAllThunkBase(Kind kind, ThunkInfo thunk_info,
                        NcclAllToAllConfig config, std::vector<Buffer> buffers);

 protected:
  Status RunAllToAll(const ExecuteParams& params, se::Stream& stream,
                     ncclComm_t comm);
  const NcclCollectiveConfig& config() const override { return config_.config; }

 private:
  const NcclAllToAllConfig config_;
  const std::vector<Buffer> buffers_;
};

class NcclAllToAllStartThunk : public NcclAllToAllThunkBase {
 public:
  NcclAllToAllStartThunk(ThunkInfo thunk_info,
                         mlir::lmhlo_gpu::AllToAllStartOp op,
                         std::vector<Buffer> buffers);

  // Returns whether the given instruction can be lowered to a nccl all-to-all
  // call.
  static Status CheckImplementable(mlir::lmhlo_gpu::AllToAllStartOp op,
                                   int64_t replica_count,
                                   int64_t partition_count);

  static const char* GetHloOpName() { return "all-to-all-start"; }
  static bool IsDegenerate(mlir::lmhlo_gpu::AllToAllStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::AllToAllStartOp op);

  static constexpr bool IsAsync() { return true; }
  AsyncExecutor& async_executor() { return async_; }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;

 private:
  AsyncExecutor async_;
};

class NcclAllToAllDoneThunk : public NcclCollectiveDoneThunk {
 public:
  NcclAllToAllDoneThunk(ThunkInfo thunk_info,
                        NcclCollectiveThunk::AsyncExecutor& async);
};

Status RunAllToAll(bool has_split_dimension,
                   std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                   ncclComm_t comm);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_TO_ALL_THUNK_H_
