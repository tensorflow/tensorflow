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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_GATHER_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_GATHER_THUNK_H_

#include <vector>

#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct NcclAllGatherConfig {
  NcclCollectiveConfig config;
};

// Base class for thunk that performs a NCCL-based All-Gather among CUDA
// GPU-based replicas.
class NcclAllGatherThunkBase : public NcclCollectiveThunk {
 public:
  NcclAllGatherThunkBase(Kind kind, ThunkInfo thunk_info,
                         NcclAllGatherConfig config,
                         std::vector<Buffer> buffers);

 protected:
  Status RunAllGather(const ExecuteParams& params, se::Stream& stream,
                      ncclComm_t comm);
  const NcclCollectiveConfig& config() const override { return config_.config; }

 private:
  const NcclAllGatherConfig config_;
  const std::vector<Buffer> buffers_;
};

class NcclAllGatherThunk : public NcclAllGatherThunkBase {
 public:
  NcclAllGatherThunk(ThunkInfo thunk_info, mlir::lmhlo::AllGatherOp op,
                     std::vector<Buffer> buffers);

  // Returns whether the given instruction can be lowered to a nccl all-gather
  // call.
  static bool CanImplement(mlir::lmhlo::AllGatherOp op);
  static const char* GetName() { return "AllGather"; }
  static bool IsDegenerate(mlir::lmhlo::AllGatherOp op, int64_t replica_count,
                           int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(mlir::lmhlo::AllGatherOp op);
  static constexpr bool IsAsync() { return false; }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;
};

class NcclAllGatherStartThunk : public NcclAllGatherThunkBase {
 public:
  NcclAllGatherStartThunk(ThunkInfo thunk_info,
                          mlir::lmhlo_gpu::AllGatherStartOp op,
                          std::vector<Buffer> buffers);

  static const char* GetName() { return "AllGatherStart"; }

  static bool CanImplement(mlir::lmhlo_gpu::AllGatherStartOp op);
  static bool IsDegenerate(mlir::lmhlo_gpu::AllGatherStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::AllGatherStartOp op);
  static constexpr bool IsAsync() { return true; }

  AsyncExecutor& async_executor() { return async_; }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;

 private:
  AsyncExecutor async_;
};

class NcclAllGatherDoneThunk : public NcclCollectiveDoneThunk {
 public:
  NcclAllGatherDoneThunk(ThunkInfo thunk_info,
                         NcclCollectiveThunk::AsyncExecutor& async);
};

Status RunAllGather(std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_GATHER_THUNK_H_
