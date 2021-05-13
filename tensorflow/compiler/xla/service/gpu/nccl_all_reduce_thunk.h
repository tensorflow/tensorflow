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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_

#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

struct NcclAllReduceConfig {
  NcclCollectiveConfig config;
  ReductionKind reduction_kind;
};

// Thunk that performs a NCCL-based All-Reduce among CUDA GPU-based replicas.
class NcclAllReduceThunk : public NcclCollectiveThunk {
 public:
  NcclAllReduceThunk(ThunkInfo thunk_info, mlir::lmhlo::AllReduceOp op,
                     std::vector<Buffer> buffers);

  // Returns whether the given instruction can be lowered to a nccl all-reduce
  // call.
  static bool CanImplement(mlir::lmhlo::AllReduceOp op);

  static const char* GetName() { return "AllReduce"; }

  static bool IsDegenerate(mlir::lmhlo::AllReduceOp op, int64 replica_count,
                           int64 partition_count) {
    return GetNcclCollectiveConfigForMlir(op, op.use_global_device_ids())
        .IsDegenerate(replica_count, partition_count);
  }

  static CollectiveOpGroupMode GetGroupMode(mlir::lmhlo::AllReduceOp op) {
    return GetNcclAllReduceConfig(op).config.group_mode;
  }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;

  const NcclCollectiveConfig& config() const override { return config_.config; }

 private:
  static NcclAllReduceConfig GetNcclAllReduceConfig(
      mlir::lmhlo::AllReduceOp op);
  const NcclAllReduceConfig config_;
  const std::vector<Buffer> buffers_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_
