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

#ifndef XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_
#define XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_

#include <optional>
#include <vector>

#include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/nccl_collective_thunk.h"

namespace xla {
namespace gpu {

struct NcclAllReduceConfig {
  NcclCollectiveConfig config;
  ReductionKind reduction_kind;
};

// Thunk that performs a NCCL-based All-Reduce or Reduce-Scatter among CUDA
// GPU-based replicas.
class NcclAllReduceReduceScatterThunkBase : public NcclCollectiveThunk {
 public:
  static std::optional<ReductionKind> MatchAllReduceComputation(
      mlir::Region& computation);

  NcclAllReduceReduceScatterThunkBase(Kind kind, ThunkInfo thunk_info,
                                      NcclAllReduceConfig config,
                                      std::vector<Buffer> buffers,
                                      bool is_sync);

 protected:
  const NcclCollectiveConfig& config() const override { return config_.config; }

  const NcclAllReduceConfig config_;
  const std::vector<Buffer> buffers_;
};

// -----------------------------------------------------------------------------
// AllReduce thunk.
// -----------------------------------------------------------------------------

class NcclAllReduceStartThunk : public NcclAllReduceReduceScatterThunkBase {
 public:
  NcclAllReduceStartThunk(ThunkInfo thunk_info,
                          mlir::lmhlo_gpu::AllReduceStartOp op,
                          std::vector<Buffer> buffers);

  static const char* GetHloOpName() { return "all-reduce-start"; }

  static Status CheckImplementable(mlir::lmhlo_gpu::AllReduceStartOp op,
                                   int64_t replica_count,
                                   int64_t partition_count);
  static bool IsDegenerate(mlir::lmhlo_gpu::AllReduceStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::AllReduceStartOp op);

 protected:
  Status RunNcclCollective(const ExecuteParams& params, se::Stream& stream,
                           ncclComm_t comm) override;
};

// -----------------------------------------------------------------------------
// ReduceScatter thunk
// -----------------------------------------------------------------------------
class NcclReduceScatterStartThunk : public NcclAllReduceReduceScatterThunkBase {
 public:
  NcclReduceScatterStartThunk(ThunkInfo thunk_info,
                              mlir::lmhlo_gpu::ReduceScatterStartOp op,
                              std::vector<Buffer> buffers);

  static const char* GetHloOpName() { return "reduce-scatter-start"; }

  static Status CheckImplementable(mlir::lmhlo_gpu::ReduceScatterStartOp op,
                                   int64_t replica_count,
                                   int64_t partition_count);
  static bool IsDegenerate(mlir::lmhlo_gpu::ReduceScatterStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::ReduceScatterStartOp op);

 protected:
  Status RunNcclCollective(const ExecuteParams& params, se::Stream& stream,
                           ncclComm_t comm) override;
};

// -----------------------------------------------------------------------------

Status RunAllReduce(ReductionKind reduction_kind,
                    std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm);

Status RunReduceScatter(ReductionKind reduction_kind,
                        std::vector<DeviceBufferPair>& buffers,
                        se::Stream& stream, ncclComm_t comm);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_
