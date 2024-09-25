/* Copyright 2023 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_FUSIONS_LEGACY_REDUCTION_H_
#define XLA_SERVICE_GPU_FUSIONS_LEGACY_REDUCTION_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "llvm/IR/IRBuilder.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/legacy/tiling_util.h"
#include "xla/service/gpu/fusions/reduction_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

class ReductionInfo {
 public:
  static ReductionInfo Create(const HloFusionAnalysis& analysis);

  const Tiling& GetTiling() const { return tiling_; }
  const ReductionGroups& GetGroups() const { return groups_; }
  Shape GetReduceOperandShape() const {
    return first_reduce_->operand(0)->shape();
  }

  bool IsRowReduction() const { return is_row_reduction_; }
  bool IsRaceFree() const { return is_race_free_; }
  int GetRowsPerWarp() const;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const;

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const;

  LaunchDimensions launch_dimensions() const;

 private:
  ReductionInfo(const HloFusionAnalysis& analysis, Tiling tiling,
                bool is_row_reduction, bool is_race_free,
                ReductionGroups groups, const HloInstruction* first_reduce)
      : analysis_(analysis),
        tiling_(tiling),
        is_row_reduction_(is_row_reduction),
        is_race_free_(is_race_free),
        groups_(std::move(groups)),
        first_reduce_(first_reduce) {}

  const HloFusionAnalysis& analysis_;
  Tiling tiling_;
  bool is_row_reduction_;
  bool is_race_free_;
  ReductionGroups groups_;
  const HloInstruction* first_reduce_;
};

// Generates code for reduction to contiguous dimensions.
//
// Row reduction uses the following algorithm described in CUDA-like
// pseudocode:
//
// ```
//  __global__ void reduce(int num_rows, float *in, float out) {
//    __shared__ float[32] cache;
//    int offset = blockDim.x * blockIdx.x + threadIdx.x;
//    if (offset >= num_rows) return;
//    int tile_bound = std::min(offset + kTileSizeX, num_rows);
//    float accum = 0;
//    for (int i=offset; i<num_rows; i+= blockDim.x) {
//      accum += in[i];
//    }
//    accum = warp_reduce(accum);
//    if (threadIdx.x % WarpSize == 0) {
//      cache[threadIdx.x / WarpSize] = accum;
//    }
//    __syncthreads();
//    if (threadIdx.x / WarpSize == 0) {
//      bool warp_exists = threadIdx.x < (blockDim.x / WarpSize);
//      float block_accum = warp_exists ? cache[threadIdx.x % WarpSize] : 0;
//      block_accum = warp_reduce(accum);
//      if (threadIdx.x == 0) {
//        out += block_accum;
//      }
//    }
//  }
// ```
//
// Column reduction uses the following algorithm:
//
// ```
// void reduce(float** in, float* out) {
//   __shared__ float[32][33] cache;
//   int thread_id = GetThreadId();
//   int block_id = GetBlockId();
//   int tile_size = 128;
//
//   float accum = 0;
//   for (int i=0; i<tile_size; i++) {
//     accum += in[thread_id.y * tile_size + i][block_id * 32 + thread_id.x];
//   }
//   cache[thread_id.x][thread_id.y] = accum;
//
//   __syncthreads();
//   accum = cache[thread_id.y][thread_id.x];
//   accum = warp_reduce(accum); // Sum all the values of `accum` in the same
//                               // warp.
//
//   if (thread_id.y % 32 == 0) {
//     out[block_id * 32 + thread_id.x] = accum;
//   }
// }
// ```
//
// Moreover, a heuristic is implemented to divide the reduce instructions
// into groups for parallelization (see `DivideOutputInstructionsIntoGroups`
// for details about the heuristic.) Reduce instructions in the same group
// will run sequentially while different groups will run in parallel.
//
// we use raw block_id_y to select the reduce groups for execution without
// complicating the index calculation in the code generation of the reduce
// instructions. In other words, a block_id_y is assigned to a group and so
// different groups can be run in parallel.
class ReductionFusion : public KernelFusionEmitterBase {
 public:
  explicit ReductionFusion(const HloFusionAnalysis& analysis)
      : analysis_(analysis), reduction_info_(ReductionInfo::Create(analysis)) {}

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override {
    return reduction_info_.ComputeThreadIdToOutputIndexing(root_index, ctx);
  }

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const override {
    return reduction_info_.ComputeThreadIdToInputIndexing(
        root_index, hero_operand_index, ctx);
  }

  LaunchDimensions launch_dimensions() const override {
    return reduction_info_.launch_dimensions();
  }

  const ReductionInfo& reduction_info() const { return reduction_info_; }

 protected:
  absl::StatusOr<FusionEmissionResult> EmitInitializers(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const override;

  absl::Status EmitKernel(IrEmitterContext& ir_emitter_context,
                          const HloFusionInstruction& fusion,
                          const LaunchDimensions& launch_dims,
                          std::vector<llvm_ir::IrArray> inputs,
                          std::vector<llvm_ir::IrArray> outputs,
                          llvm::IRBuilder<>* builder) const override;

 private:
  const HloFusionAnalysis& analysis_;
  ReductionInfo reduction_info_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_LEGACY_REDUCTION_H_
