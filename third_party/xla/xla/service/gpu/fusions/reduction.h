/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_SERVICE_GPU_FUSIONS_REDUCTION_H_
#define XLA_SERVICE_GPU_FUSIONS_REDUCTION_H_

#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla {
namespace gpu {

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
class ReductionFusion : public FusionInterface {
 public:
  explicit ReductionFusion(HloFusionAnalysis& analysis) : analysis_(analysis) {}

  StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      ElementalIrEmitter& elemental_emitter, mlir::lmhlo::FusionOp fusion_op,
      const HloFusionInstruction& fusion, KernelReuseCache& kernel_cache,
      llvm::IRBuilder<>* builder) const override;

 private:
  HloFusionAnalysis& analysis_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_REDUCTION_H_
