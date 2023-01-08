/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_TRANSFORMS_GPU_PASSES_H
#define MLIR_HLO_TRANSFORMS_GPU_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
class PassManager;
namespace gpu {
class GPUModuleOp;
}  // namespace gpu

#define GEN_PASS_DECL
#include "transforms/gpu_passes.h.inc"

// Create a pass which lowers a subset of lmhlo.fusion ops to gpu.launch_func
// plus a gpu.module containing the kernel.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createGpuFusionRewritePass();

// Returns array of bool attributes. The value of each element specifies whether
// the corresponding operand is written. This attribute is attached to
// 'gpu.launc_func' ops during the fusion rewrite pass above.
ArrayAttr getWrittenOperandsAttribute(Operation* op);

/// Pass that transforms gpu modules in standard dialect to NNVM.
std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
createGpuKernelToNvvmPass();

/// Pass that transforms gpu modules in standard dialect to ROCDL.
std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
createGpuKernelToRocdlPass();

/// Creates a pipeline that converts operations in HLO dialect to GPU kernels
/// written in a combination of LLVM and NVVM dialects, and appends the pipeline
/// to `pm`. `blockTileDim`, `warpTileDim` and `threadTileDim` indicate the
/// size of the subproblem that will be operated on by the block, warp, and
/// thread level, respectively.
void createHloToGpuPipeline(OpPassManager& pm, ArrayRef<int64_t> blockTileDim,
                            ArrayRef<int64_t> warpTileDim,
                            ArrayRef<int64_t> threadTileDim,
                            bool experimentalSoftmax);

/// Creates a pipeline that converts operations in HLO dialect to Triton
/// kernels. `blockTileDim`, indicates the block-level tile size that the
/// problem will be tiled to.
void createHloToTritonPipeline(OpPassManager& pm,
                               ArrayRef<int64_t> blockTileDim);

#define GEN_PASS_REGISTRATION
#include "transforms/gpu_passes.h.inc"

}  // namespace mlir

#endif  // MLIR_HLO_TRANSFORMS_GPU_PASSES_H
