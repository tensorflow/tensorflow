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

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class PassManager;
namespace gpu {
class GPUModuleOp;
}  // namespace gpu

/// Pass that transforms gpu modules in standard dialect to NNVM.
std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
CreateGpuKernelToNvvmPass();

/// Pass that transforms gpu modules in standard dialect to ROCDL.
std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
CreateGpuKernelToRocdlPass();

/// Creates a pipeline that converts operations in HLO dialect to GPU kernels
/// written in a combination of LLVM and NVVM dialects, and appends the pipeline
/// to `pm`. `tileSizes` and `unrollFactors` are used to control loop tiling
/// in `CreateTileLoopsPass`.
void createHloToGpuPipeline(OpPassManager &pm, ArrayRef<int64_t> tileSizes,
                            ArrayRef<int64_t> unrollFactors);

}  // namespace mlir

#endif  // MLIR_HLO_TRANSFORMS_GPU_PASSES_H
