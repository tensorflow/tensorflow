/* Copyright 2022 The OpenXLA Authors.

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

// Returns array of bool attributes. The value of each element specifies whether
// the corresponding operand is written. This attribute is attached to
// 'gpu.launc_func' ops during the fusion rewrite pass above.
ArrayAttr getWrittenOperandsAttribute(Operation* op);

/// Pass that transforms gpu modules in standard dialect to NNVM.
std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
createGpuKernelToNvvmPass(bool useBarePtrCallConv = false);

/// Pass that transforms gpu modules in standard dialect to ROCDL.
std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
createGpuKernelToRocdlPass();

#define GEN_PASS_REGISTRATION
#include "transforms/gpu_passes.h.inc"

}  // namespace mlir

#endif  // MLIR_HLO_TRANSFORMS_GPU_PASSES_H
