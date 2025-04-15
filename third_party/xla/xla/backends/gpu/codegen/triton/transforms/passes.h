/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_TRANSFORMS_PASSES_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_TRANSFORMS_PASSES_H_

#include <cstdint>
#include <memory>
#include <string>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "xla/stream_executor/device_description.h"

namespace mlir::triton::xla {

#define GEN_PASS_DECL
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

std::unique_ptr<mlir::Pass> CreateTritonXLAExtractInsertToTritonPass(
    const std::string& gpu_device_info = "", bool tma_enabled = false);
std::unique_ptr<mlir::Pass> CreateTritonXLAExtractInsertToTritonPass(
    const stream_executor::DeviceDescription& device_description,
    bool tma_enabled);
std::unique_ptr<mlir::Pass> CreateGeneralizeKernelSignaturePass();
std::unique_ptr<mlir::Pass> CreatePreventMmaV3LoopUnrollingPass();
std::unique_ptr<mlir::Pass> CreateInt4ToPackedInt4RewritePass();
std::unique_ptr<mlir::Pass> CreateRoundF32ToTF32ForTf32DotRewritePass();

// Returns true if the `op` contains an operation in it's regions that satisfies
// the `fn`.
bool ContainsOp(mlir::Operation* op,
                llvm::function_ref<bool(mlir::Operation*)> fn);

#define GEN_PASS_REGISTRATION
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

}  // namespace mlir::triton::xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TRANSFORMS_PASSES_H_
