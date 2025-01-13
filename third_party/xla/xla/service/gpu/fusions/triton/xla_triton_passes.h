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

#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_PASSES_H_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_PASSES_H_

#include <cstdint>
#include <memory>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton::xla {

#define GEN_PASS_DECL
#include "xla/service/gpu/fusions/triton/xla_triton_passes.h.inc"

std::unique_ptr<mlir::Pass> CreateSparseAddEncodingPass(
    int32_t num_warps = 4, int32_t threads_per_warp = 32, int32_t num_ctas = 1);
std::unique_ptr<mlir::Pass> CreateSparseBlockedToMMAPass();
std::unique_ptr<mlir::Pass> CreateSparseRemoveLayoutConversionPass();
std::unique_ptr<mlir::Pass> CreateSparseLocalLoadToLLVMPass();
std::unique_ptr<mlir::Pass> CreateSparseDotOpToLLVMPass();
std::unique_ptr<mlir::Pass> CreateSparseWGMMAOpToLLVMPass();
std::unique_ptr<mlir::Pass> CreatePreventMmaV3LoopUnrollingPass();
std::unique_ptr<mlir::Pass> CreateInt4ToPackedInt4RewritePass();

// Returns true if the `op` contains an operation in it's regions that satisfies
// the `fn`.
bool ContainsOp(mlir::Operation* op,
                llvm::function_ref<bool(mlir::Operation*)> fn);

#define GEN_PASS_REGISTRATION
#include "xla/service/gpu/fusions/triton/xla_triton_passes.h.inc"

}  // namespace mlir::triton::xla

#endif  // XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_PASSES_H_
