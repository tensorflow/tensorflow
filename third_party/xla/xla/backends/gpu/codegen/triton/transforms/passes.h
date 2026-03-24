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

#include <memory>

#include "absl/strings/string_view.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_dialect.h"  // IWYU pragma: keep

namespace mlir::triton::xla {

#define GEN_PASS_DECL
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

std::unique_ptr<mlir::Pass> CreateTritonXLAExtractInsertToTritonPass();
std::unique_ptr<mlir::Pass> CreateTritonXLAExtractInsertToTritonPass(
    bool allow_tma, int num_stages);
std::unique_ptr<mlir::Pass> CreateTritonXLASqueezeDimsPass();
std::unique_ptr<mlir::Pass> CreateTritonXLAFoldTransposePass();
std::unique_ptr<mlir::Pass> CreateGeneralizeKernelSignaturePass();

std::unique_ptr<mlir::Pass> CreateInt4ToPackedInt4RewritePass(
    bool enable_bf16x2);
std::unique_ptr<mlir::Pass> CreateRoundF32ToTF32ForTf32DotRewritePass();
std::unique_ptr<mlir::Pass> CreateExtractTmaInfoPass();
std::unique_ptr<mlir::Pass> CreateTritonXLAUnswitchLoopsPass();
std::unique_ptr<mlir::Pass> CreateTritonXLALowerGetTidPass();
std::unique_ptr<mlir::Pass> CreateTritonXLALowerAtomicsPass();
std::unique_ptr<mlir::Pass> CreateTritonXLALowerBlockBarrierPass();
std::unique_ptr<mlir::Pass> CreateTritonXLAConvertUnsupportedTypesPass();
std::unique_ptr<mlir::Pass> CreateTritonXLALowerRemoteAccessPass();
std::unique_ptr<mlir::Pass> CreateTritonXLALowerXTilePass();
std::unique_ptr<mlir::Pass> CreateStableHLOLowerToTritonPass(
    bool warp_specialization_allowed);
std::unique_ptr<mlir::Pass> CreateTensorLowerToTritonPass();
std::unique_ptr<mlir::Pass> CreateTritonXLAMathToLibdevicePass(
    absl::string_view libdevice_path, absl::string_view triple);
std::unique_ptr<mlir::Pass> CreateXTileLowerToTritonPass();
std::unique_ptr<mlir::Pass> CreateArithFP8ConversionToTritonPass();
std::unique_ptr<mlir::Pass> CreateUnsupportedElementwiseToTritonPass();

// Returns true if the `op` contains an operation in it's regions that satisfies
// the `fn`.
bool ContainsOp(mlir::Operation* op,
                llvm::function_ref<bool(mlir::Operation*)> fn);

#define GEN_PASS_REGISTRATION
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

}  // namespace mlir::triton::xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TRANSFORMS_PASSES_H_
