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
#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_PASSES_H_
#define XLA_SERVICE_GPU_FUSIONS_MLIR_PASSES_H_

#include <memory>
#include <optional>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DECL
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

// Returns the range of a given value, if it can be statically determined.
std::optional<Interval> GetRange(mlir::Value value);

std::unique_ptr<mlir::Pass> CreateExpandFloatOpsPass(bool pre_ampere);
std::unique_ptr<mlir::Pass> CreateConvertPureCallOpsPass();
std::unique_ptr<mlir::Pass> CreateLowerTensorsPass(
    bool is_amd_gpu = false, const std::string& gpu_arch = "6.0");
std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass();
std::unique_ptr<mlir::Pass> CreateLowerXlaGpuToScfPass();
std::unique_ptr<mlir::Pass> CreateMergePointersToSameSlicePass();
std::unique_ptr<mlir::Pass> CreatePropagateSliceIndicesPass();
std::unique_ptr<mlir::Pass> CreateSimplifyAffinePass();
std::unique_ptr<mlir::Pass> CreateSimplifyArithPass();

#define GEN_PASS_REGISTRATION
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_MLIR_PASSES_H_
