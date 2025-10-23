/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_TILED_TRANSFORMS_PASSES_H_
#define XLA_BACKENDS_CPU_CODEGEN_TILED_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/Dialect/Math/IR/Math.h"  // IWYU pragma: keep
#include "mlir/Dialect/SCF/IR/SCF.h"  // IWYU pragma: keep
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // IWYU pragma: keep
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // IWYU pragma: keep
#include "mlir/Pass/Pass.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_dialect.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_dialect.h"  // IWYU pragma: keep

namespace xla::cpu {

#define GEN_PASS_DECL
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

std::unique_ptr<mlir::Pass> CreateElementalTensorToVectorPass();
std::unique_ptr<mlir::Pass> CreateLowerXTileEntryPass();
std::unique_ptr<mlir::Pass> CreateShloToVectorPass();
std::unique_ptr<mlir::Pass> CreateXTileToVectorPass();
std::unique_ptr<mlir::Pass> CreateTensorOpsToVectorPass();
std::unique_ptr<mlir::Pass> CreateRewriteDynamicVectorExtractPass();

#define GEN_PASS_REGISTRATION
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_TILED_TRANSFORMS_PASSES_H_
