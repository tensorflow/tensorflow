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

#ifndef XLA_MLIR_MEMREF_TRANSFORMS_PASSES_H_
#define XLA_MLIR_MEMREF_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace xla {

#define GEN_PASS_DECL_ALIGNEDALLOCATIONSPASS
#include "xla/mlir/memref/transforms/passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateAlignedAllocationsPass(int64_t alignment = 64);

#define GEN_PASS_REGISTRATION
#include "xla/mlir/memref/transforms/passes.h.inc"

}  // namespace xla

#endif  // XLA_MLIR_MEMREF_TRANSFORMS_PASSES_H_
