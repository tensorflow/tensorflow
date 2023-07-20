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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_OPENXLA_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_OPENXLA_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace xla::gpu {

class ThunkSequence;  // forward declare

// Populate passes that lower MLIR modules from a combination of LMHLO and
// LMHLO_GPU dialects to the OpenXLA runtime (aka IREE input dialects + OpenXLA
// custom calls implementing library integration).
void populateOpenXlaRuntimePasses(mlir::OpPassManager& pm,
                                  ThunkSequence* thunk_sequence);

//===----------------------------------------------------------------------===//
// Conversion from LMHLO dialects to OpenXLA runtime
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertToOpenXlaPass(
    ThunkSequence* thunk_sequence = nullptr);

//===----------------------------------------------------------------------===//
// OpenXLA passes registration
//===----------------------------------------------------------------------===//

void registerOpenXlaPases();

}  // namespace xla::gpu

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_OPENXLA_TRANSFORMS_PASSES_H_
