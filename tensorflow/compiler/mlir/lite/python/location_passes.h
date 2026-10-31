/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_LOCATION_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_LOCATION_PASSES_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::TFL {

// Creates a pass that unwraps JAX op-type `NameLoc` wrappers, combines caller
// and callee scopes across inlined `CallSiteLoc`s (stripping internal
// `jit(_where)` wrappers), and enriches `stablehlo.composite` leaf names with
// the composite's semantic name.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateNormalizeLocationsPass();

// Creates a pass that removes cross-scope CSE outlier locations from
// `FusedLoc`s and updates flat `tf.entry_function` output names to unique
// hierarchical defining-op location names before FlatBuffer export.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateFinalizeExportLocationsPass();

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_LOCATION_PASSES_H_
