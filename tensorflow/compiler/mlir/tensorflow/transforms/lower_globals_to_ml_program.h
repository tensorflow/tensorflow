/* Copyright 2022 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_PASSES_LOWER_GLOBALS_TO_ML_PROGRAM_H
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_PASSES_LOWER_GLOBALS_TO_ML_PROGRAM_H

#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace tf_saved_model {

// Create a pass that removes function arguments that map to global tensors.
std::unique_ptr<Pass> CreateLowerGlobalsToMlProgramPass();

// Register this pass in the global registry of MLIR.
void RegisterLowerGlobalsToMlProgramPass();

}  // namespace tf_saved_model
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_PASSES_LOWER_GLOBALS_TO_ML_PROGRAM_H
