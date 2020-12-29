/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_XLA_MLIR_TRANSLATE_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_XLA_MLIR_TRANSLATE_H_

#include <memory>

#include "tensorflow/compiler/xla/statusor.h"

namespace llvm {
class StringRef;
}  // namespace llvm

namespace mlir {
class MLIRContext;
class OwningModuleRef;
}  // namespace mlir

namespace xla {

// Converts a HloModuleProto stored in the file with the given `input_filename`
// into a MLIR module. Creates MLIR entities into the given MLIR `context`.
mlir::OwningModuleRef HloToMlirHloTranslateFunction(llvm::StringRef input,
                                                    mlir::MLIRContext* context);

// Converts a HloModule stored in text form for a file with the given
// `input_filename` into a MLIR module. Creates MLIR entities into the given
// MLIR `context`.
mlir::OwningModuleRef HloTextToMlirHloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_XLA_MLIR_TRANSLATE_H_
