//===- mlir-translate.h - Registry & utility functions for translations ---===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Registry for user provided translations and common utility functions for
// translations.
//
//===----------------------------------------------------------------------===//
#ifndef TOOLS_MLIR_TRANSLATE_H
#define TOOLS_MLIR_TRANSLATE_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
class MLIRContext;
class Module;

using TranslateFunction =
    std::function<bool(llvm::StringRef inputFilename,
                       llvm::StringRef oututFilename, MLIRContext *)>;

// Use TranslateRegistration as a global initialiser that registers a function
// and associates it with name. This requires that a translation has not been
// registered to a given name.
//
// Usage:
//
//   // At namespace scope.
//   static TranslateRegistration Unused(&MySubCommand, [] { ... });
//
struct TranslateRegistration {
  TranslateRegistration(llvm::StringRef name,
                        const TranslateFunction &function);
};

// Returns module parsed from input filename or null in case of error.
Module *parseMLIRInput(llvm::StringRef inputFilename, MLIRContext *context);

// Prints module to outputFilename and returns whether printing module failed.
bool printMLIROutput(const Module &module, llvm::StringRef outputFilename);

} // namespace mlir

#endif // TOOLS_MLIR_TRANSLATE_H
