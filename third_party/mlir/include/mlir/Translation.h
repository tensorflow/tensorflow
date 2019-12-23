//===- Translation.h - Translation registry ---------------------*- C++ -*-===//
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
// Registry for user-provided translations.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TRANSLATION_H
#define MLIR_TRANSLATION_H

#include "llvm/ADT/StringMap.h"

#include <memory>

namespace llvm {
class MemoryBuffer;
class SourceMgr;
class StringRef;
} // namespace llvm

namespace mlir {
struct LogicalResult;
class MLIRContext;
class ModuleOp;
class OwningModuleRef;

/// Interface of the function that translates the sources managed by `sourceMgr`
/// to MLIR. The source manager has at least one buffer. The implementation
/// should create a new MLIR ModuleOp in the given context and return a pointer
/// to it, or a nullptr in case of any error.
using TranslateSourceMgrToMLIRFunction =
    std::function<OwningModuleRef(llvm::SourceMgr &sourceMgr, MLIRContext *)>;

/// Interface of the function that translates the given string to MLIR. The
/// implementation should create a new MLIR ModuleOp in the given context. If
/// source-related error reporting is required from within the function, use
/// TranslateSourceMgrToMLIRFunction instead.
using TranslateStringRefToMLIRFunction =
    std::function<OwningModuleRef(llvm::StringRef, MLIRContext *)>;

/// Interface of the function that translates MLIR to a different format and
/// outputs the result to a stream. It is allowed to modify the module.
using TranslateFromMLIRFunction =
    std::function<LogicalResult(ModuleOp, llvm::raw_ostream &output)>;

/// Interface of the function that performs file-to-file translation involving
/// MLIR. The input file is held in the given MemoryBuffer; the output file
/// should be written to the given raw_ostream. The implementation should create
/// all MLIR constructs needed during the process inside the given context. This
/// can be used for round-tripping external formats through the MLIR system.
using TranslateFunction = std::function<LogicalResult(
    llvm::SourceMgr &sourceMgr, llvm::raw_ostream &output, MLIRContext *)>;

/// Use Translate[ToMLIR|FromMLIR]Registration as a global initializer that
/// registers a function and associates it with name. This requires that a
/// translation has not been registered to a given name.
///
/// Usage:
///
///   // At namespace scope.
///   static TranslateToMLIRRegistration Unused(&MySubCommand, [] { ... });
///
/// \{
struct TranslateToMLIRRegistration {
  TranslateToMLIRRegistration(llvm::StringRef name,
                              const TranslateSourceMgrToMLIRFunction &function);
  TranslateToMLIRRegistration(llvm::StringRef name,
                              const TranslateStringRefToMLIRFunction &function);
};

struct TranslateFromMLIRRegistration {
  TranslateFromMLIRRegistration(llvm::StringRef name,
                                const TranslateFromMLIRFunction &function);
};
struct TranslateRegistration {
  TranslateRegistration(llvm::StringRef name,
                        const TranslateFunction &function);
};
/// \}

/// Get a read-only reference to the translator registry.
const llvm::StringMap<TranslateSourceMgrToMLIRFunction> &
getTranslationToMLIRRegistry();
const llvm::StringMap<TranslateFromMLIRFunction> &
getTranslationFromMLIRRegistry();
const llvm::StringMap<TranslateFunction> &getTranslationRegistry();

} // namespace mlir

#endif // MLIR_TRANSLATION_H
