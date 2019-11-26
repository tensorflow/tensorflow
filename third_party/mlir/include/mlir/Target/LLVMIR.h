//===- LLVMIR.h - MLIR to LLVM IR conversion --------------------*- C++ -*-===//
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
// This file declares the entry point for the MLIR to LLVM IR conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_H
#define MLIR_TARGET_LLVMIR_H

#include <memory>

// Forward-declare LLVM classes.
namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {

class OwningModuleRef;
class MLIRContext;
class ModuleOp;

/// Convert the given MLIR module into LLVM IR.  The LLVM context is extracted
/// from the registered LLVM IR dialect.  In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
std::unique_ptr<llvm::Module> translateModuleToLLVMIR(ModuleOp m);

/// Convert the given LLVM module into MLIR's LLVM dialect.  The LLVM context is
/// extracted from the registered LLVM IR dialect. In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `{}`.
OwningModuleRef
translateLLVMIRToModule(std::unique_ptr<llvm::Module> llvmModule,
                        MLIRContext *context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_H
