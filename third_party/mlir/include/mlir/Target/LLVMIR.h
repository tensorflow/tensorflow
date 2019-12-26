//===- LLVMIR.h - MLIR to LLVM IR conversion --------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
