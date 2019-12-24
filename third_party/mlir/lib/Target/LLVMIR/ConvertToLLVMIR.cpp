//===- ConvertToLLVMIR.cpp - MLIR to LLVM IR conversion -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR.h"

#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

std::unique_ptr<llvm::Module> mlir::translateModuleToLLVMIR(ModuleOp m) {
  return LLVM::ModuleTranslation::translateModule<>(m);
}

static TranslateFromMLIRRegistration
    registration("mlir-to-llvmir", [](ModuleOp module, raw_ostream &output) {
      auto llvmModule = LLVM::ModuleTranslation::translateModule<>(module);
      if (!llvmModule)
        return failure();

      llvmModule->print(output, nullptr);
      return success();
    });
