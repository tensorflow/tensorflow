//===- ConvertToLLVMIR.cpp - MLIR to LLVM IR conversion -------------------===//
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

static TranslateFromMLIRRegistration registration(
    "mlir-to-llvmir", [](ModuleOp module, llvm::raw_ostream &output) {
      auto llvmModule = LLVM::ModuleTranslation::translateModule<>(module);
      if (!llvmModule)
        return failure();

      llvmModule->print(output, nullptr);
      return success();
    });
