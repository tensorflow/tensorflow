//===- ConvertToNVVMIR.cpp - MLIR to LLVM IR conversion -------------------===//
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
// This file implements a translation between the MLIR LLVM + NVVM dialects and
// LLVM IR with NVVM intrinsics and metadata.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/NVVMIR.h"

#include "mlir/LLVMIR/NVVMDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {
static void createIntrinsicCall(llvm::IRBuilder<> &builder,
                                llvm::Intrinsic::ID intrinsic) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *fn = llvm::Intrinsic::getDeclaration(module, intrinsic, {});
  builder.CreateCall(fn);
}

class ModuleTranslation : public LLVM::ModuleTranslation {

public:
  explicit ModuleTranslation(Module &module)
      : LLVM::ModuleTranslation(module) {}
  ~ModuleTranslation() override {}

protected:
  bool convertOperation(Operation &opInst,
                        llvm::IRBuilder<> &builder) override {

#include "mlir/LLVMIR/NVVMConversions.inc"

    return LLVM::ModuleTranslation::convertOperation(opInst, builder);
  }
};
} // namespace

std::unique_ptr<llvm::Module> mlir::translateModuleToNVVMIR(Module &m) {
  ModuleTranslation translation(m);
  return LLVM::ModuleTranslation::translateModule<ModuleTranslation>(m);
}

static TranslateFromMLIRRegistration registration(
    "mlir-to-nvvmir", [](Module *module, llvm::StringRef outputFilename) {
      if (!module)
        return true;

      auto llvmModule =
          LLVM::ModuleTranslation::translateModule<ModuleTranslation>(*module);
      if (!llvmModule)
        return true;

      auto file = openOutputFile(outputFilename);
      if (!file)
        return true;

      llvmModule->print(file->os(), nullptr);
      file->keep();
      return false;
    });
