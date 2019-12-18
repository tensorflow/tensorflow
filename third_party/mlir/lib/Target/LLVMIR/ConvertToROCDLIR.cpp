//===- ConvertToROCDLIR.cpp - MLIR to LLVM IR conversion ------------------===//
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
// This file implements a translation between the MLIR LLVM + ROCDL dialects and
// LLVM IR with ROCDL intrinsics and metadata.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/ROCDLIR.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"

#include <iostream>

using namespace mlir;

// Create a call to llvm intrinsic
static llvm::Value *createIntrinsicCall(llvm::IRBuilder<> &builder,
                                        llvm::Intrinsic::ID intrinsic,
                                        ArrayRef<llvm::Value *> args = {}) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *fn = llvm::Intrinsic::getDeclaration(module, intrinsic);
  return builder.CreateCall(fn, args);
}

// Create a call to ROCm-Device-Library function
//   Currently this routine will work only for calling ROCDL functions that
// take a single int32 argument. It is likely that the interface of this
// function will change to make it more generic.
static llvm::Value *createDeviceFunctionCall(llvm::IRBuilder<> &builder,
                                             StringRef fn_name, int parameter) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::FunctionType *function_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(module->getContext()), // return type.
      llvm::Type::getInt32Ty(module->getContext()), // parameter type.
      false);                                       // no variadic arguments.
  llvm::Function *fn = dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fn_name, function_type).getCallee());
  llvm::Value *fn_op0 = llvm::ConstantInt::get(
      llvm::Type::getInt32Ty(module->getContext()), parameter);
  return builder.CreateCall(fn, ArrayRef<llvm::Value *>(fn_op0));
}

namespace {
class ModuleTranslation : public LLVM::ModuleTranslation {

public:
  explicit ModuleTranslation(Operation *module)
      : LLVM::ModuleTranslation(module) {}
  ~ModuleTranslation() override {}

protected:
  LogicalResult convertOperation(Operation &opInst,
                                 llvm::IRBuilder<> &builder) override {

#include "mlir/Dialect/LLVMIR/ROCDLConversions.inc"

    return LLVM::ModuleTranslation::convertOperation(opInst, builder);
  }
};
} // namespace

std::unique_ptr<llvm::Module> mlir::translateModuleToROCDLIR(Operation *m) {
  ModuleTranslation translation(m);

  // lower MLIR (with RODL Dialect) to LLVM IR (with ROCDL intrinsics)
  auto llvmModule =
      LLVM::ModuleTranslation::translateModule<ModuleTranslation>(m);

  // foreach GPU kernel
  // 1. Insert AMDGPU_KERNEL calling convention.
  // 2. Insert amdgpu-flat-workgroup-size(1, 1024) attribute.
  for (auto func :
       ModuleTranslation::getModuleBody(m).getOps<LLVM::LLVMFuncOp>()) {
    if (!func.getAttrOfType<UnitAttr>(gpu::GPUDialect::getKernelFuncAttrName()))
      continue;

    auto *llvmFunc = llvmModule->getFunction(func.getName());

    llvmFunc->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

    llvmFunc->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
  }

  return llvmModule;
}

static TranslateFromMLIRRegistration
    registration("mlir-to-rocdlir", [](ModuleOp module, raw_ostream &output) {
      auto llvmModule = mlir::translateModuleToROCDLIR(module);
      if (!llvmModule)
        return failure();

      llvmModule->print(output, nullptr);
      return success();
    });
