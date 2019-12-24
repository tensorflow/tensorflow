//===- ConvertToNVVMIR.cpp - MLIR to LLVM IR conversion -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM + NVVM dialects and
// LLVM IR with NVVM intrinsics and metadata.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/NVVMIR.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Module.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

static llvm::Value *createIntrinsicCall(llvm::IRBuilder<> &builder,
                                        llvm::Intrinsic::ID intrinsic,
                                        ArrayRef<llvm::Value *> args = {}) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *fn = llvm::Intrinsic::getDeclaration(module, intrinsic);
  return builder.CreateCall(fn, args);
}

static llvm::Intrinsic::ID getShflBflyIntrinsicId(llvm::Type *resultType,
                                                  bool withPredicate) {
  if (withPredicate) {
    resultType = cast<llvm::StructType>(resultType)->getElementType(0);
    return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_bfly_f32p
                                   : llvm::Intrinsic::nvvm_shfl_sync_bfly_i32p;
  }
  return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_bfly_f32
                                 : llvm::Intrinsic::nvvm_shfl_sync_bfly_i32;
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

#include "mlir/Dialect/LLVMIR/NVVMConversions.inc"

    return LLVM::ModuleTranslation::convertOperation(opInst, builder);
  }
};
} // namespace

std::unique_ptr<llvm::Module> mlir::translateModuleToNVVMIR(Operation *m) {
  ModuleTranslation translation(m);
  auto llvmModule =
      LLVM::ModuleTranslation::translateModule<ModuleTranslation>(m);
  if (!llvmModule)
    return llvmModule;

  // Insert the nvvm.annotations kernel so that the NVVM backend recognizes the
  // function as a kernel.
  for (auto func :
       ModuleTranslation::getModuleBody(m).getOps<LLVM::LLVMFuncOp>()) {
    if (!gpu::GPUDialect::isKernel(func))
      continue;

    auto *llvmFunc = llvmModule->getFunction(func.getName());

    llvm::Metadata *llvmMetadata[] = {
        llvm::ValueAsMetadata::get(llvmFunc),
        llvm::MDString::get(llvmModule->getContext(), "kernel"),
        llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(llvmModule->getContext()), 1))};
    llvm::MDNode *llvmMetadataNode =
        llvm::MDNode::get(llvmModule->getContext(), llvmMetadata);
    llvmModule->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvmMetadataNode);
  }

  return llvmModule;
}

static TranslateFromMLIRRegistration
    registration("mlir-to-nvvmir", [](ModuleOp module, raw_ostream &output) {
      auto llvmModule = mlir::translateModuleToNVVMIR(module);
      if (!llvmModule)
        return failure();

      llvmModule->print(output, nullptr);
      return success();
    });
