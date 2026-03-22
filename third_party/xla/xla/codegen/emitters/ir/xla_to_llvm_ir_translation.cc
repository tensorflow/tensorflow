/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/emitters/ir/xla_to_llvm_ir_translation.h"

#include <cstdint>

#include "llvm/IR/Attributes.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"

namespace xla {
namespace {

using ::mlir::LLVM::LLVMFuncOp;
using ::mlir::LLVM::ModuleTranslation;

class XlaDialectLLVMIRTranslationInterface
    : public mlir::LLVMTranslationDialectInterface {
 public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  mlir::LogicalResult convertParameterAttr(
      LLVMFuncOp funcOp, int argIdx, mlir::NamedAttribute attribute,
      ModuleTranslation& moduleTranslation) const final {
    llvm::Function* llvmFunc =
        moduleTranslation.lookupFunction(funcOp.getName());

    if (attribute.getName() == "xla.range") {
      auto arrayAttr =
          mlir::cast<mlir::DenseI64ArrayAttr>(attribute.getValue());
      if (arrayAttr.size() == 2) {
        int64_t low = arrayAttr[0];
        int64_t high = arrayAttr[1];
        // xla.range is inclusive, LLVM range is exclusive.
        llvmFunc->addParamAttr(
            argIdx,
            llvm::Attribute::get(
                moduleTranslation.getLLVMContext(), llvm::Attribute::Range,
                llvm::ConstantRange(
                    llvm::APInt(64, low, /*isSigned=*/true),
                    llvm::APInt(64, high + 1, /*isSigned=*/true))));
      }
      return mlir::success();
    }
    if (attribute.getName() == "xla.slice_index" ||
        attribute.getName() == "xla.invariant") {
      return mlir::success();
    }

    return mlir::success();
  }

  mlir::LogicalResult amendOperation(
      mlir::Operation* op, llvm::ArrayRef<llvm::Instruction*> instructions,
      mlir::NamedAttribute attribute,
      ModuleTranslation& moduleTranslation) const final {
    if (attribute.getName() == "xla.entry" ||
        attribute.getName() == "xla.range") {
      return mlir::success();
    }
    return mlir::success();
  }
};

}  // namespace

void registerXlaDialectTranslation(mlir::DialectRegistry& registry) {
  registry.insert<XlaDialect>();
  registry.addExtension(+[](mlir::MLIRContext* ctx, XlaDialect* dialect) {
    dialect->addInterfaces<XlaDialectLLVMIRTranslationInterface>();
  });
}

void registerXlaDialectTranslation(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  registerXlaDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}

}  // namespace xla
