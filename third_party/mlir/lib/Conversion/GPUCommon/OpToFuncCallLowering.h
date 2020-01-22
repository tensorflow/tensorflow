//===- OpToFuncCallLowering.h - GPU ops lowering to custom calls *- C++ -*-===//
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
#ifndef THIRD_PARTY_LLVM_LLVM_PROJECTS_GOOGLE_MLIR_LIB_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
#define THIRD_PARTY_LLVM_LLVM_PROJECTS_GOOGLE_MLIR_LIB_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"

namespace mlir {

template <typename SourceOp>
struct OpToFuncCallLowering : public LLVMOpLowering {
public:
  explicit OpToFuncCallLowering(LLVMTypeConverter &lowering_, StringRef f32Func,
                                StringRef f64Func)
      : LLVMOpLowering(SourceOp::getOperationName(),
                       lowering_.getDialect()->getContext(), lowering_),
        f32Func(f32Func), f64Func(f64Func) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;
    using LLVM::LLVMType;

    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    LLVMType resultType = lowering.convertType(op->getResult(0)->getType())
                              .template cast<LLVM::LLVMType>();
    LLVMType funcType = getFunctionType(resultType, operands);
    const std::string funcName = getFunctionName(resultType);
    if (funcName.empty()) {
      return matchFailure();
    }

    LLVMFuncOp funcOp = appendOrGetFuncOp(funcName, funcType, op);
    auto callOp = rewriter.create<LLVM::CallOp>(
        op->getLoc(), resultType, rewriter.getSymbolRefAttr(funcOp), operands);
    rewriter.replaceOp(op, {callOp.getResult(0)});
    return matchSuccess();
  }

private:
  LLVM::LLVMType getFunctionType(LLVM::LLVMType resultType,
                                 ArrayRef<Value *> operands) const {
    using LLVM::LLVMType;
    SmallVector<LLVMType, 1> operandTypes;
    for (Value *operand : operands) {
      operandTypes.push_back(operand->getType().cast<LLVMType>());
    }
    return LLVMType::getFunctionTy(resultType, operandTypes,
                                   /*isVarArg=*/false);
  }

  StringRef getFunctionName(LLVM::LLVMType type) const {
    if (type.isFloatTy())
      return f32Func;
    if (type.isDoubleTy())
      return f64Func;
    return "";
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(StringRef funcName,
                                     LLVM::LLVMType funcType,
                                     Operation *op) const {
    using LLVM::LLVMFuncOp;

    LLVMFuncOp funcOp =
        op->getParentOfType<ModuleOp>().lookupSymbol<LLVMFuncOp>(funcName);
    if (funcOp)
      return funcOp;

    mlir::OpBuilder b(op->getParentOfType<LLVMFuncOp>());
    return b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType, llvm::None);
  }

  const std::string f32Func;
  const std::string f64Func;
};

} // namespace mlir

#endif // THIRD_PARTY_LLVM_LLVM_PROJECTS_GOOGLE_MLIR_LIB_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
