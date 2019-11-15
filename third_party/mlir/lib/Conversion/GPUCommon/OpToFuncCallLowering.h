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
#ifndef MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
#define MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"

namespace mlir {

/// Rewriting that replace SourceOp with a CallOp to `f32Func` or `f64Func`
/// depending on the element type that Op operates upon. The function
/// declaration is added in case it was not added before.
///
/// Example with NVVM:
///   %exp_f32 = std.exp %arg_f32 : f32
///
/// will be transformed into
///   llvm.call @__nv_expf(%arg_f32) : (!llvm.float) -> !llvm.float
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
    StringRef funcName = getFunctionName(resultType);
    if (funcName.empty())
      return matchFailure();

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

    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcName);
    if (funcOp)
      return llvm::cast<LLVMFuncOp>(*funcOp);

    mlir::OpBuilder b(op->getParentOfType<LLVMFuncOp>());
    return b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType, llvm::None);
  }

  const std::string f32Func;
  const std::string f64Func;
};

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
