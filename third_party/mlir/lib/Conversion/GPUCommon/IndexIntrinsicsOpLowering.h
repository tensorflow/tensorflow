//===- IndexIntrinsicsOpLowering.h - GPU IndexOps Lowering class *- C++ -*-===//
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
#ifndef MLIR_CONVERSION_GPUCOMMON_INDEXINTRINSICSOPLOWERING_H_
#define MLIR_CONVERSION_GPUCOMMON_INDEXINTRINSICSOPLOWERING_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/StringSwitch.h"

namespace mlir {

// Rewriting that replaces Op with XOp, YOp, or ZOp depending on the dimension
// that Op operates on.  Op is assumed to return an `std.index` value and
// XOp, YOp and ZOp are assumed to return an `llvm.i32` value.  Depending on
// `indexBitwidth`, sign-extend or truncate the resulting value to match the
// bitwidth expected by the consumers of the value.
template <typename Op, typename XOp, typename YOp, typename ZOp>
struct GPUIndexIntrinsicOpLowering : public LLVMOpLowering {
private:
  enum dimension { X = 0, Y = 1, Z = 2, invalid };
  unsigned indexBitwidth;

  static dimension dimensionToIndex(Op op) {
    return llvm::StringSwitch<dimension>(op.dimension())
        .Case("x", X)
        .Case("y", Y)
        .Case("z", Z)
        .Default(invalid);
  }

  static unsigned getIndexBitWidth(LLVMTypeConverter &type_converter) {
    auto dialect = type_converter.getDialect();
    return dialect->getLLVMModule().getDataLayout().getPointerSizeInBits();
  }

public:
  explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter &lowering_)
      : LLVMOpLowering(Op::getOperationName(),
                       lowering_.getDialect()->getContext(), lowering_),
        indexBitwidth(getIndexBitWidth(lowering_)) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto dialect = lowering.getDialect();
    Value newOp;
    switch (dimensionToIndex(cast<Op>(op))) {
    case X:
      newOp = rewriter.create<XOp>(loc, LLVM::LLVMType::getInt32Ty(dialect));
      break;
    case Y:
      newOp = rewriter.create<YOp>(loc, LLVM::LLVMType::getInt32Ty(dialect));
      break;
    case Z:
      newOp = rewriter.create<ZOp>(loc, LLVM::LLVMType::getInt32Ty(dialect));
      break;
    default:
      return matchFailure();
    }

    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, LLVM::LLVMType::getIntNTy(dialect, indexBitwidth), newOp);
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, LLVM::LLVMType::getIntNTy(dialect, indexBitwidth), newOp);
    }

    rewriter.replaceOp(op, {newOp});
    return matchSuccess();
  }
};

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_INDEXINTRINSICSOPLOWERING_H_
