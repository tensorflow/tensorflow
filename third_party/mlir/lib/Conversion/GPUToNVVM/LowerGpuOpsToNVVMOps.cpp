//===- LowerGpuOpsToNVVMOps.cpp - MLIR GPU to NVVM lowering passes --------===//
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
// This file implements a pass to generate NVVMIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {

// Rewriting that replaces the types of a LaunchFunc operation with their
// LLVM counterparts.
struct GPULaunchFuncOpLowering : public LLVMOpLowering {
public:
  explicit GPULaunchFuncOpLowering(LLVMTypeConverter &lowering_)
      : LLVMOpLowering(gpu::LaunchFuncOp::getOperationName(),
                       lowering_.getDialect()->getContext(), lowering_) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.clone(*op)->setOperands(operands);
    return rewriter.replaceOp(op, llvm::None), matchSuccess();
  }
};

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

  static unsigned getIndexBitWidth(LLVMTypeConverter &lowering) {
    auto dialect = lowering.getDialect();
    return dialect->getLLVMModule().getDataLayout().getPointerSizeInBits();
  }

public:
  explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter &lowering_)
      : LLVMOpLowering(Op::getOperationName(),
                       lowering_.getDialect()->getContext(), lowering_),
        indexBitwidth(getIndexBitWidth(lowering_)) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto dialect = lowering.getDialect();
    Value *newOp;
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

// A pass that replaces all occurences of GPU operations with their
// corresponding NVVM equivalent.
//
// This pass does not handle launching of kernels. Instead, it is meant to be
// used on the body region of a launch or the body region of a kernel
// function.
class LowerGpuOpsToNVVMOpsPass : public ModulePass<LowerGpuOpsToNVVMOpsPass> {
public:
  void runOnModule() override {
    ModuleOp m = getModule();

    OwningRewritePatternList patterns;
    LLVMTypeConverter converter(m.getContext());
    populateGpuToNVVMConversionPatterns(converter, patterns);

    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<NVVM::NVVMDialect>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    if (failed(applyPartialConversion(m, target, patterns, &converter)))
      signalPassFailure();
  }
};

} // anonymous namespace

/// Collect a set of patterns to convert from the GPU dialect to NVVM.
void mlir::populateGpuToNVVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  patterns
      .insert<GPULaunchFuncOpLowering,
              GPUIndexIntrinsicOpLowering<gpu::ThreadId, NVVM::ThreadIdXOp,
                                          NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>,
              GPUIndexIntrinsicOpLowering<gpu::BlockDim, NVVM::BlockDimXOp,
                                          NVVM::BlockDimYOp, NVVM::BlockDimZOp>,
              GPUIndexIntrinsicOpLowering<gpu::BlockId, NVVM::BlockIdXOp,
                                          NVVM::BlockIdYOp, NVVM::BlockIdZOp>,
              GPUIndexIntrinsicOpLowering<gpu::GridDim, NVVM::GridDimXOp,
                                          NVVM::GridDimYOp, NVVM::GridDimZOp>>(
          converter);
}

std::unique_ptr<ModulePassBase> mlir::createLowerGpuOpsToNVVMOpsPass() {
  return std::make_unique<LowerGpuOpsToNVVMOpsPass>();
}

static PassRegistration<LowerGpuOpsToNVVMOpsPass>
    pass("lower-gpu-ops-to-nvvm-ops",
         "Generate NVVM operations for gpu operations");
