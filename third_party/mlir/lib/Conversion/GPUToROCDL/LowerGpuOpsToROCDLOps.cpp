//===- LowerGpuOpsToROCDLOps.cpp - MLIR GPU to ROCDL lowering passes ------===//
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
// This file implements a pass to generate ROCDLIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {

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

// A pass that replaces all occurences of GPU device operations with their
// corresponding ROCDL equivalent.
//
// This pass only handles device code and is not meant to be run on GPU host
// code.
class LowerGpuOpsToROCDLOpsPass : public ModulePass<LowerGpuOpsToROCDLOpsPass> {
public:
  void runOnModule() override {
    ModuleOp m = getModule();
    if (!m.getAttrOfType<UnitAttr>(gpu::GPUDialect::getKernelModuleAttrName()))
      return;

    OwningRewritePatternList patterns;
    LLVMTypeConverter converter(m.getContext());
    populateStdToLLVMConversionPatterns(converter, patterns);
    patterns.insert<
        GPUIndexIntrinsicOpLowering<gpu::ThreadId, ROCDL::ThreadIdXOp,
                                    ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>,
        GPUIndexIntrinsicOpLowering<gpu::BlockDim, ROCDL::BlockDimXOp,
                                    ROCDL::BlockDimYOp, ROCDL::BlockDimZOp>,
        GPUIndexIntrinsicOpLowering<gpu::BlockId, ROCDL::BlockIdXOp,
                                    ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>,
        GPUIndexIntrinsicOpLowering<gpu::GridDim, ROCDL::GridDimXOp,
                                    ROCDL::GridDimYOp, ROCDL::GridDimZOp>>(
        converter);

    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect, ROCDL::ROCDLDialect>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    if (failed(applyPartialConversion(m, target, patterns, &converter)))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createLowerGpuOpsToROCDLOpsPass() {
  return std::make_unique<LowerGpuOpsToROCDLOpsPass>();
}

static PassRegistration<LowerGpuOpsToROCDLOpsPass>
    pass("lower-gpu-ops-to-rocdl-ops",
         "Generate ROCDL operations for gpu operations");
