/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/triton_sparse_extensions.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

using namespace mlir;  // NOLINT(build/namespaces)

// The functions below are defined in AccelerateMatmul.cpp.
namespace mlir::triton::gpu {
SmallVector<unsigned, 3> getWarpsPerTile(
    Operation *dotOp, ArrayRef<int64_t> shape, int version, int numWarps,
    const SmallVector<unsigned, 3> &instrShape);
int computeOrigBitWidth(Value x);
Value getSharedMemMMAOperand(Value v, mlir::PatternRewriter &rewriter,
                             int opIdx, bool allowTranspose);
}  // namespace mlir::triton::gpu

namespace {

struct TritonSparseDotPattern
    : public OpConversionPattern<triton::gpu::SparseDotOp> {
  using OpConversionPattern<triton::gpu::SparseDotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::SparseDotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    RankedTensorType origType = cast<RankedTensorType>(op.getType());
    auto origShape = origType.getShape();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();
    int numWarps = typeConverter->getNumWarps();
    int threadsPerWarp = typeConverter->getThreadsPerWarp();
    int numCTAs = typeConverter->getNumCTAs();

    auto rank = origShape.size();
    auto numElements = product<int64_t>(origShape);
    SmallVector<unsigned> retSizePerThread(rank, 1);
    if (numElements / (numWarps * threadsPerWarp) >= 4) {
      retSizePerThread[rank - 1] = 2;
      retSizePerThread[rank - 2] = 2;
    }
    if (numElements / (numWarps * threadsPerWarp) >= 16) {
      retSizePerThread[rank - 1] = 4;
      retSizePerThread[rank - 2] = 4;
    }
    SmallVector<unsigned> retOrder(rank);
    for (unsigned i = 0; i < rank; ++i) retOrder[i] = rank - 1 - i;
    Attribute dEncoding = triton::gpu::BlockedEncodingAttr::get(
        getContext(), origShape, retSizePerThread, retOrder, numWarps,
        threadsPerWarp, numCTAs);
    RankedTensorType retType =
        RankedTensorType::get(origShape, origType.getElementType(), dEncoding);

    // a & b must be of smem layout
    auto aType = cast<RankedTensorType>(adaptor.getA().getType());
    auto bType = cast<RankedTensorType>(adaptor.getB().getType());
    Type aEltType = aType.getElementType();
    Type bEltType = bType.getElementType();
    Attribute aEncoding = aType.getEncoding();
    Attribute bEncoding = bType.getEncoding();
    if (!aEncoding || !bEncoding) return failure();
    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
    if (!isa<triton::gpu::DotOperandEncodingAttr>(aEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 0, dEncoding, aEltType);
      auto dstType =
          RankedTensorType::get(aType.getShape(), aEltType, encoding);
      a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), dstType, a);
    }
    if (!isa<triton::gpu::DotOperandEncodingAttr>(bEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 1, dEncoding, bEltType);
      auto dstType =
          RankedTensorType::get(bType.getShape(), bEltType, encoding);
      b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), dstType, b);
    }
    c = rewriter.create<triton::gpu::ConvertLayoutOp>(c.getLoc(), retType, c);

    // aMeta must be of smem layout
    auto aMetaType = cast<RankedTensorType>(adaptor.getAMeta().getType());
    Attribute aMetaEncoding = aMetaType.getEncoding();
    if (!aMetaEncoding) return failure();
    Value aMeta = adaptor.getAMeta();
    if (!isa<triton::gpu::SparseDotMetaEncodingAttr>(aMetaEncoding)) {
      Attribute encoding =
          triton::gpu::SparseDotMetaEncodingAttr::get(getContext(), dEncoding);
      auto dstType = RankedTensorType::get(
          aMetaType.getShape(), aMetaType.getElementType(), encoding);
      aMeta = rewriter.create<triton::gpu::ConvertLayoutOp>(aMeta.getLoc(),
                                                            dstType, aMeta);
    }

    auto new_op = rewriter.replaceOpWithNewOp<triton::gpu::SparseDotOp>(
        op, retType, a, b, c, aMeta);
    for (const NamedAttribute attr : op->getAttrs()) {
      if (!new_op->hasAttr(attr.getName()))
        new_op->setAttr(attr.getName(), attr.getValue());
    }

    return success();
  }
};

class AddSparseDotEncodingPass
    : public PassWrapper<AddSparseDotEncodingPass, OperationPass<ModuleOp>> {
 public:
  AddSparseDotEncodingPass() = default;
  AddSparseDotEncodingPass(int32_t num_warps, int32_t threads_per_warp,
                           int32_t num_ctas) {
    num_warps_ = num_warps;
    threads_per_warp_ = threads_per_warp;
    num_ctas_ = num_ctas;
  }
  AddSparseDotEncodingPass(const AddSparseDotEncodingPass &other) {
    num_warps_ = other.num_warps_;
    threads_per_warp_ = other.threads_per_warp_;
    num_ctas_ = other.num_ctas_;
  };

  StringRef getArgument() const override { return "add-sparse-encoding"; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    TritonGPUTypeConverter typeConverter(context, num_warps_, threads_per_warp_,
                                         num_ctas_);
    auto pattern =
        std::make_unique<TritonSparseDotPattern>(typeConverter, context);
    RewritePatternSet patterns(context, std::move(pattern));
    TritonGPUConversionTarget target(*context, typeConverter);
    target.addDynamicallyLegalOp<triton::gpu::SparseDotOp>(
        [](triton::gpu::SparseDotOp op) {
          return op.getAMeta().getType().getEncoding() != nullptr;
        });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddSparseDotEncodingPass)

 private:
  Option<int32_t> num_warps_{
      *this, "num-warps", llvm::cl::desc("number of warps"), llvm::cl::init(4)};
  Option<int32_t> threads_per_warp_{
      *this, "threads-per-warp", llvm::cl::desc("number of threads per warp"),
      llvm::cl::init(32)};
  Option<int32_t> num_ctas_{*this, "num-ctas",
                            llvm::cl::desc("number of ctas in a cga"),
                            llvm::cl::init(1)};
};

class SparseBlockedToMMA : public RewritePattern {
  using ConvertLayoutOp = triton::gpu::ConvertLayoutOp;
  using SparseDotOp = triton::gpu::SparseDotOp;
  using SparseDotMetaEncodingAttr = triton::gpu::SparseDotMetaEncodingAttr;
  using NvidiaMmaEncodingAttr = triton::gpu::NvidiaMmaEncodingAttr;

 public:
  SparseBlockedToMMA(MLIRContext *context, int compute_capability)
      : RewritePattern(SparseDotOp::getOperationName(), 2, context),
        compute_capability_(compute_capability) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto dotOp = cast<SparseDotOp>(op);
    auto ctx = op->getContext();
    Value a = dotOp.getA();
    Value b = dotOp.getB();

    // Check data-types and SM compatibility
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    assert(compute_capability_ >= 80 &&
           "SparseDot is supported on Ampere and higher");
    bool allowV3 = !triton::tools::getBoolEnv("DISABLE_MMA_V3");
    int versionMajor = compute_capability_ >= 90 && allowV3 ? 3 : 2;

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = triton::gpu::getShapePerCTA(oldRetType);
    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    auto CTALayout = triton::gpu::getCTALayout(oldRetType.getEncoding());

    auto instrShape =
        mmaVersionToInstrShape(versionMajor, retShapePerCTA,
                               cast<RankedTensorType>(a.getType()), numWarps);
    auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                        numWarps, instrShape);
    NvidiaMmaEncodingAttr mmaEnc =
        NvidiaMmaEncodingAttr::get(ctx, versionMajor, /*versionMinor=*/0,
                                   warpsPerTile, CTALayout, instrShape);
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);

    if (versionMajor == 2) {
      int minBitwidth = std::min(triton::gpu::computeOrigBitWidth(a),
                                 triton::gpu::computeOrigBitWidth(b));
      int kWidth = 32 / minBitwidth;

      // convert A operand
      auto oldAType = cast<RankedTensorType>(a.getType());
      auto newAEncoding = DotOperandEncodingAttr::get(ctx, 0, mmaEnc, kWidth);
      auto newAType = RankedTensorType::get(
          oldAType.getShape(), oldAType.getElementType(), newAEncoding);
      a = rewriter.create<ConvertLayoutOp>(a.getLoc(), newAType, a);

      // convert B operand
      auto oldBType = cast<RankedTensorType>(b.getType());
      auto newBEncoding = DotOperandEncodingAttr::get(ctx, 1, mmaEnc, kWidth);
      auto newBType = RankedTensorType::get(
          oldBType.getShape(), oldBType.getElementType(), newBEncoding);
      b = rewriter.create<ConvertLayoutOp>(b.getLoc(), newBType, b);
    } else {
      auto eltType = dotOp.getA().getType().getElementType();
      // In MMAV3 transpose is only supported for f16 and bf16.
      bool allowTranspose = eltType.isF16() || eltType.isBF16();
      a = triton::gpu::getSharedMemMMAOperand(a, rewriter, 0, allowTranspose);
      b = triton::gpu::getSharedMemMMAOperand(b, rewriter, 1, allowTranspose);
    }

    // convert metadata
    Value meta = dotOp.getAMeta();
    auto oldMetaType = cast<RankedTensorType>(meta.getType());
    auto newMetaType = RankedTensorType::get(
        oldMetaType.getShape(), oldMetaType.getElementType(),
        SparseDotMetaEncodingAttr::get(ctx, mmaEnc));
    meta = rewriter.create<ConvertLayoutOp>(meta.getLoc(), newMetaType, meta);

    // convert dot instruction
    auto newDot = rewriter.create<SparseDotOp>(dotOp.getLoc(), newRetType, a, b,
                                               newAcc, meta);

    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, oldRetType,
                                                 newDot.getResult());
    return success();
  }

 private:
  int compute_capability_;
};

class SparseBlockedToMMAPass
    : public PassWrapper<SparseBlockedToMMAPass, OperationPass<ModuleOp>> {
 public:
  SparseBlockedToMMAPass() = default;

  StringRef getArgument() const override { return "sparse-blocked-to-mma"; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    auto compute_capability = getNVIDIAComputeCapability(module);
    auto pattern =
        std::make_unique<SparseBlockedToMMA>(context, compute_capability);
    RewritePatternSet patterns(context, std::move(pattern));
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseBlockedToMMAPass)
};

}  // namespace

std::unique_ptr<Pass> xla::gpu::createAddSparseDotEncodingPass(
    int32_t num_warps, int32_t threads_per_warp, int32_t num_ctas) {
  return std::make_unique<AddSparseDotEncodingPass>(num_warps, threads_per_warp,
                                                    num_ctas);
}

std::unique_ptr<mlir::Pass> xla::gpu::createSparseBlockedToMMAPass() {
  return std::make_unique<SparseBlockedToMMAPass>();
}

void xla::gpu::registerSparsePasses() {
  registerPass([] { return std::make_unique<AddSparseDotEncodingPass>(); });
  registerPass([] { return std::make_unique<SparseBlockedToMMAPass>(); });
}
