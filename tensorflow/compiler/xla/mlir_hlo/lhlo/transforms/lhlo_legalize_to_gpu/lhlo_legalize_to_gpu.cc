/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering LHLO dialect to GPU dialect.

#include <cstdint>
#include <optional>

#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace lmhlo {

#define GEN_PASS_DEF_LHLOLEGALIZETOGPUPASS
#include "lhlo/transforms/lmhlo_passes.h.inc"

namespace {

// A simple translation of LHLO reduce operations to a corresponding gpu
// launch operation. The transformation does no tiling and also only supports
// 1d results.
class LhloReduceToGPULaunchConverter : public OpConversionPattern<ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReduceOp reduceOp, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = reduceOp.getLoc();
    // Only support 1d reductions for now.
    int64_t size = 0;
    for (auto result : reduceOp.getOut()) {
      auto shapedType = result.getType().dyn_cast<ShapedType>();
      if (!shapedType || shapedType.getRank() != 1) {
        return failure();
      }
      auto dimSize = shapedType.getDimSize(0);
      if (size && size != dimSize) {
        return failure();
      }
      size = dimSize;
    }

    auto reducingDimension = *reduceOp.getDimensions().value_begin<APInt>();

    // Require all inputs to have the same shape.
    int64_t reduceDimSize = 0;
    for (auto input : reduceOp.getInputs()) {
      auto shapedType = input.getType().dyn_cast<ShapedType>();
      if (!shapedType || !shapedType.hasStaticShape()) {
        return failure();
      }
      reduceDimSize = shapedType.getDimSize(reducingDimension.getSExtValue());
    }

    // Create a launch that is parallel in the result dimension.
    auto blockSizeX = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), size));
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    auto launchOp = rewriter.create<mlir::gpu::LaunchOp>(loc, one, one, one,
                                                         blockSizeX, one, one);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&launchOp.getBody().front());
      auto index = launchOp.getThreadIds().x;

      // Load the initial value and store it to the output.
      for (auto pair : llvm::zip(reduceOp.getInitValues(), reduceOp.getOut())) {
        auto initValue =
            rewriter.create<mlir::memref::LoadOp>(loc, std::get<0>(pair));
        rewriter.create<mlir::memref::StoreOp>(
            loc, initValue, std::get<1>(pair), ArrayRef<Value>{index});
      }

      // Insert a loop into the body to compute the reduction. The loop ranges
      // from [0.dim).
      auto zero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
      // TODO(b/137624192) Use dimOp to make it shape independent.
      auto upper = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), reduceDimSize));
      auto step = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
      auto loop = rewriter.create<mlir::scf::ForOp>(loc, zero, upper, step);

      rewriter.setInsertionPointToStart(loop.getBody());
      // Compute memrefs for the value to reduce. This makes it easier to just
      // inline the body.
      auto output = *reduceOp.getOut().begin();
      auto resType = MemRefType::get(
          std::nullopt, getElementTypeOrSelf(output.getType()),
          makeStridedLinearLayoutMap(std::nullopt, ShapedType::kDynamic,
                                     rewriter.getContext()));
      OpFoldResult offset = launchOp.getThreadIds().x;
      auto oneAttr = rewriter.getI64IntegerAttr(1);
      OpFoldResult size = oneAttr;
      OpFoldResult stride = oneAttr;
      auto accumulator = rewriter.create<memref::SubViewOp>(
          loc, resType, output, offset, size, stride);
      llvm::SmallVector<Value, 4> indexings;
      Value inputBuffer = reduceOp.getInputs().front();
      auto inputTypeRank = inputBuffer.getType().cast<MemRefType>().getRank();

      Value input = *reduceOp.operand_begin();
      SmallVector<OpFoldResult> offsets = llvm::to_vector<4>(llvm::map_range(
          llvm::seq<int>(0, inputTypeRank), [&](int dim) -> OpFoldResult {
            return dim == reducingDimension ? loop.getInductionVar()
                                            : launchOp.getThreadIds().x;
          }));
      SmallVector<OpFoldResult> sizes(inputTypeRank, oneAttr);
      SmallVector<OpFoldResult> strides(inputTypeRank, oneAttr);
      auto rhs = rewriter.create<memref::SubViewOp>(
          loc, accumulator.getType(), input, offsets, sizes, strides);

      // Now copy over the actual body of the reduction, leaving out the
      // terminator.
      IRMapping mapping;
      mapping.map(reduceOp.getBody().getArgument(0), accumulator);
      mapping.map(reduceOp.getBody().getArgument(1), rhs);
      mapping.map(reduceOp.getBody().getArgument(2), accumulator);
      for (auto& nested : reduceOp.getBody().front().without_terminator()) {
        auto* clone = rewriter.clone(nested, mapping);
        for (auto pair : llvm::zip(nested.getResults(), clone->getResults())) {
          mapping.map(std::get<0>(pair), std::get<1>(pair));
        }
      }

      // Finally, insert the terminator for the launchOp.
      rewriter.setInsertionPointToEnd(&launchOp.getBody().front());
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
    }

    rewriter.eraseOp(reduceOp);
    return success();
  };
};

struct LhloLegalizeToGpuPass
    : public impl::LhloLegalizeToGpuPassBase<LhloLegalizeToGpuPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           memref::MemRefDialect, func::FuncDialect,
                           gpu::GPUDialect, scf::SCFDialect, LmhloDialect>();
    target.addIllegalOp<ReduceOp>();
    auto func = getOperation();
    patterns.add<LhloReduceToGPULaunchConverter>(func.getContext());
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToGpuPass() {
  return std::make_unique<LhloLegalizeToGpuPass>();
}

}  // namespace lmhlo
}  // namespace mlir
