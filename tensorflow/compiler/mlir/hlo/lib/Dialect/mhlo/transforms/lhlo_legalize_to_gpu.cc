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

#include "llvm/ADT/ArrayRef.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace lmhlo {
namespace {

// A simple translation of LHLO reduce operations to a corresponding gpu
// launch operation. The transformation does no tiling and also only supports
// 1d results.
class LhloReduceToGPULaunchConverter : public OpConversionPattern<ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReduceOp reduce_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = reduce_op.getLoc();
    // Only support 1d reductions for now.
    int64_t size = 0;
    for (auto result : reduce_op.out()) {
      auto shaped_type = result.getType().dyn_cast<ShapedType>();
      if (!shaped_type || shaped_type.getRank() != 1) {
        return failure();
      }
      auto dim_size = shaped_type.getDimSize(0);
      if (size && size != dim_size) {
        return failure();
      }
      size = dim_size;
    }

    auto reducing_dimension = *reduce_op.dimensions().int_value_begin();

    // Require all inputs to have the same shape.
    int64_t reduce_dim_size = 0;
    for (auto input : reduce_op.operands()) {
      auto shaped_type = input.getType().dyn_cast<ShapedType>();
      if (!shaped_type || !shaped_type.hasStaticShape()) {
        return failure();
      }
      reduce_dim_size =
          shaped_type.getDimSize(reducing_dimension.getSExtValue());
    }

    // Create a launch that is parallel in the result dimension.
    auto block_size_x = rewriter.create<mlir::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), size));
    auto one = rewriter.create<mlir::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    auto launch_op = rewriter.create<mlir::gpu::LaunchOp>(
        loc, one, one, one, block_size_x, one, one);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&launch_op.body().front());
      auto index = launch_op.getThreadIds().x;

      // Load the initial value and store it to the output.
      for (auto pair : llvm::zip(reduce_op.init_values(), reduce_op.out())) {
        auto init_value = rewriter.create<mlir::LoadOp>(loc, std::get<0>(pair));
        rewriter.create<mlir::StoreOp>(loc, init_value, std::get<1>(pair),
                                       ArrayRef<Value>{index});
      }

      // Insert a loop into the body to compute the reduction. The loop ranges
      // from [0.dim).
      auto zero = rewriter.create<mlir::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
      // TODO(b/137624192) Use dimOp to make it shape independent.
      auto upper = rewriter.create<mlir::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), reduce_dim_size));
      auto step = rewriter.create<mlir::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
      auto loop = rewriter.create<mlir::scf::ForOp>(loc, zero, upper, step);

      rewriter.setInsertionPointToStart(loop.getBody());
      // Compute memrefs for the value to reduce. This makes it easier to just
      // inline the body.
      auto output = *reduce_op.out().begin();
      // TODO(herhut) Move this to the SliceOp builder.
      auto resType = MemRefType::get(
          llvm::None, output.getType().cast<MemRefType>().getElementType(),
          makeStridedLinearLayoutMap(llvm::None,
                                     MemRefType::getDynamicStrideOrOffset(),
                                     rewriter.getContext()));
      auto accumulator = rewriter.create<mlir::linalg::SliceOp>(
          loc, resType, output, ArrayRef<Value>{launch_op.getThreadIds().x});
      llvm::SmallVector<Value, 4> indexings;
      auto input_buffer = *reduce_op.operands().begin();
      auto input_type = input_buffer.getType().cast<MemRefType>();
      for (int64_t dim = 0; dim < input_type.getRank(); ++dim) {
        indexings.push_back(dim == reducing_dimension
                                ? loop.getInductionVar()
                                : launch_op.getThreadIds().x);
      }
      // TODO(herhut) Move this to the SliceOp builder.
      auto input = *reduce_op.operand_begin();
      auto rhs = rewriter.create<mlir::linalg::SliceOp>(
          loc,
          MemRefType::get(
              llvm::None, input_type.getElementType(),
              makeStridedLinearLayoutMap(llvm::None,
                                         MemRefType::getDynamicStrideOrOffset(),
                                         rewriter.getContext())),
          input, indexings);

      // Now copy over the actual body of the reduction, leaving out the
      // terminator.
      BlockAndValueMapping mapping;
      mapping.map(reduce_op.body().getArgument(0), accumulator);
      mapping.map(reduce_op.body().getArgument(1), rhs);
      mapping.map(reduce_op.body().getArgument(2), accumulator);
      for (auto& nested : reduce_op.body().front().without_terminator()) {
        auto clone = rewriter.clone(nested, mapping);
        for (auto pair : llvm::zip(nested.getResults(), clone->getResults())) {
          mapping.map(std::get<0>(pair), std::get<1>(pair));
        }
      }

      // Finally, insert the terminator for the launchOp.
      rewriter.setInsertionPointToEnd(&launch_op.body().front());
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
    }

    rewriter.eraseOp(reduce_op);
    return success();
  };
};

struct LhloLegalizeToGpuPass
    : public PassWrapper<LhloLegalizeToGpuPass, FunctionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, linalg::LinalgDialect,
                    scf::SCFDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           gpu::GPUDialect, scf::SCFDialect, LmhloDialect>();
    target.addIllegalOp<ReduceOp>();
    auto func = getFunction();
    patterns.insert<LhloReduceToGPULaunchConverter>(func.getContext());
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<FunctionPass> createLegalizeToGpuPass() {
  return std::make_unique<LhloLegalizeToGpuPass>();
}

}  // namespace lmhlo
}  // namespace mlir
