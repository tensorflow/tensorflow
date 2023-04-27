/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/xla_cpu/ir/xla_cpu.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace xla {
namespace cpu {
namespace {

#define GEN_PASS_DEF_LEGALIZELIBRARYOPSPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

class LegalizeLibraryOpsPass
    : public impl::LegalizeLibraryOpsPassBase<LegalizeLibraryOpsPass> {
  void runOnOperation() override;
};

std::optional<xla_cpu::ReductionKind> MatchReductionComputation(
    Region& region) {
  if (!region.hasOneBlock()) {
    return std::nullopt;
  }

  auto ret = dyn_cast<mhlo::ReturnOp>(region.front().getTerminator());
  if (!ret || ret->getNumOperands() != 1) {
    return std::nullopt;
  }

  auto computation = ret.getOperand(0).getDefiningOp();
  if (computation->getNumOperands() != 2 ||
      computation->getOperand(0) != region.front().getArgument(0) ||
      computation->getOperand(1) != region.front().getArgument(1)) {
    return std::nullopt;
  }

  if (isa<mhlo::AddOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_SUM;
  }
  if (isa<mhlo::MulOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_PRODUCT;
  }
  if (isa<mhlo::MinOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_MIN;
  }
  if (isa<mhlo::MaxOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_MAX;
  }

  auto type = computation->getOperandTypes().front().dyn_cast<ShapedType>();
  if (!type || !type.getElementType().isInteger(1)) {
    return std::nullopt;
  }

  if (isa<mhlo::AndOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_MIN;
  }
  if (isa<mhlo::OrOp>(computation)) {
    return xla_cpu::ReductionKind::ALL_REDUCE_MAX;
  }

  return std::nullopt;
}

// Returns a `tensor.empty` with the same shape as `tensor`.
Value CreateEmptyLike(OpBuilder& b, Location loc, Value tensor) {
  auto ty = tensor.getType().cast<ShapedType>();
  auto sizes = tensor::getMixedSizes(b, loc, tensor);
  return b.create<tensor::EmptyOp>(loc, sizes, ty.getElementType());
}

class AllReduceLowering : public OpRewritePattern<mhlo::AllReduceOp> {
  using OpRewritePattern<mhlo::AllReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AllReduceOp op,
                                PatternRewriter& rewriter) const override {
    auto reduction_kind = MatchReductionComputation(op.getRegion());
    if (!reduction_kind) {
      return failure();
    }

    SmallVector<Value> dsts;
    for (auto operand : op->getOperands()) {
      // The operands and results have the same shapes.
      dsts.push_back(CreateEmptyLike(rewriter, op.getLoc(), operand));
    }

    rewriter.replaceOpWithNewOp<xla_cpu::AllReduceOp>(
        op, op->getResultTypes(), op->getOperands(), dsts,
        op.getReplicaGroupsAttr(),
        rewriter.getI64IntegerAttr(op.getChannelHandle()
                                       ? op.getChannelHandle()->getHandle()
                                       : int64_t{0}),
        rewriter.getI32IntegerAttr(op.getUseGlobalDeviceIdsAttr() ? 1 : 0),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(*reduction_kind)));

    return success();
  };
};

template <typename IdOp, typename XlaIdOp>
class IdLowering : public OpRewritePattern<IdOp> {
  using OpRewritePattern<IdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IdOp op,
                                PatternRewriter& rewriter) const override {
    Value id = rewriter.create<XlaIdOp>(op.getLoc());
    // Wrap the scalar in a tensor.
    Value id_tensor = rewriter.create<tensor::FromElementsOp>(
        op.getLoc(), RankedTensorType::get({}, rewriter.getI32Type()), id);
    // And convert it to unsigned. This becomes a noop later.
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
        op,
        RankedTensorType::get({}, IntegerType::get(rewriter.getContext(), 32,
                                                   IntegerType::Unsigned)),
        id_tensor);
    return success();
  };
};

class CollectivePermuteLowering
    : public OpRewritePattern<mhlo::CollectivePermuteOp> {
  using OpRewritePattern<mhlo::CollectivePermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CollectivePermuteOp op,
                                PatternRewriter& rewriter) const override {
    // The result of collective_permute has the same shape as the operand.
    Value dst = CreateEmptyLike(rewriter, op.getLoc(), op.getOperand());
    rewriter.replaceOpWithNewOp<xla_cpu::CollectivePermuteOp>(
        op, op->getResultTypes(), op->getOperand(0), dst,
        op.getSourceTargetPairsAttr(),
        rewriter.getI64IntegerAttr(op.getChannelHandle()
                                       ? op.getChannelHandle()->getHandle()
                                       : int64_t{0}));
    return success();
  };
};

class AllToAllLowering : public OpRewritePattern<mhlo::AllToAllOp> {
  using OpRewritePattern<mhlo::AllToAllOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AllToAllOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    SmallVector<Value> dsts;

    if (!op.getConcatDimensionAttr()) {
      for (auto operand : op->getOperands()) {
        // The operands and results of TupleAllToAll the same shapes.
        dsts.push_back(CreateEmptyLike(rewriter, op.getLoc(), operand));
      }
    } else {
      auto sizes = getValueOrCreateConstantIndexOp(
          b, b.getLoc(),
          tensor::getMixedSizes(b, op.getLoc(), op->getOperand(0)));
      uint64_t split_dimension = *op.getSplitDimension();
      Value split_count = b.create<arith::ConstantIndexOp>(*op.getSplitCount());
      sizes[split_dimension] = b.createOrFold<arith::DivUIOp>(
          b.getIndexType(), sizes[split_dimension], split_count);
      uint64_t concat_dimension = *op.getConcatDimension();
      sizes[concat_dimension] =
          b.createOrFold<arith::MulIOp>(sizes[concat_dimension], split_count);

      dsts.push_back(rewriter.create<tensor::EmptyOp>(
          op.getLoc(), getAsOpFoldResult(sizes),
          op->getResultTypes()[0].cast<ShapedType>().getElementType()));
    }

    rewriter.replaceOpWithNewOp<xla_cpu::AllToAllOp>(
        op, op->getResultTypes(), op->getOperands(), dsts,
        op.getReplicaGroupsAttr(),
        rewriter.getI32IntegerAttr(op.getChannelHandle() ? 1 : 0),
        rewriter.getI64IntegerAttr(op.getChannelHandle()
                                       ? op.getChannelHandle()->getHandle()
                                       : int64_t{0}),
        op.getSplitDimensionAttr(), op.getConcatDimensionAttr(),
        op.getSplitCountAttr());
    return success();
  };
};

class FftLowering : public OpRewritePattern<mhlo::FftOp> {
  using OpRewritePattern<mhlo::FftOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::FftOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // TODO(jreiffers): Support dynamic sizes.
    auto dst = b.create<tensor::EmptyOp>(op.getLoc(), op.getType().getShape(),
                                         op.getType().getElementType());

    auto lengths =
        llvm::to_vector<3>(op.getFftLengthAttr().getValues<int64_t>());
    rewriter.replaceOpWithNewOp<xla_cpu::FftOp>(
        op, op->getResultTypes(), op->getOperand(0), dst,
        static_cast<int32_t>(op.getFftType()),
        rewriter.getI64ArrayAttr(lengths));
    return success();
  };
};

class OutfeedLowering : public OpRewritePattern<mhlo::OutfeedOp> {
  using OpRewritePattern<mhlo::OutfeedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::OutfeedOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<Attribute> result_types;
    for (auto operand : op.getInputs()) {
      result_types.push_back(
          TypeAttr::get(operand.getType().cast<ShapedType>().getElementType()));
    }
    rewriter.create<xla_cpu::OutfeedOp>(
        op.getLoc(), std::nullopt, op.getInputs(), op.getOutfeedConfigAttr(),
        ArrayAttr::get(op->getContext(), result_types));

    // Replacing the op with the token.
    rewriter.replaceOp(op, op.getToken());
    return success();
  };
};

class AfterAllLowering : public OpRewritePattern<mhlo::AfterAllOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AfterAllOp op,
                                PatternRewriter& rewriter) const override {
    // We don't reorder collective ops, so after_all is a no-op.
    rewriter.replaceOp(op, op->getOperand(0));
    return success();
  };
};

class RngBitGeneratorLowering
    : public OpRewritePattern<mhlo::RngBitGeneratorOp> {
  using OpRewritePattern<mhlo::RngBitGeneratorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::RngBitGeneratorOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto state_init = CreateEmptyLike(b, op.getLoc(), op.getOperand());
    auto output_init =
        b.create<tensor::EmptyOp>(op.getLoc(), op.getType(1), ValueRange{});

    rewriter.replaceOpWithNewOp<xla_cpu::RngBitGeneratorOp>(
        op, op->getResultTypes(), op->getOperand(0), state_init, output_init,
        op.getRngAlgorithmAttr());
    return success();
  };
};

class AddDependencyLowering : public OpRewritePattern<mhlo::AddDependencyOp> {
  using OpRewritePattern<mhlo::AddDependencyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AddDependencyOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<xla_cpu::AddDependencyOp>(
        op, op->getResultTypes(), op->getOperands());
    return success();
  };
};

class ConvolutionLowering : public OpRewritePattern<mhlo::ConvolutionOp> {
  using OpRewritePattern<mhlo::ConvolutionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto input_shape = op.getLhs().getType().dyn_cast<ShapedType>();
    auto kernel_shape = op.getRhs().getType().dyn_cast<ShapedType>();
    auto output_shape = op.getResult().getType().dyn_cast<ShapedType>();

    auto dnums = op.getDimensionNumbers();
    auto reversals = op.getWindowReversal();
    // Convolution op is implementable as Eigen convolution if:
    // - input and kernel have non-zero number of elements
    // - input is NHWC order
    // - kernel is HWIO order
    // - some other layout constraints
    auto implementable_as_eigen_convolution = [&]() {
      if (!input_shape || !kernel_shape || !output_shape ||
          !input_shape.hasStaticShape() || !kernel_shape.hasStaticShape() ||
          !output_shape.hasStaticShape()) {
        return false;
      }

      auto primitive_type = input_shape.getElementType();
      if (!(primitive_type.isF32() || primitive_type.isF16())) {
        return false;
      }

      if (llvm::is_contained(input_shape.getShape(), 0) ||
          llvm::is_contained(kernel_shape.getShape(), 0)) {
        return false;
      }

      if (reversals.has_value() &&
          llvm::is_contained(reversals.value().getValues<bool>(), true)) {
        return false;
      }

      auto numSpatialDims = dnums.getOutputSpatialDimensions().size();
      if (numSpatialDims < 1 || numSpatialDims > 3) {
        return false;
      }

      if (!llvm::equal(dnums.getInputSpatialDimensions(),
                       llvm::seq<int64_t>(1, numSpatialDims + 1))) {
        return false;
      }

      if (!llvm::equal(dnums.getKernelSpatialDimensions(),
                       llvm::seq<int64_t>(0, numSpatialDims))) {
        return false;
      }

      if (!llvm::equal(dnums.getOutputSpatialDimensions(),
                       llvm::seq<int64_t>(1, numSpatialDims + 1))) {
        return false;
      }

      if (!op.getWindowStrides().has_value() || !op.getPadding().has_value() ||
          !op.getLhsDilation().has_value() || !op.getRhsDilation().has_value())
        return false;

      auto input_rank = input_shape.getRank();
      auto kernel_rank = kernel_shape.getRank();
      auto output_rank = output_shape.getRank();
      return dnums.getInputBatchDimension() == 0 &&
             dnums.getInputFeatureDimension() == input_rank - 1 &&
             dnums.getOutputBatchDimension() == 0 &&
             dnums.getOutputFeatureDimension() == output_rank - 1 &&
             dnums.getKernelInputFeatureDimension() == kernel_rank - 2 &&
             dnums.getKernelOutputFeatureDimension() == kernel_rank - 1;
    };
    if (!implementable_as_eigen_convolution()) {
      return failure();
    }

    auto dst = b.create<tensor::EmptyOp>(op.getLoc(), op.getType().getShape(),
                                         op.getType().getElementType());

    rewriter.replaceOpWithNewOp<xla_cpu::ConvolutionOp>(
        op, op->getResultTypes(), op.getLhs(), op.getRhs(), dst,
        op.getWindowStridesAttr(), op.getPaddingAttr(), op.getLhsDilationAttr(),
        op.getRhsDilationAttr(), op.getWindowReversalAttr(),
        rewriter.getI64IntegerAttr(dnums.getInputBatchDimension()),
        rewriter.getI64IntegerAttr(dnums.getInputFeatureDimension()),
        rewriter.getI64ArrayAttr(dnums.getInputSpatialDimensions()),
        rewriter.getI64IntegerAttr(dnums.getKernelInputFeatureDimension()),
        rewriter.getI64IntegerAttr(dnums.getKernelOutputFeatureDimension()),
        rewriter.getI64ArrayAttr(dnums.getKernelSpatialDimensions()),
        rewriter.getI64IntegerAttr(dnums.getOutputBatchDimension()),
        rewriter.getI64IntegerAttr(dnums.getOutputFeatureDimension()),
        rewriter.getI64ArrayAttr(dnums.getOutputSpatialDimensions()),
        op.getFeatureGroupCountAttr(), op.getBatchGroupCountAttr(),
        op.getPrecisionConfigAttr());
    return success();
  };
};

void LegalizeLibraryOpsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

  // Convert mhlo library operations to XLA cpu ops.
  RewritePatternSet patterns(ctx);
  patterns.insert<AddDependencyLowering, AfterAllLowering, AllReduceLowering,
                  AllToAllLowering, CollectivePermuteLowering,
                  ConvolutionLowering, FftLowering,
                  IdLowering<mhlo::PartitionIdOp, xla_cpu::PartitionIdOp>,
                  IdLowering<mhlo::ReplicaIdOp, xla_cpu::ReplicaIdOp>,
                  OutfeedLowering, RngBitGeneratorLowering>(ctx);

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeLibraryOpsPass() {
  return std::make_unique<LegalizeLibraryOpsPass>();
}

}  // namespace cpu
}  // namespace xla
