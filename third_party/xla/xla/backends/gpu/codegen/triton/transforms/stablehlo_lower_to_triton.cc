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

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_STABLEHLOLOWERTOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

class LowerTranspose : public mlir::OpRewritePattern<stablehlo::TransposeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::TransposeOp op,
      mlir::PatternRewriter& rewriter) const override {
    SmallVector<int32_t> permutation =
        llvm::to_vector_of<int32_t>(op.getPermutation());
    rewriter.replaceOpWithNewOp<ttir::TransOp>(op, op.getResult().getType(),
                                               op.getOperand(), permutation);
    return mlir::success();
  }
};

class LowerIotaToMakeRange : public mlir::OpRewritePattern<stablehlo::IotaOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::IotaOp op, mlir::PatternRewriter& rewriter) const override {
    auto result_type = op.getResult().getType();

    if (result_type.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "tt.make_range is only supported for 1D outputs.");
    }

    if (!result_type.getElementType().isInteger(32)) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "tt.make_range is only supported for integer types.");
    }

    if (result_type.getElementType().isUnsignedInteger(32)) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "lowering to tt.make_range is only supported for 32 bit signed "
          "integers.");
    }

    auto iota_end = result_type.getDimSize(0);

    rewriter.replaceOpWithNewOp<ttir::MakeRangeOp>(op, result_type,
                                                   /*start=*/0, iota_end);
    return mlir::success();
  }
};

class LowerBroadcastInDim
    : public mlir::OpRewritePattern<stablehlo::BroadcastInDimOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::BroadcastInDimOp op,
      mlir::PatternRewriter& rewriter) const override {
    ::xla::EmitterLocOpBuilder builder(op.getLoc(), rewriter);
    auto input_tensor = op.getOperand();
    auto input_shape = input_tensor.getType().getShape();
    auto output_shape = op.getResult().getType().getShape();
    auto broadcast_dims = op.getBroadcastDimensions();

    if (input_shape.empty()) {
      auto broadcast_dim_input = op.getOperand();
      auto broadcast_dim_input_element_type =
          broadcast_dim_input.getType().getElementType();

      auto extracted = rewriter.create<mlir::tensor::ExtractOp>(
          op.getLoc(), broadcast_dim_input_element_type, broadcast_dim_input);

      rewriter.replaceOpWithNewOp<ttir::SplatOp>(op, op.getResult().getType(),
                                                 extracted);
      return mlir::success();
    }
    int64_t axis = 0;
    int64_t input_dim_id = 0;
    for (int output_dim_id = 0; output_dim_id < output_shape.size();
         output_dim_id++) {
      if (input_dim_id < broadcast_dims.size() &&
          output_dim_id == broadcast_dims[input_dim_id]) {
        // The dim is not broadcasted. Validate matching dim sizes.
        CHECK_EQ(input_shape[input_dim_id], output_shape[output_dim_id]);
        ++input_dim_id;
        axis = output_dim_id + 1;
        continue;
      }
      input_tensor = builder.create<ttir::ExpandDimsOp>(input_tensor, axis);
    }
    rewriter.replaceOpWithNewOp<ttir::BroadcastOp>(op, op.getResult().getType(),
                                                   input_tensor);

    return mlir::success();
  }
};

bool IsConstantAllZeros(stablehlo::ConstantOp constant_op) {
  auto float_elements = constant_op.getValue().tryGetValues<APFloat>();

  if (float_elements) {
    for (auto element : float_elements.value()) {
      if (!element.isExactlyValue(0.0)) {
        return false;
      }
    }
    return true;
  }

  auto integer_elements = constant_op.getValue().tryGetValues<APInt>();
  if (integer_elements) {
    for (auto element : integer_elements.value()) {
      if (element != 0) {
        return false;
      }
    }
    return true;
  }

  return false;
}

class LowerReduce : public mlir::OpRewritePattern<stablehlo::ReduceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::ReduceOp op, mlir::PatternRewriter& rewriter) const override {
    if (mlir::failed(VerifyOpIsCompatibleWithTritonReduce(op, rewriter))) {
      return mlir::failure();
    }

    int32_t axis = op.getDimensions()[0];

    // In case shlo returns a 0 rank tensor triton needs to return a scalar as
    // triton doesn't support 0 rank tensors.
    if (op.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "Triton supports only single result reductions.");
    }
    SmallVector<Type> adjusted_result_types;
    adjusted_result_types.reserve(op.getNumResults());
    for (auto result : op.getResults()) {
      auto shaped_type = dyn_cast<mlir::ShapedType>(result.getType());
      if (shaped_type.getRank() == 0) {
        adjusted_result_types.push_back(shaped_type.getElementType());
      } else {
        adjusted_result_types.push_back(shaped_type);
      }
    }

    auto triton_reduce_op = rewriter.create<ttir::ReduceOp>(
        op.getLoc(), adjusted_result_types, op.getInputs(), axis);

    Region& region = triton_reduce_op.getCombineOp();
    rewriter.cloneRegionBefore(op.getBody(), region, region.end());
    Block& block = region.front();
    for (mlir::BlockArgument& argument : block.getArguments()) {
      auto extract_op = cast<mlir::tensor::ExtractOp>(*argument.user_begin());

      auto scalar_type =
          dyn_cast<mlir::RankedTensorType>(argument.getType()).getElementType();
      argument.setType(scalar_type);
      rewriter.replaceOp(extract_op, argument);
    }

    Operation* terminator = block.getTerminator();
    rewriter.setInsertionPointToEnd(&block);
    SmallVector<Value> return_operands;
    for (Value operand : terminator->getOperands()) {
      auto from_elements =
          operand.getDefiningOp<mlir::tensor::FromElementsOp>();
      return_operands.append(from_elements.getElements().begin(),
                             from_elements.getElements().end());
    }
    rewriter.replaceOpWithNewOp<ttir::ReduceReturnOp>(terminator,
                                                      return_operands);

    // Replace usages of the original op results. If the original result was a
    // 0-rank tensor, we need to wrap the scalar result of tt.reduce in a
    // tensor.from_elements op.
    auto triton_result_type = adjusted_result_types[0];
    auto shaped_type = dyn_cast<mlir::ShapedType>(triton_result_type);
    rewriter.setInsertionPointAfter(triton_reduce_op);
    if (!shaped_type) {
      auto from_elements_opt_results_type =
          mlir::RankedTensorType::get(/*shape=*/{}, triton_result_type);
      auto from_elements_op = rewriter.create<mlir::tensor::FromElementsOp>(
          op.getLoc(), from_elements_opt_results_type,
          triton_reduce_op.getResult());
      rewriter.replaceOp(op, from_elements_op.getResult());
    } else {
      rewriter.replaceOp(op, triton_reduce_op);
    }
    return mlir::success();
  }

  mlir::LogicalResult VerifyOpIsCompatibleWithTritonReduce(
      stablehlo::ReduceOp op, mlir::PatternRewriter& rewriter) const {
    if (mlir::failed(VerifyArgs(op, rewriter))) {
      return mlir::failure();
    }

    if (mlir::failed(VerifyInitValues(op, rewriter))) {
      return mlir::failure();
    }
    if (mlir::failed(VerifyResults(op, rewriter))) {
      return mlir::failure();
    }

    // Check that the reduction is along a single dimension.
    auto dimensions = op.getDimensions();
    if (dimensions.size() != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "tt.reduce only supports single dimension reductions.");
    }

    return mlir::success();
  }

  mlir::LogicalResult VerifyArgs(stablehlo::ReduceOp op,
                                 mlir::PatternRewriter& rewriter) const {
    // Check that all arguments get extracted into a scalar.
    for (mlir::BlockArgument& argument : op.getBody().front().getArguments()) {
      if (!argument.hasOneUse()) {
        return rewriter.notifyMatchFailure(
            op->getLoc(),
            "Expected a single user for an argument to a reduce combiner.");
      }
      if (!dyn_cast<mlir::tensor::ExtractOp>(*argument.user_begin())) {
        return rewriter.notifyMatchFailure(op->getLoc(),
                                           "Expected a tensor extract op as "
                                           "user of reduce combiner argument.");
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult VerifyInitValues(stablehlo::ReduceOp op,
                                       mlir::PatternRewriter& rewriter) const {
    // Check that all init values are all zero constants.
    for (auto init_value : op.getInitValues()) {
      auto constant_op = init_value.getDefiningOp<stablehlo::ConstantOp>();
      if (!constant_op) {
        return rewriter.notifyMatchFailure(
            op->getLoc(),
            "Expected an init value to be a stablehlo.constant op.");
      }

      if (!IsConstantAllZeros(constant_op)) {
        return rewriter.notifyMatchFailure(
            op->getLoc(),
            "Can't lower to triton if init values are not all zero constants.");
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult VerifyResults(stablehlo::ReduceOp op,
                                    mlir::PatternRewriter& rewriter) const {
    // Check that all outputs get created by a from_elements op.
    for (Value operand : op.getBody().front().getTerminator()->getOperands()) {
      if (!operand.hasOneUse()) {
        return rewriter.notifyMatchFailure(
            op->getLoc(),
            "Expected a single user for an output of a reduce combiner.");
      }
      auto from_elements =
          operand.getDefiningOp<mlir::tensor::FromElementsOp>();
      if (!from_elements) {
        return rewriter.notifyMatchFailure(op->getLoc(),
                                           "Expected a from_elements op as "
                                           "user of reduce combiner output.");
      }
    }
    return mlir::success();
  }
};

class StableHLOLowerToTritonPass
    : public impl::StableHLOLowerToTritonPassBase<StableHLOLowerToTritonPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<LowerTranspose, LowerIotaToMakeRange, LowerBroadcastInDim,
                 LowerReduce>(mlir_context);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateStableHLOLowerToTritonPass() {
  return std::make_unique<StableHLOLowerToTritonPass>();
}

}  // namespace mlir::triton::xla
