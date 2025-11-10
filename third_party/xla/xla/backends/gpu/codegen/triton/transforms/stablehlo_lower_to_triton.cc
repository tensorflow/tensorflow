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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
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

      auto extracted = mlir::tensor::ExtractOp::create(rewriter, op.getLoc(),
                                                       broadcast_dim_input);

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
    SmallVector<Type> adjusted_result_types;
    adjusted_result_types.reserve(op.getNumResults());
    for (auto result : op.getResults()) {
      auto shaped_type = cast<mlir::ShapedType>(result.getType());
      if (shaped_type.getRank() == 0) {
        adjusted_result_types.push_back(shaped_type.getElementType());
      } else {
        adjusted_result_types.push_back(shaped_type);
      }
    }

    auto triton_reduce_op = ttir::ReduceOp::create(
        rewriter, op.getLoc(), adjusted_result_types, op.getInputs(), axis);
    Region& triton_reduce_region = triton_reduce_op.getCombineOp();

    mlir::Block& old_block = op.getBody().front();
    llvm::SmallVector<Type> arg_types;
    llvm::SmallVector<mlir::Location> arg_locs;
    for (auto old_arg_type : old_block.getArgumentTypes()) {
      arg_types.push_back(
          llvm::cast<ShapedType>(old_arg_type).getElementType());
      arg_locs.push_back(op.getLoc());
    }
    rewriter.createBlock(&triton_reduce_region, triton_reduce_region.begin(),
                         arg_types, arg_locs);

    mlir::IRMapping mapping;
    Block& triton_reduce_region_block = triton_reduce_region.front();
    rewriter.setInsertionPointToStart(&triton_reduce_region_block);
    for (auto [old_arg, new_arg] :
         llvm::zip(old_block.getArguments(),
                   triton_reduce_region_block.getArguments())) {
      auto to_tensor_op = mlir::tensor::FromElementsOp::create(
          rewriter, op.getLoc(), old_arg.getType(), new_arg);
      mapping.map(old_arg, to_tensor_op);
    }

    for (mlir::Operation& op : old_block.without_terminator()) {
      rewriter.clone(op, mapping);
    }

    SmallVector<Value> return_operands;
    for (Value operand : old_block.getTerminator()->getOperands()) {
      return_operands.push_back(mlir::tensor::ExtractOp::create(
          rewriter, op->getLoc(), mapping.lookup(operand)));
    }
    ttir::ReduceReturnOp::create(rewriter, op.getLoc(), return_operands);

    // Replace usages of the original op results. If the original result was a
    // 0-rank tensor, we need to wrap the scalar result of tt.reduce in a
    // tensor.to_tensor op.
    rewriter.setInsertionPointAfter(triton_reduce_op);
    llvm::SmallVector<Value> new_results;
    for (const auto& triton_result : triton_reduce_op.getResults()) {
      if (mlir::isa<mlir::ShapedType>(triton_result.getType())) {
        new_results.push_back(triton_result);
      } else {
        new_results.push_back(mlir::tensor::FromElementsOp::create(
            rewriter, op.getLoc(), op.getType(0), triton_result));
      }
    }

    rewriter.replaceOp(op, new_results);
    return mlir::success();
  }

  // Verifies that the stablehlo reduce op can be lowered to a triton reduce
  // op.
  // This checks that proper emitting of `tensor.from_elements` and
  // `tensor.extract` on reducer inputs and outputs has happened. It also checks
  // that `tensor.extract` was emitted on the result of the reduce operation if
  // the result is a zero rank tensor.
  mlir::LogicalResult VerifyOpIsCompatibleWithTritonReduce(
      stablehlo::ReduceOp op, mlir::PatternRewriter& rewriter) const {
    // Check that the reduction is along a single dimension.
    auto dimensions = op.getDimensions();
    if (dimensions.size() != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "tt.reduce only supports single dimension reductions.");
    }

    return mlir::success();
  }
};

class LowerReshape : public mlir::OpRewritePattern<stablehlo::ReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::ReshapeOp op, mlir::PatternRewriter& rewriter) const override {
    bool input_is_0d = op.getOperand().getType().getRank() == 0;
    bool output_is_0d = op.getType().getRank() == 0;

    if (input_is_0d && output_is_0d) {
      rewriter.replaceAllUsesWith(op, op.getOperand());
      return mlir::success();
    }

    if (input_is_0d) {
      auto to_scalar = mlir::tensor::ExtractOp::create(rewriter, op->getLoc(),
                                                       op.getOperand());
      rewriter.replaceOpWithNewOp<ttir::SplatOp>(op, op.getType(), to_scalar);
      return mlir::success();
    }

    if (output_is_0d) {
      // We know the input dimensions must be all 1s as reshape input-output
      // must have the same number of elements.
      return LowerRank0ToReduce(op, rewriter);
    }

    // Conservatively prevent Triton from reordering elements within the tile.
    // TODO(b/353637689): see if this restriction can be lifted.
    bool allow_reorder = false;
    rewriter.replaceOpWithNewOp<ttir::ReshapeOp>(
        op, op.getResult().getType(), op.getOperand(), allow_reorder);
    return mlir::success();
  }

  static mlir::LogicalResult LowerRank0ToReduce(
      stablehlo::ReshapeOp op, mlir::PatternRewriter& rewriter) {
    auto input_tensor_type = op.getOperand().getType();

    // First, reshape to a 1D tensor if not already the case. This is needed
    // because triton::ReduceOp can only reduce 1 dimension at a time.
    auto single_dim_tensor = op.getOperand();
    if (input_tensor_type.getRank() > 1) {
      Type output_tensor_type =
          mlir::RankedTensorType::get({1}, input_tensor_type.getElementType());
      single_dim_tensor = ttir::ReshapeOp::create(
          rewriter, op.getLoc(), output_tensor_type, single_dim_tensor,
          /*allow_reorder=*/true);
    }

    // Second, reduce to a scalar.
    ttir::ReduceOp reduction = ttir::ReduceOp::create(
        rewriter, op.getLoc(), single_dim_tensor, /*axis=*/0);

    auto element_type = input_tensor_type.getElementType();
    mlir::Location loc = op.getLoc();
    mlir::Block* reducer =
        rewriter.createBlock(&reduction->getRegion(0), /*insertPt=*/{},
                             /*argTypes=*/
                             {element_type, element_type},
                             /*locs=*/{loc, loc});

    rewriter.setInsertionPointToStart(reducer);
    auto create_binary_op = [&](auto op_type) -> Value {
      return op_type.create(rewriter, reducer->getArgument(0).getLoc(),
                            reducer->getArgument(0), reducer->getArgument(1));
    };
    Value result = mlir::isa<mlir::IntegerType>(element_type)
                       ? create_binary_op(arith::AddIOp())
                       : create_binary_op(arith::AddFOp());
    ttir::ReduceReturnOp::create(rewriter, result.getLoc(), {result});

    rewriter.setInsertionPointAfter(reduction);
    rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(
        op, op.getType(), reduction.getResult());

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
                 LowerReduce, LowerReshape>(mlir_context);

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
