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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/triton/transforms/lowering_utils.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_XTILELOWERTOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

absl::StatusOr<ttir::ScaleDotElemType> GetScaleDotElemType(Type value) {
  Type type = getElementTypeOrSelf(value);
  if (type == mlir::Float8E4M3FNType::get(value.getContext())) {
    return ttir::ScaleDotElemType::E4M3;
  }
  if (type == mlir::Float8E5M2Type::get(value.getContext())) {
    return ttir::ScaleDotElemType::E5M2;
  }
  if (type == mlir::Float4E2M1FNType::get(value.getContext())) {
    return ttir::ScaleDotElemType::E2M1;
  }
  if (type == mlir::BFloat16Type::get(value.getContext())) {
    return ttir::ScaleDotElemType::BF16;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported type: ", ::xla::llvm_ir::DumpToString(type)));
}

bool IsDotScaledCanonical(::xla::xtile::DotScaledOp op) {
  mlir::Attribute dims_attr = op.getDotDimensionNumbersAttr();
  if (!dims_attr ||
      !IsDotDimensionNumbersCanonical(
          mlir::cast<mlir::stablehlo::DotDimensionNumbersAttr>(dims_attr))) {
    return false;
  }

  auto is_rank_2 = [](Value v) {
    return !v || mlir::cast<ShapedType>(v.getType()).getRank() == 2;
  };

  return is_rank_2(op.getLhs()) && is_rank_2(op.getRhs()) &&
         is_rank_2(op.getLhsScale()) && is_rank_2(op.getRhsScale());
}

LogicalResult CanonicalDotScaled(::xla::xtile::DotScaledOp op,
                                 mlir::PatternRewriter& rewriter,
                                 ::xla::xtile::DotScaledOp& canonical_dot) {
  const Location op_loc = op->getLoc();
  if (IsDotScaledCanonical(op)) {
    return rewriter.notifyMatchFailure(op_loc,
                                       "Dot op is already canonicalized.");
  }

  mlir::Attribute dims_attr_raw = op.getDotDimensionNumbersAttr();
  if (!dims_attr_raw) {
    return rewriter.notifyMatchFailure(
        op_loc, "Non-canonical Dot op must have dimension numbers.");
  }
  mlir::stablehlo::DotDimensionNumbersAttr dims_attr =
      mlir::cast<mlir::stablehlo::DotDimensionNumbersAttr>(dims_attr_raw);

  mlir::ImplicitLocOpBuilder builder(op_loc, rewriter);

  Value lhs = op.getLhs();
  if (mlir::failed(CanonicalizeOperand(
          builder, lhs, dims_attr.getLhsContractingDimensions()[0],
          DotOperandSide::kLhs))) {
    return rewriter.notifyMatchFailure(op_loc, "Failed to canonicalize LHS.");
  }

  Value rhs = op.getRhs();
  if (mlir::failed(CanonicalizeOperand(
          builder, rhs, dims_attr.getRhsContractingDimensions()[0],
          DotOperandSide::kRhs))) {
    return rewriter.notifyMatchFailure(op_loc, "Failed to canonicalize RHS.");
  }

  Value lhs_scale = op.getLhsScale();
  if (lhs_scale &&
      mlir::failed(CanonicalizeOperand(
          builder, lhs_scale, dims_attr.getLhsContractingDimensions()[0],
          DotOperandSide::kLhs))) {
    return rewriter.notifyMatchFailure(op_loc,
                                       "Failed to canonicalize LHS scale.");
  }

  Value rhs_scale = op.getRhsScale();
  if (rhs_scale &&
      mlir::failed(CanonicalizeOperand(
          builder, rhs_scale, dims_attr.getRhsContractingDimensions()[0],
          DotOperandSide::kRhs))) {
    return rewriter.notifyMatchFailure(op_loc,
                                       "Failed to canonicalize RHS scale.");
  }

  RankedTensorType result_type = mlir::cast<RankedTensorType>(op.getType());
  RankedTensorType new_result_type = RankedTensorType::get(
      {mlir::cast<ShapedType>(lhs.getType()).getShape()[0],
       mlir::cast<ShapedType>(rhs.getType()).getShape()[1]},
      result_type.getElementType());

  auto canonical_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      rewriter.getContext(), {}, {}, {1}, {0});

  canonical_dot = ::xla::xtile::DotScaledOp::create(
      builder, new_result_type, lhs, rhs, lhs_scale, rhs_scale,
      op.getFastMath(), op.getLhsKPack(), op.getRhsKPack(), canonical_dims);
  return mlir::success();
}

class CanonicalizeDotScaled
    : public mlir::OpRewritePattern<::xla::xtile::DotScaledOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::DotScaledOp op,
      mlir::PatternRewriter& rewriter) const override {
    ::xla::xtile::DotScaledOp new_dot;
    if (mlir::failed(CanonicalDotScaled(op, rewriter, new_dot))) {
      return mlir::failure();
    }

    mlir::Operation* add_op;
    Value acc;
    if (mlir::failed(GetFusedAddUnit(op, rewriter, add_op, acc))) {
      return mlir::failure();
    }

    return CanonicalizeFusedAddUnit(add_op, new_dot, acc, rewriter);
  }
};

class LowerDotScaled
    : public mlir::OpRewritePattern<::xla::xtile::DotScaledOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::DotScaledOp op,
      mlir::PatternRewriter& rewriter) const override {
    const Location op_loc = op->getLoc();
    if (!IsDotScaledCanonical(op)) {
      return rewriter.notifyMatchFailure(op_loc,
                                         "Dot op must be canonicalized.");
    }

    mlir::Operation* add_op;
    Value accumulator;
    if (mlir::failed(GetFusedAddUnit(op, rewriter, add_op, accumulator))) {
      return mlir::failure();
    }

    absl::StatusOr<ttir::ScaleDotElemType> lhs_dot_elem_type =
        GetScaleDotElemType(op.getLhs().getType());
    if (!lhs_dot_elem_type.ok()) {
      return rewriter.notifyMatchFailure(
          op_loc, absl::StrCat("Failed to get dot element type for LHS: ",
                               lhs_dot_elem_type.status().message()));
    }

    absl::StatusOr<ttir::ScaleDotElemType> rhs_dot_elem_type =
        GetScaleDotElemType(op.getRhs().getType());
    if (!rhs_dot_elem_type.ok()) {
      return rewriter.notifyMatchFailure(
          op_loc, absl::StrCat("Failed to get dot element type for RHS: ",
                               rhs_dot_elem_type.status().message()));
    }

    rewriter.setInsertionPoint(add_op);
    ttir::DotScaledOp triton_dot_scaled_op = ttir::DotScaledOp::create(
        rewriter, op.getLoc(), accumulator.getType(), op.getLhs(), op.getRhs(),
        accumulator, op.getLhsScale(), op.getRhsScale(), *lhs_dot_elem_type,
        *rhs_dot_elem_type, op.getFastMath(), op.getLhsKPack(),
        op.getRhsKPack());

    rewriter.replaceOp(add_op, triton_dot_scaled_op);
    return mlir::success();
  }
};

class LowerScan : public mlir::OpRewritePattern<::xla::xtile::ScanOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::ScanOp op, mlir::PatternRewriter& rewriter) const override {
    int32_t axis = op.getDimension();
    bool reverse = op.getIsReverse();

    SmallVector<Type> adjusted_result_types;
    for (auto result : op.getOutputs()) {
      adjusted_result_types.push_back(result.getType());
    }

    auto triton_scan_op =
        ttir::ScanOp::create(rewriter, op.getLoc(), adjusted_result_types,
                             op.getInputs(), axis, reverse);
    mlir::Region& triton_scan_region = triton_scan_op.getCombineOp();

    mlir::Block& old_block = op.getBody().front();
    llvm::SmallVector<Type> arg_types;
    llvm::SmallVector<mlir::Location> arg_locs;
    for (auto old_arg_type : old_block.getArgumentTypes()) {
      arg_types.push_back(
          mlir::cast<mlir::ShapedType>(old_arg_type).getElementType());
      arg_locs.push_back(op.getLoc());
    }
    rewriter.createBlock(&triton_scan_region, triton_scan_region.begin(),
                         arg_types, arg_locs);

    mlir::IRMapping mapping;
    mlir::Block& triton_scan_region_block = triton_scan_region.front();
    rewriter.setInsertionPointToStart(&triton_scan_region_block);
    for (auto [old_arg, new_arg] :
         llvm::zip(old_block.getArguments(),
                   triton_scan_region_block.getArguments())) {
      auto to_tensor_op = mlir::tensor::FromElementsOp::create(
          rewriter, op.getLoc(), old_arg.getType(), new_arg);
      mapping.map(old_arg, to_tensor_op);
    }

    for (mlir::Operation& op_in_block : old_block.without_terminator()) {
      rewriter.clone(op_in_block, mapping);
    }

    SmallVector<Value> return_operands;
    Value zero_idx = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    // The terminator now yields (out_1, ..., out_N, acc_1, ..., acc_N) where
    // out_i and acc_i are the same values for a scan. We only need the first N
    // outputs for the Triton scan block return.
    int num_operands = op.getInputs().size();
    for (int i = 0; i < num_operands; ++i) {
      Value operand = old_block.getTerminator()->getOperand(i);
      Value mapped_op = mapping.lookupOrDefault(operand);
      auto tensor_type =
          mlir::cast<mlir::RankedTensorType>(mapped_op.getType());
      SmallVector<Value> indices(tensor_type.getRank(), zero_idx);
      return_operands.push_back(mlir::tensor::ExtractOp::create(
          rewriter, op.getLoc(), tensor_type.getElementType(), mapped_op,
          indices));
    }
    ttir::ScanReturnOp::create(rewriter, op.getLoc(), return_operands);

    // Apply the initial values if needed.
    rewriter.setInsertionPointAfter(triton_scan_op);

    bool all_neutral = true;
    bool all_add = true;
    for (int i = 0; i < num_operands; ++i) {
      Value init_val = op.getInits()[i];
      if (!mlir::matchPattern(init_val, mlir::m_Zero()) &&
          !mlir::matchPattern(init_val, mlir::m_AnyZeroFloat())) {
        all_neutral = false;
        break;
      }
    }
    for (mlir::Operation& op_in_block : old_block.without_terminator()) {
      if (!isa<stablehlo::AddOp>(op_in_block) &&
          !isa<arith::AddFOp>(op_in_block) &&
          !isa<arith::AddIOp>(op_in_block)) {
        all_add = false;
        break;
      }
    }

    mlir::IRMapping post_mapping;

    if (all_neutral && all_add) {
      // Skip applying inits, map old block results to Triton scan results.
      for (int i = 0; i < num_operands; ++i) {
        Value operand = old_block.getTerminator()->getOperand(i);
        post_mapping.map(operand, triton_scan_op->getResult(i));
      }
    } else {
      for (int i = 0; i < num_operands; ++i) {
        Value init_val = op.getInits()[i];
        auto result_type = mlir::cast<mlir::ShapedType>(
            triton_scan_op->getResult(i).getType());
        SmallVector<int64_t> bcast_dims_vec;
        for (int d = 0; d < result_type.getRank(); ++d) {
          if (d != axis) {
            bcast_dims_vec.push_back(d);
          }
        }
        auto broadcast_dims = rewriter.getDenseI64ArrayAttr(bcast_dims_vec);
        Value broadcasted_init = mlir::stablehlo::BroadcastInDimOp::create(
            rewriter, op.getLoc(), result_type, init_val, broadcast_dims);

        post_mapping.map(old_block.getArgument(i), broadcasted_init);
        post_mapping.map(old_block.getArgument(i + num_operands),
                         triton_scan_op->getResult(i));
      }

      for (mlir::Operation& op_in_block : old_block.without_terminator()) {
        // Handle stablehlo.constant separately. Cloning with a different result
        // type can cause verification failures because the 'value' attribute
        // must match the type.
        if (auto const_op = dyn_cast<stablehlo::ConstantOp>(op_in_block)) {
          auto attr = mlir::cast<mlir::DenseElementsAttr>(const_op.getValue());
          auto result_type = mlir::cast<mlir::ShapedType>(const_op.getType());
          auto triton_shape = mlir::cast<mlir::ShapedType>(
                                  triton_scan_op->getResult(0).getType())
                                  .getShape();
          auto new_result_type = mlir::RankedTensorType::get(
              triton_shape, result_type.getElementType());

          if (attr.isSplat()) {
            auto new_attr = attr.resizeSplat(new_result_type);
            auto new_const = rewriter.create<stablehlo::ConstantOp>(
                op_in_block.getLoc(), new_attr);
            post_mapping.map(const_op.getResult(), new_const.getResult());
            continue;
          }

          // Fallback for non-splat constants. We clone the constant as is and
          // then broadcast it to the desired shape.
          mlir::Operation* cloned = rewriter.clone(*const_op, post_mapping);
          Value cloned_result = cloned->getResult(0);
          SmallVector<int64_t> bcast_dims_vec;
          auto cloned_result_type =
              mlir::cast<mlir::ShapedType>(cloned_result.getType());
          for (int d = 0; d < cloned_result_type.getRank(); ++d) {
            bcast_dims_vec.push_back(d);
          }
          auto broadcast_dims = rewriter.getDenseI64ArrayAttr(bcast_dims_vec);
          Value broadcasted = mlir::stablehlo::BroadcastInDimOp::create(
              rewriter, op_in_block.getLoc(), new_result_type, cloned_result,
              broadcast_dims);
          post_mapping.map(const_op.getResult(), broadcasted);
          continue;
        }

        mlir::OperationState state(op_in_block.getLoc(), op_in_block.getName());
        for (Value operand : op_in_block.getOperands()) {
          state.addOperands(post_mapping.lookupOrDefault(operand));
        }
        for (Type old_type : op_in_block.getResultTypes()) {
          auto triton_shape = mlir::cast<mlir::ShapedType>(
                                  triton_scan_op->getResult(0).getType())
                                  .getShape();
          auto old_shaped_type = mlir::cast<mlir::ShapedType>(old_type);
          state.addTypes(mlir::RankedTensorType::get(
              triton_shape, old_shaped_type.getElementType()));
        }
        state.addAttributes(op_in_block.getAttrs());
        mlir::Operation* cloned = rewriter.create(state);
        for (int i = 0; i < cloned->getNumResults(); ++i) {
          post_mapping.map(op_in_block.getResult(i), cloned->getResult(i));
        }
      }
    }

    SmallVector<Value> final_results;
    for (int i = 0; i < num_operands; ++i) {
      Value operand = old_block.getTerminator()->getOperand(i);
      final_results.push_back(post_mapping.lookupOrDefault(operand));
    }
    for (int i = 0; i < op.getCarries().size(); ++i) {
      auto result_type =
          mlir::cast<mlir::RankedTensorType>(final_results[i].getType());
      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (int64_t d = 0; d < result_type.getRank(); ++d) {
        if (d == axis) {
          int64_t start_idx = reverse ? 0 : (result_type.getDimSize(d) - 1);
          offsets.push_back(rewriter.getIndexAttr(start_idx));
          sizes.push_back(rewriter.getIndexAttr(1));
        } else {
          offsets.push_back(rewriter.getIndexAttr(0));
          sizes.push_back(rewriter.getIndexAttr(result_type.getDimSize(d)));
        }
        strides.push_back(rewriter.getIndexAttr(1));
      }
      auto carry_type =
          mlir::cast<mlir::RankedTensorType>(op.getCarries()[i].getType());
      // The slice is extracted from the corresponding output
      // (final_results[i]).
      Value slice = mlir::tensor::ExtractSliceOp::create(
          rewriter, op.getLoc(), carry_type, final_results[i], offsets, sizes,
          strides);
      final_results.push_back(slice);
    }

    rewriter.replaceOp(op, final_results);
    return mlir::success();
  }
};

class XTileLowerToTritonPass
    : public impl::XTileLowerToTritonPassBase<XTileLowerToTritonPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();

    {
      mlir::RewritePatternSet patterns(mlir_context);
      patterns
          .add<CanonicalizeDotScaled, LowerDotScaled, LowerReshape, LowerScan>(
              mlir_context);
      if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                   std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateXTileLowerToTritonPass() {
  return std::make_unique<XTileLowerToTritonPass>();
}

}  // namespace mlir::triton::xla
