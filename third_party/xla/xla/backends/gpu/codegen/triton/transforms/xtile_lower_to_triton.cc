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
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/triton/transforms/lowering_utils.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_XTILELOWERTOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

absl::StatusOr<ttir::ScaleDotElemType> GetScaleDotElemType(Type type) {
  MLIRContext* context = type.getContext();
  if (type == mlir::Float8E4M3FNType::get(context)) {
    return ttir::ScaleDotElemType::E4M3;
  }
  if (type == mlir::Float8E5M2Type::get(context)) {
    return ttir::ScaleDotElemType::E5M2;
  }
  if (type == mlir::Float4E2M1FNType::get(context)) {
    return ttir::ScaleDotElemType::E2M1;
  }
  if (type == mlir::BFloat16Type::get(context)) {
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
      op.getFastMath(), op.getLhsKPack(), op.getRhsKPack(),
      op.getLhsElemTypeAttr().getValue(), op.getRhsElemTypeAttr().getValue(),
      canonical_dims);
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
        GetScaleDotElemType(op.getLhsElemTypeAttr().getValue());
    if (!lhs_dot_elem_type.ok()) {
      return rewriter.notifyMatchFailure(
          op_loc, absl::StrCat("Failed to get dot element type for LHS: ",
                               lhs_dot_elem_type.status().message()));
    }

    absl::StatusOr<ttir::ScaleDotElemType> rhs_dot_elem_type =
        GetScaleDotElemType(op.getRhsElemTypeAttr().getValue());
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

static void CloneScanBlockToTriton(mlir::PatternRewriter& rewriter,
                                   mlir::Location loc, int num_operands,
                                   mlir::Block& old_block,
                                   mlir::Region& triton_scan_region) {
  llvm::SmallVector<Type> arg_types;
  llvm::SmallVector<mlir::Location> arg_locs;
  for (auto type : old_block.getArgumentTypes()) {
    if (auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(type)) {
      type = shaped_type.getElementType();
    }
    arg_types.push_back(type);
    arg_locs.push_back(loc);
  }
  mlir::Block* new_block = rewriter.createBlock(
      &triton_scan_region, triton_scan_region.begin(), arg_types, arg_locs);

  mlir::IRMapping mapping;
  rewriter.setInsertionPointToStart(new_block);
  for (auto [old_arg, new_arg] :
       llvm::zip(old_block.getArguments(), new_block->getArguments())) {
    mapping.map(old_arg, new_arg);
  }

  for (mlir::Operation& op : old_block.without_terminator()) {
    rewriter.clone(op, mapping);
  }

  SmallVector<Value> return_operands;
  // The terminator now yields (out_1, ..., out_N, acc_1, ..., acc_N) where
  // out_i and acc_i are the same values for a scan. We only need the first N
  // outputs for the Triton scan block return.
  for (int i = 0; i < num_operands; ++i) {
    Value operand = old_block.getTerminator()->getOperand(i);
    Value mapped_op = mapping.lookupOrDefault(operand);
    return_operands.push_back(mapped_op);
  }

  ttir::ScanReturnOp::create(rewriter, loc, return_operands);
}

class LowerScan : public mlir::OpRewritePattern<::xla::xtile::ScanOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::ScanOp op, mlir::PatternRewriter& rewriter) const override {
    int32_t axis = op.getDimension();
    bool reverse = op.getIsReverse();
    int num_operands = op.getInputs().size();
    SmallVector<Type> adjusted_result_types;
    for (auto result : op.getOutputs()) {
      adjusted_result_types.push_back(result.getType());
    }

    auto triton_scan_op =
        ttir::ScanOp::create(rewriter, op.getLoc(), adjusted_result_types,
                             op.getInputs(), axis, reverse);
    mlir::Region& triton_scan_region = triton_scan_op.getCombineOp();

    mlir::Block& old_block = op.getBody().front();
    CloneScanBlockToTriton(rewriter, op.getLoc(), num_operands, old_block,
                           triton_scan_region);

    // Apply the initial values if needed.
    rewriter.setInsertionPointAfter(triton_scan_op);

    bool all_add = llvm::all_of(
        old_block.without_terminator(), [](mlir::Operation& op_in_block) {
          return isa<stablehlo::AddOp, arith::AddFOp, arith::AddIOp,
                     tensor::ExtractOp, tensor::FromElementsOp>(op_in_block);
        });

    if (!all_add) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "Only addition scans are supported");
    }

    SmallVector<Value> final_results;
    for (int i = 0; i < num_operands; ++i) {
      Value result = triton_scan_op->getResult(i);
      Value init_val = op.getInits()[i];
      auto result_type = mlir::cast<mlir::RankedTensorType>(result.getType());
      auto bcast_type = RankedTensorType::get(result_type.getShape(),
                                              result_type.getElementType());

      SmallVector<int64_t> bcast_dims;
      for (int64_t d = 0; d < result_type.getRank(); ++d) {
        if (d != axis) {
          bcast_dims.push_back(d);
        }
      }

      Value init_bcast = mlir::stablehlo::BroadcastInDimOp::create(
          rewriter, op.getLoc(), bcast_type, init_val,
          rewriter.getDenseI64ArrayAttr(bcast_dims));
      if (mlir::isa<mlir::FloatType>(result_type.getElementType())) {
        result =
            arith::AddFOp::create(rewriter, op.getLoc(), init_bcast, result);
      } else {
        result =
            arith::AddIOp::create(rewriter, op.getLoc(), init_bcast, result);
      }
      final_results.push_back(result);
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

}  // namespace mlir::triton::xla
