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
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
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

// Removes unit dimensions from a dot result to produce the rank-2 canonical
// form required by Triton. For example, tensor<1x128x256xf32> becomes
// tensor<128x256xf32>.
RankedTensorType CollapseUnitDims(RankedTensorType type) {
  llvm::SmallVector<int64_t> shape;
  for (int64_t dim : type.getShape()) {
    if (dim != 1) {
      shape.push_back(dim);
    }
  }
  return RankedTensorType::get(shape, type.getElementType());
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
  RankedTensorType new_result_type = CollapseUnitDims(result_type);
  if (new_result_type.getRank() != 2) {
    return rewriter.notifyMatchFailure(op_loc,
                                       "Failed to canonicalize result.");
  }

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

class LowerScan : public mlir::OpRewritePattern<::xla::xtile::ScanOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  static SmallVector<Value> CloneBlock(
      mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Block& block,
      ValueRange mapped_args,
      std::optional<ArrayRef<int64_t>> result_shape = std::nullopt) {
    mlir::IRMapping mapping;
    mapping.map(block.getArguments(), mapped_args);

    for (mlir::Operation& op : block.without_terminator()) {
      if (isa<tensor::ExtractOp, tensor::FromElementsOp>(&op)) {
        mapping.map(op.getResult(0), mapping.lookupOrDefault(op.getOperand(0)));
        continue;
      }

      if (op.hasTrait<mlir::OpTrait::ConstantLike>()) {
        mlir::Value cloned_const = rewriter.clone(op, mapping)->getResult(0);
        if (result_shape) {
          Type element_type =
              mlir::getElementTypeOrSelf(cloned_const.getType());
          auto tensor_type =
              mlir::RankedTensorType::get(*result_shape, element_type);
          cloned_const = mlir::stablehlo::BroadcastInDimOp::create(
              rewriter, loc, tensor_type, cloned_const,
              rewriter.getDenseI64ArrayAttr({}));
        }
        mapping.map(op.getResult(0), cloned_const);
        continue;
      }

      auto operands = llvm::map_to_vector(op.getOperands(), [&](Value operand) {
        return mapping.lookupOrDefault(operand);
      });
      auto types =
          llvm::map_to_vector(op.getResultTypes(), [&](Type type) -> Type {
            if (result_shape) {
              return mlir::RankedTensorType::get(
                  *result_shape, mlir::getElementTypeOrSelf(type));
            }
            return mlir::getElementTypeOrSelf(type);
          });

      mlir::OperationState state(loc, op.getName());
      state.addOperands(operands);
      state.addTypes(types);
      state.addAttributes(op.getAttrs());
      mlir::Operation* new_op = rewriter.create(state);

      mapping.map(op.getResults(), new_op->getResults());
    }

    return llvm::map_to_vector(
        block.getTerminator()->getOperands(),
        [&](Value output) { return mapping.lookupOrDefault(output); });
  }

  static ttir::ScanOp CreateTritonScan(::xla::xtile::ScanOp op,
                                       mlir::PatternRewriter& rewriter) {
    auto triton_scan_op = ttir::ScanOp::create(
        rewriter, op.getLoc(), op.getOutputs().getTypes(), op.getInputs(),
        op.getDimension(), op.getIsReverse());

    mlir::Block& old_block = op.getBody().front();

    auto arg_types = llvm::map_to_vector(
        old_block.getArgumentTypes(),
        [](Type type) { return mlir::getElementTypeOrSelf(type); });

    llvm::SmallVector<mlir::Location> arg_locs(arg_types.size(), op.getLoc());
    mlir::Block* new_block = rewriter.createBlock(
        &triton_scan_op.getCombineOp(), triton_scan_op.getCombineOp().begin(),
        arg_types, arg_locs);

    SmallVector<Value> scalar_outputs =
        CloneBlock(rewriter, op.getLoc(), old_block, new_block->getArguments());
    scalar_outputs.truncate(op.getInputs().size());
    ttir::ScanReturnOp::create(rewriter, op.getLoc(), scalar_outputs);

    return triton_scan_op;
  }

  static SmallVector<Value> FoldInitValues(::xla::xtile::ScanOp op,
                                           ttir::ScanOp triton_scan_op,
                                           mlir::PatternRewriter& rewriter) {
    SmallVector<Value> init_and_results;
    init_and_results.reserve(triton_scan_op.getNumResults() * 2);

    // The init only has a unit dimension along the scan dimension and is
    // combined with every element of the scan result.
    for (auto [result, init_val] :
         llvm::zip_equal(triton_scan_op.getResults(), op.getInits())) {
      init_and_results.push_back(ttir::BroadcastOp::create(
          rewriter, op.getLoc(), result.getType(), init_val));
    }

    llvm::append_range(init_and_results, triton_scan_op.getResults());

    auto result_shape = mlir::cast<mlir::RankedTensorType>(
                            triton_scan_op->getResult(0).getType())
                            .getShape();

    SmallVector<Value> tensor_outputs =
        CloneBlock(rewriter, op.getLoc(), op.getBody().front(),
                   init_and_results, result_shape);
    tensor_outputs.truncate(op.getInputs().size());
    return tensor_outputs;
  }

  // Returns a tensor of the same shape as `type` with the index along `axis`.
  static Value CreateIota(mlir::RankedTensorType type, int32_t axis,
                          mlir::Location loc, mlir::PatternRewriter& rewriter) {
    mlir::Type i32_type = rewriter.getI32Type();
    int64_t axis_size = type.getDimSize(axis);
    Value iota = ttir::MakeRangeOp::create(
        rewriter, loc, mlir::RankedTensorType::get({axis_size}, i32_type),
        /*start=*/0, /*end=*/axis_size);
    // Inserting the unit dimensions in increasing order leaves the range in
    // dimension `axis`.
    for (int32_t dim = 0; dim < type.getRank(); ++dim) {
      if (dim != axis) {
        iota = ttir::ExpandDimsOp::create(rewriter, loc, iota, dim);
      }
    }
    return ttir::BroadcastOp::create(rewriter, loc, type.clone(i32_type), iota);
  }

  // Returns the last element that the scan produces along `axis`, which is the
  // element at index 0 for reverse scans and at the last index otherwise,
  // keeping `axis` as a unit dimension.
  //
  // `tt.scan` does not return the final carry, and extracting it with a
  // `tt.gather` would write the entire tile to shared memory (see
  // GatherLoweringHelper::getScratchSizeInBytes()). Instead, mask all other
  // elements to zero and combine them with an `or` reduction over the bitcast
  // integer values. Zero is the neutral element of `or`, so the reduction
  // returns the single unmasked element, and because `or` is idempotent it does
  // so no matter in which order and how often a layout combines the elements.
  // PyTorch inductor uses the same approach (see triton_helpers.select_one).
  static Value ExtractCarry(Value value, int32_t axis, bool is_reverse,
                            mlir::Location loc,
                            mlir::PatternRewriter& rewriter) {
    auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
    mlir::Type element_type = type.getElementType();
    mlir::Type int_type =
        rewriter.getIntegerType(element_type.getIntOrFloatBitWidth());

    int32_t index =
        is_reverse ? 0 : static_cast<int32_t>(type.getDimSize(axis) - 1);
    Value iota = CreateIota(type, axis, loc, rewriter);
    Value indices = mlir::arith::ConstantOp::create(
        rewriter, loc,
        mlir::DenseElementsAttr::get(
            mlir::cast<mlir::ShapedType>(iota.getType()), index));
    Value mask = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::eq, iota, indices);

    mlir::RankedTensorType int_tensor_type = type.clone(int_type);
    if (int_type != element_type) {
      value = ttir::BitcastOp::create(rewriter, loc, int_tensor_type, value);
    }
    Value zero = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(int_tensor_type));
    Value masked =
        mlir::arith::SelectOp::create(rewriter, loc, mask, value, zero);

    // `tt.reduce` drops the reduced dimension and returns a scalar for rank-1
    // inputs.
    SmallVector<int64_t> reduced_shape(type.getShape());
    reduced_shape.erase(reduced_shape.begin() + axis);
    mlir::Type reduced_type =
        reduced_shape.empty()
            ? int_type
            : mlir::RankedTensorType::get(reduced_shape, int_type);
    auto reduce_op =
        ttir::ReduceOp::create(rewriter, loc, reduced_type, masked, axis);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      mlir::Block* block = rewriter.createBlock(
          &reduce_op.getCombineOp(), reduce_op.getCombineOp().begin(),
          {int_type, int_type}, {loc, loc});
      Value combined = mlir::arith::OrIOp::create(
          rewriter, loc, block->getArgument(0), block->getArgument(1));
      ttir::ReduceReturnOp::create(rewriter, loc, combined);
    }

    // Restore the scan dimension as a unit dimension.
    SmallVector<int64_t> carry_shape(type.getShape());
    carry_shape[axis] = 1;
    Value carry = reduce_op.getResult().front();
    if (reduced_shape.empty()) {
      carry = ttir::SplatOp::create(
          rewriter, loc, mlir::RankedTensorType::get(carry_shape, int_type),
          carry);
    } else {
      carry = ttir::ExpandDimsOp::create(rewriter, loc, carry, axis);
    }
    if (int_type != element_type) {
      carry = ttir::BitcastOp::create(
          rewriter, loc, mlir::RankedTensorType::get(carry_shape, element_type),
          carry);
    }
    return carry;
  }

  static SmallVector<Value> ExtractCarries(::xla::xtile::ScanOp op,
                                           ArrayRef<Value> outputs,
                                           mlir::PatternRewriter& rewriter) {
    return llvm::map_to_vector(outputs, [&](Value output) {
      return ExtractCarry(output, op.getDimension(), op.getIsReverse(),
                          op.getLoc(), rewriter);
    });
  }

  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::ScanOp op, mlir::PatternRewriter& rewriter) const override {
    // `xtile.scan` verifies that the inits and carries keep the scan dimension
    // as a unit dimension, which the lowering below relies on.
    ttir::ScanOp triton_scan_op = CreateTritonScan(op, rewriter);

    rewriter.setInsertionPointAfter(triton_scan_op);

    SmallVector<Value> results = FoldInitValues(op, triton_scan_op, rewriter);
    SmallVector<Value> carries = ExtractCarries(op, results, rewriter);
    results.append(carries.begin(), carries.end());

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

class XTileLowerToTritonPass
    : public impl::XTileLowerToTritonPassBase<XTileLowerToTritonPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<CanonicalizeDotScaled, LowerDotScaled, LowerReshape, LowerScan,
                 LowerTranspose>(mlir_context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::triton::xla
