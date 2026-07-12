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

class LowerScan : public mlir::OpRewritePattern<::xla::xtile::ScanOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  static SmallVector<Value> CloneBlock(
      mlir::PatternRewriter& rewriter, mlir::Location loc,
      mlir::Block& old_block, ValueRange mapped_args,
      std::optional<ArrayRef<int64_t>> broadcast_shape = std::nullopt) {
    mlir::IRMapping mapping;
    mapping.map(old_block.getArguments(), mapped_args);

    for (mlir::Operation& op_in_block : old_block.without_terminator()) {
      if (broadcast_shape &&
          isa<tensor::ExtractOp, tensor::FromElementsOp>(&op_in_block)) {
        mapping.map(op_in_block.getResult(0),
                    mapping.lookupOrDefault(op_in_block.getOperand(0)));
        continue;
      }

      auto mapped_operands = llvm::map_to_vector(
          op_in_block.getOperands(),
          [&](Value operand) { return mapping.lookupOrDefault(operand); });

      if (!broadcast_shape) {
        rewriter.clone(op_in_block, mapping);
        continue;
      }

      if (op_in_block.hasTrait<mlir::OpTrait::ConstantLike>()) {
        mlir::Operation* cloned_const = rewriter.clone(op_in_block, mapping);
        Type element_type =
            mlir::getElementTypeOrSelf(cloned_const->getResult(0).getType());
        auto tensor_type =
            mlir::RankedTensorType::get(*broadcast_shape, element_type);
        Value bcast = mlir::stablehlo::BroadcastInDimOp::create(
            rewriter, loc, tensor_type, cloned_const->getResult(0),
            rewriter.getDenseI64ArrayAttr({}));
        mapping.map(op_in_block.getResult(0), bcast);
        continue;
      }

      auto mapped_types = llvm::map_to_vector(
          op_in_block.getResultTypes(), [&](Type type) -> Type {
            return mlir::RankedTensorType::get(
                *broadcast_shape, mlir::getElementTypeOrSelf(type));
          });

      mlir::OperationState state(loc, op_in_block.getName());
      state.addOperands(mapped_operands);
      state.addTypes(mapped_types);
      state.addAttributes(op_in_block.getAttrs());
      mlir::Operation* new_op = rewriter.create(state);

      mapping.map(op_in_block.getResults(), new_op->getResults());
    }

    return llvm::map_to_vector(
        old_block.getTerminator()->getOperands(),
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
    int32_t axis = op.getDimension();
    SmallVector<Value> init_and_results;
    init_and_results.reserve(op.getInputs().size() * 2);

    for (auto [result, init_val] :
         llvm::zip(triton_scan_op.getResults(), op.getInits())) {
      auto result_type = mlir::cast<mlir::RankedTensorType>(result.getType());

      SmallVector<int64_t> bcast_dims;
      bcast_dims.reserve(result_type.getRank() - 1);
      for (int64_t d = 0; d < result_type.getRank(); ++d) {
        if (d != axis) {
          bcast_dims.push_back(d);
        }
      }

      init_and_results.push_back(mlir::stablehlo::BroadcastInDimOp::create(
          rewriter, op.getLoc(), result_type, init_val,
          rewriter.getDenseI64ArrayAttr(bcast_dims)));
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

  static SmallVector<Value> ExtractCarries(::xla::xtile::ScanOp op,
                                           ArrayRef<Value> outputs,
                                           mlir::PatternRewriter& rewriter) {
    int32_t axis = op.getDimension();
    bool reverse = op.getIsReverse();

    return llvm::map_to_vector(
        llvm::seq<int>(0, op.getCarries().size()), [&](int i) -> Value {
          auto result_type =
              mlir::cast<mlir::RankedTensorType>(outputs[i].getType());
          SmallVector<OpFoldResult> offsets, sizes, strides;
          offsets.reserve(result_type.getRank());
          sizes.reserve(result_type.getRank());
          strides.reserve(result_type.getRank());

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

          return mlir::tensor::ExtractSliceOp::create(rewriter, op.getLoc(),
                                                      carry_type, outputs[i],
                                                      offsets, sizes, strides);
        });
  }

  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::ScanOp op, mlir::PatternRewriter& rewriter) const override {
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
