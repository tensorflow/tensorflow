/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Analysis/shape_component_analysis.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

using ShapeOrValueInfo = ShapeComponentAnalysis::ShapeOrValueInfo;
using Symbol = ShapeComponentAnalysis::Symbol;
using SymbolicExpr = ShapeComponentAnalysis::SymbolicExpr;

namespace {

// Temporary data structure to hold a single dimension of the symbolic result of
// `shape.broadcast`.
struct SymbolicBroadcastDimension {
  size_t operand_index;
  size_t operand_dim;
  SymbolicExpr expr;
};

// Replace shape.broadcast with a shape if it's statically known.
struct SimplifyBroadcasts : public mlir::OpRewritePattern<shape::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      shape::BroadcastOp op, mlir::PatternRewriter &rewriter) const override {
    // Require successful shape analysis.
    ShapeComponentAnalysis shape_analysis;
    llvm::SmallVector<ArrayRef<SymbolicExpr>> shapes_info;
    auto shapes = op.getShapes();
    shapes_info.reserve(shapes.size());
    for (Value s : shapes) {
      auto s_info = shape_analysis.GetValueInfo(s);
      if (!s_info) return failure();
      shapes_info.push_back(*s_info);
    }

    // Find the result rank.
    size_t rank = 0;
    for (const auto &s_info : shapes_info) rank = std::max(rank, s_info.size());

    // Compute broadcast symbolically.
    SmallVector<Optional<SymbolicBroadcastDimension>> sym_result(rank,
                                                                 llvm::None);
    for (const auto &s_info : llvm::enumerate(shapes_info)) {
      size_t dim_offset = rank - s_info.value().size();
      for (const auto &sym_expr : llvm::enumerate(s_info.value())) {
        // Unit dimensions are neutral to the final result.
        if (sym_expr.value().isConstant(1)) continue;

        // Use unique expression.
        size_t i = dim_offset + sym_expr.index();
        if (!sym_result[i]) {
          sym_result[i] = {s_info.index(), sym_expr.index(), sym_expr.value()};
          continue;
        }

        // Bail if the dimensions are neither equal nor 1.
        if (sym_result[i]->expr != sym_expr.value()) return failure();
      }
    }

    // Materialize broadcast result.
    auto loc = op.getLoc();
    DenseMap<int64_t, Value> constants;
    auto find_or_create_constant = [&](int64_t c) {
      auto it = constants.find(c);
      if (it != constants.end()) return it->second;
      Value newly_created = rewriter.create<arith::ConstantIndexOp>(loc, c);
      constants[c] = newly_created;
      return newly_created;
    };
    auto elements = llvm::to_vector<8>(
        llvm::map_range(sym_result, [&](const auto &sym_result_dim) {
          // If we know the dimension statically, use a constant.
          if (!sym_result_dim) return find_or_create_constant(1);
          if (auto cexpr = sym_result_dim->expr.expr
                               .template dyn_cast<AffineConstantExpr>()) {
            return find_or_create_constant(cexpr.getValue());
          }

          // Othwerise, extract the dimension from the unique operand.
          Value operand = shapes[sym_result_dim->operand_index];
          Value operand_dim =
              find_or_create_constant(sym_result_dim->operand_dim);
          return rewriter.create<tensor::ExtractOp>(loc, operand, operand_dim)
              .getResult();
        }));
    Type index_ty = rewriter.getIndexType();
    Type concrete_result_ty = RankedTensorType::get(
        {static_cast<int64_t>(elements.size())}, index_ty);
    Value result = rewriter.create<tensor::FromElementsOp>(
        loc, concrete_result_ty, elements);

    // Insert cast, if needed.
    Type expected_ty = op.getResult().getType();
    if (result.getType() != expected_ty) {
      result = rewriter.create<tensor::CastOp>(loc, expected_ty, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

LogicalResult AnalyzeDynamicBroadcastInDimExpandingBehavior(
    ShapeComponentAnalysis &analysis, Value value, Value shape,
    llvm::SmallSetVector<int64_t, 4> *known_expanding_dims,
    llvm::SmallSetVector<int64_t, 4> *known_nonexpanding_dims) {
  // Require successful analysis of shapes.
  auto shape_in = analysis.GetShapeInfo(value);
  auto shape_out = analysis.GetValueInfo(shape);
  if (!shape_in || !shape_out) return failure();

  // Analyze per argument dimension.
  size_t rank_in = shape_in->size();
  size_t rank_out = shape_out->size();
  assert(rank_in <= rank_out);
  size_t dim_out_offset = rank_out - rank_in;
  for (size_t i = 0; i < rank_in; ++i) {
    SymbolicExpr dim_in = (*shape_in)[i];
    SymbolicExpr dim_out = (*shape_out)[dim_out_offset + i];
    if (dim_in.isConstant(1) && dim_out.isKnownNotOne())
      known_expanding_dims->insert(i);
    if (dim_in == dim_out || dim_out.isConstant(1))
      known_nonexpanding_dims->insert(i);
  }
  return success();
}

// Analyze `mhlo.dynamic_broadcast_in_dim` op and populate attributes for
// statically known expanding and non-expanding dimensions.
struct AnnotateExpandingDimensionsInDynamicBroadcastInDim
    : public mlir::OpRewritePattern<mhlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      mhlo::DynamicBroadcastInDimOp op,
      mlir::PatternRewriter &rewriter) const override {
    // Analyze shapes and identify expanding and non-expanding dims.
    ShapeComponentAnalysis analysis;
    llvm::SmallSetVector<int64_t, 4> known_expanding_dims,
        known_nonexpanding_dims;
    if (failed(AnalyzeDynamicBroadcastInDimExpandingBehavior(
            analysis, op.operand(), op.output_dimensions(),
            &known_expanding_dims, &known_nonexpanding_dims))) {
      return failure();
    }

    // Collect possibly already annotated info.
    auto insert_all = [](llvm::SmallSetVector<int64_t, 4> &dst,
                         Optional<DenseIntElementsAttr> src) {
      if (!src) return;
      for (auto it : *src) dst.insert(it.getLimitedValue());
    };
    insert_all(known_expanding_dims, op.known_expanding_dimensions());
    insert_all(known_nonexpanding_dims, op.known_nonexpanding_dimensions());

    // Fail pattern application if there is nothing new to annotate.
    auto is_equal = [](llvm::SmallSetVector<int64_t, 4> &set,
                       DenseIntElementsAttr attr) {
      return set.size() == attr.size() && llvm::all_of(attr, [&](auto it) {
               return set.count(it.getLimitedValue());
             });
    };
    if (op.known_expanding_dimensions() && op.known_nonexpanding_dimensions() &&
        is_equal(known_expanding_dims, *op.known_expanding_dimensions()) &&
        is_equal(known_nonexpanding_dims,
                 *op.known_nonexpanding_dimensions())) {
      return failure();
    }

    // Annotate op in place.
    rewriter.startRootUpdate(op);
    op.known_expanding_dimensionsAttr(
        rewriter.getI64TensorAttr(known_expanding_dims.takeVector()));
    op.known_nonexpanding_dimensionsAttr(
        rewriter.getI64TensorAttr(known_nonexpanding_dims.takeVector()));
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

// Remove compute_reshape_shape if we can prove that the dynamic shape does not
// contain a `-1` dimension.
struct RemoveComputeReshapeShape final
    : public OpRewritePattern<mhlo::ComputeReshapeShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ComputeReshapeShapeOp op,
                                PatternRewriter &rewriter) const override {
    ShapeComponentAnalysis shapeComponentAnalysis;
    auto dynamic_shape =
        shapeComponentAnalysis.GetValueInfo(op.dynamic_shape());
    if (!dynamic_shape) return failure();

    if (llvm::any_of(*dynamic_shape, [](const auto &dim) {
          return !dim.isKnownNotNegativeOne();
        })) {
      return failure();
    }
    rewriter.replaceOp(op, op.dynamic_shape());
    return success();
  }
};

bool IsProduct(AffineExpr expr,
               llvm::function_ref<void(AffineConstantExpr)> cbkConstantFactor,
               llvm::function_ref<void(AffineSymbolExpr)> cbkSymbolicFactor) {
  auto binExpr = expr.dyn_cast<AffineBinaryOpExpr>();
  if (binExpr && binExpr.getKind() == AffineExprKind::Mul) {
    return IsProduct(binExpr.getLHS(), cbkConstantFactor, cbkSymbolicFactor) &&
           IsProduct(binExpr.getRHS(), cbkConstantFactor, cbkSymbolicFactor);
  }
  if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>()) {
    cbkSymbolicFactor(symExpr);
    return true;
  }
  if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
    cbkConstantFactor(constExpr);
    return true;
  }
  return false;
}

bool IsSymbolicProduct(const SymbolicExpr &symbolicExpr,
                       llvm::function_ref<void(int64_t)> cbkConstantFactor,
                       llvm::function_ref<void(Symbol)> cbkSymbolicFactor) {
  return IsProduct(
      symbolicExpr.expr,
      [&](AffineConstantExpr cexpr) { cbkConstantFactor(cexpr.getValue()); },
      [&](AffineSymbolExpr sexpr) {
        cbkSymbolicFactor(symbolicExpr.symbols[sexpr.getPosition()]);
      });
}

// Represents a product of symbolic and concrete factors. This will allow us to
// prove product equalities symbolically.
struct SymbolicProduct {
  // Product of all concrete factors.
  int64_t concrete = 1;
  // List all symbolic factors as they can not be aggregated.
  llvm::SmallVector<Symbol> symbolic;
  bool empty() { return concrete == 1 && symbolic.empty(); }
};

bool IsSymbolicProduct(const SymbolicExpr &symbolicExpr,
                       SymbolicProduct *product) {
  return IsSymbolicProduct(
      symbolicExpr, [&](int64_t c) { product->concrete *= c; },
      [&](Symbol s) { product->symbolic.push_back(s); });
}

struct RemoveRedundantCstrReshapable final
    : public OpRewritePattern<mhlo::CstrReshapableOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::CstrReshapableOp op,
                                PatternRewriter &rewriter) const override {
    // Get shape analysis info for the number of elements.
    ShapeComponentAnalysis shapeComponentAnalysis;
    auto numElementsInfo =
        shapeComponentAnalysis.GetValueInfo(op.num_elements());
    if (!numElementsInfo) return failure();
    assert(numElementsInfo->size() == 1 && "expect one value for a scalar");
    auto numElements = numElementsInfo->front();

    // Get shape analysis info for the dynamic shape.
    auto dynShapeDims = shapeComponentAnalysis.GetValueInfo(op.dynamic_shape());
    if (!dynShapeDims) return failure();

    // We can handle two cases:
    //   - there is exactly one -1 in the dynamic shape, i.e. a unique wildcard
    //     dimension, or
    //   - there is no -1 in the dynamic shape, i.e. no wildcard dimension.
    bool unique_wildcard_dimension = false;
    for (const auto &d : *dynShapeDims) {
      if (d.isConstant(-1)) {
        if (unique_wildcard_dimension) return failure();
        unique_wildcard_dimension = true;
      } else if (!d.isKnownNotNegativeOne()) {
        return failure();
      }
    }

    // We can only handle simple products with constants and symbols. Find all
    // the factors based on the number of elements.
    SymbolicProduct numElementsRemainingFactors;
    if (!IsSymbolicProduct(numElements, &numElementsRemainingFactors)) {
      return failure();
    }
    assert(numElementsRemainingFactors.concrete >= 1 &&
           "number of elements cannot entail negative or zero factors");

    // Find all factors based on the dynamic shape.
    //   - Accumulate the conrete product to later compare it against its
    //     equivalent based on the number of elements.
    //   - Remove symbolic factors from the list and fail if we find an unknown
    //     factor, i.e. if the symbolic factors based on the dynamic shape are
    //     not a subset of the factors based on the number of elements.
    int64_t concreteProductDynShape = 1;
    for (const auto &dim : *dynShapeDims) {
      SmallVector<Symbol> partialSymbolicFactorsDynShape;
      if (!IsSymbolicProduct(
              dim,
              [&](int64_t c) {
                if (c != ShapedType::kDynamicSize) concreteProductDynShape *= c;
              },
              [&](Symbol s) { partialSymbolicFactorsDynShape.push_back(s); })) {
        return failure();
      }
      for (const Symbol &symDynShape : partialSymbolicFactorsDynShape) {
        auto *it =
            llvm::find(numElementsRemainingFactors.symbolic, symDynShape);
        if (it == numElementsRemainingFactors.symbolic.end()) return failure();
        numElementsRemainingFactors.symbolic.erase(it);
      }
    }
    assert(concreteProductDynShape >= 1 &&
           "concrete product must not aggregate negative or zero factors");

    // A wildcard dimension can subsume the remaining symbolic factors and
    // potentially also a concrete factor.
    if (unique_wildcard_dimension) {
      if (numElementsRemainingFactors.concrete % concreteProductDynShape != 0)
        return failure();
      rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
      return success();
    }

    // W/o a wildcard, the symbolic and concrete products must be equal.
    bool isReshapable =
        numElementsRemainingFactors.symbolic.empty() &&
        numElementsRemainingFactors.concrete == concreteProductDynShape;
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, isReshapable);
    return success();
  }
};

LogicalResult MaterializeReshapeAsScalarExpand(RankedTensorType operand_ty,
                                               RankedTensorType result_ty,
                                               mhlo::DynamicReshapeOp op,
                                               PatternRewriter &rewriter) {
  assert(operand_ty.getRank() == 0 && "expect scalar operand");
  auto loc = op.getLoc();
  SmallVector<int64_t> unit_dims(result_ty.getRank(), 1);
  auto expanded_ty =
      RankedTensorType::get(unit_dims, result_ty.getElementType());
  Value expanded_scalar = rewriter.create<tensor::ExpandShapeOp>(
      loc, expanded_ty, op.operand(), ArrayRef<ReassociationIndices>{});
  if (expanded_scalar.getType() != result_ty) {
    expanded_scalar =
        rewriter.create<tensor::CastOp>(loc, result_ty, expanded_scalar);
  }
  rewriter.replaceOp(op, expanded_scalar);
  return success();
}

LogicalResult MaterializeReshapeAsScalarCollapse(RankedTensorType operand_ty,
                                                 RankedTensorType result_ty,
                                                 mhlo::DynamicReshapeOp op,
                                                 PatternRewriter &rewriter) {
  assert(result_ty.getRank() == 0 && "expect scalar result");
  auto loc = op.getLoc();
  Value operand = op.operand();
  SmallVector<int64_t> unit_dims(operand_ty.getRank(), 1);
  auto casted_operand_ty =
      RankedTensorType::get(unit_dims, operand_ty.getElementType());
  if (operand.getType() != casted_operand_ty) {
    operand = rewriter.create<tensor::CastOp>(loc, casted_operand_ty, operand);
  }
  Value collapsed_scalar = rewriter.create<tensor::CollapseShapeOp>(
      loc, operand, ArrayRef<ReassociationIndices>{});
  rewriter.replaceOp(op, collapsed_scalar);
  return success();
}

enum class DimensionGroupKind {
  kNone,
  kExpanding,
  kCollapsing,
};

struct DimensionGroup {
  int64_t size = 0;
  DimensionGroupKind kind = DimensionGroupKind::kNone;
};

SymbolicProduct EliminateCommonFactors(SymbolicProduct &a, SymbolicProduct &b) {
  SymbolicProduct gcd;

  // Eliminate common concrete factors.
  gcd.concrete = llvm::GreatestCommonDivisor64(a.concrete, b.concrete);
  a.concrete /= gcd.concrete;
  b.concrete /= gcd.concrete;

  // Eliminate common symbolic factors.
  int64_t i = 0;
  while (i < a.symbolic.size()) {
    auto *it = llvm::find(b.symbolic, a.symbolic[i]);
    if (it != b.symbolic.end()) {
      gcd.symbolic.push_back(*it);
      std::swap(a.symbolic[i], a.symbolic.back());
      a.symbolic.pop_back();
      b.symbolic.erase(it);
    } else {
      i++;
    }
  }

  return gcd;
}

bool IsUnpairedUnitDimension(
    ArrayRef<ShapeComponentAnalysis::SymbolicExpr>::iterator it,
    ArrayRef<ShapeComponentAnalysis::SymbolicExpr>::iterator end,
    ArrayRef<ShapeComponentAnalysis::SymbolicExpr>::iterator other_it,
    ArrayRef<ShapeComponentAnalysis::SymbolicExpr>::iterator other_end) {
  return it != end && it->isConstant(1) &&
         !(other_it != other_end && other_it->isConstant(1));
}

int64_t GetShapedTypyDimSize(const SymbolicProduct &sym_product) {
  return sym_product.symbolic.empty() ? sym_product.concrete
                                      : ShapedType::kDynamicSize;
}

// Iterate over the operand's and the result's shape dimensions and find
// dimension groups that are collapsing, expanding, or untouched:
//   - Collapsing: Multiple dimensions of the operand shape can be collapsed
//     into a single dimension of the result shape. We must prove that the
//     product of the operand shape's dimensions is equal to the corresponding
//     result dimension.
//   - Expanding: A single dimension of the operand shape can be expanded into
//     multiple dimensions of the result shape. We must prove that the product
//     of the result shape's dimensions is equal to the corresponding operand
//     dimension. This case is limited to at most one dynamic dimension per
//     expansion group as otherwise not supported by the `expand_shape` op.
//   - Untouched: There is a 1:1 correspondance between an operand and a result
//     shape dimension.
//
// We can determine the optimal dimension groups greedily by consuming operand
// and result dimensions from left to right. If the leading operand dimension is
// a strict divisor of the leading result dimension, collapsing is required. In
// this case, we keep consuming the operand dimensions until the products are
// equal. If the leading result dimension is a strict divisor of the leading
// operand dimension, expanding is required. In this case, we keep consuming the
// result dimensions until the products are equal. Trailing unit dimensions may
// be inlcuded in the dimension group. This is useful iff they are "unpaired",
// in which case they would only limit us in the subsequent iteration.
//
LogicalResult FindExpandingAndCollapsingDimensionGroups(
    ArrayRef<SymbolicExpr> operand_shape_info,
    ArrayRef<SymbolicExpr> result_shape_info,
    SmallVector<DimensionGroup> *dimension_groups,
    SmallVector<int64_t> *expanded_interm_shape) {
  const auto *operand_shape_it = operand_shape_info.begin();
  const auto *operand_shape_end = operand_shape_info.end();
  const auto *result_shape_it = result_shape_info.begin();
  const auto *result_shape_end = result_shape_info.end();

  // Crucial iteration state.
  SymbolicProduct remaining_operand_shape_factors;
  SymbolicProduct remaining_result_shape_factors;
  auto any_remaining_factors = [&]() {
    return !remaining_operand_shape_factors.empty() ||
           !remaining_result_shape_factors.empty();
  };

  while (operand_shape_it != operand_shape_end &&
         result_shape_it != result_shape_end) {
    assert(!any_remaining_factors() &&
           "expect no remaining factors from previous iteration");
    DimensionGroup &dim_group = dimension_groups->emplace_back();

    // Consume at least one operand and result dimension.
    {
      if (!IsSymbolicProduct(*operand_shape_it++,
                             &remaining_operand_shape_factors) ||
          !IsSymbolicProduct(*result_shape_it++,
                             &remaining_result_shape_factors)) {
        return failure();
      }
      dim_group.size++;
      SymbolicProduct gcd = EliminateCommonFactors(
          remaining_operand_shape_factors, remaining_result_shape_factors);
      expanded_interm_shape->push_back(GetShapedTypyDimSize(gcd));
    }

    // Fail if there are unresolvable, contradicting factors remaining.
    if (!remaining_operand_shape_factors.empty() &&
        !remaining_result_shape_factors.empty()) {
      return failure();
    }

    // Collapsing: Create a collapsing dimension group.
    bool requires_collapsing =
        remaining_operand_shape_factors.empty() &&
        (!remaining_result_shape_factors.empty() ||
         IsUnpairedUnitDimension(operand_shape_it, operand_shape_end,
                                 result_shape_it, result_shape_end));
    if (requires_collapsing) {
      dim_group.kind = DimensionGroupKind::kCollapsing;

      // Consume operand shape dimensions until their product matches the
      // corresponding result dimension (or fail if unresolvable/contradicting
      // factors are found).
      while (operand_shape_it != operand_shape_end &&
             remaining_operand_shape_factors.empty() &&
             !remaining_result_shape_factors.empty()) {
        if (!IsSymbolicProduct(*operand_shape_it++,
                               &remaining_operand_shape_factors)) {
          return failure();
        }
        dim_group.size++;
        SymbolicProduct gcd = EliminateCommonFactors(
            remaining_operand_shape_factors, remaining_result_shape_factors);
        expanded_interm_shape->push_back(GetShapedTypyDimSize(gcd));
      }
      if (any_remaining_factors()) return failure();

      // Consume trailing, unpaired unit dimensions.
      while (IsUnpairedUnitDimension(operand_shape_it, operand_shape_end,
                                     result_shape_it, result_shape_end)) {
        operand_shape_it++;
        dim_group.size++;
        expanded_interm_shape->push_back(1);
      }

      continue;
    }

    // Expanding: Create an expanding dimension group.
    bool requires_expanding =
        remaining_result_shape_factors.empty() &&
        (!remaining_operand_shape_factors.empty() ||
         IsUnpairedUnitDimension(result_shape_it, result_shape_end,
                                 operand_shape_it, operand_shape_end));
    if (requires_expanding) {
      dim_group.kind = DimensionGroupKind::kExpanding;
      int64_t num_dynamic_dims = 0;

      // Consume result shape dimensions until their product matches the
      // corresponding operand dimension (or fail if unresolvable/contradicting
      // factors are found).
      while (result_shape_it != result_shape_end &&
             remaining_result_shape_factors.empty() &&
             !remaining_operand_shape_factors.empty()) {
        if (!IsSymbolicProduct(*result_shape_it++,
                               &remaining_result_shape_factors)) {
          return failure();
        }
        dim_group.size++;
        SymbolicProduct gcd = EliminateCommonFactors(
            remaining_operand_shape_factors, remaining_result_shape_factors);
        int64_t ty_dim_size = GetShapedTypyDimSize(gcd);

        // Allow no more than one dynamic dimension per expansion group.
        if (ty_dim_size == ShapedType::kDynamicSize) {
          num_dynamic_dims++;
          if (num_dynamic_dims > 1) return failure();
        }
        expanded_interm_shape->push_back(ty_dim_size);
      }
      if (any_remaining_factors()) return failure();

      // Consume trailing, unpaired unit dimensions.
      while (IsUnpairedUnitDimension(result_shape_it, result_shape_end,
                                     operand_shape_it, operand_shape_end)) {
        result_shape_it++;
        dim_group.size++;
        expanded_interm_shape->push_back(1);
      }

      continue;
    }

    // Untouched: 1:1 mapping between operand and result shape dimension. This
    // is neither expanding nor collapsing.
    assert(!requires_collapsing && !requires_expanding && "expect id case");
    assert(dim_group.size == 1 && dim_group.kind == DimensionGroupKind::kNone &&
           "expect simple dimension group");
  }

  // Fail if there are remaining dimensions that could not be consumed.
  assert(!any_remaining_factors() && "expect no remaining factors");
  if (operand_shape_it != operand_shape_end ||
      result_shape_it != result_shape_end) {
    return failure();
  }

  return success();
}

SmallVector<int64_t> ConcretizeOperandShape(
    ArrayRef<int64_t> operand_shape,
    ArrayRef<SymbolicExpr> operand_shape_info) {
  SmallVector<int64_t> result;
  for (auto it : llvm::zip(operand_shape, operand_shape_info)) {
    auto dim_size = std::get<0>(it);
    auto s_expr = std::get<1>(it);
    if (auto cexpr = s_expr.expr.dyn_cast<AffineConstantExpr>()) {
      int64_t also_dim_size = cexpr.getValue();
      assert((ShapedType::isDynamic(dim_size) || dim_size == also_dim_size) &&
             "expect shape analysis result to be compatible with type");
      result.push_back(also_dim_size);
      continue;
    }
    result.push_back(dim_size);
  }
  return result;
}

llvm::Optional<SmallVector<ReassociationIndices>> RequiresReassociationOfKind(
    DimensionGroupKind kind, const SmallVector<DimensionGroup> &dim_groups) {
  SmallVector<ReassociationIndices> reassociation;
  reassociation.reserve(dim_groups.size());
  bool is_strictly_reassociating = false;
  int64_t i = 0;
  for (const DimensionGroup &g : dim_groups) {
    if (g.kind == kind) {
      is_strictly_reassociating = true;
      reassociation.push_back(
          llvm::to_vector(llvm::seq<int64_t>(i, i + g.size)));
      i += g.size;
      continue;
    }
    for (int64_t j = 0; j < g.size; j++) reassociation.push_back({i++});
  }

  // Return the reassociation if expansion is required.
  if (is_strictly_reassociating) return reassociation;
  return llvm::None;
}

LogicalResult MaterializeReshapeAsExpandAndCollapse(
    ShapeComponentAnalysis &shape_analysis, RankedTensorType operand_ty,
    RankedTensorType result_ty, mhlo::DynamicReshapeOp op,
    PatternRewriter &rewriter) {
  // Require sucessful shape analysis for operand and result shape.
  auto operand_shape_info = shape_analysis.GetShapeInfo(op.operand());
  if (!operand_shape_info) return failure();
  auto result_shape_info = shape_analysis.GetValueInfo(op.output_shape());
  if (!result_shape_info) return failure();

  // Identify dimension groups and the intermediate expanded type.
  SmallVector<DimensionGroup> dimension_groups;
  SmallVector<int64_t> expanded_interm_shape;
  if (failed(FindExpandingAndCollapsingDimensionGroups(
          *operand_shape_info, *result_shape_info, &dimension_groups,
          &expanded_interm_shape))) {
    return failure();
  }

  // Materialize cast, expand, collapse, and cast, as needed.
  auto loc = op.getLoc();
  Value interm = op.operand();
  auto casted_operand_ty = RankedTensorType::get(
      ConcretizeOperandShape(operand_ty.getShape(), *operand_shape_info),
      operand_ty.getElementType());
  if (operand_ty != casted_operand_ty) {
    interm = rewriter.create<tensor::CastOp>(loc, casted_operand_ty, interm);
  }
  if (auto reassociation = RequiresReassociationOfKind(
          DimensionGroupKind::kExpanding, dimension_groups)) {
    interm = rewriter.create<tensor::ExpandShapeOp>(
        loc,
        RankedTensorType::get(expanded_interm_shape,
                              operand_ty.getElementType()),
        interm, *reassociation);
  }
  if (auto reassociation = RequiresReassociationOfKind(
          DimensionGroupKind::kCollapsing, dimension_groups)) {
    interm =
        rewriter.create<tensor::CollapseShapeOp>(loc, interm, *reassociation);
  }
  if (interm.getType() != result_ty) {
    interm = rewriter.create<tensor::CastOp>(loc, result_ty, interm);
  }
  rewriter.replaceOp(op, interm);
  return success();
}

// Tries to express `dynamic_reshape` ops through `expand_shape` and
// `collapse_shape` ops.
struct DynamicReshapeToExpandAndCollapseShape final
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto operand_ty = op.operand().getType().dyn_cast<RankedTensorType>();
    if (!operand_ty) return failure();
    auto result_ty = op.getType().dyn_cast<RankedTensorType>();
    if (!result_ty) return failure();

    // Handle degenerate scalar expand case.
    if (operand_ty.getRank() == 0) {
      return MaterializeReshapeAsScalarExpand(operand_ty, result_ty, op,
                                              rewriter);
    }

    // Handle degenerate scalar collapse case.
    if (result_ty.getRank() == 0) {
      return MaterializeReshapeAsScalarCollapse(operand_ty, result_ty, op,
                                                rewriter);
    }

    ShapeComponentAnalysis shape_analysis;
    return MaterializeReshapeAsExpandAndCollapse(shape_analysis, operand_ty,
                                                 result_ty, op, rewriter);
  }
};

// Returns true if all of bcasted_shapes can be broadcasted with output_shape.
bool IsKnownBroadcastable(ShapeComponentAnalysis &analysis,
                          ValueRange bcasted_shapes, Value output_shape) {
  auto output_shape_dims = analysis.GetValueInfo(output_shape);
  if (!output_shape_dims) return false;
  for (Value shape : bcasted_shapes) {
    auto shape_dims = analysis.GetValueInfo(shape);
    if (!shape_dims) return false;
    // Iterate backwards over the smallest input shape.
    for (auto zip : llvm::zip(llvm::reverse(*output_shape_dims),
                              llvm::reverse(*shape_dims))) {
      const auto &first = std::get<0>(zip);
      const auto &second = std::get<1>(zip);
      // TODO(ezhulenev): What to do with dimensions statically known to be
      // zero?
      // Numpy can only broadcast [0] with [1], however Tensorflow can broadcast
      // [0] with any dimension size, and produces dimension of size [0].
      // Currently we'll conservatively return failure and will not proceed with
      // a rewrite.
      if (first.isConstant(0) || second.isConstant(0)) return false;
      // If either shape has a static one dimension the broadcast will always
      // succeed.
      if (first.isConstant(1) || second.isConstant(1)) continue;
      // Otherwise dims have to be equal.
      if (first != second) return false;
    }
  }
  return true;
}

// Rewrite `shape.cstr_broadcastable` with constant witness if can prove that
// shapes are broadcastable from a symbolic analysis.
struct CstrBroadcastableOpLowering
    : public OpRewritePattern<shape::CstrBroadcastableOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::CstrBroadcastableOp op,
                                PatternRewriter &rewriter) const override {
    ShapeComponentAnalysis shape_component_analysis;
    if (!IsKnownBroadcastable(shape_component_analysis, op.getShapes(),
                              op.getShapes().front())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
    return success();
  }
};

class SymbolicShapeOptimizationPass final
    : public SymbolicShapeOptimizationBase<SymbolicShapeOptimizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    // clang-format off
    patterns.insert<
        AnnotateExpandingDimensionsInDynamicBroadcastInDim,
        CstrBroadcastableOpLowering,
        DynamicReshapeToExpandAndCollapseShape,
        RemoveComputeReshapeShape,
        RemoveRedundantCstrReshapable,
        SimplifyBroadcasts>(ctx);
    // clang-format on
    shape::AssumingOp::getCanonicalizationPatterns(patterns, ctx);

    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSymbolicShapeOptimizationPass() {
  return std::make_unique<SymbolicShapeOptimizationPass>();
}

}  // end namespace mlir
