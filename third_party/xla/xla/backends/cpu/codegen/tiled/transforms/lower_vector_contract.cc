/* Copyright 2026 The OpenXLA Authors.

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
#include <functional>
#include <iterator>
#include <numeric>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"  // IWYU pragma: keep

namespace xla::cpu {

#define GEN_PASS_DEF_LOWERVECTORCONTRACTPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;
namespace mv = ::mlir::vector;

using ::mlir::AffineDimExpr;
using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::ArrayAttr;
using ::mlir::FloatType;
using ::mlir::IntegerType;
using ::mlir::Location;
using ::mlir::LogicalResult;
using ::mlir::PatternRewriter;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::VectorType;

bool CanExtendElementType(Type src_elem, Type dst_elem) {
  if (src_elem == dst_elem) {
    return true;
  }
  if (mlir::isa<FloatType>(src_elem) && mlir::isa<FloatType>(dst_elem)) {
    return src_elem.getIntOrFloatBitWidth() < dst_elem.getIntOrFloatBitWidth();
  }
  if (mlir::isa<IntegerType>(src_elem) && mlir::isa<IntegerType>(dst_elem)) {
    return src_elem.getIntOrFloatBitWidth() < dst_elem.getIntOrFloatBitWidth();
  }
  return false;
}

Value ExtendToElementType(PatternRewriter& rewriter, Location loc, Value value,
                          Type target_elem_type) {
  auto vec_type = mlir::cast<VectorType>(value.getType());
  if (vec_type.getElementType() == target_elem_type) {
    return value;
  }
  auto target_vec_type = VectorType::get(vec_type.getShape(), target_elem_type);
  if (mlir::isa<FloatType>(vec_type.getElementType()) &&
      mlir::isa<FloatType>(target_elem_type)) {
    return ma::ExtFOp::create(rewriter, loc, target_vec_type, value);
  }
  if (mlir::isa<IntegerType>(vec_type.getElementType()) &&
      mlir::isa<IntegerType>(target_elem_type)) {
    return ma::ExtSIOp::create(rewriter, loc, target_vec_type, value);
  }
  return value;
}

// Extends `vector.contract` operands to match the accumulator element type so
// downstream `OuterProduct` lowering never receives mixed-precision operands.
LogicalResult PromoteContractionOperands(mv::ContractionOp op,
                                         PatternRewriter& rewriter) {
  Type acc_elem = mlir::getElementTypeOrSelf(op.getAccType());
  if (op.getLhsType().getElementType() == acc_elem &&
      op.getRhsType().getElementType() == acc_elem) {
    return mlir::failure();
  }

  if (!CanExtendElementType(op.getLhsType().getElementType(), acc_elem) ||
      !CanExtendElementType(op.getRhsType().getElementType(), acc_elem)) {
    return mlir::failure();
  }

  Location loc = op.getLoc();
  Value lhs = ExtendToElementType(rewriter, loc, op.getLhs(), acc_elem);
  Value rhs = ExtendToElementType(rewriter, loc, op.getRhs(), acc_elem);
  rewriter.replaceOpWithNewOp<mv::ContractionOp>(
      op, lhs, rhs, op.getAcc(), op.getIndexingMaps(), op.getIteratorTypes(),
      op.getKind());
  return mlir::success();
}

struct SqueezedOperand {
  Value value;
  AffineMap map;
};

SqueezedOperand SqueezeOperandToKeptIters(PatternRewriter& rewriter,
                                          Location loc, Value value,
                                          AffineMap map,
                                          llvm::ArrayRef<bool> keep_iter,
                                          llvm::ArrayRef<int64_t> iter_remap) {
  auto vec_type = mlir::cast<VectorType>(value.getType());
  llvm::SmallVector<int64_t> new_shape;
  llvm::SmallVector<AffineExpr> new_exprs;
  for (auto [dim_idx, expr] : llvm::enumerate(map.getResults())) {
    auto dim_expr = mlir::cast<AffineDimExpr>(expr);
    unsigned iter = dim_expr.getPosition();
    if (keep_iter[iter]) {
      new_shape.push_back(vec_type.getDimSize(dim_idx));
      new_exprs.push_back(rewriter.getAffineDimExpr(iter_remap[iter]));
    }
  }
  int64_t new_num_iters = llvm::count(keep_iter, true);
  AffineMap squeezed_map =
      AffineMap::get(new_num_iters, 0, new_exprs, rewriter.getContext());
  if (new_shape.size() == vec_type.getRank()) {
    return {value, squeezed_map};
  }
  auto new_type = VectorType::get(new_shape, vec_type.getElementType());
  Value squeezed_value =
      mv::ShapeCastOp::create(rewriter, loc, new_type, value);
  return {squeezed_value, squeezed_map};
}

struct ReductionDims {
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<unsigned> iter_order;
};

ReductionDims CollectReductionDims(
    AffineMap map, VectorType vec_type,
    llvm::ArrayRef<mv::IteratorType> iter_types) {
  ReductionDims dims;
  for (auto [idx, expr] : llvm::enumerate(map.getResults())) {
    unsigned pos = mlir::cast<AffineDimExpr>(expr).getPosition();
    if (iter_types[pos] == mv::IteratorType::reduction) {
      dims.shape.push_back(vec_type.getDimSize(idx));
      dims.iter_order.push_back(pos);
    }
  }
  return dims;
}

LogicalResult LowerAllUnitParallelContraction(
    mv::ContractionOp op, llvm::ArrayRef<mv::IteratorType> iter_types,
    PatternRewriter& rewriter) {
  Location loc = op.getLoc();
  llvm::SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  AffineMap lhs_map = maps[0];
  AffineMap rhs_map = maps[1];

  auto [lhs_red_shape, lhs_red_order] =
      CollectReductionDims(lhs_map, op.getLhsType(), iter_types);
  auto [rhs_red_shape, rhs_red_order] =
      CollectReductionDims(rhs_map, op.getRhsType(), iter_types);

  if (lhs_red_shape.empty() || lhs_red_shape.size() != rhs_red_shape.size()) {
    return mlir::failure();
  }

  Type elem_type = op.getLhsType().getElementType();
  Value lhs = op.getLhs();
  if (lhs_red_shape.size() != op.getLhsType().getRank()) {
    lhs = mv::ShapeCastOp::create(
        rewriter, loc, VectorType::get(lhs_red_shape, elem_type), lhs);
  }
  Value rhs = op.getRhs();
  if (rhs_red_shape.size() != op.getRhsType().getRank()) {
    rhs = mv::ShapeCastOp::create(
        rewriter, loc, VectorType::get(rhs_red_shape, elem_type), rhs);
  }

  if (lhs_red_order != rhs_red_order) {
    llvm::SmallVector<int64_t> perm;
    perm.reserve(lhs_red_order.size());
    for (unsigned lhs_iter : lhs_red_order) {
      auto it = llvm::find(rhs_red_order, lhs_iter);
      if (it == rhs_red_order.end()) {
        return mlir::failure();
      }
      perm.push_back(std::distance(rhs_red_order.begin(), it));
    }
    rhs = mv::TransposeOp::create(rewriter, loc, rhs, perm);
  }

  int64_t total_red =
      std::accumulate(lhs_red_shape.begin(), lhs_red_shape.end(), int64_t{1},
                      std::multiplies<>());
  auto flat_type = VectorType::get({total_red}, elem_type);
  if (lhs.getType() != flat_type) {
    lhs = mv::ShapeCastOp::create(rewriter, loc, flat_type, lhs);
  }
  if (rhs.getType() != flat_type) {
    rhs = mv::ShapeCastOp::create(rewriter, loc, flat_type, rhs);
  }

  Value product = mlir::isa<FloatType>(elem_type)
                      ? Value(ma::MulFOp::create(rewriter, loc, lhs, rhs))
                      : Value(ma::MulIOp::create(rewriter, loc, lhs, rhs));

  Value acc_scalar = op.getAcc();
  if (auto acc_vec_type = mlir::dyn_cast<VectorType>(op.getAccType())) {
    llvm::SmallVector<int64_t> zeros(acc_vec_type.getRank(), 0);
    acc_scalar = mv::ExtractOp::create(rewriter, loc, acc_scalar, zeros);
  }

  Value reduction = mv::ReductionOp::create(
      rewriter, loc, mv::CombiningKind::ADD, product, acc_scalar);
  if (mlir::isa<VectorType>(op.getResultType())) {
    Value bcast =
        mv::BroadcastOp::create(rewriter, loc, op.getResultType(), reduction);
    rewriter.replaceOp(op, bcast);
  } else {
    rewriter.replaceOp(op, reduction);
  }
  return mlir::success();
}

// Drops size-1 parallel (batch and free) dimensions from `vector.contract`.
// When all parallel dimensions are size 1, lowers directly to a 1D
// multiply + `vector.reduction<add>` to avoid scalarization.
LogicalResult DropContractionUnitDims(mv::ContractionOp op,
                                      PatternRewriter& rewriter) {
  if (op.getKind() != mv::CombiningKind::ADD || op->getNumOperands() != 3) {
    return mlir::failure();
  }
  Type acc_elem = mlir::getElementTypeOrSelf(op.getAccType());
  if (op.getLhsType().getElementType() != acc_elem ||
      op.getRhsType().getElementType() != acc_elem) {
    return mlir::failure();
  }

  llvm::SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  for (AffineMap map : maps) {
    for (AffineExpr expr : map.getResults()) {
      if (!mlir::isa<AffineDimExpr>(expr)) {
        return mlir::failure();
      }
    }
  }

  llvm::SmallVector<int64_t> bounds;
  op.getIterationBounds(bounds);
  llvm::SmallVector<mv::IteratorType> iter_types = op.getIteratorTypesArray();

  int64_t num_iters = iter_types.size();
  llvm::SmallVector<bool> keep_iter(num_iters, true);
  int64_t num_kept_parallel = 0;
  bool has_unit_parallel = false;
  for (int64_t i = 0; i < num_iters; ++i) {
    if (iter_types[i] == mv::IteratorType::parallel) {
      if (bounds[i] == 1) {
        keep_iter[i] = false;
        has_unit_parallel = true;
      } else {
        ++num_kept_parallel;
      }
    }
  }

  if (!has_unit_parallel) {
    return mlir::failure();
  }

  if (num_kept_parallel == 0) {
    return LowerAllUnitParallelContraction(op, iter_types, rewriter);
  }

  llvm::SmallVector<int64_t> iter_remap(num_iters, -1);
  llvm::SmallVector<mlir::Attribute> new_iter_types;
  int64_t next_iter = 0;
  for (int64_t i = 0; i < num_iters; ++i) {
    if (!keep_iter[i]) {
      continue;
    }
    iter_remap[i] = next_iter++;
    new_iter_types.push_back(
        rewriter.getAttr<mv::IteratorTypeAttr>(iter_types[i]));
  }

  Location loc = op.getLoc();
  auto [new_lhs, new_lhs_map] = SqueezeOperandToKeptIters(
      rewriter, loc, op.getLhs(), maps[0], keep_iter, iter_remap);
  auto [new_rhs, new_rhs_map] = SqueezeOperandToKeptIters(
      rewriter, loc, op.getRhs(), maps[1], keep_iter, iter_remap);
  auto [new_acc, new_acc_map] = SqueezeOperandToKeptIters(
      rewriter, loc, op.getAcc(), maps[2], keep_iter, iter_remap);

  ArrayAttr new_maps =
      rewriter.getAffineMapArrayAttr({new_lhs_map, new_rhs_map, new_acc_map});
  ArrayAttr new_iters = rewriter.getArrayAttr(new_iter_types);
  auto new_contract =
      mv::ContractionOp::create(rewriter, loc, new_lhs, new_rhs, new_acc,
                                new_maps, new_iters, op.getKind());
  rewriter.replaceOpWithNewOp<mv::ShapeCastOp>(op, op.getResultType(),
                                               new_contract.getResult());
  return mlir::success();
}

class LowerVectorContractPass
    : public impl::LowerVectorContractPassBase<LowerVectorContractPass> {
 public:
  using LowerVectorContractPassBase::LowerVectorContractPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add(PromoteContractionOperands, /*benefit=*/3);
    patterns.add(DropContractionUnitDims, /*benefit=*/2);
    mv::populateVectorContractLoweringPatterns(
        patterns, mv::VectorContractLowering::OuterProduct, /*benefit=*/1);
    mv::populateVectorOuterProductLoweringPatterns(patterns, /*benefit=*/1);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace xla::cpu
