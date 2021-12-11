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

#include "mlir-hlo/Analysis/shape_component_analysis.h"

#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

using SymbolicShapeConstraintsMap =
    ShapeComponentAnalysis::SymbolicShapeConstraintsMap;
using ShapeOrValueInfo = ShapeComponentAnalysis::ShapeOrValueInfo;
using Symbol = ShapeComponentAnalysis::Symbol;
using SymbolicExpr = ShapeComponentAnalysis::SymbolicExpr;
using SymbolicExprsMap = ShapeComponentAnalysis::SymbolicExprsMap;

namespace {
// Shape visitor. This implements a symbolic interpreter for MHLO with some
// shape and tensor dialect ops mixed in. We are interested in shapes (e.g., the
// dimensions of a tensor) and values (e.g, the elements of a shape tensor). The
// goal is to assign every component of a shape or value either a symbol, a
// constant, or a symbolic expression. We propagate these symbolic expressions
// through the various operations. Later optimization passes can use this
// information for optimizations, e.g., exploiting the equality of dimensions.
//
// The visitation happens in two phases:
//   1. Find the sources of a value's shape or value. This climbs up the
//      operations from a given value until an unknown op or a function argument
//      is found. These sources are assigned the initial symbols for each of
//      their components.
//   2. Propagate the initial symbols downwards. This builds symbolic
//      expressions so users of the analysis can pattern match things like
//      "two dimensions are multiplied".
//
// Conceptually, this is defined recursively. For each op, we compute the
// required shape or value information for the operands and then derive the
// resulting symbolic expression.
struct ShapeVisitor {
  ShapeVisitor(SymbolicExprsMap *symbolicExprsMap,
               SymbolicShapeConstraintsMap *symbolicShapeConstraintsMap)
      : symbolicExprsMap(symbolicExprsMap),
        symbolicShapeConstraintsMap(symbolicShapeConstraintsMap) {}

  void visit(ShapeOrValueInfo requestedInfo) {
    backwards_worklist.push_back(requestedInfo);

    // First, we climb up the operations so we get the set of all ops taking
    // part in this shape or value computation. An alternative would be
    // analyzing everything eagerly. This backwards pass allows us to be lazy.
    while (!backwards_worklist.empty()) {
      // Skip if already processed.
      ShapeOrValueInfo transitivelyRequestedInfo =
          backwards_worklist.pop_back_val();
      if (symbolicExprsMap->count(transitivelyRequestedInfo)) continue;

      // Skip irrelevant cases early.
      Value value = transitivelyRequestedInfo.value();
      Type ty = value.getType();
      if (!ty.isIntOrIndexOrFloat() && !ty.isa<RankedTensorType>()) continue;

      // Handle shapes.
      if (transitivelyRequestedInfo.isShapeInfo()) {
        if (value.getDefiningOp<shape::AssumingOp>()) {
          backwardAssumingShape(value);
        } else if (auto bcast =
                       value.getDefiningOp<mhlo::DynamicBroadcastInDimOp>()) {
          backwardDynamicBroadcastInDimShape(bcast);
        } else if (auto reshape =
                       value.getDefiningOp<mhlo::DynamicReshapeOp>()) {
          backwardDynamicReshapeShape(reshape);
        } else if (value.getDefiningOp<mhlo::ReduceOp>()) {
          backwardReduceShape(value);
        } else if (auto transpose = value.getDefiningOp<mhlo::TransposeOp>()) {
          backwardTransposeShape(transpose);
        } else if (auto select = value.getDefiningOp<mhlo::SelectOp>()) {
          backwardSelectShape(select);
        } else if (auto arg = value.dyn_cast<BlockArgument>()) {
          backwardBlockArgumentShape(arg);
        } else if (value.getDefiningOp() &&
                   value.getDefiningOp()
                       ->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
          backwardSameOperandsAndResultShape(value);
        } else {
          backwardUnknownShape(value);
        }
        continue;
      }

      // Skip irrelevant cases early.
      auto ranked_ty = ty.dyn_cast<RankedTensorType>();
      bool is_possibly_interesting_scalar = ty.isIntOrIndex();
      bool is_possibly_interesting_tensor =
          ranked_ty && ranked_ty.getRank() <= 1 && ranked_ty.hasStaticShape();
      if (!is_possibly_interesting_scalar && !is_possibly_interesting_tensor) {
        continue;
      }

      // Handle values.
      assert(transitivelyRequestedInfo.isValueInfo() &&
             "Expect value info at this point.");
      if (auto shapeof = value.getDefiningOp<shape::ShapeOfOp>()) {
        backwardShapeOf(shapeof);
      } else if (auto num_elements =
                     value.getDefiningOp<shape::NumElementsOp>()) {
        backwardNumElements(num_elements);
      } else if (auto dim = value.getDefiningOp<tensor::DimOp>()) {
        backwardDim(dim);
      } else if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
        backwardIndexCast(cast);
      } else if (auto fromElements =
                     value.getDefiningOp<tensor::FromElementsOp>()) {
        backwardTensorFromElements(fromElements);
      } else if (auto extract = value.getDefiningOp<tensor::ExtractOp>()) {
        backwardTensorExtract(extract);
      } else if (auto add = value.getDefiningOp<mhlo::AddOp>()) {
        backwardBinOp(add);
      } else if (auto mul = value.getDefiningOp<mhlo::MulOp>()) {
        backwardBinOp(mul);
      } else if (auto add = value.getDefiningOp<arith::AddIOp>()) {
        backwardBinOp(add);
      } else if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
        backwardBinOp(mul);
      } else if (auto concat = value.getDefiningOp<mhlo::ConcatenateOp>()) {
        backwardConcatenate(concat);
      } else if (auto reshape = value.getDefiningOp<mhlo::ReshapeOp>()) {
        backwardReshape(reshape);
      } else if (auto slice = value.getDefiningOp<mhlo::SliceOp>()) {
        backwardSlice(slice);
      } else if (matchPattern(value, m_Constant())) {
        backwardConstant(value);
      } else {
        backwardUnknown(value);
      }
    }

    // Second, we walk down from the defs to the uses, building symbolic
    // expressions for shape and value components.
    while (!forwards_worklist.empty()) {
      auto transitivelyRequestedInfo = forwards_worklist.pop_back_val();

      // Skip if already processed.
      if (symbolicExprsMap->count(transitivelyRequestedInfo)) continue;

      // Handle shapes.
      Value value = transitivelyRequestedInfo.value();
      if (!transitivelyRequestedInfo.isValueInfo()) {
        if (value.getDefiningOp<shape::AssumingOp>()) {
          forwardAssumingShape(value);
        } else if (auto broadcast =
                       value.getDefiningOp<mhlo::DynamicBroadcastInDimOp>()) {
          forwardDynamicBroadcastInDimShape(broadcast);
        } else if (auto reshape =
                       value.getDefiningOp<mhlo::DynamicReshapeOp>()) {
          forwardDynamicReshapeShape(reshape);
        } else if (value.getDefiningOp<mhlo::ReduceOp>()) {
          forwardReduceShape(value);
        } else if (auto transpose = value.getDefiningOp<mhlo::TransposeOp>()) {
          forwardTransposeShape(transpose);
        } else if (auto select = value.getDefiningOp<mhlo::SelectOp>()) {
          forwardSelectShape(select);
        } else if (value.getDefiningOp() &&
                   value.getDefiningOp()
                       ->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
          forwardSameOperandsShape(value);
        } else {
          forwardUnknownShape(value);
        }
        continue;
      }

      // Handle values.
      assert(transitivelyRequestedInfo.isValueInfo() &&
             "Expect value info at this point.");
      if (auto shapeof = value.getDefiningOp<shape::ShapeOfOp>()) {
        forwardShapeOf(shapeof);
      } else if (auto num_elements =
                     value.getDefiningOp<shape::NumElementsOp>()) {
        forwardNumElements(num_elements);
      } else if (auto dim = value.getDefiningOp<tensor::DimOp>()) {
        forwardDim(dim);
      } else if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
        forwardIndexCast(cast);
      } else if (auto fromElements =
                     value.getDefiningOp<tensor::FromElementsOp>()) {
        forwardTensorFromElements(fromElements);
      } else if (auto extract = value.getDefiningOp<tensor::ExtractOp>()) {
        forwardTensorExtract(extract);
      } else if (auto add = value.getDefiningOp<mhlo::AddOp>()) {
        forwardBinOp(add, [](AffineExpr a, AffineExpr b) { return a + b; });
      } else if (auto mul = value.getDefiningOp<mhlo::MulOp>()) {
        forwardBinOp(mul, [](AffineExpr a, AffineExpr b) { return a * b; });
      } else if (auto add = value.getDefiningOp<arith::AddIOp>()) {
        forwardBinOp(add, [](AffineExpr a, AffineExpr b) { return a + b; });
      } else if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
        forwardBinOp(mul, [](AffineExpr a, AffineExpr b) { return a * b; });
      } else if (auto concat = value.getDefiningOp<mhlo::ConcatenateOp>()) {
        forwardConcatenate(concat);
      } else if (auto reshape = value.getDefiningOp<mhlo::ReshapeOp>()) {
        forwardReshape(reshape);
      } else if (auto slice = value.getDefiningOp<mhlo::SliceOp>()) {
        forwardSlice(slice);
      } else if (matchPattern(value, m_Constant())) {
        forwardConstant(value);
      } else {
        forwardUnknown(value);
      }
    }
  }

 private:
  // ===
  // Functions that traverse the shapes of operations.
  // ===

  void backwardAssumingShape(Value op) {
    auto assumingOp = op.getDefiningOp<shape::AssumingOp>();
    auto number = op.cast<OpResult>().getResultNumber();
    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    backwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(
        cast<shape::AssumingYieldOp>(
            assumingOp.getDoRegion().back().getTerminator())
            .getOperand(number)));
  }
  void forwardAssumingShape(Value op) {
    auto assumingOp = op.getDefiningOp<shape::AssumingOp>();
    auto number = op.cast<OpResult>().getResultNumber();
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    dims = lookup(ShapeOrValueInfo::getShapeInfoOf(
        cast<shape::AssumingYieldOp>(
            assumingOp.getDoRegion().back().getTerminator())
            .getOperand(number)));
  }
  void backwardDynamicBroadcastInDimShape(mhlo::DynamicBroadcastInDimOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getValueInfoOf(op.output_dimensions()));
  }
  void forwardDynamicBroadcastInDimShape(mhlo::DynamicBroadcastInDimOp op) {
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    dims = lookup(ShapeOrValueInfo::getValueInfoOf(op.output_dimensions()));
  }
  void backwardDynamicReshapeShape(mhlo::DynamicReshapeOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getValueInfoOf(op.output_shape()));
  }
  void forwardDynamicReshapeShape(mhlo::DynamicReshapeOp op) {
    auto ranked_ty = op.getResult().getType().cast<RankedTensorType>();
    auto shape_dims =
        lookup(ShapeOrValueInfo::getValueInfoOf(op.output_shape()));
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    dimsFromStaticShape(ranked_ty, shape_dims, &dims);
  }
  void backwardReduceShape(Value op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    auto reduceOp = op.getDefiningOp<mhlo::ReduceOp>();
    if (reduceOp.inputs().size() == 1)
      backwards_worklist.push_back(
          ShapeOrValueInfo::getShapeInfoOf(reduceOp.inputs().back()));
  }
  void forwardReduceShape(Value op) {
    auto reduceOp = op.getDefiningOp<mhlo::ReduceOp>();
    if (reduceOp.inputs().size() != 1) return forwardUnknownShape(op);
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    for (auto dim : llvm::enumerate(lookup(
             ShapeOrValueInfo::getShapeInfoOf(reduceOp.inputs().back())))) {
      if (!llvm::is_contained(reduceOp.dimensions(), dim.index()))
        dims.push_back(dim.value());
    }
  }
  void backwardTransposeShape(mhlo::TransposeOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getShapeInfoOf(op.operand()));
  }
  void forwardTransposeShape(mhlo::TransposeOp op) {
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    auto in = lookup(ShapeOrValueInfo::getShapeInfoOf(op.operand()));
    auto elem = op.permutation().cast<DenseIntElementsAttr>();
    for (const auto &val : elem) dims.push_back(in[val.getZExtValue()]);
  }
  void backwardSelectShape(mhlo::SelectOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getShapeInfoOf(op.on_true()));
  }
  void forwardSelectShape(mhlo::SelectOp op) {
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    // Forward the `on_true` operand, it has the same shape as the output.
    dims = lookup(ShapeOrValueInfo::getShapeInfoOf(op.on_true()));
  }
  void backwardSameOperandsAndResultShape(Value v) {
    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(v));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getShapeInfoOf(v.getDefiningOp()->getOperand(0)));
  }
  void forwardSameOperandsShape(Value v) {
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(v));
    dims = lookup(
        ShapeOrValueInfo::getShapeInfoOf(v.getDefiningOp()->getOperand(0)));
  }
  void backwardBlockArgumentShape(BlockArgument argument) {
    // CPURT uses cpurt.symbolic_shape to describe identical dimensions. Make
    // use of that when it exists.
    //
    // Example:
    //   func @compute(
    //     %arg0: tensor<?xf32> {cpurt.symbolic_shape = dense<-2> :
    //     tensor<1xi64>}, %arg1: tensor<?xf32> {cpurt.symbolic_shape =
    //     dense<-2> : tensor<1xi64>})
    //   } { ... }
    //
    // Symbolic shape is a negative value smaller than `-1`. The concrete value
    // is not known at compile time, and in this particular example it is only
    // known that both arguments have the same shape.
    //
    // TODO(ezhulenev): Add symbolic shape attribute verifier to the cpurt
    // dialect.
    if (auto func =
            dyn_cast_or_null<FuncOp>(argument.getOwner()->getParentOp())) {
      if (auto shape = func.getArgAttrOfType<DenseIntElementsAttr>(
              argument.getArgNumber(), "cpurt.symbolic_shape")) {
        auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(argument));
        auto id = getAffineSymbolExpr(0, argument.getContext());
        for (auto symbol : llvm::enumerate(shape.getValues<ssize_t>())) {
          dims.emplace_back();
          auto &dim = dims.back();
          if (symbol.value() >= 0) {
            dim.expr =
                getAffineConstantExpr(symbol.value(), argument.getContext());
          } else {
            auto it = symbolicShapeConstraintsMap->try_emplace(
                symbol.value(),
                Symbol{ShapeOrValueInfo::getShapeInfoOf(argument),
                       symbol.index()});
            dim.symbols.push_back(it.first->second);
            dim.expr = id;
          }
        }
        return;
      }
    }
    forwardUnknownShape(argument);
  }
  void backwardUnknownShape(Value v) {
    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(v));
  }
  void forwardUnknownShape(Value v) {
    auto ranked_ty = v.getType().dyn_cast<RankedTensorType>();
    if (!ranked_ty) return;
    auto id = getAffineSymbolExpr(0, v.getContext());
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(v));
    return dimsFromStaticShape(
        ranked_ty,
        [&](size_t i) {
          SymbolicExpr d;
          d.symbols.push_back({ShapeOrValueInfo::getShapeInfoOf(v), i});
          d.expr = id;
          return d;
        },
        &dims);
  }

  // ===
  // Functions that traverse values. These can be shape tensors (e.g., of type
  // tensor<3xindex>) or interesting scalars (e.g., of type index).
  // ===

  void backwardShapeOf(shape::ShapeOfOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op.getArg()));
  }
  void forwardShapeOf(shape::ShapeOfOp op) {
    auto ranked_ty = op.getArg().getType().cast<RankedTensorType>();
    auto arg = lookup(ShapeOrValueInfo::getShapeInfoOf(op.getArg()));
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    return dimsFromStaticShape(ranked_ty, arg, &dims);
  }
  void backwardNumElements(shape::NumElementsOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getValueInfoOf(op.getShape()));
  }
  void forwardNumElements(shape::NumElementsOp op) {
    auto in = lookup(ShapeOrValueInfo::getValueInfoOf(op.getShape()));

    // Accumulate product symbolically and concrete where possible.
    int64_t concrete_product = 1;
    SymbolicExpr dim;
    for (auto &it : in) {
      // For constant expressions, we can accumulate a concrete product.
      if (auto cexpr = it.expr.dyn_cast<AffineConstantExpr>()) {
        assert(cexpr.getValue() > 0 && "shape value must be positive");
        concrete_product *= cexpr.getValue();
        continue;
      }

      // Simply copy the first sybolic factor.
      if (!dim.expr) {
        dim = it;
        continue;
      }

      // Multiply remaining symbolic factors.
      dim.expr = dim.expr *
                 it.expr.shiftSymbols(dim.symbols.size(), it.symbols.size());
      dim.symbols.append(it.symbols);
    }

    // Combine concrete and symbolic product.
    if (concrete_product != 1 || !dim.expr) {
      auto cexpr = getAffineConstantExpr(concrete_product, op.getContext());
      if (dim.expr)
        dim.expr = cexpr * dim.expr;
      else
        dim.expr = cexpr;
    }

    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    dims.push_back(dim);
  }
  void backwardDim(tensor::DimOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op.source()));
  }
  void forwardDim(tensor::DimOp op) {
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    if (auto index = op.index().getDefiningOp<arith::ConstantOp>()) {
      int64_t i = index.getValue().cast<IntegerAttr>().getInt();
      auto in = lookup(ShapeOrValueInfo::getShapeInfoOf(op.source()));
      dims.push_back({in[i].symbols, in[i].expr});
    } else {
      forwardUnknown(op);
    }
  }
  template <typename Op>
  void backwardBinOp(Op op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    // TODO(jpienaar): Switch to named accessors when MHLO uses prefixed form.
    backwards_worklist.append(
        {ShapeOrValueInfo::getValueInfoOf(op.getOperand(0)),
         ShapeOrValueInfo::getValueInfoOf(op.getOperand(1))});
  }
  template <typename Op, typename Combiner>
  void forwardBinOp(Op op, Combiner &&combiner) {
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    // TODO(jpienaar): Switch to named accessors when MHLO uses prefixed form.
    auto lhs = lookup(ShapeOrValueInfo::getValueInfoOf(op.getOperand(0)));
    auto rhs = lookup(ShapeOrValueInfo::getValueInfoOf(op.getOperand(1)));
    for (int64_t i = 0, e = dim0size(op.getType()); i != e; ++i) {
      dims.emplace_back();
      auto &dim = dims.back();
      dim.symbols.append(lhs[i].symbols);
      dim.symbols.append(rhs[i].symbols);
      dim.expr = combiner(lhs[i].expr,
                          rhs[i].expr.shiftSymbols(rhs[i].symbols.size(),
                                                   lhs[i].symbols.size()));
    }
  }
  void backwardIndexCast(arith::IndexCastOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op.getIn()));
  }
  void forwardIndexCast(arith::IndexCastOp op) {
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    auto in = lookup(ShapeOrValueInfo::getValueInfoOf(op.getIn()));
    for (int64_t i = 0, e = dim0size(op.getType()); i != e; ++i) {
      // This is intentionally not modelling the truncation/zero extension of
      // index_cast. While it's incorrect it doesn't really matter for shape
      // computations.
      dims.push_back({in[i].symbols, in[i].expr});
    }
  }
  void backwardTensorFromElements(tensor::FromElementsOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    for (auto operand : op.getOperands())
      backwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(operand));
  }
  void forwardTensorFromElements(tensor::FromElementsOp op) {
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    for (auto operand : op.getOperands()) {
      auto in = lookup(ShapeOrValueInfo::getValueInfoOf(operand));
      assert(in.size() == 1);
      dims.push_back({in[0].symbols, in[0].expr});
    }
  }
  void backwardTensorExtract(tensor::ExtractOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op.tensor()));
  }
  void forwardTensorExtract(tensor::ExtractOp op) {
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    assert(op.indices().size() == 1);
    if (auto index = op.indices().front().getDefiningOp<arith::ConstantOp>()) {
      int64_t i = index.getValue().cast<IntegerAttr>().getInt();
      // We asssume this is in bounds.
      auto in = lookup(ShapeOrValueInfo::getValueInfoOf(op.tensor()));
      dims.push_back({in[i].symbols, in[i].expr});
    } else {
      forwardUnknown(op);
    }
  }
  void backwardConstant(Value v) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(v));
  }
  void forwardConstant(Value v) {
    IntegerAttr intAttr;
    DenseIntElementsAttr denseAttr;
    if (matchPattern(v, m_Constant(&denseAttr))) {
      auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(v));
      for (uint64_t i = 0, e = dim0size(v.getType()); i != e; ++i) {
        dims.emplace_back();
        auto &dim = dims.back();
        dim.expr = getAffineConstantExpr(
            denseAttr.getValues<APInt>()[i].getSExtValue(), v.getContext());
      }
    } else if (matchPattern(v, m_Constant(&intAttr))) {
      auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(v));
      dims.emplace_back();
      auto &dim = dims.back();
      dim.expr = getAffineConstantExpr(intAttr.getInt(), v.getContext());
    } else {
      forwardUnknown(v);
    }
  }
  void backwardConcatenate(mhlo::ConcatenateOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    for (auto operand : op.getOperands())
      backwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(operand));
  }
  void forwardConcatenate(mhlo::ConcatenateOp op) {
    for (auto operand : op.getOperands()) {
      auto in = lookup(ShapeOrValueInfo::getValueInfoOf(operand));
      if (in.size() != 1) return forwardUnknown(op);
    }
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    for (auto operand : op.getOperands()) {
      auto in = lookup(ShapeOrValueInfo::getValueInfoOf(operand));
      dims.push_back({in[0].symbols, in[0].expr});
    }
  }
  void backwardReshape(mhlo::ReshapeOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getValueInfoOf(op.operand()));
  }
  void forwardReshape(mhlo::ReshapeOp op) {
    auto in = lookup(ShapeOrValueInfo::getValueInfoOf(op.operand()));
    if (in.size() != 1) return forwardUnknown(op);
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    dims.push_back({in[0].symbols, in[0].expr});
  }
  void backwardSlice(mhlo::SliceOp op) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getValueInfoOf(op.operand()));
  }
  void forwardSlice(mhlo::SliceOp op) {
    // Only handle slices equivalent to an extract.
    if (!op.getType().hasStaticShape({1})) {
      return forwardUnknown(op);
    }
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    auto in = lookup(ShapeOrValueInfo::getValueInfoOf(op.operand()));
    auto elem = op.start_indices().cast<DenseIntElementsAttr>();
    auto i = (*elem.begin()).getZExtValue();
    if (i >= in.size()) {  // Bounds check.
      return forwardUnknown(op);
    }
    dims.push_back({in[i].symbols, in[i].expr});
  }
  void backwardUnknown(Value v) {
    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(v));
  }
  void forwardUnknown(Value v) {
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(v));
    auto id = getAffineSymbolExpr(0, v.getContext());
    for (size_t i = 0, e = dim0size(v.getType()); i != e; ++i) {
      dims.emplace_back();
      auto &dim = dims.back();
      dim.symbols.push_back({ShapeOrValueInfo::getValueInfoOf(v), i});
      dim.expr = id;
    }
  }

  // ===
  // Helpers
  // ===

  static void dimsFromStaticShape(
      RankedTensorType ranked_ty,
      llvm::function_ref<SymbolicExpr(int64_t)> fallback,
      std::vector<SymbolicExpr> *merged_dims) {
    auto *ctx = ranked_ty.getContext();
    for (int64_t i = 0, e = ranked_ty.getRank(); i != e; ++i) {
      if (ranked_ty.isDynamicDim(i)) {
        merged_dims->push_back(fallback(i));
      } else {
        merged_dims->emplace_back();
        auto &d = merged_dims->back();
        d.expr = getAffineConstantExpr(ranked_ty.getDimSize(i), ctx);
      }
    }
  }

  static void dimsFromStaticShape(RankedTensorType ranked_ty,
                                  ArrayRef<SymbolicExpr> fallback,
                                  std::vector<SymbolicExpr> *merged_dims) {
    return dimsFromStaticShape(
        ranked_ty, [&](int64_t i) { return fallback[i]; }, merged_dims);
  }

  // Return the size of the first dimension. Returns 1 for scalars.
  static int64_t dim0size(Type type) {
    if (auto rankedType = type.dyn_cast<RankedTensorType>())
      return rankedType.getRank() == 0 ? 1 : rankedType.getDimSize(0);
    return 1;
  }

  // Retrieves the existing information from the cache.
  ArrayRef<SymbolicExpr> lookup(ShapeOrValueInfo requestedInfo) {
    auto i = symbolicExprsMap->find(requestedInfo);
    assert(i != symbolicExprsMap->end() && "op not processed yet?");
    return llvm::makeArrayRef(i->second);
  }

  // Inserts a new entry into the cache and returns a reference to its result
  // components.
  std::vector<SymbolicExpr> &insert(ShapeOrValueInfo requestedInfo) {
    auto i = symbolicExprsMap->try_emplace(requestedInfo);
    assert(i.second && "op already processed?");
    return i.first->second;
  }

  SymbolicExprsMap *symbolicExprsMap;
  SymbolicShapeConstraintsMap *symbolicShapeConstraintsMap;

  // Worklists for the forward and backward passes.
  SmallVector<ShapeOrValueInfo> backwards_worklist;
  SmallVector<ShapeOrValueInfo> forwards_worklist;
};
}  // namespace

void ShapeComponentAnalysis::compute(ShapeOrValueInfo requestedInfo) {
  ShapeVisitor(&symbolicExprsMap, &symbolicShapeConstraintsMap)
      .visit(requestedInfo);
}

Optional<ArrayRef<SymbolicExpr>>
ShapeComponentAnalysis::ShapeComponentAnalysis::GetShapeInfo(Value value) {
  auto request = ShapeOrValueInfo::getShapeInfoOf(value);
  compute(request);
  auto found = symbolicExprsMap.find(request);
  if (found == symbolicExprsMap.end()) return {};
  return llvm::makeArrayRef(found->second);
}

Optional<ArrayRef<SymbolicExpr>>
ShapeComponentAnalysis::ShapeComponentAnalysis::GetValueInfo(Value shape) {
  auto request = ShapeOrValueInfo::getValueInfoOf(shape);
  compute(request);
  auto found = symbolicExprsMap.find(request);
  if (found == symbolicExprsMap.end()) return {};
  return llvm::makeArrayRef(found->second);
}

void ShapeComponentAnalysis::reset() {
  symbolicExprsMap.clear();
  symbolicShapeConstraintsMap.clear();
}

bool SymbolicExpr::isConstant(int64_t value) const {
  return expr.isa<AffineConstantExpr>() &&
         expr.cast<AffineConstantExpr>().getValue() == value;
}

bool SymbolicExpr::isKnownNotNegativeOne() const {
  // If the symbol is coming from a shape it can't be a -1. Also allow results
  // of shape_of, compute_reshape_shape, and num_elements. This is correct, not
  // complete.
  auto isGoodSymbol = [](const Symbol &symbol) {
    if (symbol.source.isShapeInfo()) return true;
    Operation *op = symbol.source.value().getDefiningOp();
    if (op == nullptr) return false;
    return llvm::isa<shape::ShapeOfOp, mhlo::ComputeReshapeShapeOp,
                     shape::NumElementsOp>(op);
  };

  // For constants we know if it's -1 or not. Checking the sign is sufficient
  // here and allows for reuse below. This is correct, not complete.
  auto isGoodSymbolOrGoodConstantExpr = [&](AffineExpr expr) {
    if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>())
      return isGoodSymbol(symbols[symExpr.getPosition()]);
    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>())
      return constExpr.getValue() >= 0;
    return false;
  };

  if (isGoodSymbolOrGoodConstantExpr(expr)) return true;

  // Multiplying non-negative symbols and non-negative constants will always
  // give a positive result. This is correct, not complete.
  // TODO(kramerb): Could the analysis provide a generic interface for this?
  if (auto bexpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    return bexpr.getKind() == AffineExprKind::Mul &&
           isGoodSymbolOrGoodConstantExpr(bexpr.getLHS()) &&
           isGoodSymbolOrGoodConstantExpr(bexpr.getRHS());
  }

  return false;
}

llvm::Optional<Symbol> SymbolicExpr::singleton() const {
  if (expr.isa<AffineSymbolExpr>() &&
      expr.cast<AffineSymbolExpr>().getPosition() == 0) {
    assert(symbols.size() == 1);
    return symbols[0];
  }
  return llvm::None;
}

void SymbolicExpr::dump(llvm::raw_ostream &os) const {
  expr.print(os);
  if (!symbols.empty()) os << " with";
  os << "\n";
  if (symbols.empty()) return;
  for (auto sym : llvm::enumerate(symbols)) {
    os.indent(4);
    os << 's' << sym.index() << " = ";
    if (!sym.value().source.isValueInfo()) os << "shapeof(";
    sym.value().source.value().print(os);
    if (!sym.value().source.isValueInfo()) os << ")";
    os << '[' << sym.value().index << "]\n";
  }
}
