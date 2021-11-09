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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

using ShapeOrValueOfTensor = ShapeComponentAnalysis::ShapeOrValueOfTensor;

namespace {
// Shape visitor. This implements a symbolic interpreter for MHLO, which some
// Shape and Tensor dialect ops mixed in. The goal is to assign every dimension
// of a shape tensor a symbol and propagate this through various operations.
// Later optimizations passes can use this to optimize based on what dimensions
// are known to be equal.
//
// The visitation itself happens in two phases
//   1. Find the source of a dimension. This climbs up operations from a given
//      value until an unknown op or function argument is found. That value
//      becomes the initial symbol for a dimension from which others are
//      derived.
//
//.  2. Propagate symbols downward. This builds an affine expression of the
//      symbols so users of the analysis can pattern match things like
//      "two dimensions are multiplied"
//
// Conceptually, this is defined recursively. For each op, we compute the
// required shape knowledge for the operands and then derive the result affine
// expression.
struct ShapeVisitor {
  ShapeVisitor(ShapeComponentAnalysis::DimensionsMap *dimensions,
               ShapeComponentAnalysis::ConstraintsMap *symbolicShapeConstraints)
      : dimensions(dimensions),
        symbolicShapeConstraints(symbolicShapeConstraints) {}

  void visit(ShapeOrValueOfTensor v) {
    backwards_worklist.push_back(v);

    // First we climb uses so we get a list of all ops taking part in this
    // shape computation. An alternative would be analyzing everything eagerly,
    // this backwards pass allows us to be lazy.
    while (!backwards_worklist.empty()) {
      auto value = backwards_worklist.pop_back_val();
      if (dimensions->count(value)) continue;

      Value instruction = value.value();
      if (!instruction.getType().isIntOrIndexOrFloat() &&
          !instruction.getType().isa<RankedTensorType>())
        continue;

      // Handle shapes.
      if (!value.isShapeTensor()) {
        if (instruction.getDefiningOp<shape::AssumingOp>()) {
          backwardAssumingShape(instruction);
        } else if (auto broadcast =
                       instruction
                           .getDefiningOp<mhlo::DynamicBroadcastInDimOp>()) {
          backwardDynamicBroadcastInDimShape(broadcast);
        } else if (auto reshape =
                       instruction.getDefiningOp<mhlo::DynamicReshapeOp>()) {
          backwardDynamicReshapeShape(reshape);
        } else if (instruction.getDefiningOp<mhlo::ReduceOp>()) {
          backwardReduceShape(instruction);
        } else if (auto transpose =
                       instruction.getDefiningOp<mhlo::TransposeOp>()) {
          backwardTransposeShape(transpose);
        } else if (auto select = instruction.getDefiningOp<mhlo::SelectOp>()) {
          backwardSelectShape(select);
        } else if (auto arg = instruction.dyn_cast<BlockArgument>()) {
          backwardArgumentShape(arg);
        } else if (instruction.getDefiningOp() &&
                   instruction.getDefiningOp()
                       ->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
          backwardSameOperandsShape(instruction);
        } else {
          backwardUnknownShape(instruction);
        }
        continue;
      }

      // From here on we only deal with shape tensors. Filter shapes that don't
      // describe a ranked shape early.
      if (!instruction.getType().isIntOrIndex() &&
          (instruction.getType().cast<RankedTensorType>().getRank() > 1 ||
           !instruction.getType().cast<RankedTensorType>().hasStaticShape()))
        continue;

      if (auto shapeof = instruction.getDefiningOp<shape::ShapeOfOp>()) {
        backwardShapeOf(shapeof);
      } else if (auto dim = instruction.getDefiningOp<tensor::DimOp>()) {
        backwardDim(dim);
      } else if (auto cast = instruction.getDefiningOp<arith::IndexCastOp>()) {
        backwardIndexCast(cast);
      } else if (auto fromElements =
                     instruction.getDefiningOp<tensor::FromElementsOp>()) {
        backwardTensorFromElements(fromElements);
      } else if (auto extract =
                     instruction.getDefiningOp<tensor::ExtractOp>()) {
        backwardTensorExtract(extract);
      } else if (auto add = instruction.getDefiningOp<mhlo::AddOp>()) {
        backwardBinOp(add);
      } else if (auto mul = instruction.getDefiningOp<mhlo::MulOp>()) {
        backwardBinOp(mul);
      } else if (auto concat =
                     instruction.getDefiningOp<mhlo::ConcatenateOp>()) {
        backwardConcatenate(concat);
      } else if (auto reshape = instruction.getDefiningOp<mhlo::ReshapeOp>()) {
        backwardReshape(reshape);
      } else if (auto slice = instruction.getDefiningOp<mhlo::SliceOp>()) {
        backwardSlice(slice);
      } else if (matchPattern(instruction, m_Constant())) {
        backwardConstant(instruction);
      } else {
        backwardUnknown(instruction);
      }
    }

    // Now we walk down from defs to uses, building expressions for shape
    // dimensions.
    while (!forwards_worklist.empty()) {
      auto value = forwards_worklist.pop_back_val();
      if (dimensions->count(value)) continue;
      Value instruction = value.value();
      if (!value.isShapeTensor()) {
        if (instruction.getDefiningOp<shape::AssumingOp>()) {
          forwardAssumingShape(instruction);
        } else if (auto broadcast =
                       instruction
                           .getDefiningOp<mhlo::DynamicBroadcastInDimOp>()) {
          forwardDynamicBroadcastInDimShape(broadcast);
        } else if (auto reshape =
                       instruction.getDefiningOp<mhlo::DynamicReshapeOp>()) {
          forwardDynamicReshapeShape(reshape);
        } else if (instruction.getDefiningOp<mhlo::ReduceOp>()) {
          forwardReduceShape(instruction);
        } else if (auto transpose =
                       instruction.getDefiningOp<mhlo::TransposeOp>()) {
          forwardTransposeShape(transpose);
        } else if (auto select = instruction.getDefiningOp<mhlo::SelectOp>()) {
          forwardSelectShape(select);
        } else if (instruction.getDefiningOp() &&
                   instruction.getDefiningOp()
                       ->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
          forwardSameOperandsShape(instruction);
        } else {
          forwardUnknownShape(instruction);
        }
        continue;
      }

      if (auto shapeof = instruction.getDefiningOp<shape::ShapeOfOp>()) {
        forwardShapeOf(shapeof);
      } else if (auto dim = instruction.getDefiningOp<tensor::DimOp>()) {
        forwardDim(dim);
      } else if (auto cast = instruction.getDefiningOp<arith::IndexCastOp>()) {
        forwardIndexCast(cast);
      } else if (auto fromElements =
                     instruction.getDefiningOp<tensor::FromElementsOp>()) {
        forwardTensorFromElements(fromElements);
      } else if (auto extract =
                     instruction.getDefiningOp<tensor::ExtractOp>()) {
        forwardTensorExtract(extract);
      } else if (auto add = instruction.getDefiningOp<mhlo::AddOp>()) {
        forwardBinOp(add, [](AffineExpr a, AffineExpr b) { return a + b; });
      } else if (auto mul = instruction.getDefiningOp<mhlo::MulOp>()) {
        forwardBinOp(mul, [](AffineExpr a, AffineExpr b) { return a * b; });
      } else if (auto concat =
                     instruction.getDefiningOp<mhlo::ConcatenateOp>()) {
        forwardConcatenate(concat);
      } else if (auto reshape = instruction.getDefiningOp<mhlo::ReshapeOp>()) {
        forwardReshape(reshape);
      } else if (auto slice = instruction.getDefiningOp<mhlo::SliceOp>()) {
        forwardSlice(slice);
      } else if (matchPattern(instruction, m_Constant())) {
        forwardConstant(instruction);
      } else {
        forwardUnknown(instruction);
      }
    }
  }

 private:
  // ===
  // Methods to traverse shape tensors. These are always 1D integer tensors.
  // ===
  void backwardShapeOf(shape::ShapeOfOp op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op));
    backwards_worklist.push_back(ShapeOrValueOfTensor::getShapeOf(op.getArg()));
  }
  void forwardShapeOf(shape::ShapeOfOp op) {
    auto &dims = insert(ShapeOrValueOfTensor::getValueOf(op));
    auto type = op.getArg().getType().cast<RankedTensorType>();
    auto arg = lookup(ShapeOrValueOfTensor::getShapeOf(op.getArg()));
    for (int64_t i = 0, e = type.getRank(); i != e; ++i) {
      dims.emplace_back();
      auto &dim = dims.back();
      if (!type.isDynamicDim(i)) {
        dim.expr = getAffineConstantExpr(type.getDimSize(i), op.getContext());
      } else {
        dim.symbols = arg[i].symbols;
        dim.expr = arg[i].expr;
      }
    }
  }
  void backwardDim(tensor::DimOp op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op));
    backwards_worklist.push_back(ShapeOrValueOfTensor::getShapeOf(op.source()));
  }
  void forwardDim(tensor::DimOp op) {
    auto &dims = insert(ShapeOrValueOfTensor::getValueOf(op));
    if (auto index = op.index().getDefiningOp<arith::ConstantOp>()) {
      int64_t i = index.getValue().cast<IntegerAttr>().getInt();
      auto in = lookup(ShapeOrValueOfTensor::getShapeOf(op.source()));
      dims.push_back({in[i].symbols, in[i].expr});
    } else {
      forwardUnknown(op);
    }
  }
  template <typename Op>
  void backwardBinOp(Op op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op));
    backwards_worklist.append({ShapeOrValueOfTensor::getValueOf(op.lhs()),
                               ShapeOrValueOfTensor::getValueOf(op.rhs())});
  }
  template <typename Op, typename Combiner>
  void forwardBinOp(Op op, Combiner &&combiner) {
    auto &dims = insert(ShapeOrValueOfTensor::getValueOf(op));
    auto lhs = lookup(ShapeOrValueOfTensor::getValueOf(op.lhs()));
    auto rhs = lookup(ShapeOrValueOfTensor::getValueOf(op.rhs()));
    for (int i = 0, e = dim0size(op.getType()); i != e; ++i) {
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
    forwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op));
    backwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op.getIn()));
  }
  void forwardIndexCast(arith::IndexCastOp op) {
    auto &dims = insert(ShapeOrValueOfTensor::getValueOf(op));
    auto in = lookup(ShapeOrValueOfTensor::getValueOf(op.getIn()));
    for (int64_t i = 0, e = dim0size(op.getType()); i != e; ++i) {
      // This is intentionally not modelling the truncation/zero extension of
      // index_cast. While it's incorrect it doesn't really matter for shape
      // computations.
      dims.push_back({in[i].symbols, in[i].expr});
    }
  }
  void backwardTensorFromElements(tensor::FromElementsOp op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op));
    for (auto operand : op.getOperands())
      backwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(operand));
  }
  void forwardTensorFromElements(tensor::FromElementsOp op) {
    auto &dims = insert(ShapeOrValueOfTensor::getValueOf(op));
    for (auto operand : op.getOperands()) {
      auto in = lookup(ShapeOrValueOfTensor::getValueOf(operand));
      assert(in.size() == 1);
      dims.push_back({in[0].symbols, in[0].expr});
    }
  }
  void backwardTensorExtract(tensor::ExtractOp op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op));
    backwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op.tensor()));
  }
  void forwardTensorExtract(tensor::ExtractOp op) {
    auto &dims = insert(ShapeOrValueOfTensor::getValueOf(op));
    assert(op.indices().size() == 1);
    if (auto index = op.indices().front().getDefiningOp<arith::ConstantOp>()) {
      int64_t i = index.getValue().cast<IntegerAttr>().getInt();
      // We asssume this is in bounds.
      auto in = lookup(ShapeOrValueOfTensor::getValueOf(op.tensor()));
      dims.push_back({in[i].symbols, in[i].expr});
    } else {
      forwardUnknown(op);
    }
  }
  void backwardConstant(Value op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op));
  }
  void forwardConstant(Value op) {
    auto &dims = insert(ShapeOrValueOfTensor::getValueOf(op));
    IntegerAttr intAttr;
    DenseIntElementsAttr denseAttr;
    if (matchPattern(op, m_Constant(&denseAttr))) {
      for (uint64_t i = 0, e = dim0size(op.getType()); i != e; ++i) {
        dims.emplace_back();
        auto &dim = dims.back();
        dim.expr = getAffineConstantExpr(
            denseAttr.getValues<APInt>()[i].getSExtValue(), op.getContext());
      }
    } else if (matchPattern(op, m_Constant(&intAttr))) {
      dims.emplace_back();
      auto &dim = dims.back();
      dim.expr = getAffineConstantExpr(intAttr.getInt(), op.getContext());
    } else {
      forwardUnknown(op);
    }
  }
  void backwardConcatenate(mhlo::ConcatenateOp op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op));
    for (auto operand : op.getOperands())
      backwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(operand));
  }
  void forwardConcatenate(mhlo::ConcatenateOp op) {
    for (auto operand : op.getOperands()) {
      auto in = lookup(ShapeOrValueOfTensor::getValueOf(operand));
      if (in.size() != 1) return forwardUnknown(op);
    }
    auto &dims = insert(ShapeOrValueOfTensor::getValueOf(op));
    for (auto operand : op.getOperands()) {
      auto in = lookup(ShapeOrValueOfTensor::getValueOf(operand));
      dims.push_back({in[0].symbols, in[0].expr});
    }
  }
  void backwardReshape(mhlo::ReshapeOp op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op));
    backwards_worklist.push_back(
        ShapeOrValueOfTensor::getValueOf(op.operand()));
  }
  void forwardReshape(mhlo::ReshapeOp op) {
    auto in = lookup(ShapeOrValueOfTensor::getValueOf(op.operand()));
    if (in.size() != 1) return forwardUnknown(op);
    auto &dims = insert(ShapeOrValueOfTensor::getValueOf(op));
    dims.push_back({in[0].symbols, in[0].expr});
  }
  void backwardSlice(mhlo::SliceOp op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op));
    backwards_worklist.push_back(
        ShapeOrValueOfTensor::getValueOf(op.operand()));
  }
  void forwardSlice(mhlo::SliceOp op) {
    // Only handle slices equivalent to an extract.
    if (!op.getType().hasStaticShape({1})) {
      return forwardUnknown(op);
    }
    auto &dims = insert(ShapeOrValueOfTensor::getValueOf(op));
    auto in = lookup(ShapeOrValueOfTensor::getValueOf(op.operand()));
    auto elem = op.start_indices().cast<DenseIntElementsAttr>();
    auto i = (*elem.begin()).getZExtValue();
    if (i >= in.size()) {  // Bounds check.
      return forwardUnknown(op);
    }
    dims.push_back({in[i].symbols, in[i].expr});
  }
  void backwardUnknown(Value op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getValueOf(op));
  }
  void forwardUnknown(Value op) {
    auto &dims = insert(ShapeOrValueOfTensor::getValueOf(op));
    auto id = getAffineSymbolExpr(0, op.getContext());
    for (size_t i = 0, e = dim0size(op.getType()); i != e; ++i) {
      dims.emplace_back();
      auto &dim = dims.back();
      dim.symbols.push_back({ShapeOrValueOfTensor::getValueOf(op), i});
      dim.expr = id;
    }
  }

  // ===
  // Methods that traverse the shapes of operations.
  // ===

  void backwardAssumingShape(Value op) {
    auto assumingOp = op.getDefiningOp<shape::AssumingOp>();
    auto number = op.cast<OpResult>().getResultNumber();
    forwards_worklist.push_back(ShapeOrValueOfTensor::getShapeOf(op));
    backwards_worklist.push_back(ShapeOrValueOfTensor::getShapeOf(
        cast<shape::AssumingYieldOp>(
            assumingOp.getDoRegion().back().getTerminator())
            .getOperand(number)));
  }
  void forwardAssumingShape(Value op) {
    auto assumingOp = op.getDefiningOp<shape::AssumingOp>();
    auto number = op.cast<OpResult>().getResultNumber();
    auto &dims = insert(ShapeOrValueOfTensor::getShapeOf(op));
    dims = lookup(ShapeOrValueOfTensor::getShapeOf(
        cast<shape::AssumingYieldOp>(
            assumingOp.getDoRegion().back().getTerminator())
            .getOperand(number)));
  }
  void backwardDynamicBroadcastInDimShape(mhlo::DynamicBroadcastInDimOp op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getShapeOf(op));
    backwards_worklist.push_back(
        ShapeOrValueOfTensor::getValueOf(op.output_dimensions()));
  }
  void forwardDynamicBroadcastInDimShape(mhlo::DynamicBroadcastInDimOp op) {
    auto &dims = insert(ShapeOrValueOfTensor::getShapeOf(op));
    dims = lookup(ShapeOrValueOfTensor::getValueOf(op.output_dimensions()));
  }
  void backwardDynamicReshapeShape(mhlo::DynamicReshapeOp op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getShapeOf(op));
    backwards_worklist.push_back(
        ShapeOrValueOfTensor::getValueOf(op.output_shape()));
  }
  void forwardDynamicReshapeShape(mhlo::DynamicReshapeOp op) {
    auto &dims = insert(ShapeOrValueOfTensor::getShapeOf(op));
    dims = lookup(ShapeOrValueOfTensor::getValueOf(op.output_shape()));
  }
  void backwardReduceShape(Value op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getShapeOf(op));
    auto reduceOp = op.getDefiningOp<mhlo::ReduceOp>();
    if (reduceOp.inputs().size() == 1)
      backwards_worklist.push_back(
          ShapeOrValueOfTensor::getShapeOf(reduceOp.inputs().back()));
  }
  void forwardReduceShape(Value op) {
    auto reduceOp = op.getDefiningOp<mhlo::ReduceOp>();
    if (reduceOp.inputs().size() != 1) return forwardUnknownShape(op);
    auto &dims = insert(ShapeOrValueOfTensor::getShapeOf(op));
    for (auto dim : llvm::enumerate(lookup(
             ShapeOrValueOfTensor::getShapeOf(reduceOp.inputs().back())))) {
      if (!llvm::is_contained(reduceOp.dimensions(), dim.index()))
        dims.push_back(dim.value());
    }
  }
  void backwardTransposeShape(mhlo::TransposeOp op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getShapeOf(op));
    backwards_worklist.push_back(
        ShapeOrValueOfTensor::getShapeOf(op.operand()));
  }
  void forwardTransposeShape(mhlo::TransposeOp op) {
    auto &dims = insert(ShapeOrValueOfTensor::getShapeOf(op));
    auto in = lookup(ShapeOrValueOfTensor::getShapeOf(op.operand()));
    auto elem = op.permutation().cast<DenseIntElementsAttr>();
    for (const auto &val : elem) dims.push_back(in[val.getZExtValue()]);
  }
  void backwardSelectShape(mhlo::SelectOp op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getShapeOf(op));
    backwards_worklist.push_back(
        ShapeOrValueOfTensor::getShapeOf(op.on_true()));
  }
  void forwardSelectShape(mhlo::SelectOp op) {
    auto &dims = insert(ShapeOrValueOfTensor::getShapeOf(op));
    // Forward the `on_true` operand, it has the same shape as the output.
    dims = lookup(ShapeOrValueOfTensor::getShapeOf(op.on_true()));
  }
  void backwardSameOperandsShape(Value op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getShapeOf(op));
    backwards_worklist.push_back(
        ShapeOrValueOfTensor::getShapeOf(op.getDefiningOp()->getOperand(0)));
  }
  void forwardSameOperandsShape(Value op) {
    auto &dims = insert(ShapeOrValueOfTensor::getShapeOf(op));
    dims = lookup(
        ShapeOrValueOfTensor::getShapeOf(op.getDefiningOp()->getOperand(0)));
  }
  void backwardArgumentShape(BlockArgument argument) {
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
        auto &dims = insert(ShapeOrValueOfTensor::getShapeOf(argument));
        auto id = getAffineSymbolExpr(0, argument.getContext());
        for (auto symbol : llvm::enumerate(shape.getValues<ssize_t>())) {
          dims.emplace_back();
          auto &dim = dims.back();
          if (symbol.value() >= 0) {
            dim.expr =
                getAffineConstantExpr(symbol.value(), argument.getContext());
          } else {
            auto it = symbolicShapeConstraints->try_emplace(
                symbol.value(), ShapeComponentAnalysis::Symbol{
                                    ShapeOrValueOfTensor::getShapeOf(argument),
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
  void backwardUnknownShape(Value op) {
    forwards_worklist.push_back(ShapeOrValueOfTensor::getShapeOf(op));
  }
  void forwardUnknownShape(Value op) {
    auto &dims = insert(ShapeOrValueOfTensor::getShapeOf(op));
    auto type = op.getType().cast<RankedTensorType>();
    auto id = getAffineSymbolExpr(0, op.getContext());
    for (size_t i = 0, e = type.getRank(); i != e; ++i) {
      dims.emplace_back();
      auto &dim = dims.back();
      if (!type.isDynamicDim(i)) {
        dim.expr = getAffineConstantExpr(type.getDimSize(i), op.getContext());
      } else {
        dim.symbols.push_back({ShapeOrValueOfTensor::getShapeOf(op), i});
        dim.expr = id;
      }
    }
  }

  // ===
  // Helpers
  // ===

  // Return the size of the first dimension. Returns 1 for scalars.
  static int64_t dim0size(Type type) {
    if (auto rankedType = type.dyn_cast<RankedTensorType>())
      return rankedType.getRank() == 0 ? 1 : rankedType.getDimSize(0);
    return 1;
  }

  // Retrieves an existing op from the dimensions map.
  ArrayRef<ShapeComponentAnalysis::SymbolicDimension> lookup(
      ShapeOrValueOfTensor op) {
    auto i = dimensions->find(op);
    assert(i != dimensions->end() && "op not processed yet?");
    return llvm::makeArrayRef(i->second);
  }

  // Inserts a new op into the map and returns a reference to its dimensions.
  std::vector<ShapeComponentAnalysis::SymbolicDimension> &insert(
      ShapeOrValueOfTensor op) {
    auto i = dimensions->try_emplace(op);
    assert(i.second && "op already processed?");
    return i.first->second;
  }

  ShapeComponentAnalysis::DimensionsMap *dimensions;
  ShapeComponentAnalysis::ConstraintsMap *symbolicShapeConstraints;
  // Worklist for the backwards pass.
  SmallVector<ShapeOrValueOfTensor> backwards_worklist;
  // Worklist for the forwards pass.
  SmallVector<ShapeOrValueOfTensor> forwards_worklist;
};
}  // namespace

void ShapeComponentAnalysis::compute(ShapeOrValueOfTensor v) {
  ShapeVisitor(&dimensions, &symbolicShapeConstraints).visit(v);
}

Optional<ArrayRef<ShapeComponentAnalysis::SymbolicDimension>>
ShapeComponentAnalysis::ShapeComponentAnalysis::dimensionsForShape(
    Value value) {
  compute(ShapeOrValueOfTensor::getShapeOf(value));
  auto found = dimensions.find(ShapeOrValueOfTensor::getShapeOf(value));
  if (found == dimensions.end()) return {};
  return llvm::makeArrayRef(found->second);
}

Optional<ArrayRef<ShapeComponentAnalysis::SymbolicDimension>>
ShapeComponentAnalysis::ShapeComponentAnalysis::dimensionsForShapeTensor(
    Value shape) {
  compute(ShapeOrValueOfTensor::getValueOf(shape));
  auto found = dimensions.find(ShapeOrValueOfTensor::getValueOf(shape));
  if (found == dimensions.end()) return {};
  return llvm::makeArrayRef(found->second);
}

void ShapeComponentAnalysis::reset() {
  dimensions.clear();
  symbolicShapeConstraints.clear();
}

bool ShapeComponentAnalysis::SymbolicDimension::isConstant(
    int64_t value) const {
  return expr.isa<AffineConstantExpr>() &&
         expr.cast<AffineConstantExpr>().getValue() == value;
}

bool ShapeComponentAnalysis::SymbolicDimension::isKnownNotNegativeOne() const {
  // If the symbol is coming from a shape it can't be a -1. Also allow
  // chains of compute_reshape_shape.
  auto isGoodSymbol = [](const ShapeComponentAnalysis::Symbol &symbol) {
    return !symbol.source.isShapeTensor() ||
           symbol.source.value().getDefiningOp<mhlo::ComputeReshapeShapeOp>();
  };
  if (auto symbol = singleton())
    if (isGoodSymbol(*symbol)) return true;

  // For constants we know if it's -1 or not.
  if (auto cexpr = expr.dyn_cast<AffineConstantExpr>())
    if (cexpr.getValue() != -1) return true;

  // Multiplying symbols that are never negative gives a positive result.
  // TODO(kramerb): Could the analysis provide a generic interface for this?
  if (auto bexpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    auto lhs = bexpr.getLHS().dyn_cast<AffineSymbolExpr>();
    auto rhs = bexpr.getRHS().dyn_cast<AffineSymbolExpr>();

    if (bexpr.getKind() != AffineExprKind::Mul || !lhs || !rhs) return false;

    if (!llvm::all_of(symbols, isGoodSymbol)) return false;
    return true;
  }
  return false;
}

llvm::Optional<ShapeComponentAnalysis::Symbol>
ShapeComponentAnalysis::SymbolicDimension::singleton() const {
  if (expr.isa<AffineSymbolExpr>() &&
      expr.cast<AffineSymbolExpr>().getPosition() == 0) {
    assert(symbols.size() == 1);
    return symbols[0];
  }
  return llvm::None;
}

void ShapeComponentAnalysis::SymbolicDimension::dump(
    llvm::raw_ostream &os) const {
  expr.print(os);
  os << " with ";
  for (auto sym : llvm::enumerate(symbols)) {
    os << 's' << sym.index() << " = ";
    if (!sym.value().source.isShapeTensor()) os << "shapeof(";
    sym.value().source.value().print(os);
    if (!sym.value().source.isShapeTensor()) os << ")";
    os << '[' << sym.value().index << "]; ";
  }
  os << '\n';
}
