/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This file implements utilities for lowering CHLO/HLO/LHLO dialect to Linalg
// dialect.

#include "mhlo/utils/legalize_to_linalg_utils.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace mhlo {
namespace {

bool hasIntegralShapeType(Operation* op) {
  auto stp = op->getOperand(0).getType().dyn_cast<ShapedType>();
  return stp && stp.getElementType().isIntOrIndex();
}

}  // namespace

SmallVector<utils::IteratorType, 3> getParallelAndReductionIterators(
    unsigned nLoops, unsigned nReduction) {
  SmallVector<utils::IteratorType, 3> res(nLoops - nReduction,
                                          utils::IteratorType::parallel);
  res.append(nReduction, utils::IteratorType::reduction);
  return res;
}

SmallVector<utils::IteratorType, 3> getNParallelLoopsAttrs(
    unsigned nParallelLoops) {
  return getParallelAndReductionIterators(nParallelLoops, 0);
}

Value getEmptySparseTensor(OpBuilder& b, Location loc, ShapedType type,
                           ArrayRef<Value> dynSizes) {
  return b.create<bufferization::AllocTensorOp>(loc, type.cast<TensorType>(),
                                                dynSizes,
                                                /*copy=*/Value(),
                                                /*memory_space=*/IntegerAttr());
}

Value getEmptyTensor(OpBuilder& b, Location loc, ShapedType type,
                     ArrayRef<Value> dynSizes) {
  return b.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType(),
                                   dynSizes,
                                   type.cast<RankedTensorType>().getEncoding());
}

Value getEmptyTensorFor(OpBuilder& b, Location loc, ShapedType resultType,
                        Operation* op, ValueRange operands) {
  bool isSparse = sparse_tensor::getSparseTensorEncoding(resultType) != nullptr;
  // Collect the sizes for a ranked tensor to be passed as parameter to a
  // new tensor initialization operation. This operation only needs the
  // dynamic sizes.
  SmallVector<Value> sizes;
  if (resultType.hasRank() && !resultType.hasStaticShape()) {
    // Ask the op for its output shape.
    auto shapeSource = cast<InferShapedTypeOpInterface>(op);
    SmallVector<Value, 1> reifiedShapes;
    (void)shapeSource.reifyReturnTypeShapes(b, operands, reifiedShapes);
    assert(reifiedShapes.size() == 1 && "Expected one reified result");
    // Construct sizes for the required dimensions.
    for (const auto& en : llvm::enumerate(resultType.getShape())) {
      if (en.value() != ShapedType::kDynamic) continue;
      sizes.push_back(b.create<tensor::ExtractOp>(
          loc, reifiedShapes[0],
          ValueRange{b.create<arith::ConstantIndexOp>(loc, en.index())}));
    }
  }
  return isSparse ? getEmptySparseTensor(b, loc, resultType, sizes)
                  : getEmptyTensor(b, loc, resultType, sizes);
}

Value preSparsify(Operation* op, llvm::SmallVector<Value, 2>& values, Type rtp,
                  OpBuilder* b) {
  // Apply for semi-ring operations that lower to elaborate code
  // (any sign-op, or an integral abs-op).
  // TODO(peiming, ajcbik): these all can potentially be optimized by applying
  // value transform on sparse_tenosr.value memref
  if (isa<mhlo::SignOp>(op) || isa<mhlo::NegOp>(op) ||
      (isa<mhlo::AbsOp>(op) && hasIntegralShapeType(op)) ||
      isa<chlo::AsinOp>(op) || isa<chlo::AsinhOp>(op) ||
      isa<chlo::AtanOp>(op) || isa<chlo::AtanhOp>(op) ||
      isa<chlo::BesselI1eOp>(op) || isa<chlo::SinhOp>(op) ||
      isa<chlo::TanOp>(op)) {
    if (!sparse_tensor::getSparseTensorEncoding(op->getResult(0).getType()) &&
        !sparse_tensor::getSparseTensorEncoding(op->getOperand(0).getType()))
      return Value();
    Location loc = op->getLoc();
    auto semiring = b->create<sparse_tensor::UnaryOp>(loc, rtp, values[0]);
    Type itp = values[0].getType();
    Block* present = b->createBlock(&semiring.getPresentRegion(), {}, itp, loc);
    b->setInsertionPointToStart(&semiring.getPresentRegion().front());
    values[0] = present->getArgument(0);
    return semiring;
  }
  return Value();
}

Value postSparsify(Operation* op, Value semiring, Value result, OpBuilder* b) {
  if (semiring) {
    b->create<sparse_tensor::YieldOp>(op->getLoc(), result);
    b->setInsertionPointAfter(semiring.getDefiningOp());
    return semiring;
  }
  return result;
}

bool allOperandsAreScalarTensors(Operation* op) {
  return llvm::all_of(op->getOperands(), [](Value operand) {
    auto operandTy = operand.getType().dyn_cast<ShapedType>();
    return operandTy && operandTy.getRank() == 0;
  });
}

bool isInBodyOfLinalgOps(Operation* op) {
  auto* parentOp = op->getParentRegion()->getParentOp();
  return parentOp->getDialect() ==
         parentOp->getContext()->getLoadedDialect<linalg::LinalgDialect>();
}

}  // namespace mhlo

}  // namespace mlir
