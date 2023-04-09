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

#include "gml_st/utils/linalg_utils.h"

#include <iterator>

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"

namespace mlir::gml_st {

using tensor::CollapseShapeOp;
using tensor::ExpandShapeOp;

bool isCwiseGenericOp(Operation *op, int64_t *arity) {
  auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op);
  if (!genericOp || genericOp.getNumDpsInits() != 1) return false;

  // Check all-parallel iterator types.
  if (!llvm::all_of(genericOp.getIteratorTypesArray(),
                    linalg::isParallelIterator))
    return false;

  // Check all-identity maps.
  if (!llvm::all_of(genericOp.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); })) {
    return false;
  }

  // Allow for pattern matching the arity.
  if (arity != nullptr) *arity = genericOp.getNumDpsInputs();
  return true;
}

bool isSimpleBcastReduction(Operation *op, int64_t *dimension,
                            SimpleBcastReduction *chain) {
  // Match bcast.
  auto broadcastOp = llvm::dyn_cast_or_null<linalg::BroadcastOp>(op);
  if (!broadcastOp) return false;

  // Match reduction.
  auto reduceOp = llvm::dyn_cast_or_null<linalg::ReduceOp>(
      broadcastOp.getOperands().front().getDefiningOp());
  if (!reduceOp || reduceOp.getNumDpsInits() != 1) return false;

  // Check that bcast and reduction dimensions match.
  auto bcstDimensions = broadcastOp.getDimensions();
  if (!bcstDimensions.empty() && bcstDimensions != reduceOp.getDimensions())
    return false;

  // Allow for pattern matching the reduction dimension and operation chain.
  if (dimension != nullptr) *dimension = bcstDimensions.front();
  if (chain != nullptr) {
    chain->bcast = op;
    chain->reduction = reduceOp;
    chain->operand = reduceOp.getInputs().front();
  }
  return true;
}

bool isTransformableIntoMatmul(linalg::Conv2DNhwcHwcfOp convOp) {
  if (!convOp.hasTensorSemantics()) return false;

  Value input = convOp.getInputs()[0];
  auto inputType = input.getType().cast<RankedTensorType>();

  Value kernel = convOp.getInputs()[1];
  auto kernelType = kernel.getType().cast<RankedTensorType>();

  Value init = convOp.getOutputs()[0];
  auto initType = init.getType().cast<RankedTensorType>();

  if (!inputType.hasStaticShape() || !kernelType.hasStaticShape() ||
      !initType.hasStaticShape()) {
    return false;
  }

  auto allOnes = [](DenseIntElementsAttr attr) {
    return attr.isSplat() && attr.getValues<int64_t>()[0] == 1;
  };
  if (!allOnes(convOp.getDilations()) || !allOnes(convOp.getStrides()))
    return false;

  if (inputType.getDimSize(0) != 1 || inputType.getDimSize(3) != 1 ||
      kernelType.getDimSize(2) != 1 || initType.getDimSize(0) != 1 ||
      initType.getDimSize(2) != 1)
    return false;
  return true;
}

FailureOr<linalg::MatmulOp> convertConvToMatmul(linalg::Conv2DNhwcHwcfOp convOp,
                                                PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(convOp);
  Value input = convOp.getInputs()[0];
  Value kernel = convOp.getInputs()[1];
  Value init = convOp.getOutputs()[0];

  auto kernelType = kernel.getType().cast<RankedTensorType>();
  if (!isTransformableIntoMatmul(convOp) || kernelType.getDimSize(0) != 1)
    return failure();

  Location loc = convOp.getLoc();
  SmallVector<ReassociationIndices> map{{0, 1}, {2, 3}};
  Value newInput = rewriter.create<CollapseShapeOp>(loc, input, map);
  Value newKernel = rewriter.create<CollapseShapeOp>(loc, kernel, map);
  Value newInit = rewriter.create<CollapseShapeOp>(loc, init, map);

  auto matmul = rewriter.create<linalg::MatmulOp>(
      loc, newInit.getType(), ValueRange{newInput, newKernel},
      ValueRange{newInit});

  rewriter.replaceOpWithNewOp<ExpandShapeOp>(convOp, convOp.getType(0),
                                             matmul.getResult(0), map);
  return matmul;
}

FailureOr<linalg::MatmulOp> convertBatchMatmulToMatmul(
    linalg::BatchMatmulOp batchMatmulOp, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(batchMatmulOp);
  Value lhs = batchMatmulOp.getInputs()[0];
  Value rhs = batchMatmulOp.getInputs()[1];
  Value init = batchMatmulOp.getOutputs()[0];

  Location loc = batchMatmulOp.getLoc();
  SmallVector<ReassociationIndices> map{{0, 1}, {2}};
  Value newLhs = rewriter.create<CollapseShapeOp>(loc, lhs, map);
  Value newRhs = rewriter.create<CollapseShapeOp>(loc, rhs, map);
  Value newInit;
  if (auto fillOp = init.getDefiningOp<linalg::FillOp>()) {
    Value collapsedInit =
        rewriter.create<CollapseShapeOp>(loc, fillOp.getOutputs().front(), map);
    newInit = rewriter
                  .create<linalg::FillOp>(loc, fillOp.getInputs(),
                                          ValueRange{collapsedInit})
                  .getResult(0);
  } else {
    newInit = rewriter.create<CollapseShapeOp>(loc, init, map);
  }

  auto matmul = rewriter.create<linalg::MatmulOp>(
      loc, newInit.getType(), ValueRange{newLhs, newRhs}, ValueRange{newInit});

  rewriter.replaceOpWithNewOp<ExpandShapeOp>(
      batchMatmulOp, batchMatmulOp.getType(0), matmul.getResult(0), map);
  return matmul;
}

FailureOr<linalg::ReduceOp> convertDotOpToReduce(linalg::DotOp dotOp,
                                                 PatternRewriter &rewriter) {
  Location loc = dotOp.getLoc();

  // Create empty tensor for linalg.map.
  Value lhs = dotOp.getInputs().front();
  FailureOr<OpFoldResult> inputSizeOfr =
      tensor::createDimValue(rewriter, loc, lhs, 0);

  if (failed(inputSizeOfr)) {
    return rewriter.notifyMatchFailure(
        dotOp, "cannot get the size of the input tensor");
  }

  Type elementType = getElementTypeOrSelf(lhs.getType());
  Value emptyTensor =
      rewriter.create<tensor::EmptyOp>(loc, *inputSizeOfr, elementType);

  // Create linalg.map.
  Operation *arithMul = &dotOp.getBody()->front();
  auto mul = rewriter.create<linalg::MapOp>(
      loc, dotOp.getOperands().take_front(2), emptyTensor,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        auto *n = mlir::clone(b, arithMul, arithMul->getResultTypes(),
                              args.take_front(2));
        b.create<linalg::YieldOp>(loc, n->getResults());
      });

  // Create linalg.reduce.
  Operation *arithAdd = &(*std::next(dotOp.getBody()->begin()));
  auto add = rewriter.create<linalg::ReduceOp>(
      loc, ValueRange{mul.getResult()}, ValueRange{dotOp.getOperand(2)},
      SmallVector<int64_t>{0},
      [&](OpBuilder &b, Location loc, ValueRange args) {
        auto *n = mlir::clone(b, arithAdd, arithAdd->getResultTypes(),
                              {args[1], args[0]});
        b.create<linalg::YieldOp>(loc, n->getResults());
      });

  rewriter.replaceOp(dotOp, add->getResults());
  return add;
}

}  // namespace mlir::gml_st
