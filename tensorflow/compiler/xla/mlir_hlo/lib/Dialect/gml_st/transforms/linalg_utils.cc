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

#include "mlir-hlo/Dialect/gml_st/transforms/linalg_utils.h"

namespace mlir {
namespace gml_st {

namespace {

bool hasUniqueInputAndOutputMaps(linalg::GenericOp genericOp,
                                 AffineMap &inputMap, AffineMap &outputMap) {
  if (genericOp.getNumInputs() != 1 || genericOp.getNumOutputs() != 1) {
    return false;
  }
  inputMap = genericOp.getIndexingMapsArray().front();
  outputMap = genericOp.getIndexingMapsArray().back();
  return true;
}

// Checks if an affine map maps all dimensions in sequence, skipping a unique
// dimension. This can be the output map of a reduction, or the input map of a
// bcast. For example:
//   - affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//   - affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//   - affine_map<(d0, d1) -> (d0)>
//   - affine_map<(d0, d1) -> (d1)>
bool isBcastOrReductionMap(AffineMap map, int64_t &dim) {
  const auto *it = map.getResults().begin();
  const auto *end = map.getResults().end();
  auto consumeIotaSeq = [&](int64_t &i) {
    while (it != end) {
      auto expr = it->dyn_cast<AffineDimExpr>();
      if (!expr || expr.getPosition() != i) break;
      it++;
      i++;
    }
  };
  int64_t i = 0;
  consumeIotaSeq(i);
  dim = i++;
  consumeIotaSeq(i);
  return i == map.getNumDims();
}

}  // namespace

bool isSimpleReduction(Operation *op, int64_t *dimension, Value *operand) {
  auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op);
  if (!genericOp || genericOp.getNumOutputs() != 1) return false;

  // Expect monadic op.
  AffineMap inputMap, outputMap;
  if (!hasUniqueInputAndOutputMaps(genericOp, inputMap, outputMap))
    return false;

  // Check identity of operand map.
  if (!inputMap.isIdentity()) return false;

  // Check that the output map is a reduction: it maps all dimensions in
  // seqence, skipping the unique reduction dimension.
  int64_t dim;
  if (!isBcastOrReductionMap(outputMap, dim)) return false;

  // Check uniqueness of reduction dimension and remaining parallel iterator
  // types.
  auto iterTys = genericOp.getIteratorTypes();
  for (int i = 0; i < iterTys.size(); i++) {
    StringRef expectedTy = i == dim ? getReductionIteratorTypeName()
                                    : getParallelIteratorTypeName();
    StringRef actualTy =
        genericOp.getIteratorTypes()[i].cast<StringAttr>().getValue();
    if (expectedTy != actualTy) return false;
  }

  // Allow for pattern matching the reduction dimension and operand.
  if (dimension != nullptr) *dimension = dim;
  if (operand != nullptr) *operand = genericOp.getInputs().front();

  return true;
}

bool isCwiseGenericOp(Operation *op, int64_t *arity) {
  auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op);
  if (!genericOp || genericOp.getNumOutputs() != 1) return false;

  // Check all-parallel iterator types.
  if (!llvm::all_of(genericOp.getIteratorTypes(), [](Attribute it) {
        return it.cast<StringAttr>().getValue() ==
               getParallelIteratorTypeName();
      })) {
    return false;
  }

  // Check all-identity maps.
  if (!llvm::all_of(genericOp.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); })) {
    return false;
  }

  // Allow for pattern matching the arity.
  if (arity != nullptr) *arity = genericOp.getNumInputs();
  return true;
}

bool isUnaryCwiseGenericOp(Operation *op) {
  int64_t arity;
  return isCwiseGenericOp(op, &arity) && arity == 1;
}

bool isSimpleBcast(Operation *op, int64_t *dimension, Value *operand) {
  auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op);
  if (!genericOp) return false;

  // Expect monadic op.
  AffineMap inputMap, outputMap;
  if (!hasUniqueInputAndOutputMaps(genericOp, inputMap, outputMap))
    return false;

  // Check all-parallel iterator types.
  if (!llvm::all_of(genericOp.getIteratorTypes(), [](Attribute it) {
        return it.cast<StringAttr>().getValue() ==
               getParallelIteratorTypeName();
      })) {
    return false;
  }

  // Check that the operand map is a degenerate bcast: it maps all dimensions in
  // seqence, skipping the unique bcast dimension.
  int64_t dim;
  if (!isBcastOrReductionMap(inputMap, dim)) return false;

  // Check that the output map is the identity.
  if (!outputMap.isIdentity()) return false;

  // Allow for pattern matching the reduction dimension and operand.
  if (dimension != nullptr) *dimension = dim;
  if (operand != nullptr) *operand = genericOp.getInputs().front();

  return true;
}

bool isSimpleBcastReduction(Operation *op, int64_t *dimension,
                            SimpleBcastReduction *chain) {
  // Match bcast.
  int64_t bcastDim;
  Value bcastOperand;
  if (!isSimpleBcast(op, &bcastDim, &bcastOperand)) {
    return false;
  }

  // Match reduction.
  Operation *reduction = bcastOperand.getDefiningOp();
  int64_t reductionDim;
  Value operand;
  if (!isSimpleReduction(reduction, &reductionDim, &operand)) {
    return false;
  }

  // Check that bcast and reduction dimensions match.
  if (bcastDim != reductionDim) return false;

  // Allow for pattern matching the reduction dimension and operation chain.
  if (dimension != nullptr) *dimension = bcastDim;
  if (chain != nullptr) {
    chain->bcast = op;
    chain->operand = operand;
    chain->operand = operand;
  }

  return true;
}

}  // namespace gml_st
}  // namespace mlir