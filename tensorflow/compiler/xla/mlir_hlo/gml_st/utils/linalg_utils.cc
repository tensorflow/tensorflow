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

#include "mlir/Dialect/Linalg/Utils/Utils.h"

namespace mlir {
namespace gml_st {

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

}  // namespace gml_st
}  // namespace mlir
