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

#ifndef MLIR_HLO_GML_ST_UTILS_VECTOR_UTILS_H
#define MLIR_HLO_GML_ST_UTILS_VECTOR_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace gml_st {

// Matches a simple version of vector.transfer_read / vector.transfer_write
// `op`. We consider the version simple if it only represents a "cconversion":
// 1.  it reads from [0, ..., 0] index
// 2.  it has a minor identity permutation map
// 3.  it has no mask
template <typename TransferOp>
static LogicalResult matchSimpleTransferOp(TransferOp op,
                                           PatternRewriter &rewriter) {
  auto isZeroIndex = [](Value value) {
    auto constIndex = value.getDefiningOp<arith::ConstantIndexOp>();
    return constIndex && constIndex.value() == 0;
  };
  if (!llvm::all_of(op.getIndices(), isZeroIndex)) {
    return rewriter.notifyMatchFailure(op, "should have all indices set to 0");
  }
  if (!op.getPermutationMap().isMinorIdentity()) {
    return rewriter.notifyMatchFailure(op,
                                       "expected cannonical permutation map");
  }
  if (op.getMask()) {
    return rewriter.notifyMatchFailure(op, "should have no mask");
  }
  return success();
}

}  // namespace gml_st
}  // namespace mlir

#endif
