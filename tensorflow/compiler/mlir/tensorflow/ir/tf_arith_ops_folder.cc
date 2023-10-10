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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_arith_ops_folder.h"

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {

// Verifies an reduction op's `input` and reduction `dims`.
LogicalResult VerifyReductionInputAndDims(Value input, Value dims,
                                          Location loc) {
  auto dims_type = dims.getType().dyn_cast<RankedTensorType>();
  if (!dims_type) return success();
  if (dims_type.getRank() > 1)
    return emitError(loc, "dimensions can only be 0D or 1D tensor");

  auto input_type = input.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return success();
  int64_t rank = input_type.getRank();

  DenseIntElementsAttr dims_attr;
  if (!matchPattern(dims, m_Constant(&dims_attr))) return success();
  for (const auto &dim_pair : llvm::enumerate(dims_attr)) {
    int64_t cur_dim = dim_pair.value().getSExtValue();
    if (cur_dim < -rank || cur_dim >= rank)
      return emitError(loc)
             << dim_pair.index() << "-th dimension should be in the range of [-"
             << rank << ", " << rank << ")";
  }

  return success();
}

LogicalResult VerifyTypeRangesAreCompatible(Operation *op,
                                            TypeRangeWithDesc range0,
                                            TypeRangeWithDesc range1) {
  if (range0.first.size() != range1.first.size()) {
    return op->emitOpError()
           << range0.second << "s (size = " << range0.first.size() << ")"
           << " should have the same number of values as " << range1.second
           << "s (size = " << range1.first.size() << ")";
  }

  for (const auto &it :
       llvm::enumerate(llvm::zip(range0.first, range1.first))) {
    int index = it.index();
    Type type0 = std::get<0>(it.value());
    Type type1 = std::get<1>(it.value());
    if (!AreCastCompatible({type0, type1}))
      return op->emitOpError(llvm::formatv(
          "{0} type {1} is incompatible with {2} type {3} at index {4}",
          range0.second, type0, range1.second, type1, index));
  }
  return success();
}

}  // namespace TF
}  // namespace mlir
