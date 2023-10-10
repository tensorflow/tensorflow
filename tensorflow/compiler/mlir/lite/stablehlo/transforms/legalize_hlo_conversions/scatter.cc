/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/scatter.h"

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

LogicalResult CanonicalizeScatterUpdates(
    Operation* scatter_op, llvm::ArrayRef<int64_t> update_window_dims,
    const Value& indices, const ShapedType& indices_type, Value& updates,
    ShapedType& updates_type, ConversionPatternRewriter& rewriter) {
  auto canonical_update_window_dims = llvm::to_vector(
      llvm::seq<int64_t>(indices_type.getRank() - 1, updates_type.getRank()));

  if (canonical_update_window_dims == update_window_dims) return success();

  // Permute updates if `update_window_dims` are leading indices.
  // Other possibilities for `update_window_dims` are not supported yet.
  if (!IsIotaAttr(update_window_dims, update_window_dims.size()))
    return rewriter.notifyMatchFailure(
        scatter_op, "update_window_dims are not leading or trailing indices");

  SmallVector<int64_t, 4> permutation_array(updates_type.getRank());
  int64_t dim = 0;
  // Move leading indices to the back of the array.
  const auto permutation_array_size = permutation_array.size();
  for (int64_t i = update_window_dims.size(); i < permutation_array_size; ++i) {
    permutation_array[i] = dim;
    ++dim;
  }
  // Move trailing indices to the front of the array.
  for (int64_t i = 0; i < update_window_dims.size(); ++i) {
    permutation_array[i] = dim;
    ++dim;
  }

  auto permutation_and_shape = GetPermutationAndTransposedShape(
      permutation_array, updates_type, rewriter);

  auto transposed_updates = rewriter.create<mhlo::TransposeOp>(
      scatter_op->getLoc(), permutation_and_shape.shape, updates,
      permutation_and_shape.permutation);

  updates = transposed_updates;
  updates_type = permutation_and_shape.shape;
  return success();
}

}  // end namespace odml
}  // end namespace mlir
