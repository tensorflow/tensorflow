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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_LAYOUT_HELPER_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_LAYOUT_HELPER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"

namespace mlir {

class MLIRContext;

namespace TF {

SmallVector<int64_t, 4> ReversePermutation(ArrayRef<int64_t> permutation);

SmallVector<int64_t, 4> GetDataFormatPermutation(StringRef from, StringRef to);

// Shuffle elements in the `attr` according to the permutation. Optional
// `inner_size` allows to shuffle array attributes created from rank 2 tensors
// on outer dimension only.
ArrayAttr ShuffleArrayAttr(ArrayAttr attr, ArrayRef<int64_t> permutation,
                           int inner_size = 1);

// Shuffle ranked tensor dimensions according to the permutation.
Type ShuffleRankedTensorType(Type type, ArrayRef<int64_t> permutation);

bool AreCancellablePermutations(DenseIntElementsAttr perm0,
                                DenseIntElementsAttr perm1);

// Default implementation of `LayoutSensitiveInterface::UpdateDataFormat` for
// layout sensitive operations that do not have any additional layout dependent
// attributes besides `data_format` string.
template <typename Op>
LogicalResult UpdateDataFormat(StringRef data_format, Op *op) {
  auto perm = GetDataFormatPermutation(op->data_format(), data_format);
  if (perm.empty()) return failure();

  // Update data format attribute.
  (*op)->setAttr("data_format", StringAttr::get(op->getContext(), data_format));

  // Update types for all layout sensitive results.
  auto layout_sensitive = cast<LayoutSensitiveInterface>(op->getOperation());
  for (unsigned idx : layout_sensitive.GetLayoutDependentResults()) {
    OpResult result = op->getOperation()->getResult(idx);
    result.setType(ShuffleRankedTensorType(result.getType(), perm));
  }

  return success();
}

// Default implementation for folding operand transpose into the operation.
// See `FoldOperandsTransposeInterface::FoldOperandsPermutation`.
template <typename Op>
LogicalResult FoldOperandsPermutation(
    ArrayRef<int64_t> permutation, Op *op,
    ArrayRef<std::pair<StringRef, ArrayAttr>> shuffle_attrs = {}) {
  MLIRContext *context =
      (*op)->template getParentOfType<ModuleOp>().getContext();

  // We only support NHWC <-> NCHW permutations.
  static constexpr std::array<int64_t, 4> kNchwToNhwc = {0, 2, 3, 1};
  static constexpr std::array<int64_t, 4> kNhwcToNchw = {0, 3, 1, 2};

  // Operation data format after folding `permutation`.
  StringRef target_data_format = [&]() -> StringRef {
    if (op->data_format() == "NHWC" && permutation.equals(kNchwToNhwc)) {
      return "NCHW";  // cancel NCHW->NHWC operand permutation
    } else if (op->data_format() == "NCHW" && permutation.equals(kNhwcToNchw)) {
      return "NHWC";  // cancel NHWC->NCHW operand permutation
    } else {
      return "";
    }
  }();
  if (target_data_format.empty()) return failure();

  // To fold operand `permutation` into the `op` we need shuffle all layout
  // dependent attributes and types with a reverse permutation, and change
  // operation data format to `target_data_format`.
  //
  // Example:
  //   %1 = SomeOp(...)   {data_format = NHWC}
  //   %2 = Transpose(%1) {permutation = NHWC->NCHW}
  //   %3 = Op(%2)        {data_format = NCHW}
  //
  // To bypass %2 we have to change data format to shuffle data format from NCHW
  // to NHWC, which is the reverse of operand permutation (function argument).
  auto reverse_permutation =
      GetDataFormatPermutation(op->data_format(), target_data_format);
  if (reverse_permutation.empty()) return failure();

  (*op)->setAttr("data_format", StringAttr::get(context, target_data_format));

  for (auto pair : shuffle_attrs) {
    StringRef attr_name = pair.first;
    ArrayAttr attr_value = pair.second;
    (*op)->setAttr(attr_name,
                   ShuffleArrayAttr(attr_value, reverse_permutation));
  }

  auto fold = cast<FoldOperandsTransposeInterface>(op->getOperation());
  for (unsigned idx : fold.GetLayoutDependentResults()) {
    OpResult result = op->getOperation()->getResult(idx);
    result.setType(
        ShuffleRankedTensorType(result.getType(), reverse_permutation));
  }

  return success();
}

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_LAYOUT_HELPER_H_
