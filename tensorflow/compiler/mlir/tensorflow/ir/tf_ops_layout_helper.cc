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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.h"

namespace mlir {
namespace TF {

SmallVector<int64_t, 4> ReversePermutation(ArrayRef<int64_t> permutation) {
  SmallVector<int64_t, 4> reverse(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    reverse[permutation[i]] = i;
  }
  return reverse;
}

SmallVector<int64_t, 4> GetDataFormatPermutation(StringRef from, StringRef to) {
  if (from == "NHWC" && to == "NCHW") {
    return {0, 3, 1, 2};
  } else if (from == "NCHW" && to == "NHWC") {
    return {0, 2, 3, 1};
  } else {
    return {};
  }
}

// Shuffle elements in the `attr` according to the permutation. Optional
// `inner_size` allows to shuffle array attributes created from rank 2 tensors
// on outer dimension only.
ArrayAttr ShuffleArrayAttr(ArrayAttr attr, ArrayRef<int64_t> permutation,
                           int inner_size) {
  if (attr.empty()) return attr;

  assert(attr.size() % inner_size == 0);
  assert(attr.size() / inner_size == permutation.size());

  SmallVector<Attribute, 8> values{attr.begin(), attr.end()};
  SmallVector<Attribute, 8> shuffled(values.size());

  for (size_t i = 0; i < permutation.size(); ++i) {
    for (size_t j = 0; j < inner_size; ++j) {
      shuffled[i * inner_size + j] = values[permutation[i] * inner_size + j];
    }
  }

  return ArrayAttr::get(attr.getContext(), shuffled);
}

// Shuffle ranked tensor dimensions according to the permutation.
Type ShuffleRankedTensorType(Type type, ArrayRef<int64_t> permutation) {
  if (auto ranked_type = type.dyn_cast<RankedTensorType>()) {
    ArrayRef<int64_t> shape = ranked_type.getShape();
    assert(permutation.size() == shape.size());

    SmallVector<int64_t, 4> new_shape(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i)
      new_shape[i] = shape[permutation[i]];

    return RankedTensorType::get(new_shape, ranked_type.getElementType());
  }

  return type;
}

bool AreCancellablePermutations(DenseIntElementsAttr perm0,
                                DenseIntElementsAttr perm1) {
  if (perm0.getNumElements() == 0 || perm1.getNumElements() == 0) return false;
  if (perm0.getNumElements() != perm1.getNumElements()) return false;

  SmallVector<int64_t, 8> perm0_values;
  for (const auto &value : perm0.getValues<APInt>())
    perm0_values.push_back(value.getSExtValue());

  SmallVector<int64_t, 8> perm1_values;
  for (const auto &value : perm1.getValues<APInt>())
    perm1_values.push_back(value.getSExtValue());

  for (int i = 0; i < perm0_values.size(); ++i) {
    if (perm0_values[perm1_values[i]] != i) return false;
  }

  return true;
}

}  // namespace TF
}  // namespace mlir
