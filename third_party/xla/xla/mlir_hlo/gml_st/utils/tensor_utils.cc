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

#include "gml_st/utils/tensor_utils.h"

namespace mlir::gml_st {

// Returns ids of size-1 dims that were expanded or collapsed by
// tensor.expand_shape/tensor.collapse_shape.
SmallVector<int64_t> getPreservedDimensions(
    ArrayRef<int64_t> shape,
    ArrayRef<ReassociationIndices> reassociationIndices) {
  SmallVector<int64_t> result;
  for (ReassociationIndicesRef indices : reassociationIndices) {
    const auto* findIt =
        llvm::find_if(indices, [&](int64_t idx) { return shape[idx] != 1; });
    result.push_back(findIt == indices.end() ? 0 : *findIt);
  }
  return result;
}

}  // namespace mlir::gml_st
