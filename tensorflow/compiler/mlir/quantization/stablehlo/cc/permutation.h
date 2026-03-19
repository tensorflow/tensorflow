/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PERMUTATION_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PERMUTATION_H_

#include <cstdint>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"  // IWYU pragma: keep; required to include the definition of ArrayRef
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"  // IWYU pragma: keep; required to include the definition of SmallVector
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::quant {

// Permutes `values` with `permutation`. Returns the permuted values. Sizes of
// `values` and `permutation` must be equal, and the elements of `permutation`
// should be less than `values.size()`.
template <typename T,
          typename = std::enable_if_t<std::is_default_constructible_v<T>, void>>
SmallVector<T> Permute(const ArrayRef<T> values,
                       const ArrayRef<int64_t> permutation) {
  SmallVector<T> permuted_values(/*Size=*/values.size(), /*Value=*/T{});
  for (auto [i, permutation_idx] : llvm::enumerate(permutation)) {
    permuted_values[i] = std::move(values[permutation_idx]);
  }
  return permuted_values;
}

}  // namespace mlir::quant

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PERMUTATION_H_
