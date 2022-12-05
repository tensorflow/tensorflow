/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_TRANSPOSE_TEST_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_TRANSPOSE_TEST_UTILS_H_

#include <functional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

// Generates an input tensor and permutes its dimensions.
//
// The input tensor is filled with sequentially increasing values.
//
// - shape: input tensor shape.
// - perms: permutation for the dimensions. This should hold a permutation of
//   `[|0, shape.size()|]`.
//
// Returns a vector holding the transposed data.
template <typename T>
std::vector<T> RunTestPermutation(const absl::Span<const int> shape,
                                  const absl::Span<const int> perms) {
  // Count elements and allocate output.
  const int count = absl::c_accumulate(shape, 1, std::multiplies<>{});
  std::vector<T> out(count);

  // Create the dummy data
  std::vector<T> input(count);
  absl::c_iota(input, static_cast<T>(0));

  // Make input and output shapes.
  const RuntimeShape input_shape(shape.size(), shape.data());
  RuntimeShape output_shape(perms.size());
  for (int i = 0; i < perms.size(); i++) {
    output_shape.SetDim(i, input_shape.Dims(perms[i]));
  }

  TransposeParams params{};
  params.perm_count = static_cast<int8_t>(perms.size());
  absl::c_copy(perms, params.perm);

  reference_ops::Transpose(params, input_shape, input.data(), output_shape,
                           out.data());
  return out;
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_TRANSPOSE_TEST_UTILS_H_
