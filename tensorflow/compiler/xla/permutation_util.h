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

// Utilities for working with permutations.

#ifndef TENSORFLOW_COMPILER_XLA_PERMUTATION_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_PERMUTATION_UTIL_H_

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// Returns true if permutation is a permutation of the integers
// [0, permutation.size()).
bool IsPermutation(absl::Span<const int64> permutation);

// Applies `permutation` on `input` and returns the permuted array.
// For each i, output[i] = input[permutation[i]].
//
// Precondition:
// 1. `permutation` is a permutation of 0..permutation.size()-1.
// 2. permutation.size() == input.size().
template <typename Container>
std::vector<typename Container::value_type> Permute(
    const Container& input, absl::Span<const int64> permutation) {
  using T = typename Container::value_type;
  absl::Span<const T> data(input);
  CHECK_EQ(permutation.size(), data.size());
  CHECK(IsPermutation(permutation));
  std::vector<T> output(data.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    output[i] = data[permutation[i]];
  }
  return output;
}
// Applies the inverse of `permutation` on `input` and returns the permuted
// array. For each i, output[permutation[i]] = input[i].
//
// Precondition:
// 1. `permutation` is a permutation of 0..permutation.size()-1.
// 2. permutation.size() == input.size().
template <typename Container>
std::vector<typename Container::value_type> PermuteInverse(
    const Container& input, absl::Span<const int64> permutation) {
  using T = typename Container::value_type;
  absl::Span<const T> data(input);
  CHECK_EQ(permutation.size(), data.size());
  CHECK(IsPermutation(permutation));
  std::vector<T> output(data.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    output[permutation[i]] = data[i];
  }
  return output;
}

// Inverts a permutation, i.e., output_permutation[input_permutation[i]] = i.
std::vector<int64> InversePermutation(
    absl::Span<const int64> input_permutation);

// Composes two permutations: output[i] = p1[p2[i]].
std::vector<int64> ComposePermutations(absl::Span<const int64> p1,
                                       absl::Span<const int64> p2);

// Returns true iff permutation == {0, 1, 2, ...}.
bool IsIdentityPermutation(absl::Span<const int64> permutation);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PERMUTATION_UTIL_H_
