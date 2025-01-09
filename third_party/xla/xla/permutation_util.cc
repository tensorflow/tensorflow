/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/permutation_util.h"

#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"

namespace xla {

bool IsPermutation(absl::Span<const int64_t> permutation) {
  absl::InlinedVector<bool, 8> seen(permutation.size(), false);
  for (int64_t p : permutation) {
    if (p < 0 || p >= permutation.size() || seen[p]) {
      return false;
    }
    seen[p] = true;
  }
  return true;
}

std::vector<int64_t> InversePermutation(
    absl::Span<const int64_t> input_permutation) {
  DCHECK(IsPermutation(input_permutation));
  std::vector<int64_t> output_permutation(input_permutation.size(), -1);
  for (size_t i = 0; i < input_permutation.size(); ++i) {
    output_permutation[input_permutation[i]] = i;
  }
  return output_permutation;
}

std::vector<int64_t> ComposePermutations(absl::Span<const int64_t> p1,
                                         absl::Span<const int64_t> p2) {
  CHECK_EQ(p1.size(), p2.size());
  std::vector<int64_t> output;
  output.reserve(p1.size());
  for (size_t i = 0; i < p1.size(); ++i) {
    output.push_back(p1.at(p2.at(i)));
  }
  return output;
}

bool IsIdentityPermutation(absl::Span<const int64_t> permutation) {
  for (int64_t i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != i) {
      return false;
    }
  }
  return true;
}

}  // namespace xla
