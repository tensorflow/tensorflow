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

#include "tensorflow/compiler/xla/service/gather_scatter_utils.h"

#include <vector>

#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

StatusOr<HloInstruction*> MaybeTranspose(
    HloInstruction* operand, absl::Span<const int64_t> permutation) {
  if (IsIdentityPermutation(permutation)) {
    return operand;
  }
  TF_ASSIGN_OR_RETURN(auto* result, MakeTransposeHlo(operand, permutation));
  return result;
}

StatusOr<std::vector<HloInstruction*>> MaybeTranspose(
    absl::Span<HloInstruction* const> operands,
    const std::vector<int64_t>& operand_permutation) {
  std::vector<HloInstruction*> result;
  result.reserve(operands.size());
  for (auto* operand : operands) {
    TF_ASSIGN_OR_RETURN(result.emplace_back(),
                        MaybeTranspose(operand, operand_permutation));
  }
  return result;
}

StatusOr<HloInstruction*> MoveDimensionToEnd(HloInstruction* operand,
                                             size_t dimension, size_t rank) {
  std::vector<int64_t> permutation;
  for (size_t i = 0; i < rank; ++i) {
    if (i != dimension) permutation.push_back(i);
  }
  permutation.push_back(dimension);
  return MaybeTranspose(operand, permutation);
}

}  // namespace xla
