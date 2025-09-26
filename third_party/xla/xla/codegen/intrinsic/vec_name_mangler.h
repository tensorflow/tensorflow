/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_INTRINSIC_VEC_NAME_MANGLER_H_
#define XLA_CODEGEN_INTRINSIC_VEC_NAME_MANGLER_H_

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

namespace xla::codegen::intrinsic {

enum class VecParamCardinality {
  kScalar,
  kVector,
  kLinear,
};

constexpr absl::string_view VecParamCardinalityToString(
    VecParamCardinality param_cardinality) {
  switch (param_cardinality) {
    case VecParamCardinality::kScalar:
      return "s";
    case VecParamCardinality::kVector:
      return "v";
    case VecParamCardinality::kLinear:
      return "l";
  }
}

// Returns the prefix of the vectorized function name given the parameters.
// These prefixes are used to create VecDescs in LLVM:
// https://llvm.org/doxygen/classllvm_1_1VecDesc.html
inline std::string GetMangledNamePrefix(
    bool is_masked, size_t vector_width,
    std::vector<VecParamCardinality> param_cardinalities) {
  std::string mask = is_masked ? "M" : "N";

  std::vector<std::string> param_strings(param_cardinalities.size());
  std::transform(param_cardinalities.begin(), param_cardinalities.end(),
                 param_strings.begin(), VecParamCardinalityToString);
  return absl::StrCat("_ZGV_LLVM_", mask, vector_width,
                      absl::StrJoin(param_strings, ""));
}

}  // namespace xla::codegen::intrinsic

#endif  // XLA_CODEGEN_INTRINSIC_VEC_NAME_MANGLER_H_
