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

#include <cstddef>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/util.h"

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
  absl::c_transform(param_cardinalities, param_strings.begin(),
                    VecParamCardinalityToString);
  return absl::StrCat("_ZGV_LLVM_", mask, vector_width,
                      absl::StrJoin(param_strings, ""));
}

inline std::string GetMangledNamePrefix(
    bool is_masked, bool last_arg_is_return_type,
    absl::Span<const intrinsics::Type> types) {
  std::vector<VecParamCardinality> param_cardinalities;
  auto front = types.front();
  // Remove the return type if it's in the types list:
  for (const auto& type : types.first(types.size() - last_arg_is_return_type)) {
    if (type.is_scalar()) {
      param_cardinalities.push_back(VecParamCardinality::kScalar);
    } else {
      param_cardinalities.push_back(VecParamCardinality::kVector);
    }
    CHECK(type.vector_width() == front.vector_width())
        << "All types must have the same vector width.";
  }
  return GetMangledNamePrefix(is_masked, front.vector_width().value_or(1),
                              param_cardinalities);
}

inline std::string GetTypedName(bool last_arg_is_return_type,
                                absl::Span<const intrinsics::Type> types,
                                absl::string_view func_name) {
  std::vector<std::string> type_names;
  type_names.reserve(types.size());
  for (const auto& type : types) {
    type_names.push_back(type.name());
  }
  if (last_arg_is_return_type) {
    type_names.insert(--type_names.end(), "to");
  }
  return absl::StrCat("xla.", func_name, ".", absl::StrJoin(type_names, "."));
}

struct ParsedFunctionName {
  std::string base_name;
  std::vector<intrinsics::Type> types;
  bool last_arg_is_return_type;
  bool is_masked;  // TODO: Add support for masked functions.
};

inline std::string GetTypedName(const ParsedFunctionName& parsed_name) {
  return GetTypedName(parsed_name.last_arg_is_return_type, parsed_name.types,
                      parsed_name.base_name);
}

inline absl::StatusOr<ParsedFunctionName> ParseFunctionName(
    absl::string_view function_name) {
  // The `to` in a typed function name is used to specify the return type, so
  // we ignore it when parsing the function name.
  static constexpr absl::string_view kIgnoredParts[] = {"to"};
  std::vector<intrinsics::Type> types;
  auto parts = absl::StrSplit(function_name, '.');
  int i = -1;
  ParsedFunctionName result;
  result.last_arg_is_return_type = false;
  result.is_masked = false;
  for (absl::string_view part : parts) {
    // Skip the first two parts, which will be `xla.<func_name>`:
    i++;
    if (i == 0) {
      if (part != "xla") {
        return InvalidArgument("Invalid function name: %s", function_name);
      }
      // skip `xla.`
      continue;
    }
    if (i == 1) {
      result.base_name = std::string(part);
      continue;
    }
    if (bool ignored =
            absl::c_find(kIgnoredParts, part) != std::end(kIgnoredParts)) {
      if (part == "to") {
        result.last_arg_is_return_type = true;
      }
      continue;
    }
    types.push_back(intrinsics::Type::FromName(part));
  }
  if (i < 2) {
    return InvalidArgument("Invalid function name: %s", function_name);
  }
  result.types = types;
  return result;
}

}  // namespace xla::codegen::intrinsic

#endif  // XLA_CODEGEN_INTRINSIC_VEC_NAME_MANGLER_H_
