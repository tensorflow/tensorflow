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

#include "tensorflow/compiler/xla/service/name_uniquer.h"

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

bool IsAllowed(char character) {
  auto c = static_cast<unsigned char>(character);
  return (absl::ascii_isalnum(c) != 0) || c == '_' || c == '.' || c == '-';
}

}  // namespace

NameUniquer::NameUniquer(const std::string& separator) {
  CHECK(absl::c_all_of(separator, IsAllowed))
      << "separator should comprises allowed characters only";
  separator_ = separator;
}

/*static*/ std::string NameUniquer::GetSanitizedName(absl::string_view name) {
  if (name.empty()) {
    return "";
  }

  std::string result(name);
  char c = static_cast<unsigned char>(result[0]);
  if (!absl::ascii_isalpha(c) && c != '_') {
    result[0] = '_';
  }
  for (int i = 1, iter_limit = result.length(); i < iter_limit; i++) {
    if (!IsAllowed(result[i])) {
      result[i] = '_';
    }
  }

  // HLO primitive type names (with the exception of 'tuple') are keywords in
  // the HLO text representation and cannot be names, so append an underscore if
  // the name is a primitive type.
  if (primitive_util::IsPrimitiveTypeName(result) && result != "tuple") {
    result += "_";
  }

  if (absl::StartsWith(result, "__") && !absl::StartsWith(result, "__xla_")) {
    // Morph name prefix __ that is not __xla_, to avoid using name prefixes
    // reserved by the backends, such as __llvm_retpoline_ reserved by the LLVM
    // x86 backend.
    result[0] = 'a';
  }

  return result;
}

std::string NameUniquer::GetUniqueName(absl::string_view prefix) {
  std::string root =
      GetSanitizedName(prefix.empty() ? "name" : std::string(prefix));

  // Strip away numeric suffix (if any). Only recognize separator if it is in
  // the middle of the name.
  bool has_numeric_suffix = false;
  int64_t numeric_suffix = 0;
  size_t separator_index = root.rfind(separator_);
  if (separator_index != std::string::npos && (separator_index > 0) &&
      (separator_index < root.size() - 1)) {
    std::string after_suffix = root.substr(separator_index + 1);
    if (absl::SimpleAtoi(after_suffix, &numeric_suffix)) {
      has_numeric_suffix = true;
      // Remove numeric suffix from root.
      root = root.substr(0, separator_index);
    } else {
      // absl::SimpleAtoi may modify numeric_suffix even if it returns false.
      numeric_suffix = 0;
    }
  }

  SequentialIdGenerator& id_generator = generated_names_[root];
  numeric_suffix = id_generator.RegisterId(numeric_suffix);
  if (numeric_suffix == 0) {
    return has_numeric_suffix ? absl::StrCat(root, separator_, 0) : root;
  }
  absl::StrAppend(&root, separator_, numeric_suffix);
  return root;
}

}  // namespace xla
