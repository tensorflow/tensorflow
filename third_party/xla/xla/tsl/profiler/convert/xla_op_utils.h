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
#ifndef XLA_TSL_PROFILER_CONVERT_XLA_OP_UTILS_H_
#define XLA_TSL_PROFILER_CONVERT_XLA_OP_UTILS_H_

#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tsl {
namespace profiler {

// Return if a category is fusion.
inline bool IsFusion(absl::string_view category) {
  return absl::EndsWith(category, " fusion");
}

// Return a concatenation of the program name with program id.
inline std::string HloModuleNameWithProgramId(absl::string_view hlo_module_name,
                                              uint64_t program_id) {
  return absl::StrCat(hlo_module_name, "(", program_id, ")");
}

inline bool IsHloRematerialization(absl::string_view hlo_expression) {
  auto pos = hlo_expression.find_first_of('=');
  if (pos != absl::string_view::npos) {
    hlo_expression.remove_suffix(hlo_expression.size() - pos);
  }
  return absl::StrContains(hlo_expression, ".remat");
}

// Return true if framework_op is a remat.
inline bool IsFrameworkRematerialization(absl::string_view framework_op_name) {
  return absl::StrContains(framework_op_name, "/rematted_computation/");
}

// Return true if hlo_expression is a remat.
inline bool IsRematerialization(absl::string_view hlo_expression,
                                absl::string_view framework_op_name) {
  return IsHloRematerialization(hlo_expression) ||
         IsFrameworkRematerialization(framework_op_name);
}

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_CONVERT_XLA_OP_UTILS_H_
