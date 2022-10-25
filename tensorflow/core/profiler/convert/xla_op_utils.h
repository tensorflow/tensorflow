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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XLA_OP_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XLA_OP_UTILS_H_

#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
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

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XLA_OP_UTILS_H_
