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

#include "xla/backends/cpu/codegen/symbol_name_util.h"

#include <cstddef>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/util.h"

namespace xla::cpu {

bool isValidCVariableName(absl::string_view name) {
  if (name.empty()) {
    return false;
  }

  if (!absl::ascii_isalpha(name[0]) && name[0] != '_') {
    return false;
  }

  for (size_t i = 1; i < name.size(); ++i) {
    if (!absl::ascii_isalnum(name[i]) && name[i] != '_') {
      return false;
    }
  }

  return true;
}

absl::StatusOr<std::string> ConvertToCName(absl::string_view name) {
  auto maybe_c_name =
      absl::StrReplaceAll(name, {{".", "_"}, {"-", "_"}, {":", "_"}});
  if (isValidCVariableName(maybe_c_name)) {
    return maybe_c_name;
  }
  return Internal("Cannot convert %s to C name, attempt result was %s.", name,
                  maybe_c_name);
}

}  // namespace xla::cpu
