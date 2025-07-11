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

#ifndef XLA_DEBUG_OPTIONS_PARSERS_H_
#define XLA_DEBUG_OPTIONS_PARSERS_H_

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/xla.pb.h"

namespace xla {

template <typename T>
void parse_xla_backend_extra_options(T* extra_options_map,
                                     std::string comma_separated_values) {
  std::vector<std::string> extra_options_parts =
      absl::StrSplit(comma_separated_values, ',');

  // The flag contains a comma-separated list of options; some options
  // have arguments following "=", some don't.
  for (const auto& part : extra_options_parts) {
    size_t eq_pos = part.find_first_of('=');
    if (eq_pos == std::string::npos) {
      (*extra_options_map)[part] = "";
    } else {
      std::string value = "";
      if (eq_pos + 1 < part.size()) {
        value = part.substr(eq_pos + 1);
      }
      (*extra_options_map)[part.substr(0, eq_pos)] = value;
    }
  }
}

namespace details {

struct RepeatedFlagModifier {
  enum class Op {
    kAdd = 0,
    kRemove = 1,
    kClear = 2,
  };

  friend std::ostream& operator<<(std::ostream& os, Op op) {
    switch (op) {
      case Op::kAdd:
        return os << "add";
      case Op::kRemove:
        return os << "remove";
      case Op::kClear:
        return os << "clear";
    }
  }

  Op op;
  std::string value;

  bool operator==(const RepeatedFlagModifier& other) const {
    return op == other.op && value == other.value;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const RepeatedFlagModifier& modifier) {
    return os << "(" << modifier.op << ", " << modifier.value << ")";
  }
};

// Parses a comma-separated list of a repeated flag modifiers.
//
// The sequence should either be a list of values, that will replace the
// existing values, or a list of modifiers, that will be applied to the existing
// values.
//
// Uppercases the values and optionally prefixes them.
//
// For example:
// parseRepeatedEnumModifiers("a,pre_b", "pre_")
//   -> [(clear), (add "PRE_A"), (add "PRE_B")]
//
// parseRepeatedEnumModifiers("+a,-b,+c", "")
//   -> [(add, "A"), (remove, "B"), (add, "C")]
absl::StatusOr<std::vector<RepeatedFlagModifier>> parseRepeatedEnumModifiers(
    absl::string_view flag_value, absl::string_view add_prefix = "");
}  // namespace details

}  // namespace xla

#endif  // XLA_DEBUG_OPTIONS_PARSERS_H_
