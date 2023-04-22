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

#ifndef TENSORFLOW_COMPILER_XLA_DEBUG_OPTIONS_PARSERS_H_
#define TENSORFLOW_COMPILER_XLA_DEBUG_OPTIONS_PARSERS_H_

#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/xla.pb.h"

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

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_DEBUG_OPTIONS_PARSERS_H_
