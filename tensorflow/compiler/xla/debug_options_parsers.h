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

#include <vector>
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {

template <typename T>
void parse_xla_backend_extra_options(T* extra_options_map,
                                     string comma_separated_values) {
  std::vector<string> extra_options_parts =
      absl::StrSplit(comma_separated_values, ',');

  // The flag contains a comma-separated list of options; some options
  // have arguments following "=", some don't.
  for (const auto& part : extra_options_parts) {
    size_t eq_pos = part.find_first_of('=');
    if (eq_pos == string::npos) {
      (*extra_options_map)[part] = "";
    } else {
      string value = "";
      if (eq_pos + 1 < part.size()) {
        value = part.substr(eq_pos + 1);
      }
      (*extra_options_map)[part.substr(0, eq_pos)] = value;
    }
  }
}

// The --xla_reduce_precision option has the format "LOCATION=E,M:OPS;NAME",
// where LOCATION is an HloReducePrecisionOptions::location, E and M are
// integers for the exponent and matissa bit counts respectively, and OPS and
// NAMES are comma-separated of the operation types and names to which to
// attach the reduce-precision operations.  The OPS values are matches to the
// strings produced by HloOpcodeString, while the NAME values are arbitrary
// strings subject to the requirements that they not contain any of "=,:;".
// The NAME string (with its preceding semicolon) is optional.
inline bool parse_xla_reduce_precision_option(
    HloReducePrecisionOptions* options, string option_string) {
  // Split off "LOCATION" from remainder of string.
  std::vector<string> eq_split = absl::StrSplit(option_string, '=');
  if (eq_split.size() != 2) {
    return false;
  }
  string& location = eq_split[0];
  if (location == "OP_INPUTS") {
    options->set_location(HloReducePrecisionOptions::OP_INPUTS);
  } else if (location == "OP_OUTPUTS") {
    options->set_location(HloReducePrecisionOptions::OP_OUTPUTS);
  } else if (location == "UNFUSED_OP_OUTPUTS") {
    options->set_location(HloReducePrecisionOptions::UNFUSED_OP_OUTPUTS);
  } else if (location == "FUSION_INPUTS_BY_CONTENT") {
    options->set_location(HloReducePrecisionOptions::FUSION_INPUTS_BY_CONTENT);
  } else if (location == "FUSION_OUTPUTS_BY_CONTENT") {
    options->set_location(HloReducePrecisionOptions::FUSION_OUTPUTS_BY_CONTENT);
  } else {
    return false;
  }

  // Split off "E,M" from remainder of string.
  std::vector<string> colon_split = absl::StrSplit(eq_split[1], ':');
  if (colon_split.size() != 2) {
    return false;
  }

  // Split E and M, and parse.
  std::vector<int32> bitsizes;
  for (const auto& s : absl::StrSplit(colon_split[0], ',')) {
    bitsizes.emplace_back();
    if (!absl::SimpleAtoi(s, &bitsizes.back())) {
      return false;
    }
  }
  options->set_exponent_bits(bitsizes[0]);
  options->set_mantissa_bits(bitsizes[1]);

  // Split off OPS comma-separated list from remainder of string, if the
  // remainder exists.
  std::vector<string> semicolon_split = absl::StrSplit(colon_split[1], ';');
  if (semicolon_split.size() > 2) {
    return false;
  }
  // The opcode values are either 'all' (meaning all opcodes), or matches to
  // the strings returned by HloOpcodeString.  An empty string is also
  // interpreted as 'all', for convenience.  Note that 'all' may not be part
  // of a comma-separated list; it must stand alone.
  string& opcode_string = semicolon_split[0];
  if (opcode_string == "" || opcode_string == "all") {
    for (int i = 0; i < HloOpcodeCount(); i++) {
      options->add_opcodes_to_suffix(i);
    }
  } else {
    std::vector<string> opcodes = absl::StrSplit(opcode_string, ',');
    for (const string& opcode : opcodes) {
      bool found = false;
      for (int i = 0; i < HloOpcodeCount(); i++) {
        if (opcode == HloOpcodeString(static_cast<HloOpcode>(i))) {
          options->add_opcodes_to_suffix(i);
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }

  // Process the NAMES string, if it exists.
  if (semicolon_split.size() == 2) {
    std::vector<string> opnames = absl::StrSplit(semicolon_split[1], ',');
    for (const string& opname : opnames) {
      if (opname.length() > 0) {
        options->add_opname_substrings_to_suffix(opname);
      }
    }
  }

  return true;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_DEBUG_OPTIONS_PARSERS_H_
