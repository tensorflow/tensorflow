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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_HLO_MODULE_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_HLO_MODULE_UTILS_H_

#include <cstddef>
#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace tensorflow {
namespace profiler {

// Sometimes HLO produce a huge string (>100MB). Limit the name size to 1MB.
static constexpr size_t kMaxHlolNameSize = 1000000;

inline const xla::HloInstruction* FindInstruction(const xla::HloModule& module,
                                                  std::string node_name) {
  if (absl::StartsWith(node_name, "%")) {
    node_name.erase(node_name.begin());
  }
  for (const xla::HloComputation* computation : module.computations()) {
    auto instrs = computation->instructions();
    auto it = absl::c_find_if(instrs, [&](const xla::HloInstruction* instr) {
      // Try with and without "%" at the beginning of the node name.
      return absl::EqualsIgnoreCase(instr->name(), node_name) ||
             absl::EqualsIgnoreCase(instr->name(),
                                    absl::StrCat("%", node_name));
    });
    if (it != instrs.end()) {
      return *it;
    }
  }
  return nullptr;
}

inline const xla::HloComputation* FindComputation(
    const xla::HloModule& module, const std::string& comp_name) {
  for (const xla::HloComputation* computation : module.computations()) {
    if (absl::EqualsIgnoreCase(computation->name(), comp_name)) {
      return computation;
    }
  }
  return nullptr;
}

inline std::string UncachedExpression(const xla::HloInstruction* instr,
                                      bool skip_expression, size_t max_size) {
  if (skip_expression) {
    return "";
  }
  static const auto* hlo_print_options =
      new xla::HloPrintOptions(xla::HloPrintOptions()
                                   .set_print_metadata(false)
                                   .set_print_backend_config(false)
                                   .set_print_infeed_outfeed_config(false));
  std::string expression = instr->ToString(*hlo_print_options);
  if (expression.size() > max_size) {
    expression.resize(max_size);
  }
  return expression;
}
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_HLO_MODULE_UTILS_H_
