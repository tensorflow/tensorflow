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

#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace tensorflow {
namespace profiler {

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
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_HLO_MODULE_UTILS_H_
