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
#include <cstdint>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"

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
                                   .set_print_infeed_outfeed_config(false)
                                   .set_print_operand_shape(true)
                                   .set_print_large_constants(false));
  std::string expression = instr->ToString(*hlo_print_options);
  if (expression.size() > max_size) {
    expression.resize(max_size);
  }
  return expression;
}

inline std::string GetOpLocationStack(int32_t frame_id,
                                      const xla::HloInstruction* instr) {
  std::string stack_lines;
  xla::HloModule* hlo_module = instr->GetModule();
  while (frame_id != 0) {
    xla::HloModule::StackFrame frame = hlo_module->get_stack_frame(frame_id);
    if (frame.empty()) {
      break;
    }
    stack_lines.insert(0, absl::StrCat(frame.file_name, ":", frame.line, ":",
                                       frame.column, "\n"));
    frame_id = frame.parent_frame_id;
  }

  return stack_lines;
};

inline tsl::profiler::OpSourceInfo GetSourceInfo(
    const xla::HloInstruction* instr) {
  if (int32_t stack_frame_id = instr->metadata().stack_frame_id();
      stack_frame_id != 0) {
    return {.source_file = instr->metadata().source_file(),
            .source_line = instr->metadata().source_line(),
            .stack_frame = GetOpLocationStack(stack_frame_id, instr)};
  }
  return {.source_file = instr->metadata().source_file(),
          .source_line = instr->metadata().source_line()};
};
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_HLO_MODULE_UTILS_H_
