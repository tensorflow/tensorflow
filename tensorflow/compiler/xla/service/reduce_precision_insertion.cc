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

#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

StatusOr<bool> ReducePrecisionInsertion::Run(HloModule* module) {
  bool changed = false;
  VLOG(1) << "Running ReducePrecisionInsertion pass on " << module->name();

  for (auto& computation : module->computations()) {
    std::vector<HloInstruction*> instructions_to_suffix;

    for (auto& instruction : computation->instructions()) {
      VLOG(3) << "Visited instruction: " << instruction->ToString();

      // For now, ReducePrecision is only implemented for F32 arrays, so this
      // ignore instructions that produce other data.  In particular, this
      // currently ignores instructions producing tuples, even if those tuples
      // contain F32 arrays inside them.  The assumption is that in most cases
      // equivalent behavior can be obtained by adding ReducePrecision
      // instructions after the instructions that pull the F32 arrays out of
      // the tuples.
      if (instruction->shape().element_type() == PrimitiveType::F32 &&
          !ShapeUtil::IsScalar(instruction->shape()) &&
          should_reduce_output_precision_(instruction->opcode())) {
        instructions_to_suffix.push_back(instruction.get());
      }
    }

    for (auto& instruction : instructions_to_suffix) {
      HloInstruction* reduced =
          computation->AddInstruction(HloInstruction::CreateReducePrecision(
              instruction->shape(), instruction, exponent_bits_,
              mantissa_bits_));
      TF_RETURN_IF_ERROR(
          computation->ReplaceUsesOfInstruction(instruction, reduced));
      VLOG(2) << "Inserted new op after instruction: "
              << instruction->ToString();
      changed = true;
    }
  }
  return changed;
}

ReducePrecisionInsertion::OpcodeFilterFunction
ReducePrecisionInsertion::make_filter_function(
    const HloReducePrecisionOptions& reduce_precision_options) {
  // Implement the filter function with a lookup table.
  std::vector<bool> filter(HloOpcodeCount(), false);
  for (const auto& opcode : reduce_precision_options.opcodes_to_suffix()) {
    filter[opcode] = true;
  }
  return [filter](const HloOpcode opcode) {
    return filter[static_cast<unsigned int>(opcode)];
  };
}

HloReducePrecisionOptions ReducePrecisionInsertion::make_options_proto(
    const HloReducePrecisionOptions::PassTiming pass_timing,
    const int exponent_bits, const int mantissa_bits,
    const OpcodeFilterFunction& should_reduce_output_precision) {
  HloReducePrecisionOptions options;
  options.set_pass_timing(pass_timing);
  options.set_exponent_bits(exponent_bits);
  options.set_mantissa_bits(mantissa_bits);
  for (uint32_t opcode = 0; opcode < HloOpcodeCount(); opcode++) {
    if (should_reduce_output_precision(static_cast<HloOpcode>(opcode))) {
      options.add_opcodes_to_suffix(opcode);
    }
  }
  return options;
}

}  // namespace xla
