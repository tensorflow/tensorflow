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

// For now, ReducePrecision is only implemented for F32 arrays, so this
// ignores instructions that produce other data.  In particular, this
// currently ignores instructions producing tuples, even if those tuples
// contain F32 arrays inside them.  The assumption is that in most cases
// equivalent behavior can be obtained by adding ReducePrecision
// instructions after the instructions that pull the F32 arrays out of
// the tuples.
//
// TODO(b/64093391): Remove the IsScalar check once this won't cause
// failures on the GPU backend if the ReducePrecision instruction ends up
// inserted between a scalar constant and the init_value argument of a
// Reduce operation.
std::vector<HloInstruction*> ReducePrecisionInsertion::instructions_to_suffix(
    const HloComputation* computation) {
  std::vector<HloInstruction*> instructions_to_suffix;

  switch (pass_timing_) {
    case HloReducePrecisionOptions::BEFORE_OP_FUSION:
    case HloReducePrecisionOptions::AFTER_OP_FUSION:
      for (auto& instruction : computation->instructions()) {
        VLOG(4) << "Visited instruction: " << instruction->ToString();

        if (instruction->shape().element_type() == PrimitiveType::F32 &&
            !ShapeUtil::IsScalar(instruction->shape()) &&
            instruction_filter_function_(instruction.get())) {
          instructions_to_suffix.push_back(instruction.get());
        }
      }
      break;

    case HloReducePrecisionOptions::FUSION_BY_CONTENT:
      for (auto& instruction : computation->instructions()) {
        VLOG(4) << "Visited instruction: " << instruction->ToString();

        if (instruction->opcode() != HloOpcode::kFusion ||
            instruction->shape().element_type() != PrimitiveType::F32 ||
            ShapeUtil::IsScalar(instruction->shape())) {
          continue;
        }

        for (auto& fused_instruction :
             instruction->fused_instructions_computation()->instructions()) {
          VLOG(4) << "Checking sub-instruction: "
                  << fused_instruction->ToString();
          if (instruction_filter_function_(fused_instruction.get())) {
            instructions_to_suffix.push_back(instruction.get());
            break;
          }
        }
      }
      break;

    default:
      break;
  }
  VLOG(1) << "Adding " << instructions_to_suffix.size()
          << " reduce-precision operations.";

  return instructions_to_suffix;
}

StatusOr<bool> ReducePrecisionInsertion::Run(HloModule* module) {
  bool changed = false;
  VLOG(1) << "Running ReducePrecisionInsertion pass on " << module->name();

  for (auto& computation : module->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    bool computation_changed = false;
    for (auto& instruction : instructions_to_suffix(computation.get())) {
      VLOG(2) << "Adding reduce-precision operation to output of instruction: "
              << instruction->ToString();

      // Check that we haven't already inserted an equivalant reduce-precision
      // operation after this instruction.
      if (instruction->user_count() == 1) {
        HloInstruction* user = instruction->users()[0];

        if (user->opcode() == HloOpcode::kReducePrecision &&
            user->exponent_bits() == exponent_bits_ &&
            user->mantissa_bits() == mantissa_bits_) {
          VLOG(2) << "Skipped; instruction already followed by equivalent"
                     " reduce-precision instruction:"
                  << user->ToString();
          continue;
        }
      }

      if (instruction->opcode() == HloOpcode::kFusion) {
        // Insert the reduce-precision operation as the last operation inside
        // the fusion computation.
        instruction = instruction->fused_expression_root();

        VLOG(2) << "Inserting new operation after existing fusion root: "
                << instruction->ToString();

        if (instruction->opcode() == HloOpcode::kReducePrecision &&
            instruction->exponent_bits() == exponent_bits_ &&
            instruction->mantissa_bits() == mantissa_bits_) {
          VLOG(2) << "Skipped; fused computation already ends in equivalent"
                     " reduce-precision instruction:"
                  << instruction->ToString();
          continue;
        }
      }

      HloInstruction* reduced = instruction->parent()->AddInstruction(
          HloInstruction::CreateReducePrecision(instruction->shape(),
                                                instruction, exponent_bits_,
                                                mantissa_bits_));

      TF_RETURN_IF_ERROR(instruction->parent()->ReplaceUsesOfInstruction(
          instruction, reduced));
      computation_changed = true;
    }

    if (computation_changed) {
      changed = true;
      VLOG(3) << "Computation after reduce-precision insertion:";
      XLA_VLOG_LINES(3, computation->ToString());
    } else {
      VLOG(3) << "Computation " << computation->name() << " unchanged";
    }
  }

  return changed;
}

ReducePrecisionInsertion::InstructionFilterFunction
ReducePrecisionInsertion::make_filter_function(
    const HloReducePrecisionOptions& reduce_precision_options) {
  // Implement the filter function with a lookup table.
  std::vector<bool> opcode_filter(HloOpcodeCount(), false);
  for (const auto& opcode : reduce_precision_options.opcodes_to_suffix()) {
    opcode_filter[opcode] = true;
  }
  if (reduce_precision_options.opname_substrings_to_suffix_size() == 0) {
    return [opcode_filter](const HloInstruction* instruction) {
      return opcode_filter[static_cast<unsigned int>(instruction->opcode())];
    };
  } else {
    std::vector<string> opname_substrings;
    for (const auto& substring :
         reduce_precision_options.opname_substrings_to_suffix()) {
      opname_substrings.push_back(substring);
    }
    return [opcode_filter,
            opname_substrings](const HloInstruction* instruction) {
      if (!opcode_filter[static_cast<unsigned int>(instruction->opcode())]) {
        return false;
      }
      const auto& opname = instruction->metadata().op_name();
      for (const auto& substring : opname_substrings) {
        if (opname.find(substring) != string::npos) {
          return true;
        }
      }
      return false;
    };
  }
}

HloReducePrecisionOptions ReducePrecisionInsertion::make_options_proto(
    const HloReducePrecisionOptions::PassTiming pass_timing,
    const int exponent_bits, const int mantissa_bits,
    const std::function<bool(HloOpcode)>& opcode_filter_function,
    const std::vector<string>& opname_substring_list) {
  HloReducePrecisionOptions options;
  options.set_pass_timing(pass_timing);
  options.set_exponent_bits(exponent_bits);
  options.set_mantissa_bits(mantissa_bits);
  for (uint32_t opcode = 0; opcode < HloOpcodeCount(); opcode++) {
    if (opcode_filter_function(static_cast<HloOpcode>(opcode))) {
      options.add_opcodes_to_suffix(opcode);
    }
  }
  for (auto& string : opname_substring_list) {
    options.add_opname_substrings_to_suffix(string);
  }
  return options;
}

bool ReducePrecisionInsertion::AddPasses(
    HloPassPipeline* pipeline, const DebugOptions& debug_options,
    const HloReducePrecisionOptions::PassTiming pass_timing) {
  bool passes_added = false;
  for (const auto& pass_options :
       debug_options.hlo_reduce_precision_options()) {
    if (pass_options.pass_timing() == pass_timing) {
      pipeline->AddPass<ReducePrecisionInsertion>(pass_options);
      passes_added = true;
    }
  }
  return passes_added;
}

}  // namespace xla
