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

std::vector<HloInstruction*> ReducePrecisionInsertion::instructions_to_modify(
    const HloComputation* computation) {
  std::vector<HloInstruction*> instruction_list;

  switch (location_) {
    case HloReducePrecisionOptions::OP_INPUTS:
    case HloReducePrecisionOptions::OP_OUTPUTS:
    case HloReducePrecisionOptions::UNFUSED_OP_OUTPUTS:
      for (auto* instruction : computation->instructions()) {
        VLOG(4) << "Visited instruction: " << instruction->ToString();
        if (instruction_filter_function_(instruction)) {
          instruction_list.push_back(instruction);
        }
      }
      break;

    case HloReducePrecisionOptions::FUSION_INPUTS_BY_CONTENT:
    case HloReducePrecisionOptions::FUSION_OUTPUTS_BY_CONTENT:
      for (auto* instruction : computation->instructions()) {
        VLOG(4) << "Visited instruction: " << instruction->ToString();
        if (instruction->opcode() != HloOpcode::kFusion) {
          continue;
        }
        for (auto* fused_instruction :
             instruction->fused_instructions_computation()->instructions()) {
          VLOG(4) << "Checking sub-instruction: "
                  << fused_instruction->ToString();
          if (instruction_filter_function_(fused_instruction)) {
            instruction_list.push_back(instruction);
            break;
          }
        }
      }
      break;

    default:
      break;
  }
  VLOG(1) << "Found " << instruction_list.size()
          << " candidate instruction(s) for reduce-precision insertion";

  return instruction_list;
}

StatusOr<bool> ReducePrecisionInsertion::insert_after(
    HloInstruction* instruction) {
  // Check that this isn't already an equivalent operation.
  if (is_redundant(instruction)) {
    VLOG(2) << "Skipped: instruction is already an equivalent"
               " reduce-precision instruction:"
            << instruction->ToString();
    return false;
  }

  // Check that we haven't already inserted an equivalant reduce-precision
  // operation after this instruction.  (The zero-user case occurs when this is
  // the root instruction.)
  if (instruction->user_count() > 0) {
    bool redundant_followers = true;
    for (HloInstruction* user : instruction->users()) {
      if (!is_redundant(user)) {
        redundant_followers = false;
        break;
      }
    }
    if (redundant_followers) {
      VLOG(2) << "Skipped: instruction already followed by equivalent"
                 " reduce-precision instructions";
      return false;
    }
  }

  HloInstruction* reduced = instruction->parent()->AddInstruction(
      HloInstruction::CreateReducePrecision(instruction->shape(), instruction,
                                            exponent_bits_, mantissa_bits_));
  TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(reduced));
  return true;
}

StatusOr<bool> ReducePrecisionInsertion::insert_on_inputs(
    const std::vector<HloInstruction*>& instructions) {
  bool computation_changed = false;
  for (auto instruction : instructions) {
    VLOG(2) << "Adding reduce-precision operation to inputs of instruction: "
            << instruction->ToString();
    for (int64 i = 0; i < instruction->operand_count(); i++) {
      HloInstruction* operand = instruction->mutable_operand(i);
      VLOG(2) << "Adding to operand " << i << ": " << operand;

      if (!is_valid_shape(operand->shape())) {
        VLOG(2) << "Skipped: value is not an F32 vector";
        continue;
      }

      if (is_redundant(operand)) {
        VLOG(2) << "Skipped: operand is already an equivalent reduce-precision"
                   " instruction";
        continue;
      }

      if (instruction->opcode() == HloOpcode::kFusion &&
          (instruction->fusion_kind() == HloInstruction::FusionKind::kLoop ||
           instruction->fusion_kind() == HloInstruction::FusionKind::kInput)) {
        // Insert the reduce-precision operation inside the fusion computation,
        // after the corresponding parameter instruction.
        TF_ASSIGN_OR_RETURN(
            bool instruction_changed,
            insert_after(instruction->fused_instructions_computation()
                             ->parameter_instruction(i)));
        computation_changed |= instruction_changed;
      } else {
        // Look for an existing reduce-precision operation on the operand.  (We
        // need to be careful not to create a loop, though!)
        HloInstruction* reduced = nullptr;
        for (auto& user : operand->users()) {
          if (user != instruction &&
              user->opcode() == HloOpcode::kReducePrecision &&
              user->exponent_bits() == exponent_bits_ &&
              user->mantissa_bits() == mantissa_bits_) {
            reduced = user;
            break;
          }
        }
        // If there wasn't an existing reduce-precision operation, create one.
        if (!reduced) {
          reduced = instruction->parent()->AddInstruction(
              HloInstruction::CreateReducePrecision(
                  operand->shape(), operand, exponent_bits_, mantissa_bits_));
        }
        // Insert the reduce-precision operation before the operand.
        TF_RETURN_IF_ERROR(instruction->ReplaceOperandWith(i, reduced));
        computation_changed = true;
      }
    }
  }

  return computation_changed;
}

StatusOr<bool> ReducePrecisionInsertion::insert_on_outputs(
    const std::vector<HloInstruction*>& instructions) {
  bool computation_changed = false;
  for (const auto& instruction : instructions) {
    VLOG(2) << "Adding reduce-precision operation to output of instruction: "
            << instruction->ToString();

    if (!is_valid_shape(instruction->shape())) {
      VLOG(2) << "Skipped: value is not an F32 nonscalar array";
      continue;
    }

    if (instruction->opcode() == HloOpcode::kFusion &&
        (instruction->fusion_kind() == HloInstruction::FusionKind::kLoop ||
         instruction->fusion_kind() == HloInstruction::FusionKind::kOutput)) {
      // Insert the reduce-precision operation as the last operation inside
      // the fusion computation.
      HloInstruction* fusion_root = instruction->fused_expression_root();
      VLOG(2) << "Inserting new operation after existing fusion root: "
              << fusion_root->ToString();

      TF_ASSIGN_OR_RETURN(bool instruction_changed, insert_after(fusion_root));
      computation_changed |= instruction_changed;
    } else {
      // Insert the reduce-precision operation after the instruction.
      TF_ASSIGN_OR_RETURN(bool instruction_changed, insert_after(instruction));
      computation_changed |= instruction_changed;
    }
  }

  return computation_changed;
}

StatusOr<bool> ReducePrecisionInsertion::Run(HloModule* module) {
  bool changed = false;
  VLOG(1) << "Running ReducePrecisionInsertion pass on " << module->name();

  for (auto* computation : module->MakeNonfusionComputations()) {
    StatusOr<bool> computation_changed;
    switch (location_) {
      case HloReducePrecisionOptions::OP_INPUTS:
      case HloReducePrecisionOptions::FUSION_INPUTS_BY_CONTENT:
        computation_changed = ReducePrecisionInsertion::insert_on_inputs(
            instructions_to_modify(computation));
        break;

      case HloReducePrecisionOptions::FUSION_OUTPUTS_BY_CONTENT:
      case HloReducePrecisionOptions::OP_OUTPUTS:
      case HloReducePrecisionOptions::UNFUSED_OP_OUTPUTS:
        computation_changed = ReducePrecisionInsertion::insert_on_outputs(
            instructions_to_modify(computation));
        break;
      default:
        break;
    }
    TF_RETURN_IF_ERROR(computation_changed.status());

    if (computation_changed.ValueOrDie()) {
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
    const HloReducePrecisionOptions::Location location, const int exponent_bits,
    const int mantissa_bits,
    const std::function<bool(HloOpcode)>& opcode_filter_function,
    const std::vector<string>& opname_substring_list) {
  HloReducePrecisionOptions options;
  options.set_location(location);
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

bool ReducePrecisionInsertion::AddPasses(HloPassPipeline* pipeline,
                                         const DebugOptions& debug_options,
                                         const PassTiming pass_timing) {
  bool passes_added = false;
  for (const auto& pass_options :
       debug_options.hlo_reduce_precision_options()) {
    bool add_pass;
    switch (pass_options.location()) {
      case HloReducePrecisionOptions::OP_INPUTS:
      case HloReducePrecisionOptions::OP_OUTPUTS:
        add_pass = pass_timing == PassTiming::BEFORE_OPTIMIZATION;
        break;
      case HloReducePrecisionOptions::UNFUSED_OP_OUTPUTS:
      case HloReducePrecisionOptions::FUSION_INPUTS_BY_CONTENT:
      case HloReducePrecisionOptions::FUSION_OUTPUTS_BY_CONTENT:
        add_pass = pass_timing == PassTiming::AFTER_FUSION;
        break;
      default:
        add_pass = false;
    }
    if (add_pass) {
      pipeline->AddPass<ReducePrecisionInsertion>(pass_options);
      passes_added = true;
    }
  }
  return passes_added;
}

}  // namespace xla
