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

#include "xla/hlo/tools/tests/hlo_opt_test_only_passes.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/lib/prng.h"
#include "xla/hlo/builder/lib/tridiagonal.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::test_only {

namespace {

absl::StatusOr<HloComputation*> XlaComputationToHloComputation(
    XlaComputation& src_comp, HloModule* dest_module) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, src_comp.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(auto new_module,
                      HloModule::CreateFromProto(src_comp.proto(), config));
  HloCloneContext context(dest_module);
  return dest_module->DeepCloneComputation(new_module->entry_computation(),
                                           &context);
}

std::unique_ptr<HloInstruction> CreateCustomCallToBuilderMethod(
    HloInstruction* instruction, HloComputation* computation) {
  return HloInstruction::CreateCustomCall(instruction->shape(),
                                          instruction->operands(), computation,
                                          instruction->custom_call_target());
}

std::vector<XlaOp> GetParameters(XlaBuilder& builder,
                                 HloInstruction* instruction) {
  std::vector<XlaOp> parameters;
  for (int i = 0; i < instruction->operand_count(); ++i) {
    parameters.emplace_back(xla::Parameter(
        &builder, i, instruction->operand(i)->shape(), absl::StrCat("arg", i)));
    parameters[i].builder();
  }
  return parameters;
}

absl::Status VerifyOperandCounts(
    HloInstruction* instruction,
    const std::vector<int64_t>& expected_operand_counts,
    absl::string_view custom_call_target) {
  if (std::find(expected_operand_counts.begin(), expected_operand_counts.end(),
                instruction->operand_count()) ==
      expected_operand_counts.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        custom_call_target, " expected ",
        absl::StrJoin(expected_operand_counts, " or "), " operands, but got ",
        instruction->operand_count(), " operands."));
  }
  return absl::OkStatus();
}

absl::Status VerifyOperandCount(HloInstruction* instruction,
                                int64_t expected_operand_count,
                                absl::string_view custom_call_target) {
  return VerifyOperandCounts(instruction, {expected_operand_count},
                             custom_call_target);
}

absl::StatusOr<bool> BuildAndReplace(XlaBuilder& builder,
                                     HloInstruction* instruction) {
  HloComputation* computation = instruction->parent();
  HloModule* module = computation->parent();

  TF_ASSIGN_OR_RETURN(XlaComputation called_computation, builder.Build());
  TF_ASSIGN_OR_RETURN(
      HloComputation * new_computation,
      XlaComputationToHloComputation(called_computation, module));
  HloInstruction* new_instruction = computation->AddInstruction(
      CreateCustomCallToBuilderMethod(instruction, new_computation));
  TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(new_instruction));
  return true;
}

}  // namespace

absl::StatusOr<bool> XlaBuilderTestPass::ReplaceWithExpandedClientHlo(
    HloInstruction* instruction, absl::string_view custom_call_target) {
  XlaBuilder builder(
      std::string(custom_call_target.data(), custom_call_target.size()));

  std::vector<XlaOp> parameters = GetParameters(builder, instruction);

  // xla_builder.math
  if (custom_call_target == "xla_builder.math.Acos") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 1, custom_call_target));
    xla::Acos(parameters[0]);
    return BuildAndReplace(builder, instruction);
  }
  if (custom_call_target == "xla_builder.math.Acosh") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 1, custom_call_target));
    xla::Acosh(parameters[0]);
    return BuildAndReplace(builder, instruction);
  }
  if (custom_call_target == "xla_builder.math.Asin") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 1, custom_call_target));
    xla::Asin(parameters[0]);
    return BuildAndReplace(builder, instruction);
  }
  if (custom_call_target == "xla_builder.math.Asinh") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 1, custom_call_target));
    xla::Asinh(parameters[0]);
    return BuildAndReplace(builder, instruction);
  }
  if (custom_call_target == "xla_builder.math.Atan") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 1, custom_call_target));
    xla::Atan(parameters[0]);
    return BuildAndReplace(builder, instruction);
  }
  if (custom_call_target == "xla_builder.math.Cosh") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 1, custom_call_target));
    xla::Cosh(parameters[0]);
    return BuildAndReplace(builder, instruction);
  }
  if (custom_call_target == "xla_builder.math.IgammaGradA") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 2, custom_call_target));
    xla::IgammaGradA(parameters[0], parameters[1]);
    return BuildAndReplace(builder, instruction);
  }
  if (custom_call_target == "xla_builder.math.NextAfter") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 2, custom_call_target));
    xla::NextAfter(parameters[0], parameters[1]);
    return BuildAndReplace(builder, instruction);
  }
  if (custom_call_target == "xla_builder.math.Polygamma") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 2, custom_call_target));
    xla::Polygamma(parameters[0], parameters[1]);
    return BuildAndReplace(builder, instruction);
  }
  if (custom_call_target == "xla_builder.math.RandomGammaGrad") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 2, custom_call_target));
    xla::RandomGammaGrad(parameters[0], parameters[1]);
    return BuildAndReplace(builder, instruction);
  }
  if (custom_call_target == "xla_builder.math.RegularizedIncompleteBeta") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 3, custom_call_target));
    xla::RegularizedIncompleteBeta(parameters[0], parameters[1], parameters[2]);
    return BuildAndReplace(builder, instruction);
  }

  // xla_builder.matrix
  if (custom_call_target == "xla_builder.matrix.GetMatrixDiagonalViaGather") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 1, custom_call_target));
    xla::GetMatrixDiagonalViaGather(parameters[0]);
    return BuildAndReplace(builder, instruction);
  }
  if (custom_call_target == "xla_builder.matrix.Einsum") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 2, custom_call_target));
    absl::string_view einsum_config = instruction->raw_backend_config_string();
    xla::Einsum(parameters[0], parameters[1], einsum_config);
    return BuildAndReplace(builder, instruction);
  }

  // xla_builder.prng
  if (custom_call_target == "xla_builder.prng.ScramblePhiloxKey") {
    TF_RETURN_IF_ERROR(VerifyOperandCount(instruction, 1, custom_call_target));
    xla::ScramblePhiloxKey(parameters[0]);
    return BuildAndReplace(builder, instruction);
  }

  // xla_builder.tridiagonal
  if (custom_call_target == "xla_builder.tridiagonal.TridiagonalSolver") {
    TF_RETURN_IF_ERROR(
        VerifyOperandCounts(instruction, {2, 4}, custom_call_target));
    if (parameters.size() == 2) {
      TF_ASSIGN_OR_RETURN(
          std::ignore, xla::tridiagonal::TridiagonalSolver(
                           tridiagonal::SolverAlgorithm::kThomas, parameters[0],
                           parameters[1]));
      return BuildAndReplace(builder, instruction);
    }
    TF_ASSIGN_OR_RETURN(
        std::ignore, xla::tridiagonal::TridiagonalSolver(
                         tridiagonal::SolverAlgorithm::kThomas, parameters[0],
                         parameters[1], parameters[2], parameters[3]));
    return BuildAndReplace(builder, instruction);
  }

  return absl::InvalidArgumentError(absl::StrCat(
      "Unsupported xla_builder custom call target: ", custom_call_target));
}

absl::StatusOr<bool> XlaBuilderTestPass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      // Find custom calls that start with "xla_builder." and expand the HLO
      if (instruction->opcode() == HloOpcode::kCustomCall &&
          instruction->custom_call_target().rfind("xla_builder.", 0) == 0) {
        TF_ASSIGN_OR_RETURN(
            bool call_changed,
            ReplaceWithExpandedClientHlo(instruction,
                                         instruction->custom_call_target()));
        changed |= call_changed;
      }
    }
  }
  return changed;
}

}  // namespace xla::test_only
