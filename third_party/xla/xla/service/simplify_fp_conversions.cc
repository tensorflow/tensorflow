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

#include "xla/service/simplify_fp_conversions.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Simplifies floating-point conversions `A -> B -> C -> D` as `A -> D`.
StatusOr<bool> RunOnComputation(HloComputation& computation,
                                SimplifyFPConversions::Scope scope) {
  // Since the goal of this pass is to simplify type conversions by removing
  // some Convert ops, we don't want to run this pass for tests that are meant
  // to test for functionality of the Convert op itself.
  const absl::string_view comp_name = computation.name();
  const std::vector<absl::string_view> test_names{
      "ConvertF16F8e5m2Roundtrip",
      "ConvertF16F8e4m3fnRoundtrip",
      "ConvertF16F8e4m3b11fnuzRoundtrip",
      "ConvertF16F8e5m2fnuzRoundtrip",
      "ConvertF32F8e5m2fnuzRoundtrip",
      "ConvertF8e5m2fnuzRoundtripExhaustive",
      "ConvertF16F8e4m3fnuzRoundtrip",
      "ConvertF32F8e4m3fnuzRoundtrip",
      "ConvertF8e4m3fnuzRoundtripExhaustive"};
  for (const auto& test_name : test_names) {
    if (absl::StrContains(comp_name, test_name)) {
      return false;
    }
  }
  const int minimum_logical_creation_pass_id =
      (scope == SimplifyFPConversions::Scope::kSimplifyAllConversions) ? -1 : 0;
  bool changed = false;
  for (HloInstruction* instruction : computation.MakeInstructionPostOrder()) {
    HloInstruction* input = instruction;
    size_t convert_chain_length = 0;

    while ((input->opcode() == HloOpcode::kConvert) &&
           (input->metadata().logical_creation_pass_id() >=
            minimum_logical_creation_pass_id) &&
           primitive_util::IsFloatingPointType(input->shape().element_type())) {
      input = input->mutable_operand(0);
      ++convert_chain_length;
    }

    if (convert_chain_length < 2) {
      continue;
    }

    if (instruction->shape().element_type() == input->shape().element_type()) {
      TF_RETURN_IF_ERROR(
          instruction->parent()->ReplaceInstruction(instruction, input));
    } else {
      TF_RETURN_IF_ERROR(instruction->parent()->ReplaceWithNewInstruction(
          instruction,
          HloInstruction::CreateConvert(instruction->shape(), input)));
    }
    changed = true;
  }
  return changed;
}

std::string ToString(SimplifyFPConversions::Scope scope) {
  using Scope = SimplifyFPConversions::Scope;
  switch (scope) {
    case Scope::kSimplifyAllConversions:
      return "SimplifyAllConversions";
    case Scope::kOnlySimplifyCompilerGeneratedConversions:
      return "OnlySimplifyCompilerGeneratedConversions";
  }
}

}  // namespace

StatusOr<bool> SimplifyFPConversions::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2,
      absl::StrFormat("SimplifyFPConversions::Run() with scope=%s, before:\n%s",
                      ToString(scope_), module->ToString()));
  bool changed = false;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool comp_changed,
                        RunOnComputation(*computation, scope_));
    changed |= comp_changed;
  }
  XLA_VLOG_LINES(
      2,
      absl::StrFormat("SimplifyFPConversions::Run() with scope=%s, after:\n%s",
                      ToString(scope_), module->ToString()));
  return changed;
}

}  // namespace xla
