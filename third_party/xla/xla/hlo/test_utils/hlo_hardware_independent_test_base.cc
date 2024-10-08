/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/test_utils/hlo_hardware_independent_test_base.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/test_utils/filecheck.h"
#include "xla/hlo/test_utils/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {

HloHardwareIndependentTestBase::HloHardwareIndependentTestBase(
    bool verifier_layout_sensitive, bool allow_mixed_precision_in_hlo_verifier,
    HloPredicate instruction_can_change_layout_func)
    : verifier_layout_sensitive_(verifier_layout_sensitive),
      allow_mixed_precision_in_hlo_verifier_(
          allow_mixed_precision_in_hlo_verifier),
      instruction_can_change_layout_func_(instruction_can_change_layout_func) {
  hlo_verifier_ = std::make_unique<HloVerifier>(
      /*layout_sensitive=*/verifier_layout_sensitive,
      /*allow_mixed_precision=*/allow_mixed_precision_in_hlo_verifier,
      instruction_can_change_layout_func);
}

std::unique_ptr<HloModule>
HloHardwareIndependentTestBase::CreateNewUnverifiedModule(
    const std::string& name) {
  return std::make_unique<HloModule>(name, GetModuleConfigForTest());
}

std::unique_ptr<VerifiedHloModule>
HloHardwareIndependentTestBase::CreateNewVerifiedModule(const std::string& name,
                                                        int64_t replica_count) {
  return std::make_unique<VerifiedHloModule>(
      name, GetModuleConfigForTest(replica_count), verifier_layout_sensitive_,
      allow_mixed_precision_in_hlo_verifier_, ShapeUtil::ByteSizeOfElements,
      instruction_can_change_layout_func_);
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
HloHardwareIndependentTestBase::ParseAndReturnVerifiedModule(
    absl::string_view hlo_text, int64_t replica_count, int64_t num_partitions) {
  return ParseAndReturnVerifiedModule(
      hlo_text, GetModuleConfigForTest(replica_count, num_partitions));
}

absl::Status HloHardwareIndependentTestBase::
    UpdateEntryComputationLayoutToMatchProgramLayout(HloModule* module) {
  for (auto* const computation : module->computations({})) {
    if (computation->IsEntryComputation()) {
      for (int64_t i = 0; i < computation->num_parameters(); ++i) {
        const Shape& param_shape =
            computation->parameter_instruction(i)->shape();
        TF_RETURN_IF_ERROR(computation->parent()
                               ->mutable_entry_computation_layout()
                               ->mutable_parameter_layout(i)
                               ->CopyLayoutFromShape(param_shape));
      }

      TF_RETURN_IF_ERROR(
          computation->parent()
              ->mutable_entry_computation_layout()
              ->mutable_result_layout()
              ->CopyLayoutFromShape(computation->root_instruction()->shape()));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
HloHardwareIndependentTestBase::ParseAndReturnVerifiedModule(
    absl::string_view hlo_text, const HloModuleConfig& config) {
  auto module = std::make_unique<VerifiedHloModule>(
      TestName(), config, verifier_layout_sensitive_,
      allow_mixed_precision_in_hlo_verifier_, ShapeUtil::ByteSizeOfElements,
      instruction_can_change_layout_func_);
  TF_RETURN_IF_ERROR(module->ParseHloStringAndVerifyModule(hlo_text));
  return std::move(module);
}

/* static */
absl::StatusOr<bool> HloHardwareIndependentTestBase::RunHloPass(
    HloPassInterface* hlo_pass, HloModule* module) {
  const std::string module_str_before_run =
      module->ToProto().ShortDebugString();
  const auto status_or = hlo_pass->Run(module);
  if (status_or.status().ok()) {
    const std::string module_str_after_run =
        module->ToProto().ShortDebugString();
    const bool passChangedHlo = status_or.value();
    if (passChangedHlo) {
      // Check that the proto actually changed.
      EXPECT_NE(module_str_after_run, module_str_before_run);
    } else {
      // Check that the proto remains same.
      EXPECT_EQ(module_str_after_run, module_str_before_run);
    }
  }
  return status_or;
}

/* static */
absl::StatusOr<bool> HloHardwareIndependentTestBase::RunHloPass(
    HloPassInterface&& hlo_pass, HloModuleGroup* module_group) {
  const std::string module_group_str_before_run =
      module_group->ToProto().ShortDebugString();
  const auto status_or = hlo_pass.RunOnModuleGroup(module_group);
  if (status_or.status().ok()) {
    const std::string module_group_str_after_run =
        module_group->ToProto().ShortDebugString();
    const bool passChangedHlo = status_or.value();
    if (passChangedHlo) {
      // Check that the proto actually changed.
      EXPECT_NE(module_group_str_after_run, module_group_str_before_run);
    } else {
      // Check that the proto remains same.
      EXPECT_EQ(module_group_str_after_run, module_group_str_before_run);
    }
  }
  return status_or;
}

/* static */
PrecisionConfig HloHardwareIndependentTestBase::DefaultPrecisionConfig(
    int operands) {
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      operands, PrecisionConfig::DEFAULT);
  return precision_config;
}

void HloHardwareIndependentTestBase::SetAotFastMathDebugOptions(
    DebugOptions* options) {
  options->set_xla_cpu_enable_fast_math(true);
  options->set_xla_gpu_enable_fast_min_max(true);
  options->set_xla_cpu_enable_fast_min_max(true);
  options->set_xla_cpu_fast_math_honor_nans(false);
  options->set_xla_cpu_fast_math_honor_infs(false);
  options->set_xla_cpu_fast_math_honor_functions(false);
  options->set_xla_cpu_fast_math_honor_division(false);
}

DebugOptions HloHardwareIndependentTestBase::GetDebugOptionsForTest() {
  auto debug_options = GetDebugOptionsFromFlags();
  // TODO(b/38354253): Change tests to use Parameters instead of Constants.
  debug_options.add_xla_disable_hlo_passes("constant_folding");
  debug_options.set_xla_hlo_evaluator_use_fast_path(true);
  return debug_options;
}

void HloHardwareIndependentTestBase::RunAndFilecheckHloRewrite(
    absl::string_view hlo, HloPassInterface&& hlo_pass,
    std::optional<absl::string_view> expected,
    std::function<void(HloModule*)> after_pass_checks,
    const HloModuleConfig* config) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          config ? ParseAndReturnVerifiedModule(hlo, *config)
                                 : ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&hlo_pass, module.get()));
  EXPECT_EQ(changed, expected.has_value()) << module->ToString();
  if (changed) {
    TF_ASSERT_OK_AND_ASSIGN(
        bool filecheck_matches,
        RunFileCheck(
            module->ToString(HloPrintOptions{}.set_print_operand_shape(false)),
            *expected));
    EXPECT_TRUE(filecheck_matches);
    if (after_pass_checks) {
      after_pass_checks(module.get());
    }
  }
}

void HloHardwareIndependentTestBase::RunAndFilecheckHloModuleGroupRewrite(
    absl::Span<const absl::string_view> hlo_module_strs,
    HloPassInterface&& hlo_pass,
    std::optional<absl::Span<const absl::string_view>> expected) {
  std::vector<std::unique_ptr<HloModule>> modules;
  for (absl::string_view hlo : hlo_module_strs) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                            ParseAndReturnVerifiedModule(hlo));
    modules.push_back(std::move(module));
  }
  HloModuleGroup module_group("test_input_module_group", std::move(modules));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(std::move(hlo_pass), &module_group));
  EXPECT_EQ(changed, expected.has_value()) << module_group.ToString();

  if (!changed) {
    return;
  }

  EXPECT_THAT(module_group.modules(),
              ::testing::SizeIs(expected.value().size()));
  int index = 0;
  for (auto expected_str : expected.value()) {
    TF_ASSERT_OK_AND_ASSIGN(
        bool filecheck_matches,
        RunFileCheck(module_group.module(index).ToString(
                         HloPrintOptions{}.set_print_operand_shape(false)),
                     expected_str));
    EXPECT_TRUE(filecheck_matches);
    index++;
  }
}

absl::StatusOr<std::unique_ptr<HloModule>>
HloHardwareIndependentTestBase::RunAndCheckHloRewrite(
    absl::string_view hlo_template, HloPassInterface&& hlo_pass,
    bool expect_change, FixedMapping params) {
  std::string hlo_string = absl::StrReplaceAll(hlo_template, params);
  SCOPED_TRACE("Input HLO: " + hlo_string);
  VLOG(7) << "Input HLO: " << hlo_string;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSIGN_OR_RETURN(bool changed, RunHloPass(hlo_pass, module.get()));
  VLOG(7) << "Output HLO: "
          << module->ToString(HloPrintOptions::ShortParsable());
  EXPECT_EQ(changed, expect_change);
  return module;
}

std::vector<int> HloHardwareIndependentTestBase::CompareInputs(
    const HloModule& module_0, const HloModule& module_1) {
  const auto params_0 = module_0.entry_computation()->parameter_instructions();
  const auto params_1 = module_1.entry_computation()->parameter_instructions();
  std::vector<int> mismatches;
  int64_t min = std::min(params_0.size(), params_1.size());
  int64_t max = std::max(params_0.size(), params_1.size());
  for (int64_t i = 0; i < min; ++i) {
    const HloModuleConfig& module_config_0 = module_0.config();
    const Shape& param_shape_0 =
        (module_config_0.has_entry_computation_layout() &&
         module_config_0.entry_computation_layout()
             .parameter_layout(i)
             .shape()
             .is_static())
            ? module_config_0.entry_computation_layout()
                  .parameter_layout(i)
                  .shape()
            : params_0[i]->shape();

    const HloModuleConfig& module_config_1 = module_1.config();
    const Shape& param_shape_1 =
        (module_config_1.has_entry_computation_layout() &&
         module_config_1.entry_computation_layout()
             .parameter_layout(i)
             .shape()
             .is_static())
            ? module_config_1.entry_computation_layout()
                  .parameter_layout(i)
                  .shape()
            : params_1[i]->shape();

    if (!Shape::Equal().IgnoreTilesInLayout()(param_shape_0, param_shape_1)) {
      mismatches.push_back(i);
    }
  }
  for (int64_t i = min; i < max; i++) {
    mismatches.push_back(i);
  }
  return mismatches;
}

HloComputation* HloHardwareIndependentTestBase::FindComputation(
    HloModule* module, absl::string_view name) {
  return hlo_query::FindComputation(module, name);
}

HloInstruction* HloHardwareIndependentTestBase::FindInstruction(
    HloModule* module, absl::string_view name) {
  for (const HloComputation* computation : module->computations()) {
    if (HloInstruction* instruction =
            hlo_query::FindInstruction(computation, name)) {
      return instruction;
    }
  }
  return nullptr;
}

HloInstruction* HloHardwareIndependentTestBase::FindInstruction(
    HloModule* module, HloOpcode opcode) {
  for (const HloComputation* computation : module->computations()) {
    if (HloInstruction* instruction =
            hlo_query::FindInstruction(computation, opcode)) {
      return instruction;
    }
  }
  return nullptr;
}

std::vector<HloInstruction*> HloHardwareIndependentTestBase::FindInstructions(
    HloModule* module, HloOpcode opcode) {
  std::vector<HloInstruction*> instructions;
  for (const HloComputation* c : module->computations()) {
    absl::c_copy_if(c->instructions(), std::back_inserter(instructions),
                    [&](HloInstruction* i) { return i->opcode() == opcode; });
  }
  return instructions;
}

/* static */
std::string HloHardwareIndependentTestBase::TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

}  // namespace xla
