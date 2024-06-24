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

#include "xla/tests/hlo_test_base.h"

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/layout_util.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/pjrt_client_registry.h"
#include "xla/tests/test_utils.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/types.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test.h"

namespace xla {

namespace {

using absl::string_view;
using std::optional;

constexpr char kInterpreter[] = "interpreter";

bool ProgramShapesEqual(const ProgramShape& lhs, const ProgramShape& rhs) {
  if (lhs.parameters_size() != rhs.parameters_size()) {
    return false;
  }
  for (int i = 0; i < lhs.parameters_size(); i++) {
    if (!Shape::Equal().IgnoreElementSizeInLayout()(lhs.parameters(i),
                                                    rhs.parameters(i))) {
      return false;
    }
  }
  return Shape::Equal().IgnoreElementSizeInLayout()(lhs.result(), rhs.result());
}

ProgramShape GetProgramShapeWithLayout(const HloModule& module) {
  ProgramShape program_shape;
  const auto* entry = module.entry_computation();
  for (const auto* param : entry->parameter_instructions()) {
    *program_shape.add_parameters() = param->shape();
    *program_shape.add_parameter_names() = param->name();
  }
  *program_shape.mutable_result() = entry->root_instruction()->shape();
  return program_shape;
}

}  // namespace

HloTestBase::HloTestBase(bool verifier_layout_sensitive,
                         bool allow_mixed_precision_in_hlo_verifier,
                         HloPredicate instruction_can_change_layout_func)
    : HloTestBase(GetTestPlatform(), GetReferencePlatform(),
                  verifier_layout_sensitive,
                  allow_mixed_precision_in_hlo_verifier,
                  instruction_can_change_layout_func) {}

HloTestBase::HloTestBase(se::Platform* test_platform,
                         se::Platform* reference_platform,
                         bool verifier_layout_sensitive,
                         bool allow_mixed_precision_in_hlo_verifier,
                         HloPredicate instruction_can_change_layout_func)
    : test_runner_(test_platform),
      reference_runner_(reference_platform),
      verifier_layout_sensitive_(verifier_layout_sensitive),
      allow_mixed_precision_in_hlo_verifier_(
          allow_mixed_precision_in_hlo_verifier),
      instruction_can_change_layout_func_(instruction_can_change_layout_func),
      test_platform_(test_platform) {
  hlo_verifier_ = std::make_unique<HloVerifier>(
      /*layout_sensitive=*/verifier_layout_sensitive,
      /*allow_mixed_precision=*/allow_mixed_precision_in_hlo_verifier,
      instruction_can_change_layout_func);
  runner_ = GetHloRunner().value();
}

/*static*/ se::Platform* HloTestBase::GetReferencePlatform() {
  auto result = PlatformUtil::GetPlatform(kInterpreter);
  TF_CHECK_OK(result.status()) << "could not get interpreter platform";
  return result.value();
}

/*static*/ se::Platform* HloTestBase::GetTestPlatform() {
  auto result = PlatformUtil::GetDefaultPlatform();
  TF_CHECK_OK(result.status()) << "could not get test platform";
  return result.value();
}

std::unique_ptr<HloModule> HloTestBase::CreateNewUnverifiedModule(
    const std::string& name) {
  return std::make_unique<HloModule>(name, GetModuleConfigForTest());
}

std::unique_ptr<VerifiedHloModule> HloTestBase::CreateNewVerifiedModule(
    const std::string& name, int64_t replica_count) {
  return std::make_unique<VerifiedHloModule>(
      name, GetModuleConfigForTest(replica_count), verifier_layout_sensitive_,
      allow_mixed_precision_in_hlo_verifier_,
      backend().compiler()->ShapeSizeBytesFunction(),
      instruction_can_change_layout_func_);
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
HloTestBase::ParseAndReturnVerifiedModule(absl::string_view hlo_text,
                                          int64_t replica_count,
                                          int64_t num_partitions) {
  return ParseAndReturnVerifiedModule(
      hlo_text, GetModuleConfigForTest(replica_count, num_partitions));
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
HloTestBase::ParseAndReturnVerifiedModule(absl::string_view hlo_text,
                                          const HloModuleConfig& config) {
  auto module = std::make_unique<VerifiedHloModule>(
      TestName(), config, verifier_layout_sensitive_,
      allow_mixed_precision_in_hlo_verifier_,
      backend().compiler()->ShapeSizeBytesFunction(),
      instruction_can_change_layout_func_);
  TF_RETURN_IF_ERROR(module->ParseHloStringAndVerifyModule(hlo_text));
  UpdateEntryComputationLayout(module.get());
  return std::move(module);
}

HloComputation* HloTestBase::AddEntryComputationAndUpdateEntryComputationLayout(
    HloModule* module, std::unique_ptr<HloComputation> computation) {
  auto comp = module->AddEntryComputation(std::move(computation));
  UpdateEntryComputationLayout(module);
  return comp;
}

void HloTestBase::UpdateEntryComputationLayout(HloModule* module) {
  xla::UpdateEntryComputationLayout(
      module, test_runner_.device_shape_representation_fn());
}

/* static */
absl::StatusOr<bool> HloTestBase::RunHloPass(HloPassInterface* hlo_pass,
                                             HloModule* module) {
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
absl::StatusOr<bool> HloTestBase::RunHloPass(HloPassInterface&& hlo_pass,
                                             HloModuleGroup* module_group) {
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
PrecisionConfig HloTestBase::DefaultPrecisionConfig(int operands) {
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      operands, PrecisionConfig::DEFAULT);
  return precision_config;
}

void HloTestBase::SetAotFastMathDebugOptions(DebugOptions* options) {
  options->set_xla_cpu_enable_fast_math(true);
  options->set_xla_gpu_enable_fast_min_max(true);
  options->set_xla_cpu_enable_fast_min_max(true);
  options->set_xla_cpu_fast_math_honor_nans(false);
  options->set_xla_cpu_fast_math_honor_infs(false);
  options->set_xla_cpu_fast_math_honor_functions(false);
  options->set_xla_cpu_fast_math_honor_division(false);
}

DebugOptions HloTestBase::GetDebugOptionsForTest() {
  auto debug_options = GetDebugOptionsFromFlags();
  // TODO(b/38354253): Change tests to use Parameters instead of Constants.
  debug_options.add_xla_disable_hlo_passes("constant_folding");
  debug_options.set_xla_hlo_evaluator_use_fast_path(true);
  return debug_options;
}

void HloTestBase::RunAndFilecheckHloRewrite(
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

void HloTestBase::RunAndFilecheckHloModuleGroupRewrite(
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

absl::StatusOr<Literal> HloTestBase::Execute(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments) {
  return runner_->Execute(std::move(module), arguments);
}

Literal HloTestBase::ExecuteNoHloPasses(std::unique_ptr<HloModule> module,
                                        absl::Span<Literal* const> arguments) {
  return runner_
      ->Execute(std::move(module), arguments,
                /*run_hlo_passes=*/false)
      .value();
}

absl::StatusOr<std::unique_ptr<HloRunnerInterface>>
HloTestBase::GetHloRunner() {
  if (runner_ != nullptr) {
    return std::move(runner_);
  }
  absl::StatusOr<std::unique_ptr<HloRunnerInterface>> status_or_runner =
      GetHloRunnerForTest(test_platform_);

  // Test for successful creation of PjRt based Hlo Runner.
  EXPECT_TRUE(status_or_runner.ok());

  return std::move(status_or_runner.value());
}

Literal HloTestBase::ExecuteAndTransfer(std::unique_ptr<HloModule> module,
                                        absl::Span<Literal* const> arguments) {
  return runner_->Execute(std::move(module), arguments, true, nullptr).value();
}

absl::StatusOr<std::vector<Literal>> HloTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
    int64_t num_replicas, bool use_threads, bool run_hlo_passes) {
  HloRunner::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = use_threads;
  for (auto argument : arguments) {
    options.arguments.push_back(argument);
  }

  return runner_->ExecuteReplicated(std::move(module), options);
}

absl::StatusOr<std::vector<Literal>> HloTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
    int64_t num_replicas, DeviceAssignment* device_assignment,
    bool run_hlo_passes, bool use_threads) {
  HloRunner::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = use_threads;
  for (auto argument : arguments) {
    options.arguments.push_back(argument);
  }
  return runner_->ExecuteReplicated(std::move(module), options,
                                    device_assignment);
}

absl::StatusOr<std::vector<Literal>> HloTestBase::ExecuteReplicated(
    std::function<Executable*(int64_t)> executable_provider,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    int64_t num_replicas, bool run_hlo_passes,
    DeviceAssignment* device_assignment) {
  HloRunner::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;
  return runner_->ExecuteReplicated(executable_provider,
                                    argument_count_provider, argument_provider,
                                    options, device_assignment);
}

absl::StatusOr<std::unique_ptr<HloModule>> HloTestBase::MakeReferenceModule(
    const HloModule& test_module,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  std::unique_ptr<HloModule> reference_module = test_module.Clone();
  const auto& program_shape = GetProgramShapeWithLayout(test_module);

  if (reference_preprocessor != nullptr) {
    reference_preprocessor(reference_module.get());
    if (!ProgramShapesEqual(program_shape,
                            GetProgramShapeWithLayout(*reference_module))) {
      return InvalidArgument(
          "reference preprocessor must not modify the program shape");
    }
  }
  TF_RETURN_IF_ERROR(hlo_verifier_->Run(reference_module.get()).status());
  return std::move(reference_module);
}

absl::StatusOr<::testing::AssertionResult> HloTestBase::RunAndCompareInternal(
    std::unique_ptr<HloModule> module,
    const absl::Span<Literal* const> arguments,
    const optional<ErrorSpec>& error, bool run_hlo_passes,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  TF_RETURN_IF_ERROR(hlo_verifier_->Run(module.get()).status());
  TF_ASSIGN_OR_RETURN(auto reference_module,
                      MakeReferenceModule(*module, reference_preprocessor));

  // Execute on two backends.
  TF_ASSIGN_OR_RETURN(auto test, runner_->Execute(std::move(module), arguments,
                                                  run_hlo_passes));
  TF_ASSIGN_OR_RETURN(auto reference,
                      reference_runner_.Execute(std::move(reference_module),
                                                arguments, run_hlo_passes));
  if (reference.IsAll(0)) {
    LOG(WARNING) << "Reference value is only zeros.";
  }

  return LiteralTestUtil::NearOrEqual(/*expected=*/reference, /*actual=*/test,
                                      error);
}

::testing::AssertionResult HloTestBase::RunAndCompare(
    std::unique_ptr<HloModule> module,
    const absl::Span<Literal* const> arguments,
    const optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto result =
      RunAndCompareInternal(std::move(module), arguments, error,
                            /*run_hlo_passes=*/true, reference_preprocessor);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return result.value();
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPasses(
    std::unique_ptr<HloModule> module,
    const absl::Span<Literal* const> arguments,
    const optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto result =
      RunAndCompareInternal(std::move(module), arguments, error,
                            /*run_hlo_passes=*/false, reference_preprocessor);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return result.value();
}

::testing::AssertionResult HloTestBase::RunAndCompare(
    std::unique_ptr<HloModule> module, const optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor,
    std::optional<int64_t> args_max_bits_of_precision) {
  auto fake_arguments =
      MakeFakeArguments(module.get(), /*pseudo_random=*/true,
                        /*use_large_range=*/false,
                        /*treat_gte_as_data_formatting=*/false,
                        args_max_bits_of_precision)
          .value();

  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  return RunAndCompare(std::move(module), fake_argument_ptrs, error,
                       reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPasses(
    std::unique_ptr<HloModule> module, const optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  const auto fake_arguments = MakeFakeArguments(module.get()).value();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  return RunAndCompareNoHloPasses(std::move(module), fake_argument_ptrs, error,
                                  reference_preprocessor);
}

::testing::AssertionResult HloTestBase::Run(std::unique_ptr<HloModule> module,
                                            bool run_hlo_passes) {
  const auto fake_arguments = MakeFakeArguments(module.get()).value();
  const auto change = hlo_verifier_->Run(module.get());
  if (!change.ok()) {
    return ::testing::AssertionFailure() << change.status();
  }

  const auto output =
      runner_->Execute(std::move(module), fake_arguments, run_hlo_passes);
  return output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().message();
}

::testing::AssertionResult HloTestBase::RunAndCompare(
    string_view hlo_string, const std::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor,
    std::optional<int64_t> args_max_bits_of_precision) {
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }
  return RunAndCompare(std::move(module_or_status).value(), error,
                       reference_preprocessor, args_max_bits_of_precision);
}

absl::StatusOr<::testing::AssertionResult>
HloTestBase::RunAndCompareTwoModulesInternal(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const absl::Span<Literal* const> arguments,
    const std::optional<ErrorSpec>& error, bool run_hlo_passes) {
  TF_RETURN_IF_ERROR(hlo_verifier_->Run(module_0.get()).status());
  TF_RETURN_IF_ERROR(hlo_verifier_->Run(module_1.get()).status());

  // Execute the two modules.
  TF_ASSIGN_OR_RETURN(auto test_0, runner_->Execute(std::move(module_0),
                                                    arguments, run_hlo_passes));
  TF_ASSIGN_OR_RETURN(auto test_1, runner_->Execute(std::move(module_1),
                                                    arguments, run_hlo_passes));

  return LiteralTestUtil::NearOrEqual(/*expected=*/test_0, /*actual=*/test_1,
                                      error);
}

::testing::AssertionResult HloTestBase::RunAndCompareTwoModules(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const absl::Span<Literal* const> arguments,
    const optional<ErrorSpec>& error, bool run_hlo_passes) {
  auto result =
      RunAndCompareTwoModulesInternal(std::move(module_0), std::move(module_1),
                                      arguments, error, run_hlo_passes);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return result.value();
}

::testing::AssertionResult HloTestBase::RunAndCompareTwoModules(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const optional<ErrorSpec>& error, bool run_hlo_passes,
    std::optional<int64_t> args_max_bits_of_precision) {
  const auto params_0 = module_0->entry_computation()->parameter_instructions();
  const auto params_1 = module_1->entry_computation()->parameter_instructions();
  for (int i = 0; i < params_0.size(); ++i) {
    const HloModuleConfig& module_config_0 = module_0->config();
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

    const HloModuleConfig& module_config_1 = module_1->config();
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
      return ::testing::AssertionFailure()
             << "Error : mismatching parameter shapes: "
             << param_shape_0.ToString(true) << " Vs. "
             << param_shape_1.ToString(true);
    }
  }

  absl::StatusOr<std::vector<Literal>> fake_arguments = MakeFakeArguments(
      module_0.get(), /*pseudo_random=*/true, /*use_large_range=*/false,
      /*treat_gte_as_data_formatting=*/false, args_max_bits_of_precision);
  CHECK_OK(fake_arguments);

  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      *fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  return RunAndCompareTwoModules(std::move(module_0), std::move(module_1),
                                 fake_argument_ptrs, error, run_hlo_passes);
}

::testing::AssertionResult HloTestBase::RunAndCompareTwoModules(
    string_view hlo_string_module_0, string_view hlo_string_module_1,
    const std::optional<ErrorSpec>& error, bool run_hlo_passes,
    std::optional<int64_t> args_max_bits_of_precision) {
  auto module_0_or_status = ParseAndReturnVerifiedModule(hlo_string_module_0);
  if (!module_0_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0_or_status.status().ToString();
  }

  auto module_1_or_status = ParseAndReturnVerifiedModule(hlo_string_module_1);
  if (!module_1_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1_or_status.status().ToString();
  }
  return RunAndCompareTwoModules(std::move(module_0_or_status).value(),
                                 std::move(module_1_or_status).value(), error,
                                 run_hlo_passes, args_max_bits_of_precision);
}

::testing::AssertionResult HloTestBase::RunAndCompareTwoModules(
    string_view hlo_string_module_0, string_view hlo_string_module_1,
    const HloModuleConfig& config_0, const HloModuleConfig& config_1,
    const std::optional<ErrorSpec>& error, bool run_hlo_passes,
    std::optional<int64_t> args_max_bits_of_precision) {
  auto module_0_or_status =
      ParseAndReturnVerifiedModule(hlo_string_module_0, config_0);
  if (!module_0_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0_or_status.status().ToString();
  }

  auto module_1_or_status =
      ParseAndReturnVerifiedModule(hlo_string_module_1, config_1);
  if (!module_1_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1_or_status.status().ToString();
  }
  return RunAndCompareTwoModules(std::move(module_0_or_status).value(),
                                 std::move(module_1_or_status).value(), error,
                                 run_hlo_passes, args_max_bits_of_precision);
}

::testing::AssertionResult HloTestBase::RunAndCompareTwoModules(
    absl::string_view hlo_string_module_0,
    absl::string_view hlo_string_module_1,
    const absl::Span<Literal* const> arguments,
    const std::optional<ErrorSpec>& error, bool run_hlo_passes) {
  auto module_0_or_status = ParseAndReturnVerifiedModule(hlo_string_module_0);
  if (!module_0_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0_or_status.status().ToString();
  }

  auto module_1_or_status = ParseAndReturnVerifiedModule(hlo_string_module_1);
  if (!module_1_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1_or_status.status().ToString();
  }
  return RunAndCompareTwoModules(std::move(module_0_or_status).value(),
                                 std::move(module_1_or_status).value(),
                                 arguments, error, run_hlo_passes);
}

::testing::AssertionResult HloTestBase::Run(
    string_view hlo_string, bool run_hlo_passes, ExecutionProfile* profile,
    const tsl::protobuf::Message* backend_config, bool use_random_data) {
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }
  std::unique_ptr<HloModule> module = std::move(module_or_status.value());
  const auto fake_arguments =
      MakeFakeArguments(module.get(), use_random_data).value();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  if (profile != nullptr) {
    // We have to enable HLO profiling since otherwise currently the
    // ExecutionProfile is not correct.
    //
    // TODO(b/119432044): Fix collection of the ExecutionProfile
    // so that this is not necessary.
    HloModuleConfig config = module->config();
    DebugOptions debug_options = config.debug_options();
    debug_options.set_xla_hlo_profile(true);
    config.set_debug_options(debug_options);
    module->set_config(config);
  }

  if (backend_config) {
    // Set backend configuration if it is given.
    HloInstruction* instruction =
        module->entry_computation()->root_instruction();
    absl::Status s = instruction->set_backend_config(*backend_config);
    return s.ok() ? ::testing::AssertionSuccess()
                  : ::testing::AssertionFailure() << s.message();
  }

  auto output = runner_->Execute(std::move(module), fake_argument_ptrs,
                                 /*run_hlo_passes=*/run_hlo_passes,
                                 /*profile=*/profile);

  return output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().message();
}

::testing::AssertionResult HloTestBase::RunReplicated(
    string_view hlo_string, bool run_hlo_passes, int64_t num_replicas,
    const tsl::protobuf::Message* backend_config) {
  auto module_or_status =
      ParseAndReturnVerifiedModule(hlo_string, num_replicas);
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }

  std::unique_ptr<HloModule> module = std::move(module_or_status.value());
  const auto fake_arguments = MakeFakeArguments(module.get()).value();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  if (backend_config) {
    // Set backend configuration if it is given.
    HloInstruction* instruction =
        module->entry_computation()->root_instruction();
    absl::Status s = instruction->set_backend_config(*backend_config);
    return s.ok() ? ::testing::AssertionSuccess()
                  : ::testing::AssertionFailure() << s.message();
  }

  HloRunner::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;
  for (auto argument : fake_argument_ptrs) {
    options.arguments.push_back(argument);
  }
  auto output = runner_->ExecuteReplicated(std::move(module), options);

  return output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().message();
}

::testing::AssertionResult HloTestBase::RunMultipleTimes(
    string_view hlo_string, bool run_hlo_passes,
    std::vector<ExecutionProfile>* profiles,
    const tsl::protobuf::Message* backend_config, bool assert_determinism) {
  int n = profiles->size();
  std::vector<std::vector<Literal*>> fake_argument_ptrs(n);
  std::vector<std::vector<Literal>> fake_arguments(n);
  std::vector<std::unique_ptr<Executable>> executables(n);

  for (int i = 0; i < n; ++i) {
    auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
    if (!module_or_status.ok()) {
      return ::testing::AssertionFailure()
             << "Error while parsing HLO text format: "
             << module_or_status.status().ToString();
    }
    std::unique_ptr<HloModule> module = std::move(module_or_status.value());

    fake_arguments[i] = MakeFakeArguments(module.get()).value();

    if (profiles != nullptr) {
      // We have to enable HLO profiling since otherwise currently the
      // ExecutionProfile is not correct.
      //
      // TODO(b/119432044): Fix collection of the ExecutionProfile
      // so that this is not necessary.
      HloModuleConfig config = module->config();
      DebugOptions debug_options = config.debug_options();
      debug_options.set_xla_hlo_profile(true);
      config.set_debug_options(debug_options);
      module->set_config(config);
    }

    if (backend_config) {
      // Set backend configuration if it is given.
      HloInstruction* instruction =
          module->entry_computation()->root_instruction();
      absl::Status s = instruction->set_backend_config(*backend_config);
      return s.ok() ? ::testing::AssertionSuccess()
                    : ::testing::AssertionFailure() << s.message();
    }

    auto executable =
        runner_->CreateExecutable(std::move(module), run_hlo_passes);
    if (!executable.ok()) {
      return ::testing::AssertionFailure() << executable.status().message();
    }
    executables[i] = std::move(executable.value());
  }

  std::optional<Literal> canonical_output;
  for (int i = 0; i < n; ++i) {
    absl::StatusOr<Literal> output =
        runner_->ExecuteWithExecutable(executables[i].get(), fake_arguments[i],
                                       /*profile=*/&((*profiles)[i]));
    if (!output.ok()) {
      return ::testing::AssertionFailure() << output.status().message();
    }

    if (assert_determinism) {
      if (!canonical_output.has_value()) {
        canonical_output = std::move(output).value();
      } else {
        if (*canonical_output != output.value()) {
          return ::testing::AssertionFailure()
                 << "Successive runs have returned different results: "
                 << *canonical_output << " vs. " << output.value();
        }
      }
    }
  }

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HloTestBase::RunAndCompareFromFile(
    const std::string& filename, const std::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto module_or_status =
      HloRunner::ReadModuleFromHloTextFile(filename, GetDebugOptionsForTest());
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "failed reading hlo module from file";
  }
  return RunAndCompare(std::move(module_or_status).value(), error,
                       reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPasses(
    string_view hlo_string, const std::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }
  return RunAndCompareNoHloPasses(std::move(module_or_status).value(), error,
                                  reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPassesFromFile(
    const std::string& filename, const std::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto module_or_status =
      HloRunner::ReadModuleFromHloTextFile(filename, GetDebugOptionsForTest());
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "failed reading hlo module from file";
  }
  return RunAndCompareNoHloPasses(std::move(module_or_status).value(), error,
                                  reference_preprocessor);
}

HloComputation* HloTestBase::FindComputation(HloModule* module,
                                             absl::string_view name) {
  auto computations = module->computations();
  auto it = absl::c_find_if(
      computations, [&](HloComputation* c) { return c->name() == name; });
  if (it == computations.end()) {
    return nullptr;
  }
  return *it;
}

HloInstruction* HloTestBase::FindInstruction(HloModule* module,
                                             absl::string_view name) {
  for (const HloComputation* c : module->computations()) {
    auto instructions = c->instructions();
    auto it = absl::c_find_if(
        instructions, [&](HloInstruction* i) { return i->name() == name; });
    if (it != instructions.end()) {
      return *it;
    }
  }
  return nullptr;
}

HloInstruction* HloTestBase::FindInstruction(HloModule* module,
                                             HloOpcode opcode) {
  for (const HloComputation* c : module->computations()) {
    auto instructions = c->instructions();
    auto it = absl::c_find_if(
        instructions, [&](HloInstruction* i) { return i->opcode() == opcode; });
    if (it != instructions.end()) {
      return *it;
    }
  }
  return nullptr;
}

se::DeviceMemoryAllocator* HloTestBase::GetAllocator() {
  if (allocator_ == nullptr) {
    allocator_ = std::make_unique<se::StreamExecutorMemoryAllocator>(
        backend().default_stream_executor());
  }
  return allocator_.get();
}

Backend& HloTestBase::backend() { return test_runner_.backend(); }

/* static */
std::string HloTestBase::TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

void HloTestBase::MatchOptimizedHlo(absl::string_view hlo,
                                    absl::string_view pattern,
                                    bool print_operand_shape) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(hlo));
  HloPrintOptions print_opts;
  print_opts.set_print_operand_shape(print_operand_shape);
  absl::StatusOr<bool> filecheck_result =
      RunFileCheck(optimized_module->ToString(print_opts), pattern);
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.value());
}

absl::StatusOr<std::unique_ptr<HloModule>> HloTestBase::GetOptimizedModule(
    absl::string_view hlo) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  return backend().compiler()->RunHloPasses(
      std::move(module), backend().default_stream_executor(), GetAllocator());
}

absl::StatusOr<std::unique_ptr<HloModule>> HloTestBase::GetOptimizedModule(
    std::unique_ptr<HloModule> hlo_module) {
  return backend().compiler()->RunHloPasses(std::move(hlo_module),
                                            backend().default_stream_executor(),
                                            GetAllocator());
}

absl::StatusOr<std::unique_ptr<HloRunnerInterface>>
HloTestBase::GetHloRunnerForTest(se::Platform* test_platform) {
  if (ShouldUsePjRt()) {
    PjRtClientTestFactoryRegistry& pjrt_registry =
        GetGlobalPjRtClientTestFactory();
    TF_ASSIGN_OR_RETURN(auto client, pjrt_registry.Get()());

    auto device_shape_representation_fn =
        pjrt_registry.GetDeviceShapeRepresentationFn(client.get());

    return std::unique_ptr<HloRunnerInterface>(
        new HloRunnerPjRt(std::move(client), device_shape_representation_fn));
  } else {
    return std::unique_ptr<HloRunnerInterface>(new HloRunner(test_platform));
  }
}

}  // namespace xla
