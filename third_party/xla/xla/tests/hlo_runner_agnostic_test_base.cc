/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/tests/hlo_runner_agnostic_test_base.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {

namespace {

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

HloRunnerAgnosticTestBase::HloRunnerAgnosticTestBase(
    absl::Nonnull<std::unique_ptr<HloRunnerInterface>> test_runner,
    absl::Nonnull<std::unique_ptr<HloRunnerInterface>> reference_runner,
    const bool verifier_layout_sensitive,
    const bool allow_mixed_precision_in_hlo_verifier,
    const HloPredicate instruction_can_change_layout_func)
    : HloHardwareIndependentTestBase(verifier_layout_sensitive,
                                     allow_mixed_precision_in_hlo_verifier,
                                     instruction_can_change_layout_func),
      test_runner_(std::move(test_runner)),
      reference_runner_(std::move(reference_runner)) {}

std::unique_ptr<VerifiedHloModule>
HloRunnerAgnosticTestBase::CreateNewVerifiedModule(
    const std::string& name, const int64_t replica_count) {
  return std::make_unique<VerifiedHloModule>(
      name, GetModuleConfigForTest(replica_count), verifier_layout_sensitive(),
      allow_mixed_precision_in_hlo_verifier(),
      test_runner_->device_shape_size_fn(),
      instruction_can_change_layout_func());
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
HloRunnerAgnosticTestBase::ParseAndReturnVerifiedModule(
    absl::string_view hlo_text, int64_t replica_count, int64_t num_partitions) {
  return ParseAndReturnVerifiedModule(
      hlo_text, GetModuleConfigForTest(replica_count, num_partitions));
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
HloRunnerAgnosticTestBase::ParseAndReturnVerifiedModule(
    absl::string_view hlo_text, const HloModuleConfig& config) {
  auto module = std::make_unique<VerifiedHloModule>(
      TestName(), config, verifier_layout_sensitive(),
      allow_mixed_precision_in_hlo_verifier(),
      test_runner_->device_shape_size_fn(),
      instruction_can_change_layout_func());
  TF_RETURN_IF_ERROR(module->ParseHloStringAndVerifyModule(hlo_text));
  UpdateEntryComputationLayout(module.get());
  return std::move(module);
}

HloComputation*
HloRunnerAgnosticTestBase::AddEntryComputationAndUpdateEntryComputationLayout(
    HloModule* const module, std::unique_ptr<HloComputation> computation) {
  HloComputation* const comp =
      module->AddEntryComputation(std::move(computation));
  UpdateEntryComputationLayout(module);
  return comp;
}

void HloRunnerAgnosticTestBase::UpdateEntryComputationLayout(
    HloModule* const module) const {
  xla::UpdateEntryComputationLayout(
      module, test_runner_->device_shape_representation_fn());
}

absl::StatusOr<Literal> HloRunnerAgnosticTestBase::Execute(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
    bool run_hlo_passes) {
  return test_runner_->Execute(std::move(module), arguments, run_hlo_passes);
}

Literal HloRunnerAgnosticTestBase::ExecuteNoHloPasses(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments) {
  absl::StatusOr<Literal> result = Execute(std::move(module), arguments,
                                           /*run_hlo_passes=*/false);
  CHECK_OK(result.status());
  return *std::move(result);
}

Literal HloRunnerAgnosticTestBase::ExecuteAndTransfer(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments) {
  absl::StatusOr<Literal> result =
      test_runner_->Execute(std::move(module), arguments, true, nullptr);
  CHECK_OK(result.status());
  return *std::move(result);
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const absl::Span<Literal* const> arguments, const int64_t num_replicas,
    const bool use_threads, const bool run_hlo_passes) {
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.arguments = {arguments.begin(), arguments.end()};
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = use_threads;
  return test_runner_->ExecuteReplicated(std::move(module), std::move(options));
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const absl::Span<Literal* const> arguments, const int64_t num_replicas,
    DeviceAssignment* const device_assignment, const bool run_hlo_passes,
    const bool use_threads) {
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.arguments = {arguments.begin(), arguments.end()};
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = use_threads;
  return test_runner_->ExecuteReplicated(std::move(module), std::move(options),
                                         device_assignment);
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    const std::function<Executable*(int64_t)> executable_provider,
    const std::function<int64_t(int64_t)> argument_count_provider,
    const std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const int64_t num_replicas, const bool run_hlo_passes,
    DeviceAssignment* const device_assignment) {
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;
  return test_runner_->ExecuteReplicated(
      executable_provider, argument_count_provider, argument_provider,
      std::move(options), device_assignment);
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const std::vector<std::vector<Literal*>> arguments,
    const int64_t num_replicas, const bool run_hlo_passes,
    DeviceAssignment* const device_assignment) {
  CHECK(num_replicas > 0 && "expect at least one replica");
  CHECK(num_replicas == arguments.size() &&
        "expect arguments for each replica");
  int64_t argument_count = arguments.front().size();
  TF_ASSIGN_OR_RETURN(
      const std::unique_ptr<Executable> executable,
      test_runner_->CreateExecutable(std::move(module), run_hlo_passes));
  return ExecuteReplicated(
      /*executable_provider=*/[&](int64_t) { return executable.get(); },
      /*argument_count_provider=*/[&](int64_t) { return argument_count; },
      /*argument_provider=*/
      [&](int64_t replica_idx, int64_t argument_idx) -> const Literal* {
        return arguments[replica_idx][argument_idx];
      },
      num_replicas, /*run_hlo_passes=*/run_hlo_passes,
      /*device_assignment=*/device_assignment);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompare(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
    const std::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor,
    const std::function<void(HloModule*)>& test_preprocessor) {
  const absl::StatusOr<::testing::AssertionResult> result =
      RunAndCompareInternal(std::move(module), arguments, error,
                            /*run_hlo_passes=*/true, reference_preprocessor,
                            test_preprocessor);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return *result;
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareNoHloPasses(
    std::unique_ptr<HloModule> module,
    const absl::Span<Literal* const> arguments,
    const std::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor,
    const std::function<void(HloModule*)>& test_preprocessor) {
  const absl::StatusOr<::testing::AssertionResult> result =
      RunAndCompareInternal(std::move(module), arguments, error,
                            /*run_hlo_passes=*/false, reference_preprocessor,
                            test_preprocessor);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return *result;
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompare(
    std::unique_ptr<HloModule> module, const std::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor,
    const std::function<void(HloModule*)>& test_preprocessor,
    const std::optional<int64_t> args_max_bits_of_precision) {
  const std::vector<Literal> fake_arguments =
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
                       reference_preprocessor, test_preprocessor);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareNoHloPasses(
    std::unique_ptr<HloModule> module, const std::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor,
    const std::function<void(HloModule*)>& test_preprocessor) {
  const std::vector<Literal> fake_arguments =
      MakeFakeArguments(module.get()).value();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });
  return RunAndCompareNoHloPasses(std::move(module), fake_argument_ptrs, error,
                                  reference_preprocessor, test_preprocessor);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::Run(
    std::unique_ptr<HloModule> module, const bool run_hlo_passes,
    const std::function<void(HloModule*)>& test_preprocessor) {
  const std::vector<Literal> fake_arguments =
      MakeFakeArguments(module.get()).value();
  if (const absl::StatusOr<bool> change = verifier().Run(module.get());
      !change.ok()) {
    return ::testing::AssertionFailure() << change.status();
  }
  if (absl::Status status = PreprocessModuleForTestRunner(module.get());
      !status.ok()) {
    return ::testing::AssertionFailure() << status;
  }
  if (test_preprocessor != nullptr) {
    test_preprocessor(module.get());
  }

  const absl::StatusOr<Literal> output =
      test_runner_->Execute(std::move(module), fake_arguments, run_hlo_passes);
  return output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().message();
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompare(
    const absl::string_view hlo_string, const std::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor,
    const std::function<void(HloModule*)>& test_preprocessor,
    const std::optional<int64_t> args_max_bits_of_precision) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
      ParseAndReturnVerifiedModule(hlo_string);
  if (!module.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module.status().ToString();
  }
  return RunAndCompare(*std::move(module), error, reference_preprocessor,
                       test_preprocessor, args_max_bits_of_precision);
}

::testing::AssertionResult
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const HloRunnerInterface::ReplicatedExecuteOptions options,
    const std::optional<ErrorSpec>& error) {
  const int replica_count = module_0->config().replica_count();
  if (replica_count != module_1->config().replica_count()) {
    return ::testing::AssertionFailure()
           << "Number of replicas is not the same: " << replica_count << " Vs "
           << module_1->config().replica_count();
  }
  if (options.num_replicas != replica_count) {
    return ::testing::AssertionFailure()
           << "Number of execution replicas is different from number of "
              "replicas in the module: requested number of replicas = "
           << options.num_replicas
           << ", number of replicas in hlo = " << replica_count;
  }

  if (const std::vector<int> mismatches = CompareInputs(*module_0, *module_1);
      !mismatches.empty()) {
    return ::testing::AssertionFailure()
           << "Error: parameter mismatch at indices: "
           << absl::StrJoin(mismatches, ",");
  }
  if (const int64_t num_args = module_0->entry_computation()->num_parameters();
      num_args != options.arguments.size()) {
    return ::testing::AssertionFailure()
           << "Mismatch in number of arguments passed while running replicated "
              "hlo module. Expected: "
           << num_args << ", actual: " << options.arguments.size();
  }
  const absl::StatusOr<::testing::AssertionResult> result =
      RunAndCompareTwoModulesInternalReplicated(
          std::move(module_0), std::move(module_1), options, error);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return *result;
}

::testing::AssertionResult
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const bool run_hlo_passes, const bool use_threads,
    const std::optional<ErrorSpec>& error) {
  const absl::StatusOr<std::vector<Literal>> fake_arguments = MakeFakeArguments(
      /*module=*/module_0.get(), /*pseudo_random=*/true,
      /*use_large_range=*/false,
      /*treat_gte_as_data_formatting=*/false,
      /*max_bits_of_precision=*/std::nullopt);
  CHECK_OK(fake_arguments);

  return RunAndCompareTwoModulesReplicated(std::move(module_0),
                                           std::move(module_1), *fake_arguments,
                                           run_hlo_passes, use_threads, error);
}

::testing::AssertionResult
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const std::vector<Literal>& fake_arguments, const bool run_hlo_passes,
    const bool use_threads, const std::optional<ErrorSpec>& error) {
  std::vector<const Literal*> fake_argument_ptrs;
  absl::c_transform(
      /*input=*/fake_arguments,
      /*output=*/std::back_inserter(fake_argument_ptrs),
      /*unary_op=*/[](const Literal& literal) -> Literal* {
        return const_cast<Literal*>(&literal);
      });
  const HloRunnerInterface::ReplicatedExecuteOptions options{
      /*num_replicas=*/module_0->config().replica_count(),
      /*arguments=*/fake_argument_ptrs,
      /*infeed_values=*/{},
      /*infeed_steps=*/-1,
      /*outfeed_shape=*/{},
      /*outfeed_values=*/nullptr,
      /*run_hlo_passes=*/run_hlo_passes,
      /*use_threads=*/use_threads};
  return RunAndCompareTwoModulesReplicated(std::move(module_0),
                                           std::move(module_1), options, error);
}

::testing::AssertionResult
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesReplicated(
    const absl::string_view module_0_str, const absl::string_view module_1_str,
    const bool run_hlo_passes, const bool use_threads,
    const std::optional<ErrorSpec>& error) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_0 =
      ParseAndReturnVerifiedModule(module_0_str);
  if (!module_0.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0.status().ToString();
  }

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_1 =
      ParseAndReturnVerifiedModule(module_1_str);
  if (!module_1.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1.status().ToString();
  }
  return RunAndCompareTwoModulesReplicated(*std::move(module_0),
                                           *std::move(module_1), run_hlo_passes,
                                           use_threads, error);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareTwoModules(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const absl::Span<Literal* const> arguments,
    const std::optional<ErrorSpec>& error, bool run_hlo_passes) {
  const absl::StatusOr<::testing::AssertionResult> result =
      RunAndCompareTwoModulesInternal(std::move(module_0), std::move(module_1),
                                      arguments, error, run_hlo_passes);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return *result;
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareTwoModules(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const std::optional<ErrorSpec>& error, const bool run_hlo_passes,
    const std::optional<int64_t> args_max_bits_of_precision) {
  if (const std::vector<int> mismatches = CompareInputs(*module_0, *module_1);
      !mismatches.empty()) {
    return ::testing::AssertionFailure()
           << "Error : mismatching parameter shapes for parameters "
           << absl::StrJoin(mismatches, ", ");
  }

  const absl::StatusOr<std::vector<Literal>> fake_arguments = MakeFakeArguments(
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

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareTwoModules(
    const absl::string_view hlo_string_module_0,
    const absl::string_view hlo_string_module_1,
    const std::optional<ErrorSpec>& error, const bool run_hlo_passes,
    const std::optional<int64_t> args_max_bits_of_precision) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_0 =
      ParseAndReturnVerifiedModule(hlo_string_module_0);
  if (!module_0.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0.status().ToString();
  }

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_1 =
      ParseAndReturnVerifiedModule(hlo_string_module_1);
  if (!module_1.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1.status().ToString();
  }
  return RunAndCompareTwoModules(*std::move(module_0), *std::move(module_1),
                                 error, run_hlo_passes,
                                 args_max_bits_of_precision);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareTwoModules(
    const absl::string_view hlo_string_module_0,
    const absl::string_view hlo_string_module_1,
    const HloModuleConfig& config_0, const HloModuleConfig& config_1,
    const std::optional<ErrorSpec>& error, const bool run_hlo_passes,
    const std::optional<int64_t> args_max_bits_of_precision) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_0 =
      ParseAndReturnVerifiedModule(hlo_string_module_0, config_0);
  if (!module_0.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0.status().ToString();
  }

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_1 =
      ParseAndReturnVerifiedModule(hlo_string_module_1, config_1);
  if (!module_1.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1.status().ToString();
  }
  return RunAndCompareTwoModules(*std::move(module_0), *std::move(module_1),
                                 error, run_hlo_passes,
                                 args_max_bits_of_precision);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareTwoModules(
    absl::string_view hlo_string_module_0,
    absl::string_view hlo_string_module_1,
    const absl::Span<Literal* const> arguments,
    const std::optional<ErrorSpec>& error, const bool run_hlo_passes) {
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

::testing::AssertionResult HloRunnerAgnosticTestBase::Run(
    const absl::string_view hlo_string, const bool run_hlo_passes,
    ExecutionProfile* const profile,
    const tsl::protobuf::Message* backend_config, const bool use_random_data) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
      ParseAndReturnVerifiedModule(hlo_string);
  if (!module.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module.status().ToString();
  }
  if (absl::Status status = PreprocessModuleForTestRunner(module->get());
      !status.ok()) {
    return ::testing::AssertionFailure() << status;
  }
  const std::vector<Literal> fake_arguments =
      MakeFakeArguments(module->get(), use_random_data).value();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  if (profile != nullptr) {
    // We have to enable HLO profiling since otherwise currently the
    // ExecutionProfile is not correct.
    HloModuleConfig config = (*module)->config();
    DebugOptions debug_options = config.debug_options();
    debug_options.set_xla_hlo_profile(true);
    config.set_debug_options(debug_options);
    (*module)->set_config(config);
  }

  if (backend_config) {
    // Set backend configuration if it is given.
    HloInstruction* instruction =
        (*module)->entry_computation()->root_instruction();
    absl::Status s = instruction->set_backend_config(*backend_config);
    return s.ok() ? ::testing::AssertionSuccess()
                  : ::testing::AssertionFailure() << s.message();
  }

  auto output = test_runner_->Execute(*std::move(module), fake_argument_ptrs,
                                      /*run_hlo_passes=*/run_hlo_passes,
                                      /*profile=*/profile);

  return output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().message();
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunReplicated(
    const absl::string_view hlo_string, const bool run_hlo_passes,
    const int64_t num_replicas, const tsl::protobuf::Message* backend_config) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
      ParseAndReturnVerifiedModule(hlo_string, num_replicas);
  if (!module.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module.status().ToString();
  }

  const std::vector<Literal> fake_arguments =
      MakeFakeArguments(module->get()).value();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  if (backend_config) {
    // Set backend configuration if it is given.
    HloInstruction* instruction =
        (*module)->entry_computation()->root_instruction();
    if (const absl::Status s = instruction->set_backend_config(*backend_config);
        !s.ok()) {
      return ::testing::AssertionFailure() << s.message();
    }
    return ::testing::AssertionSuccess();
  }

  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.arguments = {fake_argument_ptrs.begin(), fake_argument_ptrs.end()};
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;
  const absl::StatusOr<std::vector<Literal>> output =
      test_runner_->ExecuteReplicated(*std::move(module), std::move(options));
  if (output.ok()) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << output.status().message();
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunMultipleTimes(
    const absl::string_view hlo_string, const bool run_hlo_passes,
    std::vector<ExecutionProfile>* const profiles,
    const tsl::protobuf::Message* const backend_config,
    const bool assert_determinism) {
  const int n = profiles->size();
  std::vector<std::vector<Literal*>> fake_argument_ptrs(n);
  std::vector<std::vector<Literal>> fake_arguments(n);
  std::vector<std::unique_ptr<Executable>> executables(n);

  for (int i = 0; i < n; ++i) {
    absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
        ParseAndReturnVerifiedModule(hlo_string);
    if (!module.ok()) {
      return ::testing::AssertionFailure()
             << "Error while parsing HLO text format: "
             << module.status().ToString();
    }

    fake_arguments[i] = MakeFakeArguments(module->get()).value();

    if (profiles != nullptr) {
      // We have to enable HLO profiling since otherwise currently the
      // ExecutionProfile is not correct.
      HloModuleConfig config = (*module)->config();
      DebugOptions debug_options = config.debug_options();
      debug_options.set_xla_hlo_profile(true);
      config.set_debug_options(debug_options);
      (*module)->set_config(config);
    }

    if (backend_config) {
      // Set backend configuration if it is given.
      HloInstruction* instruction =
          (*module)->entry_computation()->root_instruction();
      absl::Status s = instruction->set_backend_config(*backend_config);
      return s.ok() ? ::testing::AssertionSuccess()
                    : ::testing::AssertionFailure() << s.message();
    }

    absl::StatusOr<std::unique_ptr<Executable>> executable =
        test_runner_->CreateExecutable(*std::move(module), run_hlo_passes);
    if (!executable.ok()) {
      return ::testing::AssertionFailure() << executable.status().message();
    }
    executables[i] = *std::move(executable);
  }

  std::optional<Literal> canonical_output;
  for (int i = 0; i < n; ++i) {
    absl::StatusOr<Literal> output = test_runner_->ExecuteWithExecutable(
        executables[i].get(), fake_arguments[i],
        /*profile=*/&((*profiles)[i]));
    if (!output.ok()) {
      return ::testing::AssertionFailure() << output.status().message();
    }

    if (assert_determinism) {
      if (!canonical_output.has_value()) {
        canonical_output = *std::move(output);
      } else {
        if (*canonical_output != *output) {
          return ::testing::AssertionFailure()
                 << "Successive runs have returned different results: "
                 << *canonical_output << " vs. " << *output;
        }
      }
    }
  }

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareNoHloPasses(
    const absl::string_view hlo_string, const std::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor,
    const std::function<void(HloModule*)>& test_preprocessor) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
      ParseAndReturnVerifiedModule(hlo_string);
  if (!module.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module.status().ToString();
  }
  return RunAndCompareNoHloPasses(*std::move(module), error,
                                  reference_preprocessor, test_preprocessor);
}

absl::StatusOr<std::unique_ptr<HloModule>>
HloRunnerAgnosticTestBase::MakeReferenceModule(
    const HloModule& test_module,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  std::unique_ptr<HloModule> reference_module = test_module.Clone();
  const ProgramShape program_shape = GetProgramShapeWithLayout(test_module);

  if (reference_preprocessor != nullptr) {
    reference_preprocessor(reference_module.get());
    if (!ProgramShapesEqual(program_shape,
                            GetProgramShapeWithLayout(*reference_module))) {
      return InvalidArgument(
          "reference preprocessor must not modify the program shape");
    }
  }
  TF_RETURN_IF_ERROR(verifier().Run(reference_module.get()).status());
  return std::move(reference_module);
}

absl::StatusOr<::testing::AssertionResult>
HloRunnerAgnosticTestBase::RunAndCompareInternal(
    std::unique_ptr<HloModule> module,
    const absl::Span<Literal* const> arguments,
    const std::optional<ErrorSpec>& error, const bool run_hlo_passes,
    const std::function<void(HloModule*)>& reference_preprocessor,
    const std::function<void(HloModule*)>& test_preprocessor) {
  TF_RETURN_IF_ERROR(verifier().Run(module.get()).status());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> reference_module,
                      MakeReferenceModule(*module, reference_preprocessor));
  TF_RETURN_IF_ERROR(PreprocessModuleForTestRunner(module.get()));
  if (test_preprocessor != nullptr) {
    test_preprocessor(module.get());
  }
  // Execute on two backends.
  TF_ASSIGN_OR_RETURN(
      const Literal test,
      test_runner_->Execute(std::move(module), arguments, run_hlo_passes));
  TF_ASSIGN_OR_RETURN(const Literal reference,
                      reference_runner_->Execute(std::move(reference_module),
                                                 arguments, run_hlo_passes));
  if (reference.IsAll(0)) {
    LOG(WARNING) << "Reference value is only zeros.";
  }

  return LiteralTestUtil::NearOrEqual(/*expected=*/reference, /*actual=*/test,
                                      error);
}

absl::StatusOr<::testing::AssertionResult>
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesInternalReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const HloRunnerInterface::ReplicatedExecuteOptions options,
    const std::optional<ErrorSpec>& error) {
  TF_RETURN_IF_ERROR(verifier().Run(module_0.get()).status());
  TF_RETURN_IF_ERROR(verifier().Run(module_1.get()).status());

  // Execute the two modules.
  TF_ASSIGN_OR_RETURN(auto test_0, test_runner_->ExecuteReplicated(
                                       std::move(module_0), options));
  TF_ASSIGN_OR_RETURN(auto test_1, test_runner_->ExecuteReplicated(
                                       std::move(module_1), options));

  for (const auto& [expected, actual] : llvm::zip_equal(test_0, test_1)) {
    if (::testing::AssertionResult result =
            LiteralTestUtil::NearOrEqual(expected, actual, error);
        !result) {
      return result;
    }
  }
  return ::testing::AssertionSuccess();
}

absl::StatusOr<::testing::AssertionResult>
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesInternal(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const absl::Span<Literal* const> arguments,
    const std::optional<ErrorSpec>& error, bool run_hlo_passes) {
  TF_RETURN_IF_ERROR(verifier().Run(module_0.get()).status());
  TF_RETURN_IF_ERROR(verifier().Run(module_1.get()).status());

  // Execute the two modules.
  TF_ASSIGN_OR_RETURN(
      const Literal test_0,
      test_runner_->Execute(std::move(module_0), arguments, run_hlo_passes));
  TF_ASSIGN_OR_RETURN(
      const Literal test_1,
      test_runner_->Execute(std::move(module_1), arguments, run_hlo_passes));

  return LiteralTestUtil::NearOrEqual(/*expected=*/test_0, /*actual=*/test_1,
                                      error);
}

}  // namespace xla
