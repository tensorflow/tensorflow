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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/test_utils/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/service/backend.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/pjrt_client_registry.h"
#include "xla/tests/test_utils.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
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
    : HloHardwareIndependentTestBase(verifier_layout_sensitive,
                                     allow_mixed_precision_in_hlo_verifier,
                                     instruction_can_change_layout_func),
      test_runner_(test_platform),
      reference_runner_(reference_platform),
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

absl::StatusOr<Literal> HloTestBase::Execute(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
    bool run_hlo_passes) {
  return runner_->Execute(std::move(module), arguments, run_hlo_passes);
}

Literal HloTestBase::ExecuteNoHloPasses(std::unique_ptr<HloModule> module,
                                        absl::Span<Literal* const> arguments) {
  return Execute(std::move(module), arguments,
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

absl::StatusOr<std::vector<Literal>> HloTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    std::vector<std::vector<Literal*>> arguments, int64_t num_replicas,
    bool run_hlo_passes) {
  CHECK(num_replicas > 0 && "expect at least one replica");
  CHECK(num_replicas == arguments.size() &&
        "expect arguments for each replica");
  int64_t argument_count = arguments.front().size();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      runner_->CreateExecutable(std::unique_ptr<HloModule>(std::move(module)),
                                run_hlo_passes));
  return ExecuteReplicated(
      /*executable_provider=*/[&](int64_t) { return executable.get(); },
      /*argument_count_provider=*/[&](int64_t) { return argument_count; },
      /*argument_provider=*/
      [&](int64_t replica_idx, int64_t argument_idx) -> const Literal* {
        return arguments[replica_idx][argument_idx];
      },
      num_replicas, /*run_hlo_passes=*/run_hlo_passes,
      /*device_assignment=*/nullptr);
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
    const std::function<void(HloModule*)>& reference_preprocessor,
    const std::function<void(HloModule*)>& test_preprocessor) {
  TF_RETURN_IF_ERROR(hlo_verifier_->Run(module.get()).status());
  TF_ASSIGN_OR_RETURN(auto reference_module,
                      MakeReferenceModule(*module, reference_preprocessor));
  if (test_preprocessor) {
    test_preprocessor(module.get());
  }
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
    const std::function<void(HloModule*)>& reference_preprocessor,
    const std::function<void(HloModule*)>& test_preprocessor) {
  auto result = RunAndCompareInternal(
      std::move(module), arguments, error,
      /*run_hlo_passes=*/false, reference_preprocessor, test_preprocessor);
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
    const std::function<void(HloModule*)>& reference_preprocessor,
    const std::function<void(HloModule*)>& test_preprocessor) {
  const auto fake_arguments = MakeFakeArguments(module.get()).value();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  auto assertion_result =
      RunAndCompareNoHloPasses(std::move(module), fake_argument_ptrs, error,
                               reference_preprocessor, test_preprocessor);
  if (!assertion_result) {
    for (const auto& literal : fake_arguments) {
      uint64_t total_elements = 1;
      absl::c_for_each(literal.shape().dimensions(),
                       [&](int64_t dim) { total_elements *= dim; });
      if (total_elements > 1000) {
        assertion_result << "argument literal is too large to print: "
                         << literal.shape().ToString();
        continue;
      }
      assertion_result << "argument literal: " << literal.ToString();
    }
  }
  return assertion_result;
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
HloTestBase::RunAndCompareTwoModulesInternalReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    HloRunner::ReplicatedExecuteOptions options,
    const std::optional<ErrorSpec>& error) {
  TF_RETURN_IF_ERROR(hlo_verifier_->Run(module_0.get()).status());
  TF_RETURN_IF_ERROR(hlo_verifier_->Run(module_1.get()).status());

  // Execute the two modules.
  TF_ASSIGN_OR_RETURN(auto test_0,
                      runner_->ExecuteReplicated(std::move(module_0), options));
  TF_ASSIGN_OR_RETURN(auto test_1,
                      runner_->ExecuteReplicated(std::move(module_1), options));

  for (auto [expected, actual] : llvm::zip_equal(test_0, test_1)) {
    auto compare_result = LiteralTestUtil::NearOrEqual(expected, actual, error);
    if (!compare_result) {
      return compare_result;
    }
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HloTestBase::RunAndCompareTwoModulesReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    HloRunner::ReplicatedExecuteOptions options,
    const optional<ErrorSpec>& error) {
  int replica_count = module_0->config().replica_count();
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

  std::vector<int> mismatches = CompareInputs(*module_0, *module_1);
  if (!mismatches.empty()) {
    return ::testing::AssertionFailure()
           << "Error: parameter mismatch at indices: "
           << absl::StrJoin(mismatches, ",");
  }
  auto num_args = module_0->entry_computation()->num_parameters();
  if (num_args != options.arguments.size()) {
    return ::testing::AssertionFailure()
           << "Mismatch in number of arguments passed while running "
              "replicated "
              "hlo module. Expected: "
           << num_args << ", actual: " << options.arguments.size();
  }
  auto result = RunAndCompareTwoModulesInternalReplicated(
      std::move(module_0), std::move(module_1), options, error);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return result.value();
}

::testing::AssertionResult HloTestBase::RunAndCompareTwoModulesReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    bool run_hlo_passes, bool use_threads,
    const std::optional<ErrorSpec>& error) {
  absl::StatusOr<std::vector<Literal>> fake_arguments = MakeFakeArguments(
      /*module=*/module_0.get(), /*pseudo_random=*/true,
      /*use_large_range=*/false,
      /*treat_gte_as_data_formatting=*/false,
      /*max_bits_of_precision=*/std::nullopt);
  CHECK_OK(fake_arguments);
  std::vector<const Literal*> fake_argument_ptrs;
  absl::c_transform(
      /*input=*/*fake_arguments,
      /*output=*/std::back_inserter(fake_argument_ptrs),
      /*unary_op=*/[](const Literal& literal) -> Literal* {
        return const_cast<Literal*>(&literal);
      });
  HloRunner::ReplicatedExecuteOptions options{
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

::testing::AssertionResult HloTestBase::RunAndCompareTwoModulesReplicated(
    absl::string_view module_0, absl::string_view module_1, bool run_hlo_passes,
    bool use_threads, const std::optional<ErrorSpec>& error) {
  auto module_0_or_status = ParseAndReturnVerifiedModule(module_0);
  if (!module_0_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0_or_status.status().ToString();
  }

  auto module_1_or_status = ParseAndReturnVerifiedModule(module_1);
  if (!module_1_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1_or_status.status().ToString();
  }
  return RunAndCompareTwoModulesReplicated(
      std::move(module_0_or_status).value(),
      std::move(module_1_or_status).value(), run_hlo_passes, use_threads,
      error);
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
  std::vector<int> mismatches = CompareInputs(*module_0, *module_1);
  if (!mismatches.empty()) {
    return ::testing::AssertionFailure()
           << "Error : mismatching parameter shapes for parameters "
           << absl::StrJoin(mismatches, ", ");
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
    const std::function<void(HloModule*)>& reference_preprocessor,
    const std::function<void(HloModule*)>& test_preprocessor) {
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }
  return RunAndCompareNoHloPasses(std::move(module_or_status).value(), error,
                                  reference_preprocessor, test_preprocessor);
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

se::DeviceMemoryAllocator* HloTestBase::GetAllocator() {
  if (allocator_ == nullptr) {
    allocator_ = std::make_unique<se::StreamExecutorMemoryAllocator>(
        backend().default_stream_executor());
  }
  return allocator_.get();
}

Backend& HloTestBase::backend() { return test_runner_.backend(); }
const Backend& HloTestBase::backend() const { return test_runner_.backend(); }

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
