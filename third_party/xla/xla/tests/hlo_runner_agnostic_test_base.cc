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
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "google/protobuf/message.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {

HloRunnerAgnosticTestBase::HloRunnerAgnosticTestBase(
    absl_nonnull std::unique_ptr<HloRunnerInterface> test_runner,
    DeviceShapeRepresentationFn device_shape_representation_fn,
    DeviceShapeSizeFn device_shape_size_fn,
    HloRunnerAgnosticTestBaseOptions options)
    : HloHardwareIndependentTestBase(
          options.verifier_layout_sensitive,
          options.allow_mixed_precision_in_hlo_verifier,
          std::move(options.instruction_can_change_layout_func)),
      test_runner_(std::move(test_runner)),
      device_shape_representation_fn_(
          std::move(device_shape_representation_fn)),
      device_shape_size_fn_(std::move(device_shape_size_fn)),
      swallow_execution_errors_(options.swallow_execution_errors) {}

HloRunnerAgnosticTestBase::HloRunnerAgnosticTestBase(
    absl_nonnull std::unique_ptr<HloRunnerInterface> test_runner,
    DeviceShapeRepresentationFn device_shape_representation_fn,
    DeviceShapeSizeFn device_shape_size_fn,
    const bool verifier_layout_sensitive,
    const bool allow_mixed_precision_in_hlo_verifier,
    const HloPredicate instruction_can_change_layout_func)
    : HloHardwareIndependentTestBase(verifier_layout_sensitive,
                                     allow_mixed_precision_in_hlo_verifier,
                                     instruction_can_change_layout_func),
      test_runner_(std::move(test_runner)),
      device_shape_representation_fn_(
          std::move(device_shape_representation_fn)),
      device_shape_size_fn_(std::move(device_shape_size_fn)) {}

std::unique_ptr<VerifiedHloModule>
HloRunnerAgnosticTestBase::CreateNewVerifiedModule(
    const std::string& name, const int64_t replica_count) {
  return std::make_unique<VerifiedHloModule>(
      name, GetModuleConfigForTest(replica_count), verifier_layout_sensitive(),
      allow_mixed_precision_in_hlo_verifier(), device_shape_size_fn_,
      instruction_can_change_layout_func());
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
HloRunnerAgnosticTestBase::ParseAndReturnVerifiedModule(
    absl::string_view hlo_text, const HloModuleConfig& config,
    const HloParserOptions& parser_options) const {
  return HloHardwareIndependentTestBase::ParseAndReturnVerifiedModule(
      hlo_text, config, parser_options, device_shape_size_fn_);
}

absl::StatusOr<std::unique_ptr<HloModule>>
HloRunnerAgnosticTestBase::HloModuleFromXlaComputation(
    const XlaComputation& computation,
    const ExecutionOptions& execution_options) const {
  TF_ASSIGN_OR_RETURN(
      HloModuleConfig module_config,
      HloModule::CreateModuleConfigFromProto(computation.proto(),
                                             execution_options.debug_options(),
                                             &execution_options));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      HloModule::CreateFromProto(computation.proto(), module_config));
  TF_RETURN_IF_ERROR(verifier().Run(module.get()).status());
  return module;
}

absl::StatusOr<std::unique_ptr<HloModule>>
HloRunnerAgnosticTestBase::HloModuleFromXlaBuilder(
    XlaBuilder* builder, const ExecutionOptions& execution_options) const {
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder->Build());
  return HloModuleFromXlaComputation(computation, execution_options);
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
  // TODO - b/391868033: Remove UpdateEntryComputationLayout from this class.
  xla::UpdateEntryComputationLayout(module, device_shape_representation_fn_);
}

absl::StatusOr<Literal> HloRunnerAgnosticTestBase::Execute(
    std::unique_ptr<HloModule> module,
    absl::Span<const Literal* const> arguments, bool run_hlo_passes) {
  TF_RETURN_IF_ERROR(PreprocessModuleForTestRunner(module.get()));
  return test_runner_->Execute(std::move(module), arguments, run_hlo_passes);
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const absl::Span<const Literal* const> arguments, const int64_t num_devices,
    const bool use_threads, const bool run_hlo_passes) {
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices = num_devices;
  options.arguments = {arguments.begin(), arguments.end()};
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = use_threads;
  TF_RETURN_IF_ERROR(PreprocessModuleForTestRunner(module.get()));
  return test_runner_->ExecuteReplicated(std::move(module), std::move(options));
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const absl::Span<const Literal* const> arguments, const int64_t num_devices,
    DeviceAssignment* const device_assignment, const bool run_hlo_passes,
    const bool use_threads) {
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices = num_devices;
  options.arguments = {arguments.begin(), arguments.end()};
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = use_threads;
  TF_RETURN_IF_ERROR(PreprocessModuleForTestRunner(module.get()));
  return test_runner_->ExecuteReplicated(std::move(module), std::move(options),
                                         device_assignment);
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    absl::AnyInvocable<OpaqueExecutable*(int64_t)> executable_provider,
    absl::AnyInvocable<int64_t(int64_t)> argument_count_provider,
    absl::AnyInvocable<const Literal*(int64_t, int64_t)> argument_provider,
    const int64_t num_devices, const bool run_hlo_passes,
    DeviceAssignment* const device_assignment) {
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices = num_devices;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;
  return test_runner_->ExecuteReplicated(
      std::move(executable_provider), std::move(argument_count_provider),
      std::move(argument_provider), std::move(options), device_assignment);
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const std::vector<std::vector<Literal*>> arguments,
    const int64_t num_devices, const bool run_hlo_passes,
    DeviceAssignment* const device_assignment) {
  CHECK(num_devices > 0 && "expected at least one device");
  CHECK(num_devices == arguments.size() && "expect arguments for each device");
  int64_t argument_count = arguments.front().size();
  TF_RETURN_IF_ERROR(PreprocessModuleForTestRunner(module.get()));
  TF_ASSIGN_OR_RETURN(
      const std::unique_ptr<OpaqueExecutable> executable,
      test_runner_->CreateExecutable(std::move(module), run_hlo_passes));
  return ExecuteReplicated(
      /*executable_provider=*/[&](int64_t) { return executable.get(); },
      /*argument_count_provider=*/[&](int64_t) { return argument_count; },
      /*argument_provider=*/
      [&](int64_t replica_idx, int64_t argument_idx) -> const Literal* {
        return arguments[replica_idx][argument_idx];
      },
      num_devices, /*run_hlo_passes=*/run_hlo_passes,
      /*device_assignment=*/device_assignment);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::Run(
    std::unique_ptr<HloModule> module, const bool run_hlo_passes,
    const std::function<void(HloModule*)>& test_preprocessor,
    BufferAssignmentProto* buffer_assignment_proto) {
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
      buffer_assignment_proto == nullptr
          ? test_runner_->Execute(std::move(module), fake_arguments,
                                  run_hlo_passes)
          : test_runner_->ExecuteWithBufferAssignment(
                std::move(module), buffer_assignment_proto, fake_arguments,
                run_hlo_passes);

  return swallow_execution_errors_ || output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().message();
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
  if (options.num_devices != replica_count) {
    return ::testing::AssertionFailure()
           << "Number of execution replicas is different from number of "
              "replicas in the module: requested number of replicas = "
           << options.num_devices
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
  if (!fake_arguments.ok()) {
    return ::testing::AssertionFailure() << fake_arguments.status();
  }

  return RunAndCompareTwoModulesReplicated(std::move(module_0),
                                           std::move(module_1), *fake_arguments,
                                           run_hlo_passes, use_threads, error);
}

::testing::AssertionResult
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const std::vector<Literal>& fake_arguments, const bool run_hlo_passes,
    const bool use_threads, const std::optional<ErrorSpec>& error) {
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices =
      module_0->config().replica_count() * module_0->config().num_partitions();
  options.arguments = LiteralUtil::MakePointers(fake_arguments);
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = use_threads;
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
    const absl::Span<const Literal* const> arguments,
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
  if (!fake_arguments.ok()) {
    return ::testing::AssertionFailure() << fake_arguments.status();
  }

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
    const absl::Span<const Literal* const> arguments,
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

::testing::AssertionResult
HloRunnerAgnosticTestBase::RunAndCompareTwoExecutables(
    OpaqueExecutable* executable_0, OpaqueExecutable* executable_1,
    const std::optional<ErrorSpec>& error) {
  absl::StatusOr<const HloModule*> module_0 =
      test_runner_->HloModuleFromWrapped(executable_0);
  if (!module_0.ok()) {
    return ::testing::AssertionFailure() << module_0.status();
  }
  absl::StatusOr<const HloModule*> module_1 =
      test_runner_->HloModuleFromWrapped(executable_1);
  if (!module_1.ok()) {
    return ::testing::AssertionFailure() << module_1.status();
  }
  if (const std::vector<int> mismatches =
          CompareInputs(*module_0.value(), *module_1.value());
      !mismatches.empty()) {
    return ::testing::AssertionFailure()
           << "Error : mismatching parameter shapes for parameters "
           << absl::StrJoin(mismatches, ", ");
  }
  absl::StatusOr<std::vector<Literal>> fake_arguments = MakeFakeArguments(
      /*module=*/module_0.value(), /*pseudo_random=*/true,
      /*use_large_range=*/false,
      /*treat_gte_as_data_formatting=*/false,
      /*max_bits_of_precision=*/std::nullopt);
  if (!fake_arguments.ok()) {
    return ::testing::AssertionFailure() << fake_arguments.status();
  }
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments.value(), std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  absl::StatusOr<::testing::AssertionResult> result =
      RunAndCompareTwoExecutablesInternal(executable_0, executable_1,
                                          fake_argument_ptrs, error);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return *result;
}

::testing::AssertionResult HloRunnerAgnosticTestBase::Run(
    const absl::string_view hlo_string, const bool run_hlo_passes,
    const tsl::protobuf::Message* backend_config, const bool use_random_data,
    BufferAssignmentProto* buffer_assignment_proto) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
      ParseAndReturnVerifiedModule(hlo_string);
  if (!module.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module.status().ToString();
  }
  const std::vector<Literal> fake_arguments =
      MakeFakeArguments(module->get(), use_random_data).value();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  if (backend_config) {
    // Set backend configuration if it is given.
    HloInstruction* instruction =
        (*module)->entry_computation()->root_instruction();
    absl::Status s = instruction->set_backend_config(*backend_config);
    return s.ok() ? ::testing::AssertionSuccess()
                  : ::testing::AssertionFailure() << s.message();
  }

  if (const absl::Status status = PreprocessModuleForTestRunner(module->get());
      !status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while preprocessing module: " << status;
  }

  const absl::StatusOr<Literal> output =
      buffer_assignment_proto == nullptr
          ? test_runner_->Execute(*std::move(module), fake_argument_ptrs,
                                  /*run_hlo_passes=*/run_hlo_passes)
          : test_runner_->ExecuteWithBufferAssignment(
                *std::move(module), buffer_assignment_proto, fake_argument_ptrs,
                /*run_hlo_passes=*/run_hlo_passes);
  return swallow_execution_errors_ || output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().message();
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunReplicated(
    const absl::string_view hlo_string, const bool run_hlo_passes,
    const int64_t num_devices, const tsl::protobuf::Message* backend_config) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
      ParseAndReturnVerifiedModule(hlo_string, /*num_replicas=*/num_devices,
                                   /*num_partitions=*/1);
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
  options.num_devices = num_devices;
  options.arguments = {fake_argument_ptrs.begin(), fake_argument_ptrs.end()};
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;
  if (const absl::Status status = PreprocessModuleForTestRunner(module->get());
      !status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while preprocessing module: " << status;
  }
  const absl::StatusOr<std::vector<Literal>> output =
      test_runner_->ExecuteReplicated(*std::move(module), std::move(options));
  if (swallow_execution_errors_ || output.ok()) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << output.status().message();
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunMultipleTimes(
    const absl::string_view hlo_string, const bool run_hlo_passes,
    const int64_t num_runs, const tsl::protobuf::Message* const backend_config,
    const bool assert_determinism) {
  std::vector<std::vector<Literal*>> fake_argument_ptrs(num_runs);
  std::vector<std::vector<Literal>> fake_arguments(num_runs);
  std::vector<std::unique_ptr<OpaqueExecutable>> executables(num_runs);

  for (int i = 0; i < num_runs; ++i) {
    absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
        ParseAndReturnVerifiedModule(hlo_string);
    if (!module.ok()) {
      return ::testing::AssertionFailure()
             << "Error while parsing HLO text format: "
             << module.status().ToString();
    }

    fake_arguments[i] = MakeFakeArguments(module->get()).value();

    if (backend_config) {
      // Set backend configuration if it is given.
      HloInstruction* instruction =
          (*module)->entry_computation()->root_instruction();
      absl::Status s = instruction->set_backend_config(*backend_config);
      return s.ok() ? ::testing::AssertionSuccess()
                    : ::testing::AssertionFailure() << s.message();
    }

    if (const absl::Status status =
            PreprocessModuleForTestRunner(module->get());
        !status.ok()) {
      return ::testing::AssertionFailure()
             << "Error while preprocessing module: " << status;
    }
    absl::StatusOr<std::unique_ptr<OpaqueExecutable>> executable =
        test_runner_->CreateExecutable(*std::move(module), run_hlo_passes);
    if (!executable.ok()) {
      return ::testing::AssertionFailure() << executable.status().message();
    }
    executables[i] = *std::move(executable);
  }

  std::optional<Literal> canonical_output;
  for (int i = 0; i < num_runs; ++i) {
    absl::StatusOr<Literal> output = test_runner_->ExecuteWithExecutable(
        executables[i].get(), fake_arguments[i]);
    if (!swallow_execution_errors_ && !output.ok()) {
      return ::testing::AssertionFailure() << output.status().message();
    }

    // Swallowing errors implies determinism.
    if (assert_determinism && !swallow_execution_errors_) {
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

absl::StatusOr<::testing::AssertionResult>
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesInternalReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const HloRunnerInterface::ReplicatedExecuteOptions options,
    const std::optional<ErrorSpec>& error) {
  TF_RETURN_IF_ERROR(PreprocessModuleForTestRunner(module_0.get()));
  TF_RETURN_IF_ERROR(PreprocessModuleForTestRunner(module_1.get()));
  TF_RETURN_IF_ERROR(verifier().Run(module_0.get()).status());
  TF_RETURN_IF_ERROR(verifier().Run(module_1.get()).status());

  // Execute the two modules.
  const absl::StatusOr<std::vector<Literal>> test_0 =
      test_runner_->ExecuteReplicated(std::move(module_0), options);
  if (!swallow_execution_errors_ && !test_0.ok()) {
    // Exit early if we aren't swallowing errors.
    return test_0.status();
  }
  const absl::StatusOr<std::vector<Literal>> test_1 =
      test_runner_->ExecuteReplicated(std::move(module_1), options);
  if (swallow_execution_errors_ && !test_0.ok()) {
    return ::testing::AssertionSuccess();
  }
  if (!test_1.ok()) {
    if (swallow_execution_errors_) {
      return ::testing::AssertionSuccess();
    }
    return test_1.status();
  }

  for (const auto& [expected, actual] : llvm::zip_equal(*test_0, *test_1)) {
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
    absl::Span<const Literal* const> arguments,
    const std::optional<ErrorSpec>& error, bool run_hlo_passes) {
  TF_RETURN_IF_ERROR(PreprocessModuleForTestRunner(module_0.get()));
  TF_RETURN_IF_ERROR(PreprocessModuleForTestRunner(module_1.get()));
  TF_RETURN_IF_ERROR(verifier().Run(module_0.get()).status());
  TF_RETURN_IF_ERROR(verifier().Run(module_1.get()).status());

  // Compile and execute the two modules. We compile both before running either
  // to allow caching to work better.
  TF_ASSIGN_OR_RETURN(
      const std::unique_ptr<OpaqueExecutable> executable_0,
      test_runner_->CreateExecutable(std::move(module_0), run_hlo_passes));
  TF_ASSIGN_OR_RETURN(
      const std::unique_ptr<OpaqueExecutable> executable_1,
      test_runner_->CreateExecutable(std::move(module_1), run_hlo_passes));

  return RunAndCompareTwoExecutablesInternal(
      executable_0.get(), executable_1.get(), arguments, error);
}

absl::StatusOr<::testing::AssertionResult>
HloRunnerAgnosticTestBase::RunAndCompareTwoExecutablesInternal(
    OpaqueExecutable* executable_0, OpaqueExecutable* executable_1,
    const absl::Span<const Literal* const> arguments,
    const std::optional<ErrorSpec>& error) {
  const absl::StatusOr<Literal> test_0 =
      test_runner_->ExecuteWithExecutable(executable_0, arguments);
  if (!swallow_execution_errors_ && !test_0.ok()) {
    // Exit early if we aren't swallowing errors.
    return test_0.status();
  }
  const absl::StatusOr<Literal> test_1 =
      test_runner_->ExecuteWithExecutable(executable_1, arguments);
  if (swallow_execution_errors_ && !test_0.ok()) {
    return ::testing::AssertionSuccess();
  }
  if (!test_1.ok()) {
    if (swallow_execution_errors_) {
      return ::testing::AssertionSuccess();
    }
    return test_1.status();
  }

  return LiteralTestUtil::NearOrEqual(/*expected=*/*test_0, /*actual=*/*test_1,
                                      error);
}

}  // namespace xla
