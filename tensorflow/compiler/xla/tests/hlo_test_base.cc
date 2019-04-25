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

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#include <memory>
#include <set>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

using absl::optional;
using absl::string_view;

constexpr char kInterpreter[] = "interpreter";

// Helper functions to get test and reference platforms.
se::Platform* GetReferencePlatform() {
  auto result = PlatformUtil::GetPlatform(kInterpreter);
  TF_CHECK_OK(result.status()) << "could not get interpreter platform";
  return result.ValueOrDie();
}

se::Platform* GetTestPlatform() {
  auto result = PlatformUtil::GetDefaultPlatform();
  TF_CHECK_OK(result.status()) << "could not get test platform";
  return result.ValueOrDie();
}

bool ProgramShapesEqual(const ProgramShape& lhs, const ProgramShape& rhs) {
  if (lhs.parameters_size() != rhs.parameters_size()) {
    return false;
  }
  for (int i = 0; i < lhs.parameters_size(); i++) {
    if (!ShapeUtil::Equal(lhs.parameters(i), rhs.parameters(i))) {
      return false;
    }
  }
  return ShapeUtil::Equal(lhs.result(), rhs.result());
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

Status VerifiedHloModule::Verify() {
  if (computation_count() == 0) {
    // The computation was never built. Nothing to verify.
    return Status::OK();
  }
  return verifier_.Run(this).status();
}

void VerifiedHloModule::VerifyOrAddFailure(const string& message) {
  Status status = Verify();
  if (!status.ok()) {
    ADD_FAILURE() << "HloVerifier failed on module " << name()
                  << (message.empty() ? "" : absl::StrCat(" (", message, ")"))
                  << ": " << status;
    LOG(ERROR) << "Contents of bad module:";
    XLA_LOG_LINES(tensorflow::ERROR, ToString());
  }
}

HloTestBase::HloTestBase(bool verifier_layout_sensitive,
                         bool allow_mixed_precision_in_hlo_verifier,
                         std::function<bool(const HloInstruction*)>
                             instruction_can_change_layout_func)
    : HloTestBase(GetTestPlatform(), GetReferencePlatform(),
                  verifier_layout_sensitive,
                  allow_mixed_precision_in_hlo_verifier,
                  instruction_can_change_layout_func) {}

HloTestBase::HloTestBase(se::Platform* test_platform,
                         se::Platform* reference_platform,
                         bool verifier_layout_sensitive,
                         bool allow_mixed_precision_in_hlo_verifier,
                         std::function<bool(const HloInstruction*)>
                             instruction_can_change_layout_func)
    : test_runner_(test_platform),
      reference_runner_(reference_platform),
      verifier_layout_sensitive_(verifier_layout_sensitive),
      allow_mixed_precision_in_hlo_verifier_(
          allow_mixed_precision_in_hlo_verifier) {
  hlo_verifier_ = absl::make_unique<HloVerifier>(
      /*layout_sensitive=*/verifier_layout_sensitive,
      /*allow_mixed_precision=*/allow_mixed_precision_in_hlo_verifier,
      instruction_can_change_layout_func);
}

std::unique_ptr<HloModule> HloTestBase::CreateNewUnverifiedModule(
    const string& name) {
  return absl::make_unique<HloModule>(name, GetModuleConfigForTest());
}

std::unique_ptr<VerifiedHloModule> HloTestBase::CreateNewVerifiedModule(
    const string& name) {
  return absl::make_unique<VerifiedHloModule>(
      name, GetModuleConfigForTest(), verifier_layout_sensitive_,
      allow_mixed_precision_in_hlo_verifier_,
      backend().compiler()->ShapeSizeBytesFunction());
}

StatusOr<std::unique_ptr<VerifiedHloModule>>
HloTestBase::ParseAndReturnVerifiedModule(absl::string_view hlo_text,
                                          const HloModuleConfig& config) {
  auto module = absl::make_unique<VerifiedHloModule>(
      TestName(), config, verifier_layout_sensitive_,
      allow_mixed_precision_in_hlo_verifier_,
      backend().compiler()->ShapeSizeBytesFunction());
  TF_RETURN_IF_ERROR(ParseHloString(hlo_text, module.get()));
  TF_RETURN_IF_ERROR(module->Verify());
  return std::move(module);
}

/* static */
StatusOr<bool> HloTestBase::RunHloPass(HloPassInterface* hlo_pass,
                                       HloModule* module) {
  const string module_str_before_run = module->ToProto().ShortDebugString();
  const auto status_or = hlo_pass->Run(module);
  if (status_or.status().ok()) {
    const string module_str_after_run = module->ToProto().ShortDebugString();
    if (!status_or.ValueOrDie()) {
      // Check that the proto remains same.
      EXPECT_EQ(module_str_after_run, module_str_before_run);
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

DebugOptions HloTestBase::GetDebugOptionsForTest() {
  auto debug_options = GetDebugOptionsFromFlags();
  // TODO(b/38354253): Change tests to use Parameters instead of Constants.
  debug_options.add_xla_disable_hlo_passes("constant_folding");
  debug_options.set_xla_gpu_max_kernel_unroll_factor(1);
  debug_options.set_xla_hlo_evaluator_use_fast_path(true);
  return debug_options;
}

StatusOr<Literal> HloTestBase::Execute(std::unique_ptr<HloModule> module,
                                       absl::Span<Literal* const> arguments) {
  return test_runner_.Execute(std::move(module), arguments);
}

Literal HloTestBase::ExecuteNoHloPasses(std::unique_ptr<HloModule> module,
                                        absl::Span<Literal* const> arguments) {
  return test_runner_
      .Execute(std::move(module), arguments,
               /*run_hlo_passes=*/false)
      .ValueOrDie();
}

Literal HloTestBase::ExecuteAndTransfer(std::unique_ptr<HloModule> module,
                                        absl::Span<Literal* const> arguments) {
  return test_runner_.Execute(std::move(module), arguments).ValueOrDie();
}

StatusOr<std::vector<Literal>> HloTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
    int64 num_replicas, bool use_threads) {
  HloRunner::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  for (auto argument : arguments) {
    options.arguments.push_back(argument);
  }
  return test_runner_.ExecuteReplicated(std::move(module), options,
                                        use_threads);
}

StatusOr<std::vector<Literal>> HloTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
    int64 num_replicas, DeviceAssignment* device_assignment,
    bool run_hlo_passes, bool use_threads) {
  HloRunner::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.run_hlo_passes = run_hlo_passes;
  for (auto argument : arguments) {
    options.arguments.push_back(argument);
  }
  return test_runner_.ExecuteReplicated(std::move(module), options,
                                        device_assignment, use_threads);
}

StatusOr<std::unique_ptr<HloModule>> HloTestBase::MakeReferenceModule(
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

StatusOr<::testing::AssertionResult> HloTestBase::RunAndCompareInternal(
    std::unique_ptr<HloModule> module,
    const absl::Span<Literal* const> arguments,
    const optional<ErrorSpec>& error, bool run_hlo_passes,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  TF_RETURN_IF_ERROR(hlo_verifier_->Run(module.get()).status());
  TF_ASSIGN_OR_RETURN(auto reference_module,
                      MakeReferenceModule(*module, reference_preprocessor));

  // Execute on two backends.
  TF_ASSIGN_OR_RETURN(
      auto test,
      test_runner_.Execute(std::move(module), arguments, run_hlo_passes));
  TF_ASSIGN_OR_RETURN(auto reference,
                      reference_runner_.Execute(std::move(reference_module),
                                                arguments, run_hlo_passes));
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
  return result.ValueOrDie();
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
  return result.ValueOrDie();
}

::testing::AssertionResult HloTestBase::RunAndCompare(
    std::unique_ptr<HloModule> module, const optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto fake_arguments = MakeFakeArguments(module.get()).ConsumeValueOrDie();

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
  const auto& fake_arguments =
      MakeFakeArguments(module.get()).ConsumeValueOrDie();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  return RunAndCompareNoHloPasses(std::move(module), fake_argument_ptrs, error,
                                  reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompare(
    string_view hlo_string, const absl::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto module_or_status =
      HloRunner::CreateModuleFromString(hlo_string, GetDebugOptionsForTest());
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }
  return RunAndCompare(module_or_status.ConsumeValueOrDie(), error,
                       reference_preprocessor);
}

::testing::AssertionResult HloTestBase::Run(string_view hlo_string,
                                            bool run_hlo_passes,
                                            ExecutionProfile* profile,
                                            string backend_config) {
  auto module_or_status =
      HloRunner::CreateModuleFromString(hlo_string, GetDebugOptionsForTest());
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }

  std::unique_ptr<HloModule> module = std::move(module_or_status.ValueOrDie());
  const auto& fake_arguments =
      MakeFakeArguments(module.get()).ConsumeValueOrDie();
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

  if (!backend_config.empty()) {
    // Set backend configuration if it is given.
    HloInstruction* instruction =
        module->entry_computation()->root_instruction();
    instruction->set_raw_backend_config_string(backend_config);
  }

  // return ::testing::AssertionSuccess();
  auto output = test_runner_.Execute(std::move(module), fake_argument_ptrs,
                                     /*run_hlo_passes=*/run_hlo_passes,
                                     /*profile=*/profile);

  return output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().error_message();
}

::testing::AssertionResult HloTestBase::RunMultipleTimes(
    string_view hlo_string, bool run_hlo_passes,
    std::vector<ExecutionProfile>* profiles, string backend_config) {
  int n = profiles->size();
  std::vector<std::vector<Literal*>> fake_argument_ptrs(n);
  std::vector<std::vector<Literal>> fake_arguments(n);
  std::vector<std::unique_ptr<Executable>> executables(n);

  for (int i = 0; i < n; ++i) {
    auto module_or_status =
        HloRunner::CreateModuleFromString(hlo_string, GetDebugOptionsForTest());
    if (!module_or_status.ok()) {
      return ::testing::AssertionFailure()
             << "Error while parsing HLO text format: "
             << module_or_status.status().ToString();
    }
    std::unique_ptr<HloModule> module =
        std::move(module_or_status.ValueOrDie());

    fake_arguments[i] = MakeFakeArguments(module.get()).ConsumeValueOrDie();
    absl::c_transform(
        fake_arguments[i], std::back_inserter(fake_argument_ptrs[i]),
        [](const Literal& literal) { return const_cast<Literal*>(&literal); });

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

    if (!backend_config.empty()) {
      // Set backend configuration if it is given.
      HloInstruction* instruction =
          module->entry_computation()->root_instruction();
      instruction->set_raw_backend_config_string(backend_config);
    }

    auto executable =
        test_runner_.CreateExecutable(std::move(module), run_hlo_passes);
    if (!executable.ok()) {
      return ::testing::AssertionFailure()
             << executable.status().error_message();
    }
    executables[i] = std::move(executable.ValueOrDie());
  }

  for (int i = 0; i < n; ++i) {
    auto output =
        test_runner_.Execute(std::move(executables[i]), fake_argument_ptrs[i],
                             /*profile=*/&((*profiles)[i]));
    if (!output.ok()) {
      return ::testing::AssertionFailure() << output.status().error_message();
    }
  }

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HloTestBase::RunAndCompareFromFile(
    const string& filename, const absl::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto module_or_status =
      HloRunner::ReadModuleFromHloTextFile(filename, GetDebugOptionsForTest());
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "failed reading hlo module from file";
  }
  return RunAndCompare(module_or_status.ConsumeValueOrDie(), error,
                       reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPasses(
    string_view hlo_string, const absl::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto module_or_status =
      HloRunner::CreateModuleFromString(hlo_string, GetDebugOptionsForTest());
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }
  return RunAndCompareNoHloPasses(module_or_status.ConsumeValueOrDie(), error,
                                  reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPassesFromFile(
    const string& filename, const absl::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto module_or_status =
      HloRunner::ReadModuleFromHloTextFile(filename, GetDebugOptionsForTest());
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "failed reading hlo module from file";
  }
  return RunAndCompareNoHloPasses(module_or_status.ConsumeValueOrDie(), error,
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

Backend& HloTestBase::backend() { return test_runner_.backend(); }

/* static */
string HloTestBase::TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

}  // namespace xla
