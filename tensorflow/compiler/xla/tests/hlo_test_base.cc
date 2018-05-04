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

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

using tensorflow::StringPiece;
using tensorflow::gtl::ArraySlice;
using tensorflow::gtl::optional;

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

HloTestBase::HloTestBase()
    : HloTestBase(GetTestPlatform(), GetReferencePlatform()) {}

HloTestBase::HloTestBase(se::Platform* test_platform,
                         se::Platform* reference_platform)
    : test_runner_(test_platform), reference_runner_(reference_platform) {
  hlo_verifier_ = MakeUnique<HloVerifier>(/*allow_mixed_precision=*/true);
}

/* static */
std::unique_ptr<HloModule> HloTestBase::CreateNewModule() {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  return MakeUnique<HloModule>(TestName(), VersionedComputationHandle(),
                               config);
}

/*static*/ DebugOptions HloTestBase::GetDebugOptionsForTest() {
  auto debug_options = legacy_flags::GetDebugOptionsFromFlags();
  // TODO(b/38354253): Change tests to use Parameters instead of Constants.
  debug_options.add_xla_disable_hlo_passes("constant_folding");
  return debug_options;
}

StatusOr<std::unique_ptr<Literal>> HloTestBase::Execute(
    std::unique_ptr<HloModule> module,
    tensorflow::gtl::ArraySlice<Literal*> arguments) {
  return test_runner_.Execute(std::move(module), arguments);
}

std::unique_ptr<Literal> HloTestBase::ExecuteNoHloPasses(
    std::unique_ptr<HloModule> module,
    tensorflow::gtl::ArraySlice<Literal*> arguments) {
  return test_runner_
      .Execute(std::move(module), arguments,
               /*run_hlo_passes=*/false)
      .ValueOrDie();
}

std::unique_ptr<Literal> HloTestBase::ExecuteAndTransfer(
    std::unique_ptr<HloModule> module,
    tensorflow::gtl::ArraySlice<Literal*> arguments) {
  return test_runner_.Execute(std::move(module), arguments).ValueOrDie();
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
    std::unique_ptr<HloModule> module, const ArraySlice<Literal*> arguments,
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
  return LiteralTestUtil::NearOrEqual(/*expected=*/*reference, /*actual=*/*test,
                                      error);
}

::testing::AssertionResult HloTestBase::RunAndCompare(
    std::unique_ptr<HloModule> module, const ArraySlice<Literal*> arguments,
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
    std::unique_ptr<HloModule> module, const ArraySlice<Literal*> arguments,
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
  const auto& fake_arguments =
      MakeFakeArguments(module.get()).ConsumeValueOrDie();

  std::vector<Literal*> fake_argument_ptrs;
  c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const std::unique_ptr<Literal>& literal) { return literal.get(); });

  return RunAndCompare(std::move(module), fake_argument_ptrs, error,
                       reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPasses(
    std::unique_ptr<HloModule> module, const optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  const auto& fake_arguments =
      MakeFakeArguments(module.get()).ConsumeValueOrDie();
  std::vector<Literal*> fake_argument_ptrs;
  c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const std::unique_ptr<Literal>& literal) { return literal.get(); });

  return RunAndCompareNoHloPasses(std::move(module), fake_argument_ptrs, error,
                                  reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompare(
    const StringPiece hlo_string,
    const tensorflow::gtl::optional<ErrorSpec>& error,
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

::testing::AssertionResult HloTestBase::RunAndCompareFromFile(
    const string& filename, const tensorflow::gtl::optional<ErrorSpec>& error,
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
    const StringPiece hlo_string,
    const tensorflow::gtl::optional<ErrorSpec>& error,
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
    const string& filename, const tensorflow::gtl::optional<ErrorSpec>& error,
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
                                             tensorflow::StringPiece name) {
  auto it = c_find_if(module->computations(),
                      [&](HloComputation* c) { return c->name() == name; });
  if (it == module->computations().end()) {
    return nullptr;
  }
  return *it;
}

HloInstruction* HloTestBase::FindInstruction(HloModule* module,
                                             tensorflow::StringPiece name) {
  for (const HloComputation* c : module->computations()) {
    auto it = c_find_if(c->instructions(),
                        [&](HloInstruction* i) { return i->name() == name; });
    if (it != c->instructions().end()) {
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
