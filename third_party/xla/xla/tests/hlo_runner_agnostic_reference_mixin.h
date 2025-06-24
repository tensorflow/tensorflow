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

#ifndef XLA_TESTS_HLO_RUNNER_AGNOSTIC_REFERENCE_MIXIN_H_
#define XLA_TESTS_HLO_RUNNER_AGNOSTIC_REFERENCE_MIXIN_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {

ProgramShape GetProgramShapeWithLayout(const HloModule& module);

bool ProgramShapesEqual(const ProgramShape& lhs, const ProgramShape& rhs);

// This class is designed to be used as a mixin for tests that want to run
// against a reference implementation via a runner implementing
// HloRunnerInterface.
//
// The mixin requires that that the test class is a subclass of
// HloRunnerAgnosticTestBase.
template <typename T>
class HloRunnerAgnosticReferenceMixin : public T {
  static_assert(
      std::is_base_of_v<HloRunnerAgnosticTestBase, T>,
      "Mixin must be used with a subclass of HloRunnerAgnosticTestBase.");

 public:
  // A little helper to make sure that error messages are clear when the mixin
  // is not used correctly.
  using has_reference_runner_mixin = std::true_type;

 protected:
  template <typename... BaseArgs>
  explicit HloRunnerAgnosticReferenceMixin(
      absl_nonnull std::unique_ptr<HloRunnerInterface> reference_runner,
      BaseArgs&&... base_args)
      : T(std::forward<BaseArgs>(base_args)...),
        reference_runner_(std::move(reference_runner)) {}
  ~HloRunnerAgnosticReferenceMixin() override = default;

  // Executes the given hlo module on two backends and compares results.
  //
  // 'arguments': the input of the hlo module.
  //
  // 'error': if has value, expects the results to be near (within the error
  // bound). Otherwise, expects the results to be equal.
  //
  // 'reference_preprocessor': the module should be ready to run on the test
  // backend, but it might need to be tailored so that it is able to run on the
  // reference backend. Note that the program shape of the module must not be
  // modified.
  ::testing::AssertionResult RunAndCompare(
      std::unique_ptr<HloModule> module,
      absl::Span<const Literal* const> arguments,
      const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr) {
    const absl::StatusOr<::testing::AssertionResult> result =
        RunAndCompareInternal(std::move(module), arguments, error,
                              /*run_hlo_passes=*/true, reference_preprocessor,
                              test_preprocessor);
    if (!result.ok()) {
      return ::testing::AssertionFailure() << result.status();
    }
    return *result;
  }

  // Same as above, except that the module will be executed without Hlo
  // optimization.
  ::testing::AssertionResult RunAndCompareNoHloPasses(
      std::unique_ptr<HloModule> module,
      absl::Span<const Literal* const> arguments,
      const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr) {
    const absl::StatusOr<::testing::AssertionResult> result =
        RunAndCompareInternal(std::move(module), arguments, error,
                              /*run_hlo_passes=*/false, reference_preprocessor,
                              test_preprocessor);
    if (!result.ok()) {
      return ::testing::AssertionFailure() << result.status();
    }
    return *result;
  }

  // Executes an hlo module with fake inputs and compares the results.
  ::testing::AssertionResult RunAndCompare(
      std::unique_ptr<HloModule> module, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr,
      const std::optional<int64_t> args_max_bits_of_precision = std::nullopt) {
    const absl::StatusOr<std::vector<Literal>> fake_arguments =
        MakeFakeArguments(module.get(), /*pseudo_random=*/true,
                          /*use_large_range=*/false,
                          /*treat_gte_as_data_formatting=*/false,
                          args_max_bits_of_precision);
    if (!fake_arguments.ok()) {
      return ::testing::AssertionFailure() << fake_arguments.status().message();
    }
    return RunAndCompare(std::move(module),
                         LiteralUtil::MakePointers(*fake_arguments), error,
                         reference_preprocessor, test_preprocessor);
  }

  // Same as above, except that the module will be executed without Hlo
  // optimization.
  ::testing::AssertionResult RunAndCompareNoHloPasses(
      std::unique_ptr<HloModule> module, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr) {
    const absl::StatusOr<std::vector<Literal>> fake_arguments =
        MakeFakeArguments(module.get());
    if (!fake_arguments.ok()) {
      return ::testing::AssertionFailure() << fake_arguments.status().message();
    }
    return RunAndCompareNoHloPasses(
        std::move(module), LiteralUtil::MakePointers(*fake_arguments), error,
        reference_preprocessor, test_preprocessor);
  }

  // Convenient wrapper for executing and comparing an hlo module with fake
  // input. Module can be passed in directly, or parsed from an hlo_string,
  // or loaded from a file.
  ::testing::AssertionResult RunAndCompare(
      const absl::string_view hlo_string, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr,
      const std::optional<int64_t> args_max_bits_of_precision = std::nullopt) {
    absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
        this->ParseAndReturnVerifiedModule(hlo_string);
    if (!module.ok()) {
      return ::testing::AssertionFailure()
             << "Error while parsing HLO text format: "
             << module.status().ToString();
    }
    return RunAndCompare(*std::move(module), error, reference_preprocessor,
                         test_preprocessor, args_max_bits_of_precision);
  }

  ::testing::AssertionResult RunAndCompareNoHloPasses(
      const absl::string_view hlo_string, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr) {
    absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
        this->ParseAndReturnVerifiedModule(hlo_string);
    if (!module.ok()) {
      return ::testing::AssertionFailure()
             << "Error while parsing HLO text format: "
             << module.status().ToString();
    }
    return RunAndCompareNoHloPasses(*std::move(module), error,
                                    reference_preprocessor, test_preprocessor);
  }

  HloRunnerInterface& reference_runner() const { return *reference_runner_; }

 private:
  // Given the test module, makes a reference module that is ready to run on the
  // reference platform. This assumes that the given module is ready to run on
  // the test platform.
  absl::StatusOr<std::unique_ptr<HloModule>> MakeReferenceModule(
      const HloModule& test_module,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr) {
    std::unique_ptr<HloModule> reference_module = test_module.Clone();
    const ProgramShape program_shape = GetProgramShapeWithLayout(test_module);

    if (reference_preprocessor != nullptr) {
      reference_preprocessor(reference_module.get());
      if (!ProgramShapesEqual(program_shape,
                              GetProgramShapeWithLayout(*reference_module))) {
        return absl::InvalidArgumentError(
            "reference preprocessor must not modify the program shape");
      }
    }
    TF_RETURN_IF_ERROR(this->verifier().Run(reference_module.get()).status());
    return std::move(reference_module);
  }

  // Runs the module on two platforms with or without running hlo passes and
  // compares the results. Returns whether the results are near or equal. If any
  // error happens before the results are computed, returns the error status.
  absl::StatusOr<::testing::AssertionResult> RunAndCompareInternal(
      std::unique_ptr<HloModule> module,
      absl::Span<const Literal* const> arguments,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr) {
    TF_RETURN_IF_ERROR(this->verifier().Run(module.get()).status());
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> reference_module,
                        MakeReferenceModule(*module, reference_preprocessor));
    TF_RETURN_IF_ERROR(this->PreprocessModuleForTestRunner(module.get()));
    if (test_preprocessor != nullptr) {
      test_preprocessor(module.get());
    }
    // Execute on two backends.
    TF_ASSIGN_OR_RETURN(const Literal test,
                        this->test_runner().Execute(std::move(module),
                                                    arguments, run_hlo_passes));
    TF_ASSIGN_OR_RETURN(const Literal reference,
                        reference_runner_->Execute(std::move(reference_module),
                                                   arguments, run_hlo_passes));
    if (reference.IsAll(0)) {
      LOG(WARNING) << "Reference value is only zeros.";
    }

    return LiteralTestUtil::NearOrEqual(/*expected=*/reference, /*actual=*/test,
                                        error);
  }

  std::unique_ptr<HloRunnerInterface> reference_runner_;
};

}  // namespace xla

#endif  // XLA_TESTS_HLO_RUNNER_AGNOSTIC_REFERENCE_MIXIN_H_
