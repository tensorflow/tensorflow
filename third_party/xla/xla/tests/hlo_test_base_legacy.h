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

#ifndef XLA_TESTS_HLO_TEST_BASE_LEGACY_H_
#define XLA_TESTS_HLO_TEST_BASE_LEGACY_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/service/backend.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_legacy.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_runner_agnostic_reference_mixin.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

class ABSL_DEPRECATED(
    "Please avoid introducing new tests that use this class. This class exists "
    "to support existing tests that cannot be easily migrated to use "
    "HloRunnerPjRt") HloTestBaseLegacy
    : public HloRunnerAgnosticReferenceMixin<HloRunnerAgnosticTestBase> {
 public:
  // Compiles the given `hlo` with optimizations, and verifies that optimized
  // HLO matches the given FileCheck pattern.
  void MatchOptimizedHlo(absl::string_view hlo, absl::string_view pattern,
                         bool print_operand_shape = false);

  // Like MatchOptimizedHlo, but checks operand shapes as well.
  void MatchOptimizedHloWithShapes(absl::string_view hlo,
                                   absl::string_view pattern) {
    MatchOptimizedHlo(hlo, pattern, /*print_operand_shape=*/true);
  }

  // Compiles and returns module with optimizations from a given HLO.
  absl::StatusOr<std::unique_ptr<HloModule>> GetOptimizedModule(
      absl::string_view hlo);

  absl::StatusOr<std::unique_ptr<HloModule>> GetOptimizedModule(
      std::unique_ptr<HloModule> hlo_module);

  using HloRunnerAgnosticTestBase::ParseAndReturnVerifiedModule;

 protected:
  // This uses the interpreter backend as the reference backend and
  // automatically finds another supported backend as the test backend. If the
  // interpreter is the only supported backend, it will be both the test backend
  // and the reference backend.
  explicit HloTestBaseLegacy(
      bool verifier_layout_sensitive = false,
      bool allow_mixed_precision_in_hlo_verifier = true,
      HloPredicate instruction_can_change_layout_func = {},
      absl::FunctionRef<std::unique_ptr<HloRunnerLegacy>(se::Platform*)>
          runner_factory = DefaultRunnerFactory);

  // If your test doesn't use interpreter as the reference backend, you can use
  // this constructor. Note that your test target is responsible for linking in
  // both needed backends.
  HloTestBaseLegacy(
      se::Platform* test_platform, se::Platform* reference_platform,
      bool verifier_layout_sensitive = false,
      bool allow_mixed_precision_in_hlo_verifier = true,
      HloPredicate instruction_can_change_layout_func = {},
      absl::FunctionRef<std::unique_ptr<HloRunnerLegacy>(se::Platform*)>
          runner_factory = DefaultRunnerFactory);

  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[nodiscard]] [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  absl::StatusOr<std::vector<Literal>> ExecuteReplicatedWithHloRunner(
      OpaqueExecutable* executable,
      const HloRunnerInterface::ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) {
    return test_runner_as_hlo_runner().ExecuteReplicated(
        executable, options, device_assignment, nullptr);
  }

  ::testing::AssertionResult RunAndCompareFromFile(
      const std::string& filename, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);
  ::testing::AssertionResult RunAndCompareNoHloPassesFromFile(
      const std::string& filename, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);

  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  const Backend& backend() const {
    return test_runner_as_hlo_runner().backend();
  }
  // Returns the backend owned by the test runner.
  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  Backend& backend() {
    return test_runner_as_hlo_runner().backend();
  }

  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  const HloRunnerLegacy& test_runner_as_hlo_runner() const {
    return *static_cast<HloRunnerLegacy*>(&test_runner());
  }
  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  HloRunnerLegacy& test_runner_as_hlo_runner() {
    return *static_cast<HloRunnerLegacy*>(&test_runner());
  }

  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  int64_t num_devices() {
    return backend().device_count();
  }

  // Helper functions to get test and reference platforms.
  static se::Platform* GetReferencePlatform();
  static se::Platform* GetTestPlatform();

  // Creates or retrieves the allocator.
  se::DeviceAddressAllocator* GetAllocator();

  ErrorSpec error_spec_{0.0001};

 private:
  HloTestBaseLegacy(
      std::tuple<std::unique_ptr<HloRunnerInterface>,
                 HloRunnerAgnosticTestBase::DeviceShapeRepresentationFn,
                 HloRunnerAgnosticTestBase::DeviceShapeSizeFn>
          test_runner_and_functions,
      std::unique_ptr<HloRunnerInterface> reference_runner,
      bool verifier_layout_sensitive,
      bool allow_mixed_precision_in_hlo_verifier,
      HloPredicate instruction_can_change_layout_func);

  static std::unique_ptr<HloRunnerLegacy> DefaultRunnerFactory(
      se::Platform* platform) {
    return std::make_unique<HloRunnerLegacy>(platform);
  }

  std::unique_ptr<se::DeviceAddressAllocator> allocator_;
};

}  // namespace xla

#endif  // XLA_TESTS_HLO_TEST_BASE_LEGACY_H_
