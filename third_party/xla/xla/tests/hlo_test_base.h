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

#ifndef XLA_TESTS_HLO_TEST_BASE_H_
#define XLA_TESTS_HLO_TEST_BASE_H_

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/service/backend.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/new_hlo_test_base.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {

// A base class for tests which build and/or run HLO code. The class includes
// support for running an HLO module on two platforms and compare the results.
// This is a lower level of abstraction than using the client interface and
// enables, for one, explicitly building a graph of HLO instructions to run.
//
// This can also be used to write text/file-based test cases. Note that the test
// target is responsible for linking the needed backends. A convenient way to do
// this is to make it an xla_test: it will generate test targets linking with
// the respective backends, which will be used as the test backend; the
// interpreter backend is already linked with hlo_test_base so it will be the
// default reference backend. For example, if you want to compare both cpu vs.
// interpreter, and gpu vs. interpreter, you can:
//
//  xla_test (
//    name = "sample_text_test",
//    srcs = ["sample_text_test.cc"],
//    backends = [
//      "cpu",
//      "gpu",
//    ],
//    deps = [
//      "//xla/tests:hlo_test_base",
//      ...
//    ],
//  )
//
// For a more detailed example, see "../tests/sample_text_test.cc".
//
// ** NOTE **
// This class will soon be deprecated in favor of NewHloTestBase. We are in the
// process of incrementally migrating tests to use this new base class.
// HloTestBase remains as a shim on tests during this migration process. While
// we would prefer if you can avoid introducing new tests that use this class,
// we are still working on documenting the exact migration procedure.
class HloTestBase : public NewHloTestBase {
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

  using NewHloTestBase::ParseAndReturnVerifiedModule;

 protected:
  // This uses the interpreter backend as the reference backend and
  // automatically finds another supported backend as the test backend. If the
  // interpreter is the only supported backend, it will be both the test backend
  // and the reference backend.
  explicit HloTestBase(bool verifier_layout_sensitive = false,
                       bool allow_mixed_precision_in_hlo_verifier = true,
                       HloPredicate instruction_can_change_layout_func = {});

  // If your test doesn't use interpreter as the reference backend, you can use
  // this constructor. Note that your test target is responsible for linking in
  // both needed backends.
  HloTestBase(se::Platform* test_platform, se::Platform* reference_platform,
              bool verifier_layout_sensitive = false,
              bool allow_mixed_precision_in_hlo_verifier = true,
              HloPredicate instruction_can_change_layout_func = {});

  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[nodiscard]] [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  absl::StatusOr<std::vector<Literal>> ExecuteReplicatedWithHloRunner(
      Executable* executable,
      const HloRunnerInterface::ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment,
      ExecutionProfile* profile = nullptr) {
    return test_runner_as_hlo_runner().ExecuteReplicated(
        executable, options, device_assignment, profile);
  }

  [[nodiscard]] ::testing::AssertionResult RunAndCompareFromFile(
      const std::string& filename, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);
  [[nodiscard]] ::testing::AssertionResult RunAndCompareNoHloPassesFromFile(
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
  const HloRunner& test_runner_as_hlo_runner() const {
    return *static_cast<HloRunner*>(&test_runner());
  }
  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  HloRunner& test_runner_as_hlo_runner() {
    return *static_cast<HloRunner*>(&test_runner());
  }

  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  int64_t num_devices() {
    return backend().device_count();
  }

  absl::StatusOr<std::unique_ptr<HloRunnerInterface>> GetHloRunner();

  // Helper functions to get test and reference platforms.
  static se::Platform* GetReferencePlatform();
  static se::Platform* GetTestPlatform();

  // Creates or retrieves the allocator.
  se::DeviceMemoryAllocator* GetAllocator();

  ErrorSpec error_spec_{0.0001};

 private:
  se::Platform* test_platform_;
  std::unique_ptr<se::DeviceMemoryAllocator> allocator_;
};

#define SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(x)                      \
  int64_t num_devices = backend().device_count();                  \
  if (num_devices < x) {                                           \
    GTEST_SKIP() << "Test requires at least " << x << " devices (" \
                 << num_devices << " available)";                  \
  }

}  // namespace xla

#endif  // XLA_TESTS_HLO_TEST_BASE_H_
