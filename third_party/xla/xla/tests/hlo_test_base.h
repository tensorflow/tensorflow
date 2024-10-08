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
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/test_utils/hlo_hardware_independent_test_base.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/service/backend.h"
#include "xla/service/computation_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/platform_util.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/verified_hlo_module.h"
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
class HloTestBase : public HloHardwareIndependentTestBase {
 public:
  // Like CreateNewUnverifiedModule, except the HloModule returned here runs the
  // HLO verifier on destruction.
  std::unique_ptr<VerifiedHloModule> CreateNewVerifiedModule(
      const std::string& name = TestName(), int64_t replica_count = 1);

  // Parses the given string and returns module as a VerifiedHloModule.
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
  ParseAndReturnVerifiedModule(absl::string_view hlo_text,
                               int64_t replica_count = 1,
                               int64_t num_partitions = 1);
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
  ParseAndReturnVerifiedModule(absl::string_view hlo_text,
                               const HloModuleConfig& config);

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

  ~HloTestBase() override = default;

  // Executes the given module and return the result as a Literal.
  absl::StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                                  absl::Span<Literal* const> arguments,
                                  bool run_hlo_passes = true);

  // Same as above, except the module will be executed without running any HLO
  // passes on it.
  Literal ExecuteNoHloPasses(std::unique_ptr<HloModule> module,
                             absl::Span<Literal* const> arguments);

  Literal ExecuteAndTransfer(std::unique_ptr<HloModule> module,
                             absl::Span<Literal* const> arguments);

  // Compile the given module to an executable.
  absl::StatusOr<std::unique_ptr<Executable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) {
    return runner_->CreateExecutable(std::move(module), run_hlo_passes);
  }

  // Executes the given module on multiple replicas.
  //
  // use_threads indicates whether this replicated computation will be executed
  // with a thread-per-replica, vs using an implicitly async call such as
  // Executable::ExecuteOnStreams.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
      int64_t num_replicas, bool use_threads, bool run_hlo_passes = false);

  // Same as above, but uses specified device assignment.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
      int64_t num_replicas, DeviceAssignment* device_assignment,
      bool run_hlo_passes, bool use_threads);

  // Same as above, but allows passing different programs for replicas.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::function<Executable*(int64_t)> executable_provider,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      int64_t num_replicas, bool run_hlo_passes,
      DeviceAssignment* device_assignment = nullptr);

  // Convenience function for above. Allows passing different inputs to
  // different replicas of the same program.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      std::vector<std::vector<Literal*>> arguments, int64_t num_replicas,
      bool run_hlo_passes);

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
  [[nodiscard]] ::testing::AssertionResult RunAndCompare(
      std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);

  // Same as above, except that the module will be executed without Hlo
  // optimization.
  [[nodiscard]] ::testing::AssertionResult RunAndCompareNoHloPasses(
      std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr);

  // Executes an hlo module with fake inputs and compares the results.
  [[nodiscard]] ::testing::AssertionResult RunAndCompare(
      std::unique_ptr<HloModule> module, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      std::optional<int64_t> args_max_bits_of_precision = std::nullopt);

  // Same as above, except that the module will be executed without Hlo
  // optimization.
  [[nodiscard]] ::testing::AssertionResult RunAndCompareNoHloPasses(
      std::unique_ptr<HloModule> module, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr);

  // Executes an hlo module with fake inputs and checks that the execution is
  // successful.
  [[nodiscard]] ::testing::AssertionResult Run(
      std::unique_ptr<HloModule> module, bool run_hlo_passes);

  // Convenient wrappers for executing and comparing an hlo module with fake
  // input. Module can be passed in directly, or parsed from an hlo_string,
  // or loaded from a file.
  [[nodiscard]] ::testing::AssertionResult RunAndCompare(
      absl::string_view hlo_string, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      std::optional<int64_t> args_max_bits_of_precision = std::nullopt);
  [[nodiscard]] ::testing::AssertionResult Run(
      absl::string_view hlo_string, bool run_hlo_passes = true,
      ExecutionProfile* profile = nullptr,
      const tsl::protobuf::Message* backend_config = nullptr,
      bool use_random_data = true);

  // Same as below, except that it requires all the options to be passed.
  ::testing::AssertionResult RunAndCompareTwoModulesReplicated(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      HloRunner::ReplicatedExecuteOptions options,
      const std::optional<ErrorSpec>& error);

  // Same as below, except that it requires the parsed modules to be passed.
  ::testing::AssertionResult RunAndCompareTwoModulesReplicated(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      bool run_hlo_passes, bool use_threads,
      const std::optional<ErrorSpec>& error);

  // Parses the modules, and executes them based on `run_hlo_passes` and
  // `use_threads` flags. The replica count should be mentioned in the module
  // itself.
  ::testing::AssertionResult RunAndCompareTwoModulesReplicated(
      absl::string_view module_0, absl::string_view module_1,
      bool run_hlo_passes, bool use_threads,
      const std::optional<ErrorSpec>& error);

  // Same as below, except requires passing fake arguments.
  ::testing::AssertionResult RunAndCompareTwoModules(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes = true);

  // Same as below, except requires passing the modules.
  ::testing::AssertionResult RunAndCompareTwoModules(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes = true,
      std::optional<int64_t> args_max_bits_of_precision = std::nullopt);

  // Convenient wrapper for executing and comparing results of two hlo modules
  // with fake input. By default compares unoptimized modules. If the modules
  // are already optimized, set |run_hlo_passes| to false.
  ::testing::AssertionResult RunAndCompareTwoModules(
      absl::string_view hlo_string_module_0,
      absl::string_view hlo_string_module_1,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes = true,
      std::optional<int64_t> args_max_bits_of_precision = std::nullopt);

  // Same as above but allows running with different configs.
  ::testing::AssertionResult RunAndCompareTwoModules(
      absl::string_view hlo_string_module_0,
      absl::string_view hlo_string_module_1, const HloModuleConfig& config_0,
      const HloModuleConfig& config_1, const std::optional<ErrorSpec>& error,
      bool run_hlo_passes = true,
      std::optional<int64_t> args_max_bits_of_precision = std::nullopt);

  // Same as above but requires explicit arguments.
  ::testing::AssertionResult RunAndCompareTwoModules(
      absl::string_view hlo_string_module_0,
      absl::string_view hlo_string_module_1,
      absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes = true);

  // Executes an hlo module with fake inputs on multiple replicas.
  [[nodiscard]] ::testing::AssertionResult RunReplicated(
      absl::string_view hlo_string, bool run_hlo_passes = true,
      int64_t num_replicas = 1,
      const tsl::protobuf::Message* backend_config = nullptr);

  // If assert_determinism is true, the assertion will fail unless all runs
  // produce exactly the same output.
  [[nodiscard]] ::testing::AssertionResult RunMultipleTimes(
      absl::string_view hlo_string, bool run_hlo_passes,
      std::vector<ExecutionProfile>* profiles,
      const tsl::protobuf::Message* backend_config = nullptr,
      bool assert_determinism = false);
  [[nodiscard]] ::testing::AssertionResult RunAndCompareFromFile(
      const std::string& filename, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);
  [[nodiscard]] ::testing::AssertionResult RunAndCompareNoHloPasses(
      absl::string_view hlo_string, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr);
  [[nodiscard]] ::testing::AssertionResult RunAndCompareNoHloPassesFromFile(
      const std::string& filename, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);

  // Returns the backend owned by the test runner.
  Backend& backend();
  const Backend& backend() const;

  int64_t num_devices() { return backend().device_count(); }

  HloRunner test_runner_;
  HloRunner reference_runner_;

  ErrorSpec error_spec_{0.0001};

  HloComputation* AddEntryComputationAndUpdateEntryComputationLayout(
      HloModule*, std::unique_ptr<HloComputation> computation);
  void UpdateEntryComputationLayout(HloModule* module);

  absl::StatusOr<std::unique_ptr<HloRunnerInterface>> GetHloRunner();

 protected:
  // Helper functions to get test and reference platforms.
  static se::Platform* GetReferencePlatform();
  static se::Platform* GetTestPlatform();

  // Creates or retrieves the allocator.
  se::DeviceMemoryAllocator* GetAllocator();

 private:
  // Either an HloRunner or HloRunnerPjRt depending on if ShouldUsePjRt()
  std::unique_ptr<HloRunnerInterface> runner_;
  se::Platform* test_platform_;
  std::unique_ptr<se::DeviceMemoryAllocator> allocator_;

  // Given the test module, makes a reference module that is ready to run on the
  // reference platform. This assumes that the given module is ready to run on
  // the test platform.
  absl::StatusOr<std::unique_ptr<HloModule>> MakeReferenceModule(
      const HloModule& test_module,
      const std::function<void(HloModule*)>& reference_preprocessor);

  // Runs the module on two platforms with or without running hlo passes and
  // compares the results. Returns whether the results are near or equal. If any
  // error happens before the results are computed, returns the error status.
  absl::StatusOr<::testing::AssertionResult> RunAndCompareInternal(
      std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes,
      const std::function<void(HloModule*)>& reference_preprocessor,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr);

  // Runs the two module with or without running hlo passes and compares
  // the results. Returns whether the results are near or equal. If any
  // error happens before the results are computed, returns the error status.
  absl::StatusOr<::testing::AssertionResult>
  RunAndCompareTwoModulesInternalReplicated(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      HloRunner::ReplicatedExecuteOptions options,
      const std::optional<ErrorSpec>& error);

  // Runs the two module on with or without running hlo passes and
  // compares the results. Returns whether the results are near or equal. If any
  // error happens before the results are computed, returns the error status.
  absl::StatusOr<::testing::AssertionResult> RunAndCompareTwoModulesInternal(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes);

  // Returns either an HloRunner or HloRunnerPjRt implementation depending if
  // there exists a registered PjRtClientFactory.
  absl::StatusOr<std::unique_ptr<HloRunnerInterface>> GetHloRunnerForTest(
      se::Platform* test_platform);
};

#define SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(x)                      \
  int64_t num_devices = backend().device_count();                  \
  if (num_devices < x) {                                           \
    GTEST_SKIP() << "Test requires at least " << x << " devices (" \
                 << num_devices << " available)";                  \
  }

}  // namespace xla

#endif  // XLA_TESTS_HLO_TEST_BASE_H_
