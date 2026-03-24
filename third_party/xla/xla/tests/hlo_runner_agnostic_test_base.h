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

#ifndef XLA_TESTS_HLO_RUNNER_AGNOSTIC_TEST_BASE_H_
#define XLA_TESTS_HLO_RUNNER_AGNOSTIC_TEST_BASE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

struct HloRunnerAgnosticTestBaseOptions {
  bool verifier_layout_sensitive = false;
  bool allow_mixed_precision_in_hlo_verifier = true;
  HloPredicate instruction_can_change_layout_func;
  // If true, execution errors (any non-OK absl::StatusOr originating from the
  // test runner) are swallowed. This only applies to the Run* methods, as these
  // do not return literals themselves.
  bool swallow_execution_errors = false;
};

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
//      "//xla/tests:hlo_runner_agnostic_test_base",
//      ...
//    ],
//  )
//
// Unlike HloTestBase, which relies on StreamExecutor via HloRunner, this class
// relies on HloRunnerInterface. HloRunnerInterface supports HloRunner among
// other implementations. We plan to incrementally migrate tests to this class
// and away from HloTestBase.
class HloRunnerAgnosticTestBase : public HloHardwareIndependentTestBase {
 public:
  using DeviceShapeRepresentationFn = std::function<Shape(const Shape&)>;
  using DeviceShapeSizeFn = std::function<int64_t(const Shape&)>;

  static constexpr ErrorSpec kDefaultErrorSpec{0.0001};

 protected:
  // Preferred constructor, has more options.
  explicit HloRunnerAgnosticTestBase(
      absl_nonnull std::unique_ptr<HloRunnerInterface> test_runner,
      DeviceShapeRepresentationFn device_shape_representation_fn,
      DeviceShapeSizeFn device_shape_size_fn,
      HloRunnerAgnosticTestBaseOptions options = {});
  // Legacy constructor with old defaults. Do not add new options.
  explicit HloRunnerAgnosticTestBase(
      absl_nonnull std::unique_ptr<HloRunnerInterface> test_runner,
      DeviceShapeRepresentationFn device_shape_representation_fn,
      DeviceShapeSizeFn device_shape_size_fn,
      bool verifier_layout_sensitive = false,
      bool allow_mixed_precision_in_hlo_verifier = true,
      HloPredicate instruction_can_change_layout_func = {});

  // Creates a new HLO module for a test. The module created will have
  // TestName() for its name; it will also automatically populate its debug
  // options from command-line flags. If you want a fresh HloModule object and
  // then add HloComputations to it, it's recommended to use this method in your
  // tests.
  //
  // This returns a VerifiedHloModule that runs the HLO verifier on
  // destruction.
  std::unique_ptr<VerifiedHloModule> CreateNewVerifiedModule(
      const std::string& name = TestName(), int64_t replica_count = 1);

  // Parses the given string and returns module as a VerifiedHloModule.
  using HloHardwareIndependentTestBase::ParseAndReturnVerifiedModule;

  // To obtain a HloModuleConfig with a specific replica and partition count and
  // no further customization, either use the overload above or use
  // GetModuleConfigForTest. The latter option may be useful if you want to pass
  // custom HloParserOptions as well.
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
  ParseAndReturnVerifiedModule(
      absl::string_view hlo_text, const HloModuleConfig& config,
      const HloParserOptions& parser_options = HloParserOptions()) const;

  // Builds an HLO module from the given XlaComputation using the given
  // execution options.
  absl::StatusOr<std::unique_ptr<HloModule>> HloModuleFromXlaComputation(
      const XlaComputation& computation,
      const ExecutionOptions& execution_options) const;

  // Builds an HLO module from the given XlaBuilder using the given
  // execution options.
  absl::StatusOr<std::unique_ptr<HloModule>> HloModuleFromXlaBuilder(
      XlaBuilder* builder, const ExecutionOptions& execution_options) const;

  HloComputation* AddEntryComputationAndUpdateEntryComputationLayout(
      HloModule*, std::unique_ptr<HloComputation> computation);
  void UpdateEntryComputationLayout(HloModule* module) const;

  // Executes the given module and return the result as a Literal.
  absl::StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                                  absl::Span<const Literal* const> arguments,
                                  bool run_hlo_passes = true);

  // Compile the given module to an executable.
  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) {
    return test_runner_->CreateExecutable(std::move(module), run_hlo_passes);
  }

  // Parse the given hlo_text and compile the resulting module to an executable.
  // Returns the HloModule and the corresponding executable that owns it.
  absl::StatusOr<std::pair<const HloModule*, std::unique_ptr<OpaqueExecutable>>>
  GetOptimizedModuleForExecutable(absl::string_view hlo_text,
                                  const HloModuleConfig& config) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                        ParseAndReturnVerifiedModule(hlo_text, config));
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<OpaqueExecutable> executable,
        CreateExecutable(std::move(module), /*run_hlo_passes=*/true));
    TF_ASSIGN_OR_RETURN(const HloModule* optimized_module,
                        test_runner_->HloModuleFromWrapped(executable.get()));
    return {{optimized_module, std::move(executable)}};
  }

  // Compiles the given hlo_text to an executable, and returns a clone of the
  // optimized HloModule.
  absl::StatusOr<std::unique_ptr<HloModule>> GetOptimizedModule(
      absl::string_view hlo_text, const HloModuleConfig& config) {
    TF_ASSIGN_OR_RETURN(auto module_and_executable,
                        GetOptimizedModuleForExecutable(hlo_text, config));
    return module_and_executable.first->Clone();
  }

  absl::StatusOr<std::unique_ptr<HloModule>> GetOptimizedModule(
      absl::string_view hlo_text) {
    return GetOptimizedModule(hlo_text, GetModuleConfigForTest());
  }

  void MatchOptimizedHlo(absl::string_view hlo, absl::string_view pattern,
                         bool print_operand_shape = false) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo));
    HloPrintOptions print_opts;
    print_opts.set_print_operand_shape(print_operand_shape);
    absl::StatusOr<bool> filecheck_result =
        RunFileCheck(optimized_module->ToString(print_opts), pattern);
    TF_ASSERT_OK(filecheck_result.status());
    EXPECT_TRUE(filecheck_result.value());
  }

  // Executes the given module on multiple devices.
  //
  // use_threads indicates whether this replicated computation will be executed
  // with a thread-per-device, vs using an implicitly async call such as
  // Executable::ExecuteOnStreams.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      absl::Span<const Literal* const> arguments, int64_t num_devices,
      bool use_threads, bool run_hlo_passes = false);

  // Same as above, but uses specified device assignment.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      absl::Span<const Literal* const> arguments, int64_t num_devices,
      DeviceAssignment* device_assignment, bool run_hlo_passes,
      bool use_threads);

  // Same as above, but allows passing different programs for devices.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      absl::AnyInvocable<OpaqueExecutable*(int64_t)> executable_provider,
      absl::AnyInvocable<int64_t(int64_t)> argument_count_provider,
      absl::AnyInvocable<const Literal*(int64_t, int64_t)> argument_provider,
      int64_t num_devices, bool run_hlo_passes,
      DeviceAssignment* device_assignment = nullptr);

  // Convenience function for above. Allows passing different inputs to
  // different devices of the same program.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      std::vector<std::vector<Literal*>> arguments, int64_t num_devices,
      bool run_hlo_passes, DeviceAssignment* device_assignment = nullptr);

  // Executes an hlo module with fake inputs and checks that the execution is
  // successful.
  ::testing::AssertionResult Run(
      std::unique_ptr<HloModule> module, bool run_hlo_passes,
      const std::function<void(HloModule*)>& test_preprocessor = nullptr,
      BufferAssignmentProto* buffer_assignment_proto = nullptr);

  // Convenient wrapper for executing and comparing an hlo module with fake
  // input. Module can be passed in directly, or parsed from an hlo_string,
  // or loaded from a file.
  ::testing::AssertionResult Run(
      absl::string_view hlo_string, bool run_hlo_passes = true,
      const tsl::protobuf::Message* backend_config = nullptr,
      bool use_random_data = true,
      BufferAssignmentProto* buffer_assignment_proto = nullptr);

  // Same as below, except that it requires all the options to be passed.
  ::testing::AssertionResult RunAndCompareTwoModulesReplicated(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      HloRunnerInterface::ReplicatedExecuteOptions options,
      const std::optional<ErrorSpec>& error);

  // Same as below, except that it requires the parsed modules to be passed.
  ::testing::AssertionResult RunAndCompareTwoModulesReplicated(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      bool run_hlo_passes, bool use_threads,
      const std::optional<ErrorSpec>& error);

  ::testing::AssertionResult RunAndCompareTwoModulesReplicated(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      const std::vector<Literal>& fake_arguments, bool run_hlo_passes,
      bool use_threads, const std::optional<ErrorSpec>& error);

  // Parses the modules, and executes them based on `run_hlo_passes` and
  // `use_threads` flags. The replica + partition count should be set in the
  // module itself.
  ::testing::AssertionResult RunAndCompareTwoModulesReplicated(
      absl::string_view module_0_str, absl::string_view module_1_str,
      bool run_hlo_passes, bool use_threads,
      const std::optional<ErrorSpec>& error);

  // Same as below, except requires passing fake arguments.
  ::testing::AssertionResult RunAndCompareTwoModules(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      absl::Span<const Literal* const> arguments,
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
      absl::Span<const Literal* const> arguments,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes = true);

  // Executes the two executables using fake arguments and compares the results.
  // Returns whether the results are near or equal.
  ::testing::AssertionResult RunAndCompareTwoExecutables(
      OpaqueExecutable* executable_0, OpaqueExecutable* executable_1,
      const std::optional<ErrorSpec>& error);

  // Executes an hlo module with fake inputs on multiple devices.
  ::testing::AssertionResult RunReplicated(
      absl::string_view hlo_string, bool run_hlo_passes = true,
      int64_t num_devices = 1,
      const tsl::protobuf::Message* backend_config = nullptr);

  // If assert_determinism is true, the assertion will fail unless all runs
  // produce exactly the same output.
  ::testing::AssertionResult RunMultipleTimes(
      absl::string_view hlo_string, bool run_hlo_passes, int64_t num_runs,
      const tsl::protobuf::Message* backend_config = nullptr,
      bool assert_determinism = false);

  // Override this method to add a default preprocessing step that is applied to
  // the test module in all Run* methods. The intended usecase for this is to
  // adapt existing test cases to be compatible with runners that don't support
  // certain features. Does nothing and returns OK by default.
  //
  // This method is called before any additional preprocessing steps performed
  // by the optional `test_preprocessor` argument.
  virtual absl::Status PreprocessModuleForTestRunner(HloModule* module) const {
    return absl::OkStatus();
  }

  HloRunnerInterface& test_runner() const { return *test_runner_; }
  bool swallow_execution_errors() const { return swallow_execution_errors_; }

 private:
  // Runs the two module with or without running hlo passes and compares
  // the results. Returns whether the results are near or equal. If any
  // error happens before the results are computed, returns the error status.
  absl::StatusOr<::testing::AssertionResult>
  RunAndCompareTwoModulesInternalReplicated(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      HloRunnerInterface::ReplicatedExecuteOptions options,
      const std::optional<ErrorSpec>& error);

  // Runs the two module on with or without running hlo passes and
  // compares the results. Returns whether the results are near or equal. If any
  // error happens before the results are computed, returns the error status.
  absl::StatusOr<::testing::AssertionResult> RunAndCompareTwoModulesInternal(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      absl::Span<const Literal* const> arguments,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes);

  // Executes the two executables and compares the results. Returns whether the
  // results are near or equal. If any error happens before the results are
  // computed, returns the error status.
  absl::StatusOr<::testing::AssertionResult>
  RunAndCompareTwoExecutablesInternal(
      OpaqueExecutable* executable_0, OpaqueExecutable* executable_1,
      absl::Span<const Literal* const> arguments,
      const std::optional<ErrorSpec>& error);

  std::unique_ptr<HloRunnerInterface> test_runner_;
  DeviceShapeRepresentationFn device_shape_representation_fn_;
  DeviceShapeSizeFn device_shape_size_fn_;
  bool swallow_execution_errors_ = false;
};

}  // namespace xla

#endif  // XLA_TESTS_HLO_RUNNER_AGNOSTIC_TEST_BASE_H_
