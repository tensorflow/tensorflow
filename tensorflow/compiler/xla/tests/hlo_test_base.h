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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_HLO_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_HLO_TEST_BASE_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module_group.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/manifest_checking_test.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/test.h"

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
//      "//third_party/tensorflow/compiler/xla/tests:hlo_test_base",
//      ...
//    ],
//  )
//
// For a more detailed example, see "../tests/sample_text_test.cc".
class HloTestBase : public ManifestCheckingTest {
 public:
  // Creates a new HLO module for a test. The module created will have
  // TestName() for its name; it will also automatically populate its debug
  // options from command-line flags. If you want a fresh HloModule object and
  // then add HloComputations to it, it's recommended to use this method in your
  // tests.
  //
  // This returns a vanilla HloModule that doesn't run the HLO verifier on
  // destruction.
  ABSL_DEPRECATED("Use CreateNewVerifiedModule instead.")
  std::unique_ptr<HloModule> CreateNewUnverifiedModule(
      const std::string& name = TestName());

  // Like CreateNewUnverifiedModule, except the HloModule returned here runs the
  // HLO verifier on destruction.
  std::unique_ptr<VerifiedHloModule> CreateNewVerifiedModule(
      const std::string& name = TestName(), int64_t replica_count = 1);

  // Parses the given string and returns module as a VerifiedHloModule.
  StatusOr<std::unique_ptr<VerifiedHloModule>> ParseAndReturnVerifiedModule(
      absl::string_view hlo_text, int64_t replica_count = 1,
      int64_t num_partitions = 1);
  StatusOr<std::unique_ptr<VerifiedHloModule>> ParseAndReturnVerifiedModule(
      absl::string_view hlo_text, const HloModuleConfig& config);

  // Runs the hlo_pass with the provided module and returns the result. This
  // function also verifies that the module remains unchanged when hlo_pass
  // returns false as the StatusOr value.
  //
  // These three overloads all do the same thing.  The && overload lets you do
  // `RunHloPass(MyPass(), module)` all in one line.  The reason for the
  // overload that takes a pointer is that, at one point in the past, non-const
  // lvalue references were banned in Google code.
  static StatusOr<bool> RunHloPass(HloPassInterface* hlo_pass,
                                   HloModule* module);
  static StatusOr<bool> RunHloPass(HloPassInterface& hlo_pass,
                                   HloModule* module) {
    return RunHloPass(&hlo_pass, module);
  }
  static StatusOr<bool> RunHloPass(HloPassInterface&& hlo_pass,
                                   HloModule* module) {
    return RunHloPass(&hlo_pass, module);
  }

  // Runs the hlo_pass with the provided module group and returns the result.
  // This method runs the input HLO module group pass for a `HloModuleGroup` and
  // it also verifies the module group remains unchanged when hlo_pass returns
  // false as the StatusOr value.
  static StatusOr<bool> RunHloPass(HloPassInterface&& hlo_pass,
                                   HloModuleGroup* module_group);

  static PrecisionConfig DefaultPrecisionConfig(int operands);

  // Sets most fath math options to be enabled to model the fast math flags
  // generally used for CPU:AOT compilation.
  static void SetAotFastMathDebugOptions(DebugOptions* options);

  // Compiles the given `hlo` with optimizations, and verifies that optimized
  // HLO matches the given FileCheck pattern.
  void MatchOptimizedHlo(absl::string_view hlo, absl::string_view pattern,
                         bool print_operand_shape = false);

  // LikeMatchOptimizedHlo, but checks operand shapes as well.
  void MatchOptimizedHloWithShapes(absl::string_view hlo,
                                   absl::string_view pattern) {
    MatchOptimizedHlo(hlo, pattern, /*print_operand_shape=*/true);
  }

  // Compiles and returns module with optimizations from a given HLO.
  StatusOr<std::unique_ptr<HloModule>> GetOptimizedModule(
      absl::string_view hlo);

  StatusOr<std::unique_ptr<HloModule>> GetOptimizedModule(
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

  ~HloTestBase() override {}

  // Runs pass `hlo_pass` on input HLO module `hlo`, and FileChecks the result
  // against `expected`.
  //
  // If the rewrite has changed the module, also runs `additional_checks` on the
  // result.
  void RunAndFilecheckHloRewrite(
      absl::string_view hlo, HloPassInterface&& hlo_pass,
      std::optional<absl::string_view> expected,
      std::function<void(HloModule*)> after_pass_checks = nullptr);

  // Runs pass `hlo_pass` on a group of input HLO modules `hlo_module_strs`,
  // and FileChecks the result against `expected`.
  void RunAndFilecheckHloModuleGroupRewrite(
      absl::Span<const absl::string_view> hlo_module_strs,
      HloPassInterface&& hlo_pass,
      std::optional<absl::Span<const absl::string_view>> expected);

  // Populates debug options from command-line flags and adjusts the options for
  // testing. It is recommended to use this when you need to pass in
  // DebugOptions, e.g. when creating a module from a string or a file.
  //
  // This function is virtual so tests can specify an alternative set of debug
  // options (e.g. disabling additional passes).
  virtual DebugOptions GetDebugOptionsForTest();

  // Gets an HloModuleConfig with options appropriate for tests.
  HloModuleConfig GetModuleConfigForTest(int64_t replica_count = 1,
                                         int64_t num_partitions = 1) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    config.set_replica_count(replica_count);
    config.set_num_partitions(num_partitions);
    return config;
  }

  // Executes the given module and return the result as a Literal.
  StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                            absl::Span<Literal* const> arguments);

  // Same as above, except the module will be executed without running any HLO
  // passes on it.
  Literal ExecuteNoHloPasses(std::unique_ptr<HloModule> module,
                             absl::Span<Literal* const> arguments);

  Literal ExecuteAndTransfer(std::unique_ptr<HloModule> module,
                             absl::Span<Literal* const> arguments);

  // Executes the given module on multiple replicas.
  //
  // use_threads indicates whether this replicated computation will be executed
  // with a thread-per-replica, vs using an implicitly async call such as
  // Executable::ExecuteOnStreams.
  StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
      int64_t num_replicas, bool use_threads, bool run_hlo_passes = false);

  // Same as above, but uses specified device assignment.
  StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
      int64_t num_replicas, DeviceAssignment* device_assignment,
      bool run_hlo_passes, bool use_threads);

  // Same as above, but allows passing different programs for replicas.
  StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::function<Executable*(int64_t)> executable_provider,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      int64_t num_replicas, bool run_hlo_passes,
      DeviceAssignment* device_assignment = nullptr);

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
      std::unique_ptr<HloModule> module,
      const absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);

  // Same as above, except that the module will be executed without Hlo
  // optimization.
  [[nodiscard]] ::testing::AssertionResult RunAndCompareNoHloPasses(
      std::unique_ptr<HloModule> module,
      const absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);

  // Executes an hlo module with fake inputs and compares the results.
  [[nodiscard]] ::testing::AssertionResult RunAndCompare(
      std::unique_ptr<HloModule> module, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);

  // Same as above, except that the module will be executed without Hlo
  // optimization.
  [[nodiscard]] ::testing::AssertionResult RunAndCompareNoHloPasses(
      std::unique_ptr<HloModule> module, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);

  // Executes an hlo module with fake inputs and checks that the execution is
  // successful.
  [[nodiscard]] ::testing::AssertionResult Run(
      std::unique_ptr<HloModule> module, bool run_hlo_passes);

  // Convenient wrappers for executing and comparing an hlo module with fake
  // input. Module can be passed in directly, or parsed from an hlo_string,
  // or loaded from a file.
  [[nodiscard]] ::testing::AssertionResult RunAndCompare(
      const absl::string_view hlo_string, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);
  [[nodiscard]] ::testing::AssertionResult Run(
      const absl::string_view hlo_string, bool run_hlo_passes = true,
      ExecutionProfile* profile = nullptr,
      const tsl::protobuf::Message* backend_config = nullptr);

  // Same as below, except requires passing fake arguments.
  ::testing::AssertionResult RunAndCompareTwoModules(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      const absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes = true);

  // Same as below, except requires passing the modules.
  ::testing::AssertionResult RunAndCompareTwoModules(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes = true);

  // Convenient wrapper for executing and comparing results of two hlo modules
  // with fake input. By default compares unoptimized modules. If the modules
  // are already optimized, set |run_hlo_passes| to false.
  ::testing::AssertionResult RunAndCompareTwoModules(
      absl::string_view hlo_string_module_0,
      absl::string_view hlo_string_module_1,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes = true);

  // Executes an hlo module with fake inputs on multiple replicas.
  [[nodiscard]] ::testing::AssertionResult RunReplicated(
      const absl::string_view hlo_string, bool run_hlo_passes = true,
      int64_t num_replicas = 1,
      const tsl::protobuf::Message* backend_config = nullptr);

  // If assert_determinism is true, the assertion will fail unless all runs
  // produce exactly the same output.
  [[nodiscard]] ::testing::AssertionResult RunMultipleTimes(
      const absl::string_view hlo_string, bool run_hlo_passes,
      std::vector<ExecutionProfile>* profiles,
      const tsl::protobuf::Message* backend_config = nullptr,
      bool assert_determinism = false);
  [[nodiscard]] ::testing::AssertionResult RunAndCompareFromFile(
      const std::string& filename, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);
  [[nodiscard]] ::testing::AssertionResult RunAndCompareNoHloPasses(
      const absl::string_view hlo_string, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);
  [[nodiscard]] ::testing::AssertionResult RunAndCompareNoHloPassesFromFile(
      const std::string& filename, const std::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);

  // Convenience method to force the layout of a given parameter in a module.
  // The layout of parameter number 'param_no' in the 'module' is set to
  // 'layout'.
  void ForceParameterLayout(HloModule* module, int64_t param_no,
                            const Layout& layout) {
    ASSERT_LT(param_no,
              module->mutable_entry_computation_layout()->parameter_count());
    module->mutable_entry_computation_layout()
        ->mutable_parameter_layout(param_no)
        ->ResetLayout(layout);
  }

  // Convenience method to force the layout of the computation result in a
  // module. The result layout of 'module' is set to 'layout'.
  void ForceResultLayout(HloModule* module, const Layout& layout) {
    module->mutable_entry_computation_layout()
        ->mutable_result_layout()
        ->ResetLayout(layout);
  }

  void ForceResultLayout(HloModule* module, const Layout& layout,
                         ShapeIndexView shape_index) {
    module->mutable_entry_computation_layout()
        ->mutable_result_layout()
        ->ResetLayout(layout, shape_index);
  }

  // Convenience method to clear the layout of the computation result in
  // 'module'.
  void ForceClearResultLayout(HloModule* module) {
    module->mutable_entry_computation_layout()
        ->mutable_result_layout()
        ->Clear();
  }

  // Gets the computation/instruction from the given module with the given name.
  //
  // This is useful for tests which create HLOs from a string and then want to
  // inspect a particular computation or instruction.
  HloComputation* FindComputation(HloModule* module, absl::string_view name);
  HloInstruction* FindInstruction(HloModule* module, absl::string_view name);
  // Gets the instruction from the given module with the given opcode.
  HloInstruction* FindInstruction(HloModule* module, HloOpcode opcode);

  // Return an HLO verifier constructed for the test backend.
  HloVerifier& verifier() const { return *hlo_verifier_; }

  static std::string TestName();

  // Returns the backend owned by the test runner.
  Backend& backend();

  HloRunner test_runner_;
  HloRunner reference_runner_;

  bool verifier_layout_sensitive_;
  bool allow_mixed_precision_in_hlo_verifier_;
  std::unique_ptr<HloVerifier> hlo_verifier_;

  ErrorSpec error_spec_{0.0001};

  HloComputation* AddEntryComputationAndUpdateEntryComputationLayout(
      HloModule*, std::unique_ptr<HloComputation> computation);
  void UpdateEntryComputationLayout(HloModule* module);

  StatusOr<std::unique_ptr<HloRunnerInterface>> GetHloRunner();

 protected:
  // Helper functions to get test and reference platforms.
  static se::Platform* GetReferencePlatform();
  static se::Platform* GetTestPlatform();

 private:
  // Either an HloRunner or HloRunnerPjRt depending on if ShouldUsePjRt()
  std::unique_ptr<HloRunnerInterface> runner_;
  se::Platform* test_platform_;

  // Given the test module, makes a reference module that is ready to run on the
  // reference platform. This assumes that the given module is ready to run on
  // the test platform.
  StatusOr<std::unique_ptr<HloModule>> MakeReferenceModule(
      const HloModule& test_module,
      const std::function<void(HloModule*)>& reference_preprocessor);

  // Runs the module on two platforms with or without running hlo passes and
  // compares the results. Returns whether the results are near or equal. If any
  // error happens before the results are computed, returns the error status.
  StatusOr<::testing::AssertionResult> RunAndCompareInternal(
      std::unique_ptr<HloModule> module,
      const absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes,
      const std::function<void(HloModule*)>& reference_preprocessor);

  // Runs the two module on with or without running hlo passes and
  // compares the results. Returns whether the results are near or equal. If any
  // error happens before the results are computed, returns the error status.
  StatusOr<::testing::AssertionResult> RunAndCompareTwoModulesInternal(
      std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
      const absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec>& error, bool run_hlo_passes);

  // Returns either an HloRunner or HloRunnerPjRt implementation depending if
  // there exists a registered PjRtClientFactory.
  StatusOr<std::unique_ptr<HloRunnerInterface>> GetHloRunnerForTest(
      se::Platform* test_platform);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_HLO_TEST_BASE_H_
