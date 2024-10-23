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

#ifndef XLA_HLO_TESTLIB_HLO_HARDWARE_INDEPENDENT_TEST_BASE_H_
#define XLA_HLO_TESTLIB_HLO_HARDWARE_INDEPENDENT_TEST_BASE_H_

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
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/layout.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {

// A base class for tests which build and manipulate HLO without running it.
//
class HloHardwareIndependentTestBase : public ::testing::Test {
 public:
  static PrecisionConfig DefaultPrecisionConfig(int operands);

 protected:
  explicit HloHardwareIndependentTestBase(
      bool verifier_layout_sensitive = false,
      bool allow_mixed_precision_in_hlo_verifier = true,
      HloPredicate instruction_can_change_layout_func = {});

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
      const std::string& name = TestName()) const;

  // Like CreateNewUnverifiedModule, except the HloModule returned here runs the
  // HLO verifier on destruction.
  std::unique_ptr<VerifiedHloModule> CreateNewVerifiedModule(
      const std::string& name = TestName(), int64_t replica_count = 1) const;

  // Parses the given string and returns module as a VerifiedHloModule.
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
  ParseAndReturnVerifiedModule(absl::string_view hlo_text,
                               int64_t replica_count = 1,
                               int64_t num_partitions = 1) const;
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
  ParseAndReturnVerifiedModule(absl::string_view hlo_text,
                               const HloModuleConfig& config) const;

  // Runs the hlo_pass with the provided module and returns the result. This
  // function also verifies that the module remains unchanged when hlo_pass
  // returns false as the absl::StatusOr value.
  //
  // These three overloads all do the same thing.  The && overload lets you do
  // `RunHloPass(MyPass(), module)` all in one line.  The reason for the
  // overload that takes a pointer is that, at one point in the past, non-const
  // lvalue references were banned in Google code.
  static absl::StatusOr<bool> RunHloPass(HloPassInterface* hlo_pass,
                                         HloModule* module);
  static absl::StatusOr<bool> RunHloPass(HloPassInterface& hlo_pass,
                                         HloModule* module) {
    return RunHloPass(&hlo_pass, module);
  }
  static absl::StatusOr<bool> RunHloPass(HloPassInterface&& hlo_pass,
                                         HloModule* module) {
    return RunHloPass(&hlo_pass, module);
  }

  // Runs the hlo_pass with the provided module group and returns the result.
  // This method runs the input HLO module group pass for a `HloModuleGroup` and
  // it also verifies the module group remains unchanged when hlo_pass returns
  // false as the absl::StatusOr value.
  static absl::StatusOr<bool> RunHloPass(HloPassInterface&& hlo_pass,
                                         HloModuleGroup* module_group);

  // Sets most fath math options to be enabled to model the fast math flags
  // generally used for CPU:AOT compilation.
  static void SetAotFastMathDebugOptions(DebugOptions* options);

  // Runs pass `hlo_pass` on input HLO module `hlo` with optional config, and
  // FileChecks the result against `expected`.
  //
  // If the rewrite has changed the module, also runs `additional_checks` on the
  // result.
  void RunAndFilecheckHloRewrite(
      absl::string_view hlo, HloPassInterface&& hlo_pass,
      std::optional<absl::string_view> expected,
      std::function<void(HloModule*)> after_pass_checks = nullptr,
      const HloModuleConfig* config = nullptr) const;

  // Runs pass `hlo_pass` on a group of input HLO modules `hlo_module_strs`,
  // and FileChecks the result against `expected`.
  void RunAndFilecheckHloModuleGroupRewrite(
      absl::Span<const absl::string_view> hlo_module_strs,
      HloPassInterface&& hlo_pass,
      std::optional<absl::Span<const absl::string_view>> expected) const;

  using FixedMapping =
      std::initializer_list<std::pair<absl::string_view, absl::string_view>>;

  // Creates an HLO module from a template and an optional replacement map and
  // runs the given hlo_pass on the module. Validates whether the pass has
  // changed the module or not based on expect_change flag.  Returns unique_ptr
  // to the HLO module for further inspection.
  absl::StatusOr<std::unique_ptr<HloModule>> RunAndCheckHloRewrite(
      absl::string_view hlo_template, HloPassInterface&& hlo_pass,
      bool expect_change = true, FixedMapping params = {}) const;

  // Populates debug options from command-line flags and adjusts the options for
  // testing. It is recommended to use this when you need to pass in
  // DebugOptions, e.g. when creating a module from a string or a file.
  //
  // This function is virtual so tests can specify an alternative set of debug
  // options (e.g. disabling additional passes).
  virtual DebugOptions GetDebugOptionsForTest() const;

  // Gets an HloModuleConfig with options appropriate for tests.
  HloModuleConfig GetModuleConfigForTest(int64_t replica_count = 1,
                                         int64_t num_partitions = 1) const {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    config.set_replica_count(replica_count);
    config.set_num_partitions(num_partitions);
    return config;
  }

  // Convenience method to force the layout of a given parameter in a module.
  // The layout of parameter number 'param_no' in the 'module' is set to
  // 'layout'.
  static void ForceParameterLayout(HloModule* module, int64_t param_no,
                                   const Layout& layout) {
    ASSERT_LT(param_no,
              module->mutable_entry_computation_layout()->parameter_count());
    module->mutable_entry_computation_layout()
        ->mutable_parameter_layout(param_no)
        ->ResetLayout(layout);
  }

  // Convenience method to force the layout of the computation result in a
  // module. The result layout of 'module' is set to 'layout'.
  static void ForceResultLayout(HloModule* module, const Layout& layout) {
    module->mutable_entry_computation_layout()
        ->mutable_result_layout()
        ->ResetLayout(layout);
  }

  static void ForceResultLayout(HloModule* module, const Layout& layout,
                                ShapeIndexView shape_index) {
    module->mutable_entry_computation_layout()
        ->mutable_result_layout()
        ->ResetLayout(layout, shape_index);
  }

  // Convenience method to clear the layout of the computation result in
  // 'module'.
  static void ForceClearResultLayout(HloModule* module) {
    module->mutable_entry_computation_layout()
        ->mutable_result_layout()
        ->Clear();
  }

  // Gets the computation/instruction from the given module with the given name.
  // Note that it is encouraged to use these functions directly via the
  // hlo_query.h header instead since they are independent from any test-time
  // variables or contexts.

  // This is useful for tests which create HLOs from a string and then want to
  // inspect a particular computation or instruction.
  static HloComputation* FindComputation(HloModule* module,
                                         absl::string_view name);
  static HloInstruction* FindInstruction(HloModule* module,
                                         absl::string_view name);
  // Gets the instruction from the given module with the given opcode.
  static HloInstruction* FindInstruction(HloModule* module, HloOpcode opcode);
  // Gets all the instructions from the given module with the given opcode.
  static std::vector<HloInstruction*> FindInstructions(HloModule* module,
                                                       HloOpcode opcode);

  bool verifier_layout_sensitive() const { return verifier_layout_sensitive_; }
  void set_verifier_layout_sensitive(bool verifier_layout_sensitive) {
    verifier_layout_sensitive_ = verifier_layout_sensitive;
  }
  HloPredicate instruction_can_change_layout_func() const {
    return instruction_can_change_layout_func_;
  }
  void set_instruction_can_change_layout_func(
      HloPredicate instruction_can_change_layout_func) {
    instruction_can_change_layout_func_ =
        std::move(instruction_can_change_layout_func);
  }
  // Return an HLO verifier constructed for the test backend.
  HloVerifier& verifier() const { return *hlo_verifier_; }
  void set_hlo_verifier(std::unique_ptr<HloVerifier> hlo_verifier) {
    hlo_verifier_ = std::move(hlo_verifier);
  }
  bool allow_mixed_precision_in_hlo_verifier() const {
    return allow_mixed_precision_in_hlo_verifier_;
  }

  static std::string TestName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  // Updates the entry computation layout to match the program shape. Useful
  // when tiling assignment has been run to update the latter and we want those
  // changes propagated into the former.
  static absl::Status UpdateEntryComputationLayoutToMatchProgramLayout(
      HloModule* module);

  // Compares the inputs shapes of two modules and returns the list of parameter
  // indices that mismatch. The mismatch could be either in shape or datatype.
  // If there is no mismatch, an empty vector is returned.
  [[nodiscard]] std::vector<int> CompareInputs(const HloModule& module_0,
                                               const HloModule& module_1);

 private:
  bool verifier_layout_sensitive_;
  bool allow_mixed_precision_in_hlo_verifier_;
  HloPredicate instruction_can_change_layout_func_;
  std::unique_ptr<HloVerifier> hlo_verifier_;
};

}  // namespace xla

#endif  // XLA_HLO_TESTLIB_HLO_HARDWARE_INDEPENDENT_TEST_BASE_H_
