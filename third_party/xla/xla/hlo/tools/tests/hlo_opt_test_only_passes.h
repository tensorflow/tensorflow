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

#ifndef XLA_HLO_TOOLS_TESTS_HLO_OPT_TEST_ONLY_PASSES_H_
#define XLA_HLO_TOOLS_TESTS_HLO_OPT_TEST_ONLY_PASSES_H_

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"

namespace xla::test_only {

// A module pass which renames instructions named 'foo' to 'bar'.
class FooToBarModulePass : public HloModulePass {
 public:
  absl::string_view name() const override { return "test-only-foo2bar"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    bool changed = false;
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->name() == "foo") {
          instruction->SetAndSanitizeName("bar");
          changed = true;
        }
      }
    }
    return changed;
  }
};

// A module pass which renames instructions named 'bar' to 'hello'.
class BarToHelloModulePass : public HloModulePass {
 public:
  absl::string_view name() const override { return "test-only-bar2hello"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    bool changed = false;
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->name() == "bar") {
          instruction->SetAndSanitizeName("hello");
          changed = true;
        }
      }
    }
    return changed;
  }
};

// Allows `AlgebraicSimplifier::PromoteConvolutionToF32IfNotOnednnCompatible` to
// be tested regardless of whether the build configuration uses OneDNN.
class AlgebraicSimplifierWithOnednnEnabled : public AlgebraicSimplifier {
 public:
  absl::string_view name() const override {
    return "test-only-algebraic-simplifier-with-onednn-enabled";
  }

  explicit AlgebraicSimplifierWithOnednnEnabled()
      : AlgebraicSimplifier(TestSpecificOptions()) {}

 private:
  static AlgebraicSimplifierOptions TestSpecificOptions() {
    AlgebraicSimplifierOptions options;
    options.set_enable_onednn_support(true);
    options.set_executing_on_cpu(true);
    return options;
  }
};

// Test XLA Builder methods using lit tests.
// Transforms custom calls that start with `xla_builder.some_method` into
// expanded HLO by calling the client methods:
// Example:
//  custom-call @xla_builder.add(operand1, operand2)
//  ==>
//  add(operand1, operand2)
class XlaBuilderTestPass : public HloModulePass {
 public:
  absl::string_view name() const override { return "test-only-xla-builder"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> ReplaceWithExpandedClientHlo(
      HloInstruction* instruction, absl::string_view custom_call_target);
};

}  // namespace xla::test_only

#endif  // XLA_HLO_TOOLS_TESTS_HLO_OPT_TEST_ONLY_PASSES_H_
