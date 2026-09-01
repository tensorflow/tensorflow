/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/pass/hlo_pass_pipeline.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;
using ::testing::StrEq;

using HloPassPipelineTest = HloHardwareIndependentTestBase;

// A module pass which renames instructions named 'foo' to 'bar'.
class FooToBarModulePass : public HloModulePass {
  absl::string_view name() const override { return "foo2bar"; }

 protected:
  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    bool changed = false;
    for (HloComputation* computation :
         module->computations(execution_threads)) {
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

// A module pass with a configurable name that appends a unique tag to a shared
// log every time it runs. Used to verify the richer xla_disable_hlo_passes /
// xla_enable_hlo_passes_only entry syntax (occurrence, pipeline scope, and
// pass_id).
class RecordingPass : public HloModulePass {
 public:
  RecordingPass(absl::string_view name, absl::string_view tag,
                std::vector<std::string>* run_log)
      : name_(name), tag_(tag), run_log_(run_log) {}

  absl::string_view name() const override { return name_; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* /*module*/,
      const absl::flat_hash_set<absl::string_view>& /*execution_threads*/)
      override {
    run_log_->push_back(tag_);
    return false;
  }

 private:
  std::string name_;
  std::string tag_;
  std::vector<std::string>* run_log_;
};

// A module pass which renames root instructions names in reverse string order,
// e.g. "xyz" becomes "zyx".
class ReverseStringModulePass : public HloModulePass {
  absl::string_view name() const override { return "reverse"; }

 protected:
  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    bool changed = false;
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      HloInstruction* root = computation->root_instruction();
      std::string name(root->name());
      absl::c_reverse(name);
      root->SetAndSanitizeName(name);
      changed = true;
    }
    return changed;
  }
};

// A module pass which renames instructions named 'baz' to 'qux'.
class BazToQuxModulePass : public HloModulePass {
  absl::string_view name() const override { return "baz2qux"; }

  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    bool changed = false;
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->name() == "baz") {
          instruction->SetAndSanitizeName("qux");
          changed = true;
        }
      }
    }
    return changed;
  }
};

// An invariant checker pass which returns an error if there exists an
// instruction named 'bar'.
class BarBlowerUpper : public HloModulePass {
  absl::string_view name() const override { return "bar-blower-upper"; }

 protected:
  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->name() == "bar") {
          return Internal("Module has instruction named bar");
        }
      }
    }
    return false;
  }
};

TEST_F(HloPassPipelineTest, ModulePassChanged) {
  // Test an HLO module pass which changes a module.
  const std::string module_str = R"(
HloModule ModulePassChanged

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT foo = f32[] multiply(a, b)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<FooToBarModulePass>();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->name(), "foo");
  ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(root->name(), "bar");
}

TEST_F(HloPassPipelineTest, ModulePassUnchanged) {
  // Test an HLO module pass which does not change a module.
  const std::string module_str = R"(
HloModule ModulePassUnchanged

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT blahblah = f32[] multiply(a, b)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<FooToBarModulePass>();

  ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(HloPassPipelineTest, ModulePassChangedForParallelThread) {
  // Test an HLO module pass which changes a module.
  const std::string module_str = R"(
HloModule ModulePassChanged
%async_builder {
  %p0 = f32[10] parameter(0)
  %p1 = f32[10] parameter(1)
  ROOT %foo = add(%p0, %p1)
}, execution_thread="parallel_thread"


ENTRY %Entry (p0: f32[10], p1: f32[10]) -> f32[10] {
  %p0 = f32[10] parameter(0)
  %p1 = f32[10] parameter(1)
  %async-start = ((f32[10], f32[10]), f32[10], s32[]) async-start(f32[10] %p0, f32[10] %p1), async_execution_thread="parallel_thread",calls=%async_builder
  ROOT %baz = f32[10]{0} async-done(((f32[10], f32[10]), f32[10], s32[]) %async-start), async_execution_thread="parallel_thread", calls=%async_builder
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<ReverseStringModulePass>();

  HloInstruction* main_root = module->entry_computation()->root_instruction();
  HloInstruction* parallel_thread_root =
      main_root->async_wrapped_computation()->root_instruction();
  EXPECT_EQ(main_root->name(), "baz");
  EXPECT_EQ(parallel_thread_root->name(), "foo");
  ASSERT_OK_AND_ASSIGN(bool changed,
                       pipeline.Run(module.get(), {"parallel_thread"}));
  EXPECT_TRUE(changed);
  EXPECT_EQ(main_root->name(), "baz");
  EXPECT_EQ(parallel_thread_root->name(), "oof");
}

TEST_F(HloPassPipelineTest, ModulePassChangedForAllexecution_threads) {
  // Test an HLO module pass which changes a module.
  const std::string module_str = R"(
HloModule ModulePassChanged
%async_builder {
  %p0 = f32[10] parameter(0)
  %p1 = f32[10] parameter(1)
  ROOT %foo = add(%p0, %p1)

}, execution_thread="parallel_thread"


ENTRY %Entry (p0: f32[10], p1: f32[10]) -> f32[10] {
  %p0 = f32[10] parameter(0)
  %p1 = f32[10] parameter(1)
  %async-start = ((f32[10], f32[10]), f32[10], s32[]) async-start(f32[10] %p0, f32[10] %p1), async_execution_thread="parallel_thread",calls=%async_builder
  ROOT %baz = f32[10]{0} async-done(((f32[10], f32[10]), f32[10], s32[]) %async-start), async_execution_thread="parallel_thread", calls=%async_builder
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<ReverseStringModulePass>();

  HloInstruction* main_root = module->entry_computation()->root_instruction();
  HloInstruction* parallel_thread_root =
      main_root->async_wrapped_computation()->root_instruction();
  EXPECT_EQ(main_root->name(), "baz");
  EXPECT_EQ(parallel_thread_root->name(), "foo");
  ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(main_root->name(), "zab");
  EXPECT_EQ(parallel_thread_root->name(), "oof");
}

TEST_F(HloPassPipelineTest, MixedPipeline) {
  const std::string module_0_str = R"(
HloModule MixedPipeline.1

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT baz = f32[] multiply(a, b)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(module_0_str));

  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<BazToQuxModulePass>();
  pipeline.AddPass<FooToBarModulePass>();

  HloInstruction* root0 = module->entry_computation()->root_instruction();
  EXPECT_EQ(root0->name(), "baz");

  ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(root0->name(), "qux");
}

TEST_F(HloPassPipelineTest, InvariantChecker) {
  const std::string module_str = R"(
HloModule InvariantChecker

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT foo = f32[] multiply(a, b)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  {
    // Run a pipeline with just the invariant checker. It should not fail
    // because there is no 'bar' instruction in the module.
    HloPassPipeline pipeline(TestName());
    pipeline.AddInvariantChecker<BarBlowerUpper>();

    ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
    EXPECT_FALSE(changed);
  }

  {
    // Run a pipeline which renames 'foo' to 'bar' then an invariant checker
    // which fails if there is an instruction named 'bar'.
    HloPassPipeline pipeline(TestName());
    pipeline.AddInvariantChecker<BarBlowerUpper>();
    pipeline.AddPass<FooToBarModulePass>();

    absl::Status status = pipeline.Run(module.get()).status();
    ASSERT_IS_NOT_OK(status);
    EXPECT_THAT(status.message(),
                ::testing::HasSubstr("Module has instruction named bar"));
    EXPECT_THAT(status.message(), ::testing::HasSubstr("Failed after foo2bar"));
  }

  {
    // Run the invariant-checker only pipeline again. It should fail this time.
    HloPassPipeline pipeline(TestName());
    pipeline.AddInvariantChecker<BarBlowerUpper>();

    absl::Status status = pipeline.Run(module.get()).status();
    ASSERT_IS_NOT_OK(status);
    EXPECT_THAT(status.message(),
                ::testing::HasSubstr("Module has instruction named bar"));
    EXPECT_THAT(status.message(),
                ::testing::HasSubstr("Failed after pipeline-start"));
  }
}

// Test that metadata is set when a module goes through a pass pipeline.
TEST_F(HloPassPipelineTest, SetHloModuleMetadata) {
  std::unique_ptr<VerifiedHloModule> module = CreateNewVerifiedModule();

  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<BazToQuxModulePass>();
  pipeline.AddPass<FooToBarModulePass>();
  TF_ASSERT_OK(pipeline.Run(module.get()).status());

  std::vector<std::string> pass_names = {"pipeline-start", "baz2qux",
                                         "foo2bar"};
  std::string pipeline_name = std::string(pipeline.name());
  const HloModuleMetadataProto& metadata = module->metadata()->proto();
  EXPECT_EQ(metadata.canonical_module_id(), module->unique_id());

  ASSERT_THAT(metadata.pass_metadata(), SizeIs(3));
  for (int pass = 0; pass < metadata.pass_metadata().size(); pass++) {
    const HloPassMetadata& pass_metadata = metadata.pass_metadata(pass);
    EXPECT_NE(pass_metadata.pass_id(), 0);
    EXPECT_THAT(pass_metadata.pass_name(), StrEq(pass_names[pass]));
    EXPECT_THAT(pass_metadata.pipeline_name(), StrEq(pipeline_name));
    EXPECT_FALSE(pass_metadata.module_changed());
    EXPECT_EQ(pass_metadata.module_id(), module->unique_id());
    EXPECT_GT(pass_metadata.start_timestamp_usec(), 0);
    EXPECT_LE(pass_metadata.start_timestamp_usec(),
              pass_metadata.end_timestamp_usec());
  }
}

class NoOpModulePass : public HloModulePass {
  absl::string_view name() const override { return "noop"; }

 protected:
  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    return false;
  }
};

TEST_F(HloPassPipelineTest, NoCrashOnNoChange) {
  const std::string module_str = R"(
HloModule ModuleGroupPassOnModule

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT foo = f32[] multiply(a, b)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_unsupported_crash_on_hlo_pass_silent_hlo_change(true);
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<NoOpModulePass>();

  absl::Status status = pipeline.Run(module.get()).status();
  TF_EXPECT_OK(status);
}

class AppendPass : public HloModulePass {
 public:
  AppendPass(std::string name, std::string suffix)
      : name_(name), suffix_(suffix) {}
  absl::string_view name() const override { return name_; }

 protected:
  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    HloInstruction* root = module->entry_computation()->root_instruction();
    root->SetAndSanitizeName(absl::StrCat(root->name(), "_", suffix_));
    return true;
  }

 private:
  std::string name_;
  std::string suffix_;
};

TEST_F(HloPassPipelineTest, RunPassesStartingFrom) {
  const std::string module_str = R"(
HloModule test_module

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT foo = f32[] multiply(p0, p1)
}
)";

  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(module_str));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_run_hlo_passes_starting_from("reverse");

    HloPassPipeline pipeline(TestName());
    pipeline.AddPass<FooToBarModulePass>();
    pipeline.AddPass<ReverseStringModulePass>();

    ASSERT_OK(pipeline.Run(module.get()).status());

    EXPECT_EQ(module->entry_computation()->root_instruction()->name(), "oof");
  }

  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(module_str));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_run_hlo_passes_starting_from("foo2bar");

    HloPassPipeline pipeline(TestName());
    pipeline.AddPass<FooToBarModulePass>();
    pipeline.AddPass<ReverseStringModulePass>();

    ASSERT_OK(pipeline.Run(module.get()).status());

    EXPECT_EQ(module->entry_computation()->root_instruction()->name(), "rab");
  }

  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(module_str));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_run_hlo_passes_starting_from("non-existent");

    HloPassPipeline pipeline(TestName());
    pipeline.AddPass<FooToBarModulePass>();
    pipeline.AddPass<ReverseStringModulePass>();

    ASSERT_OK(pipeline.Run(module.get()).status());

    EXPECT_EQ(module->entry_computation()->root_instruction()->name(), "foo");
  }
}

TEST_F(HloPassPipelineTest, RunPassesStartingFromNested) {
  const std::string module_str = R"(
HloModule test_module

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT foo = f32[] multiply(p0, p1)
}
)";

  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(module_str));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_run_hlo_passes_starting_from("B");

    HloPassPipeline pipeline(TestName());
    pipeline.AddPass<AppendPass>("A", "A");
    auto& sub = pipeline.AddPass<HloPassPipeline>("sub");
    sub.AddPass<AppendPass>("A2", "A2");
    sub.AddPass<AppendPass>("B", "B");
    sub.AddPass<AppendPass>("C", "C");
    pipeline.AddPass<AppendPass>("D", "D");

    ASSERT_OK(pipeline.Run(module.get()).status());

    EXPECT_EQ(module->entry_computation()->root_instruction()->name(),
              "foo_B_C_D");
  }

  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(module_str));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_run_hlo_passes_starting_from("sub");

    HloPassPipeline pipeline(TestName());
    pipeline.AddPass<AppendPass>("A", "A");
    auto& sub = pipeline.AddPass<HloPassPipeline>("sub");
    sub.AddPass<AppendPass>("A2", "A2");
    sub.AddPass<AppendPass>("B", "B");
    sub.AddPass<AppendPass>("C", "C");
    pipeline.AddPass<AppendPass>("D", "D");

    ASSERT_OK(pipeline.Run(module.get()).status());

    EXPECT_EQ(module->entry_computation()->root_instruction()->name(),
              "foo_A2_B_C_D");
  }
}

// Tests for the richer xla_disable_hlo_passes / xla_enable_hlo_passes_only
// entry syntax. RecordingPass instances share the same pass name but log a
// distinct tag, so we can tell exactly which invocation ran.
constexpr absl::string_view kSimpleModule = R"(
HloModule test_module

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT foo = f32[] multiply(p0, p1)
}
)";

// Backward compatibility: a plain name disables every invocation of that pass.
TEST_F(HloPassPipelineTest, DisablePlainNameSkipsAllInvocations) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "algsimp");

  std::vector<std::string> run_log;
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<RecordingPass>("algsimp", "a0", &run_log);
  pipeline.AddPass<RecordingPass>("dce", "d0", &run_log);
  pipeline.AddPass<RecordingPass>("algsimp", "a1", &run_log);

  ASSERT_OK(pipeline.Run(module.get()).status());
  EXPECT_THAT(run_log, ElementsAre("d0"));
}

// Disabling a pipeline's own name skips all its passes (backward compatibility
// with directly-Run()'d pipelines like the GPU "fusion" pipeline).
TEST_F(HloPassPipelineTest, DisablePipelineOwnNameSkipsAllPasses) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "my-pipeline");

  std::vector<std::string> run_log;
  HloPassPipeline pipeline("my-pipeline");
  pipeline.AddPass<RecordingPass>("algsimp", "a0", &run_log);
  pipeline.AddPass<RecordingPass>("dce", "d0", &run_log);

  ASSERT_OK(pipeline.Run(module.get()).status());
  EXPECT_THAT(run_log, ::testing::IsEmpty());
}

// "algsimp:1" disables the global (per-module) 0-based invocation index 1.
TEST_F(HloPassPipelineTest, DisableGlobalOccurrence) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "algsimp:1");

  std::vector<std::string> run_log;
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<RecordingPass>("algsimp", "a0", &run_log);
  pipeline.AddPass<RecordingPass>("algsimp", "a1", &run_log);
  pipeline.AddPass<RecordingPass>("algsimp", "a2", &run_log);

  ASSERT_OK(pipeline.Run(module.get()).status());
  EXPECT_THAT(run_log, ElementsAre("a0", "a2"));
}

// "simplification/algsimp" disables algsimp only under its immediate parent
// pipeline named "simplification".
TEST_F(HloPassPipelineTest, DisablePipelineScope) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "simplification/algsimp");

  std::vector<std::string> run_log;
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<RecordingPass>("algsimp", "top", &run_log);
  auto& sub = pipeline.AddPass<HloPassPipeline>("simplification");
  sub.AddPass<RecordingPass>("algsimp", "in_sub", &run_log);

  ASSERT_OK(pipeline.Run(module.get()).status());
  // The top-level algsimp runs (its parent is not "simplification"); the one
  // inside the "simplification" pipeline is skipped.
  EXPECT_THAT(run_log, ElementsAre("top"));
}

// "simplification/algsimp:1" disables the invocation at index 1 counted WITHIN
// the "simplification" pipeline instance (not the global index).
TEST_F(HloPassPipelineTest, DisableScopedOccurrence) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "simplification/algsimp:1");

  std::vector<std::string> run_log;
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<RecordingPass>("algsimp", "t0", &run_log);
  auto& sub = pipeline.AddPass<HloPassPipeline>("simplification");
  sub.AddPass<RecordingPass>("algsimp", "s0", &run_log);
  sub.AddPass<RecordingPass>("algsimp", "s1", &run_log);
  sub.AddPass<RecordingPass>("algsimp", "s2", &run_log);
  pipeline.AddPass<RecordingPass>("algsimp", "t1", &run_log);

  ASSERT_OK(pipeline.Run(module.get()).status());
  // Within "simplification": s0=0, s1=1, s2=2; only s1 is skipped. The global
  // occurrence index of s1 is 2, which is intentionally NOT used here.
  EXPECT_THAT(run_log, ElementsAre("t0", "s0", "s2", "t1"));
}

// "@N" disables the pass with the exact raw pass_id N. In a flat pipeline the
// pipeline-start pseudo-pass takes id 1, so the first real pass is id 2.
TEST_F(HloPassPipelineTest, DisableByPassId) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "@3");

  std::vector<std::string> run_log;
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<RecordingPass>("pa", "a", &run_log);  // pass_id 2
  pipeline.AddPass<RecordingPass>("pb", "b", &run_log);  // pass_id 3
  pipeline.AddPass<RecordingPass>("pc", "c", &run_log);  // pass_id 4

  ASSERT_OK(pipeline.Run(module.get()).status());
  EXPECT_THAT(run_log, ElementsAre("a", "c"));
}

// Backward compatibility: a plain enable-only list keeps the exact legacy
// behavior (only the named passes run).
TEST_F(HloPassPipelineTest, EnableOnlyPlainName) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  // The test base disables "constant_folding" by default; clear it so the
  // enable-only list is not rejected by the disable/enable exclusivity check.
  module->mutable_config()
      .mutable_debug_options()
      .clear_xla_disable_hlo_passes();
  module->mutable_config()
      .mutable_debug_options()
      .add_xla_enable_hlo_passes_only("dce");

  std::vector<std::string> run_log;
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<RecordingPass>("algsimp", "a0", &run_log);
  pipeline.AddPass<RecordingPass>("dce", "d0", &run_log);

  ASSERT_OK(pipeline.Run(module.get()).status());
  EXPECT_THAT(run_log, ElementsAre("d0"));
}

// Enable-only with the richer occurrence syntax: only the matching invocation
// runs.
TEST_F(HloPassPipelineTest, EnableOnlyOccurrence) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->mutable_config()
      .mutable_debug_options()
      .clear_xla_disable_hlo_passes();
  module->mutable_config()
      .mutable_debug_options()
      .add_xla_enable_hlo_passes_only("algsimp:1");

  std::vector<std::string> run_log;
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<RecordingPass>("algsimp", "a0", &run_log);
  pipeline.AddPass<RecordingPass>("algsimp", "a1", &run_log);
  pipeline.AddPass<RecordingPass>("algsimp", "a2", &run_log);

  ASSERT_OK(pipeline.Run(module.get()).status());
  EXPECT_THAT(run_log, ElementsAre("a1"));
}

// Enable-only with rich syntax descends into nested pipelines so a scoped entry
// can match a pass nested inside a sub-pipeline.
TEST_F(HloPassPipelineTest, EnableOnlyScopedDescendsIntoNestedPipeline) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->mutable_config()
      .mutable_debug_options()
      .clear_xla_disable_hlo_passes();
  module->mutable_config()
      .mutable_debug_options()
      .add_xla_enable_hlo_passes_only("simplification/algsimp");

  std::vector<std::string> run_log;
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<RecordingPass>("algsimp", "top", &run_log);
  auto& sub = pipeline.AddPass<HloPassPipeline>("simplification");
  sub.AddPass<RecordingPass>("algsimp", "in_sub", &run_log);

  ASSERT_OK(pipeline.Run(module.get()).status());
  // Only the algsimp scoped to "simplification" runs; the top-level algsimp is
  // gated out even though the sub-pipeline is still entered.
  EXPECT_THAT(run_log, ElementsAre("in_sub"));
}

// Enable-only with a mix of a plain entry and a rich entry in the same list:
// the union of both matchers runs (every dce invocation plus algsimp #1).
TEST_F(HloPassPipelineTest, EnableOnlyMixedPlainAndRich) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->mutable_config()
      .mutable_debug_options()
      .clear_xla_disable_hlo_passes();
  module->mutable_config()
      .mutable_debug_options()
      .add_xla_enable_hlo_passes_only("dce");
  module->mutable_config()
      .mutable_debug_options()
      .add_xla_enable_hlo_passes_only("algsimp:1");

  std::vector<std::string> run_log;
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<RecordingPass>("algsimp", "a0", &run_log);
  pipeline.AddPass<RecordingPass>("dce", "d0", &run_log);
  pipeline.AddPass<RecordingPass>("algsimp", "a1", &run_log);
  pipeline.AddPass<RecordingPass>("dce", "d1", &run_log);
  pipeline.AddPass<RecordingPass>("algsimp", "a2", &run_log);

  ASSERT_OK(pipeline.Run(module.get()).status());
  // All dce invocations run (plain "dce"); among algsimp only global index 1
  // (a1) runs.
  EXPECT_THAT(run_log, ElementsAre("d0", "a1", "d1"));
}

// "@N" matches the raw pass_id even across nested sub-pipelines. Ids are
// assigned in start order including the pipeline-start pseudo-passes:
//   1 top pipeline-start
//   2 pa (top leaf)
//   3 sub (sub-pipeline pass itself)
//   4 sub pipeline-start
//   5 sb (sub leaf)   <-- target
//   6 sc (sub leaf)
//   7 pd (top leaf)
TEST_F(HloPassPipelineTest, DisableByPassIdNested) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "@5");

  std::vector<std::string> run_log;
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<RecordingPass>("pa", "pa", &run_log);
  auto& sub = pipeline.AddPass<HloPassPipeline>("sub");
  sub.AddPass<RecordingPass>("sb", "sb", &run_log);
  sub.AddPass<RecordingPass>("sc", "sc", &run_log);
  pipeline.AddPass<RecordingPass>("pd", "pd", &run_log);

  ASSERT_OK(pipeline.Run(module.get()).status());
  EXPECT_THAT(run_log, ElementsAre("pa", "sc", "pd"));
}

}  // namespace
}  // namespace xla
