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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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
      std::reverse(name.begin(), name.end());
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<FooToBarModulePass>();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->name(), "foo");
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<FooToBarModulePass>();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<ReverseStringModulePass>();

  HloInstruction* main_root = module->entry_computation()->root_instruction();
  HloInstruction* parallel_thread_root =
      main_root->async_wrapped_computation()->root_instruction();
  EXPECT_EQ(main_root->name(), "baz");
  EXPECT_EQ(parallel_thread_root->name(), "foo");
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<ReverseStringModulePass>();

  HloInstruction* main_root = module->entry_computation()->root_instruction();
  HloInstruction* parallel_thread_root =
      main_root->async_wrapped_computation()->root_instruction();
  EXPECT_EQ(main_root->name(), "baz");
  EXPECT_EQ(parallel_thread_root->name(), "foo");
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_0_str));

  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<BazToQuxModulePass>();
  pipeline.AddPass<FooToBarModulePass>();

  HloInstruction* root0 = module->entry_computation()->root_instruction();
  EXPECT_EQ(root0->name(), "baz");

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  {
    // Run a pipeline with just the invariant checker. It should not fail
    // because there is no 'bar' instruction in the module.
    HloPassPipeline pipeline(TestName());
    pipeline.AddInvariantChecker<BarBlowerUpper>();

    TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
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
  ASSERT_OK(pipeline.Run(module.get()).status());

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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_unsupported_crash_on_hlo_pass_silent_hlo_change(true);
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<NoOpModulePass>();

  absl::Status status = pipeline.Run(module.get()).status();
  EXPECT_OK(status);
}

// TODO: Add test.

}  // namespace
}  // namespace xla
