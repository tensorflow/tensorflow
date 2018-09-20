/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class HloPassPipelineTest : public HloVerifiedTestBase {
 protected:
  StatusOr<HloModuleGroup> ParseModuleGroup(
      absl::Span<const string> hlo_strings) {
    HloModuleGroup group(TestName());
    for (const string& hlo_string : hlo_strings) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
      group.push_back(std::move(module));
    }
    return std::move(group);
  }
};

// A module pass which renames instructions named 'foo' to 'bar'.
class FooToBarModulePass : public HloModulePass {
  absl::string_view name() const override { return "foo2bar"; }

  StatusOr<bool> Run(HloModule* module) override {
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

// A module group pass which renames instructions named 'baz' to 'qux'.
class BazToQuxModuleGroupPass : public HloModuleGroupPass {
  absl::string_view name() const override { return "baz2qux"; }

  StatusOr<bool> RunOnModuleGroup(HloModuleGroup* module_group) override {
    bool changed = false;
    for (HloModule* module : module_group->modules()) {
      for (HloComputation* computation : module->computations()) {
        for (HloInstruction* instruction : computation->instructions()) {
          if (instruction->name() == "baz") {
            instruction->SetAndSanitizeName("qux");
            changed = true;
          }
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

  StatusOr<bool> Run(HloModule* module) override {
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->name() == "bar") {
          return InternalError("Module has instruction named bar");
        }
      }
    }
    return false;
  }
};

TEST_F(HloPassPipelineTest, ModulePassChanged) {
  // Test an HLO module pass which changes a module.
  const string module_str = R"(
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
  const string module_str = R"(
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

TEST_F(HloPassPipelineTest, MixedPipeline) {
  // Test a pipeline with both a module pass and a module group pass.
  const string module_0_str = R"(
HloModule MixedPipeline.1

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT baz = f32[] multiply(a, b)
}
)";
  const string module_1_str = R"(
HloModule MixedPipeline.0

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT foo = f32[] multiply(a, b)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModuleGroup module_group,
                          ParseModuleGroup({module_0_str, module_1_str}));

  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<BazToQuxModuleGroupPass>();
  pipeline.AddPass<FooToBarModulePass>();

  HloInstruction* root0 =
      module_group.module(0).entry_computation()->root_instruction();
  HloInstruction* root1 =
      module_group.module(1).entry_computation()->root_instruction();
  EXPECT_EQ(root0->name(), "baz");
  EXPECT_EQ(root1->name(), "foo");

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          pipeline.RunOnModuleGroup(&module_group));
  EXPECT_TRUE(changed);

  EXPECT_EQ(root0->name(), "qux");
  EXPECT_EQ(root1->name(), "bar");
}

TEST_F(HloPassPipelineTest, InvariantChecker) {
  const string module_str = R"(
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

    Status status = pipeline.Run(module.get()).status();
    ASSERT_IS_NOT_OK(status);
    EXPECT_THAT(status.error_message(),
                ::testing::HasSubstr("Module has instruction named bar"));
    EXPECT_THAT(status.error_message(),
                ::testing::HasSubstr("Failed after foo2bar"));
  }

  {
    // Run the invariant-checker only pipeline again. It should fail this time.
    HloPassPipeline pipeline(TestName());
    pipeline.AddInvariantChecker<BarBlowerUpper>();

    Status status = pipeline.Run(module.get()).status();
    ASSERT_IS_NOT_OK(status);
    EXPECT_THAT(status.error_message(),
                ::testing::HasSubstr("Module has instruction named bar"));
    EXPECT_THAT(status.error_message(),
                ::testing::HasSubstr("Failed after pipeline-start"));
  }
}

TEST_F(HloPassPipelineTest, ModuleGroupPassOnModule) {
  // Running a module group pass on a module should produce an error.
  const string module_str = R"(
HloModule ModuleGroupPassOnModule

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT foo = f32[] multiply(a, b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<BazToQuxModuleGroupPass>();

  Status status = pipeline.Run(module.get()).status();
  ASSERT_IS_NOT_OK(status);
  EXPECT_THAT(
      status.error_message(),
      ::testing::HasSubstr("Module group pass cannot be run on a module"));
}

}  // namespace
}  // namespace xla
