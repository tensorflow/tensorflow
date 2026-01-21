/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/separate_compilation/hlo_module_splitting.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::separate_compilation {
namespace {

using ::testing::UnorderedElementsAreArray;

class SplittingTest : public HloHardwareIndependentTestBase {};

TEST_F(SplittingTest, AllComputationsInBuckets) {
  constexpr absl::string_view module_text = R"(
HloModule simple_module


// Simple alpha equivalence examples

%add.0 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%add.1 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.2 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.3 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

ENTRY %main() -> f32[] {

  %p = f32[] constant(3.3)
  %q = f32[] constant(-1.0)
  %r = f32[] constant(7.1)
  %s = f32[] constant(0.2)

  %res.0 = call(%p, %q), to_apply=%add.0
  %res.1 = call(%p, %q), to_apply=%add.1
  %res.2 = call(%r, %s), to_apply=%add.2
  %res.3 = call(%p, %s), to_apply=%add.3

  %res.01 = f32[] add(f32[] %res.0, f32[] %res.1)
  %res.23 = f32[] add(f32[] %res.2, f32[] %res.3)

  ROOT %result = f32[] add(f32[] %res.01, f32[] %res.23)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_text));
  TF_ASSERT_OK_AND_ASSIGN(auto splits, GroupComputationsForSplitting(*module));

  absl::flat_hash_set<const HloComputation*> all_computations_in_splits;
  for (const auto& bucket : splits) {
    all_computations_in_splits.insert(bucket.begin(), bucket.end());
  }
  EXPECT_THAT(all_computations_in_splits,
              UnorderedElementsAreArray(module->computations()));
}

TEST_F(SplittingTest, CreateModule) {
  constexpr absl::string_view module_text = R"(
HloModule simple_module


// Simple alpha equivalence examples

%add.0 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%add.1 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.2 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.3 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

ENTRY %main() -> f32[] {

  %p = f32[] constant(3.3)
  %q = f32[] constant(-1.0)
  %r = f32[] constant(7.1)
  %s = f32[] constant(0.2)

  %res.0 = call(%p, %q), to_apply=%add.0
  %res.1 = call(%p, %q), to_apply=%add.1
  %res.2 = call(%r, %s), to_apply=%add.2
  %res.3 = call(%p, %s), to_apply=%add.3

  %res.01 = f32[] add(f32[] %res.0, f32[] %res.1)
  %res.23 = f32[] add(f32[] %res.2, f32[] %res.3)

  ROOT %result = f32[] add(f32[] %res.01, f32[] %res.23)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_text));
  auto* main = FindComputation(module.get(), "main");

  TF_ASSERT_OK_AND_ASSIGN(auto module_split,
                          CreateHloModuleSplit(*module, {main}));

  const int kMainModuleComputationCount = 5;  // the main + 4 stubs
  // const int kMainModuleClonedComputations = 1;
  const int kMainModuleCallSitesCount = 4;
  EXPECT_EQ(module_split->module.computation_count(),
            kMainModuleComputationCount);
  EXPECT_EQ(module_split->stub_map.size(), kMainModuleCallSitesCount);
  EXPECT_EQ(module_split->call_sites.size(), kMainModuleCallSitesCount);
}

TEST_F(SplittingTest, SplitModule) {
  constexpr absl::string_view module_text = R"(
HloModule simple_module


// Simple alpha equivalence examples

%fusion.1 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %add = f32[] negate(f32[] %x)
}

%add.0 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%add.1 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.2 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%fusion.2 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %add = f32[] negate(f32[] %x)
}

ENTRY %main() -> f32[] {

  %p = f32[] constant(3.3)
  %q = f32[] constant(-1.0)
  %r = f32[] constant(7.1)
  %s = f32[] constant(0.2)

  %res.0 = call(%p, %q), to_apply=%add.0
  %res.1 = call(%p, %q), to_apply=%add.1
  %res.2 = call(%r, %s), to_apply=%add.2
  %res.3 = call(%p, %s), to_apply=%add.2
  %fusion = f32[] fusion(%res.3), kind=kLoop, calls=%fusion.2

  %res.01 = f32[] add(f32[] %res.0, f32[] %res.1)
  %res.23 = f32[] add(f32[] %res.2, f32[] %res.3)
  %res.fu23 = f32[] add(f32[] %fusion, f32[] %res.23)

  ROOT %result = f32[] add(f32[] %res.01, f32[] %res.fu23)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_text));
  TF_ASSERT_OK_AND_ASSIGN(auto module_split_group,
                          CreateHloModuleSplitGroup(*module));

  TF_ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                          PlatformUtil::GetPlatform("cpu"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler,
                          Compiler::GetForPlatform(platform->id()));
  ASSERT_OK(compiler->RunHloPasses(module->Clone(), /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
  for (auto& split : module_split_group->module_splits) {
    EXPECT_OK(compiler->RunHloPasses(split->submodule->Clone(),
                                     /*executor=*/nullptr,
                                     Compiler::CompileOptions{}));
  }
}

TEST_F(SplittingTest, SplitDiamondGraphModule) {
  constexpr absl::string_view module_text = R"(
    HloModule shared_callee_module

    %y {
      %p = f32[] parameter(0)
      ROOT %result = f32[] exponential(%p)
    }

    %x {
      %p = f32[] parameter(0)
      %call_y = f32[] call(%p), to_apply=%y
      ROOT %result = f32[] cosine(%call_y)
    }

    %a {
      %p = f32[] parameter(0)
      %c = f32[] constant(5.0)
      %call_x = f32[] call(%p), to_apply=%x
      ROOT %result = f32[] add(%call_x, %c)
    }

    %b {
      %p = f32[] parameter(0)
      %c = f32[] constant(10.0)
      %call_x = f32[] call(%p), to_apply=%x
      ROOT %result = f32[] subtract(%call_x, %c)
    }

    ENTRY %entry {
      %p_entry = f32[] parameter(0)
      %call_a = f32[] call(%p_entry), to_apply=%a
      %call_b = f32[] call(%p_entry), to_apply=%b
      ROOT %result = f32[] add(%call_a, %call_b)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_text));
  TF_ASSERT_OK_AND_ASSIGN(auto module_split_group,
                          CreateHloModuleSplitGroup(*module));

  // Expect all computations to be assigned somewhere.
  for (auto original_comp : module->computations()) {
    EXPECT_TRUE(module_split_group->address_book.contains(original_comp));
  }

  EXPECT_EQ(module_split_group->module_splits.size(),
            module->computation_count());

  TF_ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                          PlatformUtil::GetPlatform("cpu"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler,
                          Compiler::GetForPlatform(platform->id()));
  ASSERT_OK(compiler->RunHloPasses(module->Clone(), /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
  for (auto& split : module_split_group->module_splits) {
    EXPECT_OK(compiler->RunHloPasses(split->submodule->Clone(),
                                     /*executor=*/nullptr,
                                     Compiler::CompileOptions{}));
  }
}

TEST_F(SplittingTest, SplitModuleWithSharedComputations) {
  constexpr absl::string_view module_text = R"(
HloModule simple_module


// Simple alpha equivalence examples

%fusion.1 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %add = f32[] negate(f32[] %x)
}

%fusion.2 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %add = f32[] negate(f32[] %x)
}

%add.0 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%add.1 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.2 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

ENTRY %main() -> f32[] {

  %p = f32[] constant(3.3)
  %q = f32[] constant(-1.0)
  %r = f32[] constant(7.1)
  %s = f32[] constant(0.2)

  %res.0 = call(%p, %q), to_apply=%add.0
  %res.1 = call(%p, %q), to_apply=%add.1
  %res.1f = f32[] fusion(%res.1), kind=kLoop, calls=%fusion.1
  %res.2 = call(%r, %s), to_apply=%add.2
  %res.3 = call(%p, %s), to_apply=%add.2
  %fusion = f32[] fusion(%res.3), kind=kLoop, calls=%fusion.2

  %res.01 = f32[] add(f32[] %res.0, f32[] %res.1f)
  %res.23 = f32[] add(f32[] %res.2, f32[] %res.3)
  %res.f = f32[] add(f32[] %fusion, f32[], %res.23)

  ROOT %result = f32[] add(f32[] %res.01, f32[] %res.f)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_text));
  TF_ASSERT_OK_AND_ASSIGN(auto module_split_group,
                          CreateHloModuleSplitGroup(*module));

  TF_ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                          PlatformUtil::GetPlatform("cpu"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler,
                          Compiler::GetForPlatform(platform->id()));
  ASSERT_OK(compiler->RunHloPasses(module->Clone(), /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
  for (auto& split : module_split_group->module_splits) {
    EXPECT_OK(compiler->RunHloPasses(split->submodule->Clone(),
                                     /*executor=*/nullptr,
                                     Compiler::CompileOptions{}));
  }
}

}  // namespace
}  // namespace xla::separate_compilation
