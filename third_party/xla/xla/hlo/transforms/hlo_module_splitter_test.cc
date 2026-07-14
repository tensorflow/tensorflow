/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/hlo_module_splitter.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"

namespace xla {
namespace {

class HloModuleSplitterTest : public HloHardwareIndependentTestBase {
 protected:
  struct SplitResult {
    std::unique_ptr<HloModule> module;
    std::vector<std::unique_ptr<HloModule>> submodules;
  };

  absl::StatusOr<SplitResult> RunSplitter(absl::string_view hlo_string) {
    std::unique_ptr<HloModule> module;
    ASSIGN_OR_RETURN(module, ParseAndReturnVerifiedModule(hlo_string));
    HloModuleSplitter splitter;
    bool changed = false;
    ASSIGN_OR_RETURN(changed, splitter.Run(module.get()));
    if (!changed) {
      return Internal("Expected splitter to run and change the module");
    }
    for (auto& submodule : splitter.submodules()) {
      RETURN_IF_ERROR(HloVerifier(/*layout_sensitive=*/false,
                                  /*allow_mixed_precision=*/true)
                          .Run(submodule.get())
                          .status());
    }
    return SplitResult{std::move(module), std::move(splitter.submodules())};
  }
};

TEST_F(HloModuleSplitterTest, SplitNonInlineableComputation) {
  const char* hlo_string = R"(
HloModule module
callee {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}
ENTRY entry {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT call = f32[] call(p0, p1), to_apply=callee, frontend_attributes={compilation_unit="my_callee"}
}
)";

  ASSERT_OK_AND_ASSIGN(auto split_result, RunSplitter(hlo_string));
  auto module = std::move(split_result.module);
  auto submodules = std::move(split_result.submodules);

  const char* expected_hlo = R"(
CHECK: ENTRY %entry
CHECK:   ROOT {{.*}} custom-call({{.*}}), custom_call_target="_xla_multi_module_call",{{.*}}backend_config="my_callee"
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_ok,
                       RunFileCheck(module->ToString(), expected_hlo));
  EXPECT_TRUE(filecheck_ok);

  EXPECT_EQ(submodules.size(), 1);

  const char* expected_submodule_hlo = R"(
CHECK: ENTRY %callee
CHECK:   ROOT {{.*}} add(
)";

  ASSERT_OK_AND_ASSIGN(
      bool sub_filecheck_ok,
      RunFileCheck(submodules[0]->ToString(), expected_submodule_hlo));
  EXPECT_TRUE(sub_filecheck_ok);

  EXPECT_TRUE(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .ok());
}

TEST_F(HloModuleSplitterTest, NestedNonInlineableComputations) {
  const char* hlo_string = R"(
HloModule module
inner {
  p0 = f32[] parameter(0)
  ROOT neg = f32[] negate(p0)
}
outer {
  p0 = f32[] parameter(0)
  ROOT call = f32[] call(p0), to_apply=inner, frontend_attributes={compilation_unit="my_inner"}
}
ENTRY entry {
  p0 = f32[] parameter(0)
  ROOT call = f32[] call(p0), to_apply=outer, frontend_attributes={compilation_unit="my_outer"}
}
)";

  ASSERT_OK_AND_ASSIGN(auto split_result, RunSplitter(hlo_string));
  auto module = std::move(split_result.module);
  auto submodules = std::move(split_result.submodules);

  // We should have 2 submodules: inner and outer.
  EXPECT_EQ(submodules.size(), 2);

  const char* expected_hlo = R"(
CHECK: ENTRY %entry
CHECK:   ROOT {{.*}} custom-call({{.*}}), custom_call_target="_xla_multi_module_call",{{.*}}backend_config="my_outer"
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_ok,
                       RunFileCheck(module->ToString(), expected_hlo));
  EXPECT_TRUE(filecheck_ok);

  HloModule* outer_mod = nullptr;
  for (const auto& m : submodules) {
    if (m->name() == "my_outer") {
      outer_mod = m.get();
    }
  }
  ASSERT_NE(outer_mod, nullptr);

  const char* expected_outer_hlo = R"(
CHECK: ENTRY %outer
CHECK:   ROOT {{.*}} custom-call({{.*}}), custom_call_target="_xla_multi_module_call",{{.*}}backend_config="my_inner"
)";

  ASSERT_OK_AND_ASSIGN(bool outer_filecheck_ok,
                       RunFileCheck(outer_mod->ToString(), expected_outer_hlo));
  EXPECT_TRUE(outer_filecheck_ok);

  EXPECT_TRUE(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .ok());
}

TEST_F(HloModuleSplitterTest, SideEffectComputation) {
  const char* hlo_string = R"(
HloModule module
callee {
  p0 = f32[] parameter(0)
  ROOT outfeed = token[] outfeed(p0, token[] after-all()), outfeed_config="abc"
}
ENTRY entry {
  p0 = f32[] parameter(0)
  ROOT call = token[] call(p0), to_apply=callee, frontend_attributes={compilation_unit="my_callee"}
}
)";

  ASSERT_OK_AND_ASSIGN(auto split_result, RunSplitter(hlo_string));
  auto module = std::move(split_result.module);
  auto submodules = std::move(split_result.submodules);

  const char* expected_hlo = R"(
CHECK: ENTRY %entry
CHECK:   ROOT {{.*}} custom-call({{.*}}), custom_call_target="_xla_multi_module_call",{{.*}}custom_call_has_side_effect=true,{{.*}}backend_config="my_callee"
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_ok,
                       RunFileCheck(module->ToString(), expected_hlo));
  EXPECT_TRUE(filecheck_ok);

  EXPECT_TRUE(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .ok());
}

TEST_F(HloModuleSplitterTest, ReduceComputation) {
  const char* hlo_string = R"(
HloModule module
reducer {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}
callee {
  p0 = f32[10] parameter(0)
  init = f32[] constant(0.0)
  ROOT reduce = f32[] reduce(p0, init), dimensions={0}, to_apply=reducer
}
ENTRY entry {
  p0 = f32[10] parameter(0)
  ROOT call = f32[] call(p0), to_apply=callee, frontend_attributes={compilation_unit="my_callee"}
}
)";

  ASSERT_OK_AND_ASSIGN(auto split_result, RunSplitter(hlo_string));
  auto module = std::move(split_result.module);
  auto submodules = std::move(split_result.submodules);

  const char* expected_hlo = R"(
CHECK: ENTRY %entry
CHECK:   ROOT {{.*}} custom-call({{.*}}), custom_call_target="_xla_multi_module_call",{{.*}}backend_config="my_callee"
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_ok,
                       RunFileCheck(module->ToString(), expected_hlo));
  EXPECT_TRUE(filecheck_ok);
}

TEST_F(HloModuleSplitterTest, SplitterClearSubmodulesOnReuse) {
  const char* hlo_string = R"(
HloModule module
callee {
  p0 = f32[] parameter(0)
  ROOT neg = f32[] negate(p0)
}
ENTRY entry {
  p0 = f32[] parameter(0)
  ROOT call = f32[] call(p0), to_apply=callee, frontend_attributes={compilation_unit="my_callee"}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module1, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto module2, ParseAndReturnVerifiedModule(hlo_string));

  HloModuleSplitter splitter;
  ASSERT_OK_AND_ASSIGN(bool changed1, splitter.Run(module1.get()));
  EXPECT_TRUE(changed1);
  EXPECT_EQ(splitter.submodules().size(), 1);

  ASSERT_OK_AND_ASSIGN(bool changed2, splitter.Run(module2.get()));
  EXPECT_TRUE(changed2);
  // Verify that submodules from the first run were cleared, so size is still 1
  // (not 2).
  EXPECT_EQ(splitter.submodules().size(), 1);
}

TEST_F(HloModuleSplitterTest, IgnoreNonCallWithInlineableFalse) {
  const char* hlo_string = R"(
HloModule module
ENTRY entry {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1), frontend_attributes={compilation_unit="add"}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  HloModuleSplitter splitter;
  ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  // Verify that the splitter ignored the non-kCall instruction.
  EXPECT_FALSE(changed);
  EXPECT_TRUE(splitter.submodules().empty());
}

TEST_F(HloModuleSplitterTest, IgnoreCallWithInlineableFalseOnly) {
  const char* hlo_string = R"(
HloModule module
callee {
  p0 = f32[] parameter(0)
  ROOT neg = f32[] negate(p0)
}
ENTRY entry {
  p0 = f32[] parameter(0)
  ROOT call = f32[] call(p0), to_apply=callee, frontend_attributes={inlineable="false"}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  HloModuleSplitter splitter;
  ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  // Verify that the splitter ignored the call because it only has
  // inlineable="false" (no compilation_unit)
  EXPECT_FALSE(changed);
  EXPECT_TRUE(splitter.submodules().empty());
}

}  // namespace
}  // namespace xla
