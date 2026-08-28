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

#include "xla/hlo/transforms/hlo_module_stitcher.h"

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

class HloModuleStitcherTest : public HloHardwareIndependentTestBase {};

TEST_F(HloModuleStitcherTest, SuccessfullyStitchesModule) {
  const char* main_hlo_string = R"(
HloModule main

ENTRY main {
  param0 = f32[100] parameter(0)
  ROOT custom-call = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="optimized_sub_module",
    api_version=API_VERSION_TYPED_FFI,
    frontend_attributes={inlineable="false"}
}
)";

  const char* sub_hlo_string = R"(
HloModule optimized_sub_module

ENTRY sub_entry {
  param0 = f32[100] parameter(0)
  ROOT add = f32[100] add(param0, param0)
}
)";

  ASSERT_OK_AND_ASSIGN(auto main_module,
                       ParseAndReturnVerifiedModule(main_hlo_string));
  ASSERT_OK_AND_ASSIGN(auto sub_module,
                       ParseAndReturnVerifiedModule(sub_hlo_string));

  absl::flat_hash_map<std::string, HloModule*> optimized_modules;
  optimized_modules["optimized_sub_module"] = sub_module.get();

  HloModuleStitcher stitcher(optimized_modules);
  EXPECT_THAT(stitcher.Run(main_module.get()),
              absl_testing::IsOkAndHolds(true));

  const char* expected_hlo = R"(
CHECK: %sub_entry
CHECK:   %[[PARAM_SUB:.*]] = f32[100]{{.*}} parameter(0)
CHECK:   ROOT %[[ADD:.*]] = f32[100]{{.*}} add(%[[PARAM_SUB]], %[[PARAM_SUB]])

CHECK: ENTRY %main
CHECK:   %[[PARAM0:.*]] = f32[100]{{.*}} parameter(0)
CHECK:   ROOT %[[CALL:.*]] = f32[100]{{.*}} call(%[[PARAM0]]),
CHECK-SAME: to_apply=%sub_entry, frontend_attributes={inlineable="false"}
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_ok,
                       RunFileCheck(main_module->ToString(), expected_hlo));
  EXPECT_TRUE(filecheck_ok);

  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision=*/false);
  EXPECT_TRUE(verifier.Run(main_module.get()).status().ok());
}

TEST_F(HloModuleStitcherTest, SubModuleNotFoundReturnsError) {
  const char* main_hlo_string = R"(
HloModule main

ENTRY main {
  param0 = f32[100] parameter(0)
  ROOT custom-call = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="missing_sub_module",
    api_version=API_VERSION_TYPED_FFI
}
)";

  ASSERT_OK_AND_ASSIGN(auto main_module,
                       ParseAndReturnVerifiedModule(main_hlo_string));

  absl::flat_hash_map<std::string, HloModule*> optimized_modules;

  HloModuleStitcher stitcher(optimized_modules);
  EXPECT_THAT(stitcher.Run(main_module.get()),
              absl_testing::StatusIs(
                  absl::StatusCode::kNotFound,
                  HasSubstr("Sub-module missing_sub_module not found")));
}

TEST_F(HloModuleStitcherTest, OperandCountMismatchReturnsError) {
  const char* main_hlo_string = R"(
HloModule main

ENTRY main {
  param0 = f32[100] parameter(0)
  param1 = f32[100] parameter(1)
  ROOT custom-call = f32[100] custom-call(param0, param1),
    custom_call_target="_xla_multi_module_call",
    backend_config="optimized_sub_module",
    api_version=API_VERSION_TYPED_FFI
}
)";

  const char* sub_hlo_string = R"(
HloModule optimized_sub_module

ENTRY sub_entry {
  param0 = f32[100] parameter(0)
  ROOT add = f32[100] add(param0, param0)
}
)";

  ASSERT_OK_AND_ASSIGN(auto main_module,
                       ParseAndReturnVerifiedModule(main_hlo_string));
  ASSERT_OK_AND_ASSIGN(auto sub_module,
                       ParseAndReturnVerifiedModule(sub_hlo_string));

  absl::flat_hash_map<std::string, HloModule*> optimized_modules;
  optimized_modules["optimized_sub_module"] = sub_module.get();

  HloModuleStitcher stitcher(optimized_modules);
  EXPECT_THAT(stitcher.Run(main_module.get()),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Operand count mismatch")));
}

TEST_F(HloModuleStitcherTest, NullSubModuleReturnsError) {
  const char* main_hlo_string = R"(
HloModule main

ENTRY main {
  param0 = f32[100] parameter(0)
  ROOT custom-call = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="null_sub_module",
    api_version=API_VERSION_TYPED_FFI
}
)";

  ASSERT_OK_AND_ASSIGN(auto main_module,
                       ParseAndReturnVerifiedModule(main_hlo_string));

  absl::flat_hash_map<std::string, HloModule*> optimized_modules;
  optimized_modules["null_sub_module"] = nullptr;

  HloModuleStitcher stitcher(optimized_modules);
  EXPECT_THAT(stitcher.Run(main_module.get()),
              absl_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(HloModuleStitcherTest, UniquelyClonesSharedComputations) {
  const char* main_hlo_string = R"(
HloModule main

ENTRY main {
  param0 = f32[100] parameter(0)
  call1 = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="optimized_sub_module",
    api_version=API_VERSION_TYPED_FFI
  ROOT call2 = f32[100] custom-call(call1),
    custom_call_target="_xla_multi_module_call",
    backend_config="optimized_sub_module",
    api_version=API_VERSION_TYPED_FFI
}
)";

  const char* sub_hlo_string = R"(
HloModule optimized_sub_module

ENTRY sub_entry {
  param0 = f32[100] parameter(0)
  ROOT add = f32[100] add(param0, param0)
}
)";

  ASSERT_OK_AND_ASSIGN(auto main_module,
                       ParseAndReturnVerifiedModule(main_hlo_string));
  ASSERT_OK_AND_ASSIGN(auto sub_module,
                       ParseAndReturnVerifiedModule(sub_hlo_string));

  absl::flat_hash_map<std::string, HloModule*> optimized_modules;
  optimized_modules["optimized_sub_module"] = sub_module.get();

  HloModuleStitcher stitcher(optimized_modules);
  EXPECT_THAT(stitcher.Run(main_module.get()),
              absl_testing::IsOkAndHolds(true));

  // Count occurrences of sub_entry. Since we do not cache cloned computations
  // across custom calls, it should appear twice (uniquely cloned for each call)
  // to avoid BufferAssignment crashes on shared stitched computations.
  int sub_entry_count = 0;
  for (auto* comp : main_module->computations()) {
    if (absl::StrContains(comp->name(), "sub_entry")) {
      sub_entry_count++;
    }
  }
  EXPECT_EQ(sub_entry_count, 2);
}

TEST_F(HloModuleStitcherTest, LayoutReconciliationAddsKCopy) {
  const char* main_hlo_string = R"(
HloModule main

ENTRY main {
  param0 = f32[10,20]{1,0} parameter(0)
  ROOT custom-call = f32[10,20]{1,0} custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="optimized_sub_module",
    api_version=API_VERSION_TYPED_FFI,
    frontend_attributes={inlineable="false"}
}
)";

  const char* sub_hlo_string = R"(
HloModule optimized_sub_module

ENTRY sub_entry {
  param0 = f32[10,20]{0,1} parameter(0)
  ROOT add = f32[10,20]{0,1} add(param0, param0)
}
)";

  ASSERT_OK_AND_ASSIGN(auto main_module,
                       ParseAndReturnVerifiedModule(main_hlo_string));
  ASSERT_OK_AND_ASSIGN(auto sub_module,
                       ParseAndReturnVerifiedModule(sub_hlo_string));

  absl::flat_hash_map<std::string, HloModule*> optimized_modules;
  optimized_modules["optimized_sub_module"] = sub_module.get();

  HloModuleStitcher stitcher(optimized_modules);
  EXPECT_THAT(stitcher.Run(main_module.get()),
              absl_testing::IsOkAndHolds(true));

  const char* expected_hlo = R"(
CHECK: ENTRY %main
CHECK:   %[[PARAM0:.*]] = f32[10,20]{1,0} parameter(0)
CHECK:   %[[OP_COPY:.*]] = f32[10,20]{0,1} copy(%[[PARAM0]])
CHECK:   %[[CALL:.*]] = f32[10,20]{0,1} call(%[[OP_COPY]]), to_apply=%sub_entry
CHECK:   ROOT %[[ROOT_COPY:.*]] = f32[10,20]{1,0} copy(%[[CALL]])
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_ok,
                       RunFileCheck(main_module->ToString(), expected_hlo));
  EXPECT_TRUE(filecheck_ok);

  HloVerifier verifier(/*layout_sensitive=*/true,
                       /*allow_mixed_precision=*/false);
  EXPECT_TRUE(verifier.Run(main_module.get()).status().ok());
}

TEST_F(HloModuleStitcherTest, SuccessfullyStitchesNestedSubmodules) {
  const char* main_hlo_string = R"(
HloModule main

ENTRY main {
  param0 = f32[100] parameter(0)
  ROOT custom-call = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="outer_sub_module",
    api_version=API_VERSION_TYPED_FFI
}
)";

  const char* outer_sub_hlo_string = R"(
HloModule outer_sub_module

ENTRY outer_entry {
  param0 = f32[100] parameter(0)
  ROOT custom-call = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="inner_sub_module",
    api_version=API_VERSION_TYPED_FFI
}
)";

  const char* inner_sub_hlo_string = R"(
HloModule inner_sub_module

ENTRY inner_entry {
  param0 = f32[100] parameter(0)
  ROOT add = f32[100] add(param0, param0)
}
)";

  ASSERT_OK_AND_ASSIGN(auto main_module,
                       ParseAndReturnVerifiedModule(main_hlo_string));
  ASSERT_OK_AND_ASSIGN(auto outer_module,
                       ParseAndReturnVerifiedModule(outer_sub_hlo_string));
  ASSERT_OK_AND_ASSIGN(auto inner_module,
                       ParseAndReturnVerifiedModule(inner_sub_hlo_string));

  absl::flat_hash_map<std::string, HloModule*> optimized_modules;
  optimized_modules["outer_sub_module"] = outer_module.get();
  optimized_modules["inner_sub_module"] = inner_module.get();

  HloModuleStitcher stitcher(optimized_modules);
  EXPECT_THAT(stitcher.Run(main_module.get()),
              absl_testing::IsOkAndHolds(true));

  const char* expected_hlo = R"(
CHECK: %inner_entry
CHECK:   %[[PARAM_INNER:.*]] = f32[100]{{.*}} parameter(0)
CHECK:   ROOT %[[ADD:.*]] = f32[100]{{.*}} add(%[[PARAM_INNER]], %[[PARAM_INNER]])

CHECK: %outer_entry
CHECK:   %[[PARAM_OUTER:.*]] = f32[100]{{.*}} parameter(0)
CHECK:   ROOT %[[CALL_INNER:.*]] = f32[100]{{.*}} call(%[[PARAM_OUTER]]),
CHECK-SAME: to_apply=%inner_entry

CHECK: ENTRY %main
CHECK:   %[[PARAM0:.*]] = f32[100]{{.*}} parameter(0)
CHECK:   ROOT %[[CALL_OUTER:.*]] = f32[100]{{.*}} call(%[[PARAM0]]),
CHECK-SAME: to_apply=%outer_entry
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_ok,
                       RunFileCheck(main_module->ToString(), expected_hlo));
  EXPECT_TRUE(filecheck_ok);

  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision=*/false);
  EXPECT_TRUE(verifier.Run(main_module.get()).status().ok());
}

TEST_F(HloModuleStitcherTest, StitcherDetectsCircularDependency) {
  const char* main_hlo = R"(
HloModule main
ENTRY main {
  param0 = f32[100] parameter(0)
  ROOT custom-call = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="sub1",
    api_version=API_VERSION_TYPED_FFI
}
)";

  const char* sub1_hlo = R"(
HloModule sub1
ENTRY sub1_entry {
  param0 = f32[100] parameter(0)
  ROOT custom-call = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="sub2",
    api_version=API_VERSION_TYPED_FFI
}
)";

  const char* sub2_hlo = R"(
HloModule sub2
ENTRY sub2_entry {
  param0 = f32[100] parameter(0)
  ROOT custom-call = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="sub1",
    api_version=API_VERSION_TYPED_FFI
}
)";

  ASSERT_OK_AND_ASSIGN(auto main_module,
                       ParseAndReturnVerifiedModule(main_hlo));
  ASSERT_OK_AND_ASSIGN(auto sub1_module,
                       ParseAndReturnVerifiedModule(sub1_hlo));
  ASSERT_OK_AND_ASSIGN(auto sub2_module,
                       ParseAndReturnVerifiedModule(sub2_hlo));

  absl::flat_hash_map<std::string, HloModule*> optimized_modules;
  optimized_modules["sub1"] = sub1_module.get();
  optimized_modules["sub2"] = sub2_module.get();

  HloModuleStitcher stitcher(optimized_modules);
  absl::Status status = stitcher.Run(main_module.get()).status();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("Circular dependency detected"));
}

TEST_F(HloModuleStitcherTest, StitcherHandlesDiamondDependency) {
  const char* main_hlo = R"(
HloModule main
ENTRY main {
  param0 = f32[100] parameter(0)
  c1 = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="sub1",
    api_version=API_VERSION_TYPED_FFI
  c2 = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="sub2",
    api_version=API_VERSION_TYPED_FFI
  ROOT add = f32[100] add(c1, c2)
}
)";

  const char* sub1_hlo = R"(
HloModule sub1
ENTRY sub1_entry {
  param0 = f32[100] parameter(0)
  ROOT custom-call = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="shared_sub",
    api_version=API_VERSION_TYPED_FFI
}
)";

  const char* sub2_hlo = R"(
HloModule sub2
ENTRY sub2_entry {
  param0 = f32[100] parameter(0)
  ROOT custom-call = f32[100] custom-call(param0),
    custom_call_target="_xla_multi_module_call",
    backend_config="shared_sub",
    api_version=API_VERSION_TYPED_FFI
}
)";

  const char* shared_sub_hlo = R"(
HloModule shared_sub
ENTRY shared_entry {
  param0 = f32[100] parameter(0)
  ROOT add = f32[100] add(param0, param0)
}
)";

  ASSERT_OK_AND_ASSIGN(auto main_module,
                       ParseAndReturnVerifiedModule(main_hlo));
  ASSERT_OK_AND_ASSIGN(auto sub1_module,
                       ParseAndReturnVerifiedModule(sub1_hlo));
  ASSERT_OK_AND_ASSIGN(auto sub2_module,
                       ParseAndReturnVerifiedModule(sub2_hlo));
  ASSERT_OK_AND_ASSIGN(auto shared_sub_module,
                       ParseAndReturnVerifiedModule(shared_sub_hlo));

  absl::flat_hash_map<std::string, HloModule*> optimized_modules;
  optimized_modules["sub1"] = sub1_module.get();
  optimized_modules["sub2"] = sub2_module.get();
  optimized_modules["shared_sub"] = shared_sub_module.get();

  HloModuleStitcher stitcher(optimized_modules);
  EXPECT_THAT(stitcher.Run(main_module.get()),
              absl_testing::IsOkAndHolds(true));

  // Verify it is fully stitched (no custom calls left)
  for (const auto* comp : main_module->computations()) {
    for (const auto* inst : comp->instructions()) {
      EXPECT_NE(inst->opcode(), HloOpcode::kCustomCall)
          << "Found remaining custom call: " << inst->ToString();
    }
  }

  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision=*/false);
  EXPECT_TRUE(verifier.Run(main_module.get()).status().ok());
}

}  // namespace
}  // namespace xla
