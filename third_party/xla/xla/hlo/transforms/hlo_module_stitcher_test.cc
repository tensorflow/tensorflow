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
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
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
  ROOT custom-call = f32[100] custom-call(param0), custom_call_target="_xla_multi_module_call", backend_config="optimized_sub_module", api_version=API_VERSION_STATUS_RETURNING_UNIFIED, frontend_attributes={inlineable="false"}
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

  absl::flat_hash_map<std::string, const HloModule*> optimized_modules;
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
CHECK:   ROOT %[[CALL:.*]] = f32[100]{{.*}} call(%[[PARAM0]]), to_apply=%sub_entry, frontend_attributes={inlineable="false"}
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
  ROOT custom-call = f32[100] custom-call(param0), custom_call_target="_xla_multi_module_call", backend_config="missing_sub_module", api_version=API_VERSION_STATUS_RETURNING_UNIFIED
}
)";

  ASSERT_OK_AND_ASSIGN(auto main_module,
                       ParseAndReturnVerifiedModule(main_hlo_string));

  absl::flat_hash_map<std::string, const HloModule*> optimized_modules;

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
  ROOT custom-call = f32[100] custom-call(param0, param1), custom_call_target="_xla_multi_module_call", backend_config="optimized_sub_module", api_version=API_VERSION_STATUS_RETURNING_UNIFIED
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

  absl::flat_hash_map<std::string, const HloModule*> optimized_modules;
  optimized_modules["optimized_sub_module"] = sub_module.get();

  HloModuleStitcher stitcher(optimized_modules);
  EXPECT_THAT(stitcher.Run(main_module.get()),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Operand count mismatch")));
}

TEST_F(HloModuleStitcherTest, LayoutReconciliationAddsKCopy) {
  const char* main_hlo_string = R"(
HloModule main

ENTRY main {
  param0 = f32[10,20]{1,0} parameter(0)
  ROOT custom-call = f32[10,20]{1,0} custom-call(param0), custom_call_target="_xla_multi_module_call", backend_config="optimized_sub_module", api_version=API_VERSION_STATUS_RETURNING_UNIFIED, frontend_attributes={inlineable="false"}
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

  absl::flat_hash_map<std::string, const HloModule*> optimized_modules;
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
}  // namespace
}  // namespace xla
