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

#include "xla/tools/prepare_reference_module.h"

#include "xla/hlo/ir/hlo_module.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

const char* const kModuleStr = R"(
  HloModule jit_step

  %fused_computation (param_0.2: f32[1,4]) -> f32[1,3] {
    %param_0.2 = f32[1,4]{1,0} parameter(0)
    ROOT %slice.11 = f32[1,3]{1,0} slice(f32[1,4]{1,0} %param_0.2),
      slice={[0:1], [0:3]}
  }

  ENTRY %main.3491 (Arg_0.0: f32[1,4]) -> f32[1,3] {
    %Arg_0.0 = f32[1,4]{1,0} parameter(0)
    ROOT %fusion = f32[1,3]{1,0} fusion(f32[1,4]{1,0} %Arg_0.0), kind=kLoop,
      calls=%fused_computation
  }
)";

using PrepareReferenceModuleTest = HloTestBase;

// Ideally 'Despecializer' pass should be mocked. Because it is not feasible
// with the current design, despecialization tests in this file are based on
// Despecializer's implementation (Despecializer removes fusion op from the
// module).
TEST_F(PrepareReferenceModuleTest, PerformDespecialization) {
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto reference_module,
      PrepareReferenceModule(*test_module, nullptr, {}, {},
                             /*skip_despecialization=*/false));

  // Fusion op should have been removed.
  EXPECT_THAT(reference_module->ToString(),
              Not(::testing::HasSubstr("fusion")));
}

TEST_F(PrepareReferenceModuleTest, SkipDespecialization) {
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto reference_module,
      PrepareReferenceModule(*test_module, nullptr, {}, {},
                             /*skip_despecialization=*/true));

  // Fusion op should be there.
  EXPECT_THAT(reference_module->ToString(), ::testing::HasSubstr("fusion"));
}

}  // namespace
}  // namespace xla
