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

#include "xla/service/cpu/cpu_float_support.h"

#include <memory>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

class SkipInstructionTest : public HloRunnerAgnosticTestBase {
 public:
  SkipInstructionTest()
      : HloRunnerAgnosticTestBase(std::make_unique<HloRunner>(
            PlatformUtil::GetDefaultPlatform().value())) {}
  void SetUp() override { HloRunnerAgnosticTestBase::SetUp(); }
};

TEST_F(SkipInstructionTest, SkipDot) {
  constexpr absl::string_view kHlo = R"(
    HloModule test
    
    ENTRY main {
      p0 = bf16[100,100] parameter(0)
      p1 = bf16[100,100] parameter(1)
      ROOT dot = f32[100,100] dot(p0, p1), 
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  CpuFloatSupport::DotStrategyChecker call_library_for_dot =
      [](const HloInstruction& hlo) { return true; };
  CpuFloatSupport cpu_float_support(BF16, call_library_for_dot);
  FloatNormalization float_normalization(&cpu_float_support);
  TF_ASSERT_OK_AND_ASSIGN(bool upcast, float_normalization.Run(module.get()));
  EXPECT_EQ(upcast, false);
}

TEST_F(SkipInstructionTest, UpcastAdd) {
  constexpr absl::string_view kHlo = R"(
    HloModule test
    
    ENTRY main {
      p0 = bf16[100,100] parameter(0)
      p1 = bf16[100,100] parameter(1)
      ROOT add = f32[100,100] add(p0, p1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  CpuFloatSupport::DotStrategyChecker call_library_for_dot =
      [](const HloInstruction& hlo) { return true; };
  CpuFloatSupport cpu_float_support(BF16, call_library_for_dot);
  FloatNormalization float_normalization(&cpu_float_support);
  TF_ASSERT_OK_AND_ASSIGN(bool upcast, float_normalization.Run(module.get()));
  EXPECT_EQ(upcast, true);
}

}  // namespace
}  // namespace xla::cpu
