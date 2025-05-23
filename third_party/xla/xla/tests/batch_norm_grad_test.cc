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

#include <utility>

#include "xla/tests/xla_test_backend_predicates.h"
#include "absl/status/status.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

const char* const kModuleStr = R"(
    HloModule BatchNormGrad
    ENTRY BatchNormGrad.v6 {
     input       = f32[2,2] parameter(0)
     scale       = f32[2]   parameter(1)
     mean        = f32[2]   parameter(2)
     variance    = f32[2]   parameter(3)
     grad_output = f32[2,2] parameter(4)
     ROOT batch-norm-grad = (f32[2,2]{1,0}, f32[2]{0}, f32[2]{0})
      batch-norm-grad(input, scale, mean, variance, grad_output), epsilon=0, feature_index=1
    }
  )";

class BatchNormGradTest : public HloPjRtTestBase {};

TEST_F(BatchNormGradTest, CorrectComputation) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // Define input parameters
  auto input = LiteralUtil::CreateR2<float>({{1.2, 2.1}, {1.3, 2.4}});
  auto scale = LiteralUtil::CreateR1<float>({1.0, 1.0});
  auto mean = LiteralUtil::CreateR1<float>({0.0, 0.0});
  auto variance = LiteralUtil::CreateR1<float>({1.0, 1.0});
  auto grad_output = LiteralUtil::CreateR2<float>({{1.0, 1.0}, {1.0, 1.0}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, Execute(std::move(module),
                           {&input, &scale, &mean, &variance, &grad_output}));

  auto expected_input_grad =
      LiteralUtil::CreateR2<float>({{-1.5, -4.725}, {-1.625, -5.4}});
  auto expected_scale_grad = LiteralUtil::CreateR1<float>({2.5, 4.5});
  auto expected_mean_grad = LiteralUtil::CreateR1<float>({2.0, 2.0});

  EXPECT_EQ(result,
            LiteralUtil::MakeTuple({&expected_input_grad, &expected_scale_grad,
                                    &expected_mean_grad}));
}

TEST_F(BatchNormGradTest, DISABLED_ON_TPU(ReturnsErrorWhenHloPassesDisabled)) {
  if (test::DeviceIsOneOf({test::kGpu, test::kInterpreter})) {
    GTEST_SKIP();
  }
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  auto status_or_result =
      Execute(std::move(module), {}, /*run_hlo_passes=*/false);
  EXPECT_EQ(status_or_result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(
      status_or_result.status().message(),
      ::testing::HasSubstr("BatchNormGrad should be rewritten for CPU"));
}

}  // namespace
}  // namespace xla
