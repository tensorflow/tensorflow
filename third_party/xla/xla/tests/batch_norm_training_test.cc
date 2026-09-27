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

#include <cmath>
#include <utility>

#include "xla/tests/xla_test_backend_predicates.h"
#include "absl/status/status.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

const char* const kModuleStr = R"(
HloModule module
ENTRY entry {
  %input  = f32[2,1] parameter(0)
  %scale  = f32[1]   parameter(1)
  %offset = f32[1]   parameter(2)
  ROOT %batch-norm-training = (f32[2,1], f32[1], f32[1])
    batch-norm-training(f32[2,1] %input, f32[1] %scale, f32[1] %offset),
    epsilon=0.001, feature_index=1
}
)";

class BatchNormTrainingTest : public HloTestBase {};

TEST_F(BatchNormTrainingTest, CorrectComputation) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  auto input = LiteralUtil::CreateR2<float>({{1.0}, {2.0}});
  auto scale = LiteralUtil::CreateR1<float>({0.5});
  auto offset = LiteralUtil::CreateR1<float>({0.1});

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, Execute(std::move(module), {&input, &scale, &offset}));

  // Decompose result tuple
  auto result_tuple = result.DecomposeTuple();

  auto expected_output =
      LiteralUtil::CreateR2<float>({{-0.399003029f}, {0.599003f}});
  auto expected_mean = LiteralUtil::CreateR1<float>({1.5f});
  auto expected_var = LiteralUtil::CreateR1<float>({0.25f});

  const float tolerance = 1e-5f;  // for floating-point comparison

  // Compare each element using EXPECT_NEAR instead of EXPECT_EQ to avoid
  // floating-point comparison issues, otherwise the test will be flaky.
  for (int i = 0; i < expected_output.element_count(); ++i) {
    EXPECT_NEAR(result_tuple[0].data<float>()[i],
                expected_output.data<float>()[i], tolerance);
  }

  for (int i = 0; i < expected_mean.element_count(); ++i) {
    EXPECT_NEAR(result_tuple[1].data<float>()[i],
                expected_mean.data<float>()[i], tolerance);
  }

  for (int i = 0; i < expected_var.element_count(); ++i) {
    EXPECT_NEAR(result_tuple[2].data<float>()[i], expected_var.data<float>()[i],
                tolerance);
  }
}

TEST_F(BatchNormTrainingTest, LargeOffset) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));

  auto input =
      LiteralUtil::CreateR2<float>({{10000.0f + 1.0f}, {10000.0f + 2.0f}});
  auto scale = LiteralUtil::CreateR1<float>({0.5f});
  auto offset = LiteralUtil::CreateR1<float>({0.1f});

  ASSERT_OK_AND_ASSIGN(auto result,
                       Execute(std::move(module), {&input, &scale, &offset}));

  auto result_tuple = result.DecomposeTuple();

  for (int i = 0; i < result_tuple[0].element_count(); ++i) {
    EXPECT_FALSE(std::isnan(result_tuple[0].data<float>()[i]));
  }

  for (int i = 0; i < result_tuple[1].element_count(); ++i) {
    EXPECT_FALSE(std::isnan(result_tuple[1].data<float>()[i]));
    EXPECT_NEAR(result_tuple[1].data<float>()[i], 10000.0f + 1.5f, 1e-4f);
  }

  for (int i = 0; i < result_tuple[2].element_count(); ++i) {
    EXPECT_FALSE(std::isnan(result_tuple[2].data<float>()[i]));
    EXPECT_GE(result_tuple[2].data<float>()[i], 0.0f);
  }
}

TEST_F(BatchNormTrainingTest, ExtremeOffset) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));

  auto input = LiteralUtil::CreateR2<float>({{1e8f + 1.0f}, {1e8f + 2.0f}});
  auto scale = LiteralUtil::CreateR1<float>({0.5f});
  auto offset = LiteralUtil::CreateR1<float>({0.1f});

  ASSERT_OK_AND_ASSIGN(auto result,
                       Execute(std::move(module), {&input, &scale, &offset}));

  auto result_tuple = result.DecomposeTuple();

  for (int i = 0; i < result_tuple[0].element_count(); ++i) {
    EXPECT_FALSE(std::isnan(result_tuple[0].data<float>()[i]));
  }

  for (int i = 0; i < result_tuple[1].element_count(); ++i) {
    EXPECT_FALSE(std::isnan(result_tuple[1].data<float>()[i]));
    EXPECT_NEAR(result_tuple[1].data<float>()[i], 1e8f, 1e2f);
  }

  for (int i = 0; i < result_tuple[2].element_count(); ++i) {
    EXPECT_FALSE(std::isnan(result_tuple[2].data<float>()[i]));
    EXPECT_GE(result_tuple[2].data<float>()[i], 0.0f);
  }
}
TEST_F(BatchNormTrainingTest, ReturnsErrorWhenHloPassesDisabled) {
  if (test::DeviceTypeIsOneOf({test::kGpu, test::kInterpreter, test::kTpu})) {
    GTEST_SKIP();
  }
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  auto status_or_result =
      Execute(std::move(module), {}, /*run_hlo_passes=*/false);
  EXPECT_EQ(status_or_result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(
      status_or_result.status().message(),
      ::testing::HasSubstr("BatchNormTraining should be rewritten for CPU"));
}

}  // namespace
}  // namespace xla
