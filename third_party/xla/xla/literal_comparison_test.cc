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

#include "xla/literal_comparison.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace {

template <typename T>
class LiteralComparisonTest : public ::testing::Test {};

using TestedTypes =
    ::testing::Types<tsl::float4_e2m1fn, tsl::float8_e3m4, tsl::float8_e4m3,
                     tsl::float8_e4m3b11fnuz, tsl::float8_e4m3fn,
                     tsl::float8_e4m3fnuz, tsl::float8_e5m2,
                     tsl::float8_e5m2fnuz, tsl::float8_e8m0fnu>;
TYPED_TEST_SUITE(LiteralComparisonTest, TestedTypes);

TYPED_TEST(LiteralComparisonTest, CompareNear_Equal) {
  auto actual = LiteralUtil::CreateR0<TypeParam>(TypeParam(1.0));
  auto expected = LiteralUtil::CreateR0<TypeParam>(TypeParam(1.0));
  TF_EXPECT_OK(literal_comparison::Near(expected, actual, ErrorSpec(0.0, 0.0),
                                        /*detailed_message=*/false,
                                        /*miscompare_callback=*/nullptr));
}

TYPED_TEST(LiteralComparisonTest, CompareNear_NotEqual_1ulp) {
  PrimitiveType type = primitive_util::NativeToPrimitiveType<TypeParam>();
  auto actual = LiteralUtil::CreateR0<TypeParam>(TypeParam(1.0));
  float expV = 1.125;  // F8E4M3*
  if (type == F8E5M2 || type == F8E5M2FNUZ)
    expV = 1.25;
  else if (type == F8E3M4)
    expV = 1.0625;
  else if (type == F4E2M1FN)
    expV = 1.5;
  else if (type == F8E8M0FNU)
    expV = 2.0;
  auto expected = LiteralUtil::CreateR0<TypeParam>(TypeParam{expV});
  auto error_spec = ErrorSpec(0.0, 0.0);
  EXPECT_IS_NOT_OK(literal_comparison::Near(expected, actual, error_spec,
                                            /*detailed_message=*/false,
                                            /*miscompare_callback=*/nullptr));
  error_spec.low_precision_fp_error_spec.type = type;
  error_spec.low_precision_fp_error_spec.within_n_values = 1;
  EXPECT_IS_OK(literal_comparison::Near(expected, actual, error_spec,
                                        /*detailed_message=*/false,
                                        /*miscompare_callback=*/nullptr));
}

TYPED_TEST(LiteralComparisonTest, CompareNear_NotEqual_4ulps) {
  PrimitiveType type = primitive_util::NativeToPrimitiveType<TypeParam>();
  auto actual = LiteralUtil::CreateR0<TypeParam>(TypeParam(1.0));
  float expV = 1.5;  // F8E4M3*
  if (type == F8E5M2 || type == F8E5M2FNUZ)
    expV = 2.0;
  else if (type == F8E3M4)
    expV = 1.25;
  else if (type == F4E2M1FN)
    expV = 4.0;
  else if (type == F8E8M0FNU)
    expV = 16.0;
  auto expected = LiteralUtil::CreateR0<TypeParam>(TypeParam{expV});
  auto error_spec = ErrorSpec(0.0, 0.0);
  error_spec.low_precision_fp_error_spec.type = type;
  error_spec.low_precision_fp_error_spec.within_n_values = 1;
  EXPECT_IS_NOT_OK(literal_comparison::Near(expected, actual, error_spec,
                                            /*detailed_message=*/false,
                                            /*miscompare_callback=*/nullptr));
  error_spec.low_precision_fp_error_spec.type = type;
  error_spec.low_precision_fp_error_spec.within_n_values = 4;
  EXPECT_IS_OK(literal_comparison::Near(expected, actual, error_spec,
                                        /*detailed_message=*/false,
                                        /*miscompare_callback=*/nullptr));
}

TYPED_TEST(LiteralComparisonTest, FloatUsingCompareNear_NotEqual_4ulps) {
  PrimitiveType type = primitive_util::NativeToPrimitiveType<TypeParam>();
  auto actual = LiteralUtil::CreateR0<float>(1.0);
  float expV = 1.51;  // F8E4M3*
  if (type == F8E5M2 || type == F8E5M2FNUZ)
    expV = 2.01;
  else if (type == F8E3M4)
    expV = 1.26;
  else if (type == F4E2M1FN)
    expV = 4.1;
  else if (type == F8E8M0FNU)
    expV = 16.5;
  auto expected = LiteralUtil::CreateR0<float>(expV);
  auto error_spec = ErrorSpec(0.0, 0.0);
  error_spec.low_precision_fp_error_spec.type = type;
  error_spec.low_precision_fp_error_spec.within_n_values = 1;
  EXPECT_IS_NOT_OK(literal_comparison::Near(expected, actual, error_spec,
                                            /*detailed_message=*/false,
                                            /*miscompare_callback=*/nullptr));
  error_spec.low_precision_fp_error_spec.type = type;
  error_spec.low_precision_fp_error_spec.within_n_values = 4;
  EXPECT_IS_OK(literal_comparison::Near(expected, actual, error_spec,
                                        /*detailed_message=*/false,
                                        /*miscompare_callback=*/nullptr));
}

TEST(LiteralComparisonSuggestionsTest, SuggestErrorSpec) {
  // Create expected and actual literals that will fail standard comparison
  // in a way that generates multiple distinct candidates.
  // Element 1: actual = 1.002,  expected = 1.0   -> abs_err = 0.002,  rel_err =
  // 0.002 Element 2: actual = 20.04,  expected = 20.0  -> abs_err = 0.04,
  // rel_err = 0.002 Element 3: actual = 0.0511, expected = 0.05  -> abs_err =
  // 0.0011, rel_err = 0.022

  auto expected = LiteralUtil::CreateR1<float>({1.0f, 20.0f, 0.05f});
  auto actual = LiteralUtil::CreateR1<float>({1.002f, 20.04f, 0.0511f});

  ErrorSpec error_spec(0.001, 0.001);

  absl::Status status =
      literal_comparison::Near(expected, actual, error_spec,
                               /*detailed_message=*/true,
                               /*miscompare_callback=*/nullptr);

  EXPECT_FALSE(status.ok());
  std::string error_message = std::string(status.message());

  // Verify that suggestions are present
  EXPECT_TRUE(absl::StrContains(
      error_message, "Suggested ErrorSpec adjustments to make this test pass:"))
      << "Actual message:\n"
      << error_message;

  // Verify specific options (rounded to 1 sig fig)
  EXPECT_TRUE(absl::StrContains(error_message, "ErrorSpec{0.05, 0.001}"))
      << "Actual message:\n"
      << error_message;
  EXPECT_TRUE(absl::StrContains(error_message, "ErrorSpec{0.002, 0.003}"))
      << "Actual message:\n"
      << error_message;
  EXPECT_TRUE(absl::StrContains(error_message, "ErrorSpec{0.001, 0.03}"))
      << "Actual message:\n"
      << error_message;

  // Verify current spec is printed
  EXPECT_TRUE(absl::StrContains(error_message,
                                "Current ErrorSpec: ErrorSpec{0.001, 0.001}"))
      << "Actual message:\n"
      << error_message;
}

TEST(LiteralComparisonSuggestionsTest,
     SuggestErrorSpecPerformanceOnLargeMismatches) {
  // If the test times out then the algorithm gets broken.
  constexpr int N = 100000;
  std::vector<float> expected_vec(N, 100.0f);
  std::vector<float> actual_vec(N);
  for (int i = 0; i < N; ++i) {
    // Generate trade-off errors: abs_error = (i + 1), rel_error = (N - i) / 100
    actual_vec[i] = 100.0f + static_cast<float>(i + 1);
  }

  Literal expected = LiteralUtil::CreateR1<float>(expected_vec);
  Literal actual = LiteralUtil::CreateR1<float>(actual_vec);
  ErrorSpec error_spec(0.1, 0.001);

  absl::Status status =
      literal_comparison::Near(expected, actual, error_spec,
                               /*detailed_message=*/true,
                               /*miscompare_callback=*/nullptr);

  EXPECT_FALSE(status.ok());
  std::string error_message = std::string(status.message());
  EXPECT_TRUE(absl::StrContains(
      error_message,
      "Suggested ErrorSpec adjustments to make this test pass:"));
}

}  // namespace
}  // namespace xla
