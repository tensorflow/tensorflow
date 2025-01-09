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

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/literal_util.h"
#include "xla/test_helpers.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace {

template <typename T>
class LiteralComparisonTest : public ::testing::Test {};

using TestedTypes =
    ::testing::Types<tsl::float8_e3m4, tsl::float8_e4m3, tsl::float8_e4m3fn,
                     tsl::float8_e4m3b11fnuz, tsl::float8_e5m2>;
TYPED_TEST_SUITE(LiteralComparisonTest, TestedTypes);

TYPED_TEST(LiteralComparisonTest, CompareNear_Equal) {
  auto actual = LiteralUtil::CreateR0<TypeParam>(TypeParam(8.0));
  auto expected = LiteralUtil::CreateR0<TypeParam>(TypeParam(8.0));
  TF_EXPECT_OK(literal_comparison::Near(expected, actual, ErrorSpec(0.0, 0.0),
                                        /*detailed_message=*/false,
                                        /*miscompare_callback=*/nullptr));
}

TYPED_TEST(LiteralComparisonTest, CompareNear_NotEqual_1ulp) {
  PrimitiveType type = primitive_util::NativeToPrimitiveType<TypeParam>();
  auto actual = LiteralUtil::CreateR0<TypeParam>(TypeParam(8.0));
  float expV = 9.0;  // F8E4M3*
  if (type == F8E5M2)
    expV = 10.0;
  else if (type == F8E3M4)
    expV = 8.5;
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
  auto actual = LiteralUtil::CreateR0<TypeParam>(TypeParam(8.0));
  float expV = 12.0;  // F8E4M3*
  if (type == F8E5M2)
    expV = 14.0;
  else if (type == F8E3M4)
    expV = 10.0;
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
  auto actual = LiteralUtil::CreateR0<float>(8.0);
  float expV = 12.1;  // F8E4M3*
  if (type == F8E5M2)
    expV = 13.0;
  else if (type == F8E3M4)
    expV = 10.125;
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

}  // namespace
}  // namespace xla
