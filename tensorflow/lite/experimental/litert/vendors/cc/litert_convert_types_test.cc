// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert_types.h"

#include <initializer_list>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert_test_util.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_convert_types_impl.h"

namespace litert {
namespace {

using ::litert::example::ExampleLegalizer;
using ::litert::example::ExampleOp;
using ::litert::example::ExampleOpLegalization;
using ::litert::example::Type;
using ::litert::testing::TestOpContext;

TEST(LegalizeTest, SimpleLegalization) {
  TestOpContext op(kLiteRtOpCodeTflCustom, {"input_1", "input_2"},
                   {"output_1"});

  ExampleOpLegalization<kLiteRtOpCodeTflCustom> legal;
  ASSERT_EQ(legal.OpToMatch(), kLiteRtOpCodeTflCustom);
  auto result = legal.Legalize(op.GetOp());
  ASSERT_TRUE(result);

  ASSERT_TRUE(LegalizationMatched(*result));
  auto legalized = GetSimpleConversionResult(*result);
  ASSERT_TRUE(legalized);

  EXPECT_EQ(legalized->code, static_cast<int>(kLiteRtOpCodeTflCustom));
  ASSERT_EQ(legalized->input_types.size(), 2);
  ASSERT_EQ(legalized->output_types.size(), 1);
  EXPECT_EQ(legalized->input_types.at(0), "input_1");
  EXPECT_EQ(legalized->input_types.at(1), "input_2");
  EXPECT_EQ(legalized->output_types.at(0), "output_1");
}

TEST(LegalizeTest, NoMatch) {
  TestOpContext op(kLiteRtOpCodeTflCustom, {}, {"output_1"});

  ExampleOpLegalization<kLiteRtOpCodeTflCustom> legal;
  ASSERT_EQ(legal.OpToMatch(), kLiteRtOpCodeTflCustom);
  auto result = legal.Legalize(op.GetOp());
  ASSERT_TRUE(result);

  ASSERT_FALSE(LegalizationMatched(*result));

  auto legalized = GetSimpleConversionResult(*result);
  ASSERT_FALSE(legalized);
}

TEST(LegalizeTest, NoLegalization) {
  TestOpContext op(kLiteRtOpCodeTflCustom, {}, {"output_1"});

  ExampleOpLegalization<kLiteRtOpCodeShloAbs> legal;
  ASSERT_EQ(legal.OpToMatch(), kLiteRtOpCodeShloAbs);

  auto result = legal.Legalize(op.GetOp());
  ASSERT_TRUE(result);
  ASSERT_FALSE(LegalizationMatched(*result));
  auto legalized = GetSimpleConversionResult(*result);
  ASSERT_FALSE(legalized);
}

TEST(LegalizerTest, NoMatch) {
  TestOpContext op(kLiteRtOpCodeTflCustom, {}, {"output_1"});
  ExampleLegalizer legalizer;
  LITERT_ASSERT_STATUS_OK(legalizer.Register(
      {ExampleOpLegalization<kLiteRtOpCodeTflCustom>::Create()}));

  auto result = legalizer.Legalize(op.GetOp());
  ASSERT_TRUE(result);
  ASSERT_FALSE(LegalizationMatched(*result));
  auto legalized = GetSimpleConversionResult(*result);
  ASSERT_FALSE(legalized);
}

TEST(LegalizerTest, NoLegalization) {
  TestOpContext op(kLiteRtOpCodeTflCustom, {}, {"output_1"});
  ExampleLegalizer legalizer;

  auto result = legalizer.Legalize(op.GetOp());
  ASSERT_FALSE(result);
}

TEST(LegalizerTest, DuplicateLegalizationFail) {
  ExampleLegalizer legalizer;
  LITERT_ASSERT_STATUS_OK(legalizer.Register(
      {ExampleOpLegalization<kLiteRtOpCodeTflCustom>::Create()}));
  EXPECT_NE(legalizer.Register(
                {ExampleOpLegalization<kLiteRtOpCodeTflCustom>::Create()}),
            kLiteRtStatusOk);
}

TEST(LegalizerTest, Match) {
  TestOpContext op(kLiteRtOpCodeTflCustom, {"input_1", "input_2"},
                   {"output_1"});

  ExampleLegalizer legalizer;
  LITERT_ASSERT_STATUS_OK(legalizer.Register(
      {ExampleOpLegalization<kLiteRtOpCodeTflCustom>::Create()}));

  auto res = legalizer.Legalize(op.GetOp());
  ASSERT_TRUE(res);
}

}  // namespace
}  // namespace litert
