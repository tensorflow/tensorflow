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

#include "xla/tools/compare_literals/compare_literals_impl.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/error_spec.h"
#include "xla/literal_util.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::AllOf;
using ::testing::ContainsRegex;
using ::testing::HasSubstr;
using ::testing::status::StatusIs;

TEST(DiffTest, CompareLiteralsRank1FloatValuesMultipleDifferent) {
  LiteralProto lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f}).ToProto();
  LiteralProto lit2 = LiteralUtil::CreateR1<float>({3.0f, 4.0f}).ToProto();

  EXPECT_THAT(
      CompareLiterals(lit1, lit2, ErrorSpec(0, 0)),
      StatusIs(
          absl::StatusCode::kDataLoss,
          AllOf(HasSubstr("Mismatch count 2 (100.0000%) in shape f32[2]"),
                ContainsRegex(
                    "actual\\s+3,\\s+expected\\s+1,\\s+index\\s+\\{0\\}"),
                ContainsRegex(
                    "actual\\s+4,\\s+expected\\s+2,\\s+index\\s+\\{1\\}"))));
}

TEST(DiffTest, CompareLiteralsDifferentShape) {
  LiteralProto lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f}).ToProto();
  LiteralProto lit2 =
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f}).ToProto();

  EXPECT_THAT(CompareLiterals(lit1, lit2, ErrorSpec(0, 0)),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("mismatch in dimension #0")));
}

TEST(DiffTest, CompareLiteralsRank1ComplexValuesEqual) {
  LiteralProto lit1 =
      LiteralUtil::CreateR1<complex64>({{1.0f, 2.0f}, {3.0f, 4.0f}}).ToProto();
  LiteralProto lit2 =
      LiteralUtil::CreateR1<complex64>({{1.0f, 2.0f}, {3.0f, 4.0f}}).ToProto();

  EXPECT_OK(CompareLiterals(lit1, lit2, ErrorSpec(0, 0)));
}

TEST(DiffTest, CompareLiteralsRank1ComplexValuesDifferent) {
  LiteralProto lit1 =
      LiteralUtil::CreateR1<complex64>({{1.0f, 2.0f}, {3.0f, 4.0f}}).ToProto();
  LiteralProto lit2 =
      LiteralUtil::CreateR1<complex64>({{1.0f, 2.0f}, {3.0f, 5.0f}}).ToProto();

  EXPECT_THAT(
      CompareLiterals(lit1, lit2, ErrorSpec(0, 0)),
      StatusIs(
          absl::StatusCode::kDataLoss,
          AllOf(HasSubstr(
                    "Mismatch count 1 (50.0000%) in shape c64[2] (2 elements)"),
                ContainsRegex(
                    "actual\\s+5\\s+\\+\\s+0,\\s+expected\\s+4\\s+\\+\\s+0,"
                    "\\s+index\\s+\\{1\\}"))));
}

TEST(DiffTest, CompareLiteralsRank1S32ValuesEqual) {
  LiteralProto lit1 = LiteralUtil::CreateR1<int32_t>({1, 2, 3}).ToProto();
  LiteralProto lit2 = LiteralUtil::CreateR1<int32_t>({1, 2, 3}).ToProto();

  EXPECT_OK(CompareLiterals(lit1, lit2, ErrorSpec(0, 0)));
}

TEST(DiffTest, CompareLiteralsRank1S32ValuesDifferent) {
  LiteralProto lit1 = LiteralUtil::CreateR1<int32_t>({1, 2, 3}).ToProto();
  LiteralProto lit2 = LiteralUtil::CreateR1<int32_t>({1, 2, 4}).ToProto();

  EXPECT_THAT(CompareLiterals(lit1, lit2, ErrorSpec(0, 0)),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("first mismatch at array index {2}")));
}

TEST(DiffTest, CompareLiteralsRank2FloatValuesEqual) {
  LiteralProto lit1 =
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}}).ToProto();
  LiteralProto lit2 =
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}}).ToProto();

  EXPECT_OK(CompareLiterals(lit1, lit2, ErrorSpec(0, 0)));
}

TEST(DiffTest, CompareLiteralsRank2FloatValuesDifferent) {
  LiteralProto lit1 =
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}}).ToProto();
  LiteralProto lit2 =
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 5.0f}}).ToProto();

  EXPECT_THAT(CompareLiterals(lit1, lit2, ErrorSpec(0, 0)),
              StatusIs(absl::StatusCode::kDataLoss,
                       ContainsRegex("actual\\s+5,\\s+expected\\s+4,"
                                     "\\s+index\\s+\\{1,1\\}")));
}

TEST(DiffTest, CompareLiteralsRank1Bf16ValuesEqual) {
  LiteralProto lit1 = LiteralUtil::CreateR1<bfloat16>(
                          {bfloat16(1.0f), bfloat16(2.0f), bfloat16(3.0f)})
                          .ToProto();
  LiteralProto lit2 = LiteralUtil::CreateR1<bfloat16>(
                          {bfloat16(1.0f), bfloat16(2.0f), bfloat16(3.0f)})
                          .ToProto();

  EXPECT_OK(CompareLiterals(lit1, lit2, ErrorSpec(0, 0)));
}

TEST(DiffTest, CompareLiteralsRank1Bf16ValuesDifferent) {
  LiteralProto lit1 = LiteralUtil::CreateR1<bfloat16>(
                          {bfloat16(1.0f), bfloat16(2.0f), bfloat16(3.0f)})
                          .ToProto();
  LiteralProto lit2 = LiteralUtil::CreateR1<bfloat16>(
                          {bfloat16(1.0f), bfloat16(2.0f), bfloat16(4.0f)})
                          .ToProto();

  EXPECT_THAT(
      CompareLiterals(lit1, lit2, ErrorSpec(0, 0)),
      StatusIs(
          absl::StatusCode::kDataLoss,
          ContainsRegex("actual\\s+4,\\s+expected\\s+3,\\s+index\\s+\\{2\\}")));
}

}  // namespace
}  // namespace xla
