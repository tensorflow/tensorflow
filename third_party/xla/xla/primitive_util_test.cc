/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/primitive_util.h"

#include <initializer_list>
#include <string>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(PrimitiveUtilTest, StringToPrimitiveType) {
  auto expect_ok_and_equal = [](const std::string& str,
                                PrimitiveType expected) {
    TF_ASSERT_OK_AND_ASSIGN(PrimitiveType actual,
                            primitive_util::StringToPrimitiveType(str));
    EXPECT_EQ(expected, actual);
  };
  expect_ok_and_equal("f32", F32);
  expect_ok_and_equal("tuple", TUPLE);
  expect_ok_and_equal("pred", PRED);
  expect_ok_and_equal("s32", S32);

  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("F32").status());
  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("Pred").status());
  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("preD").status());
}

TEST(PrimitiveUtilTest, FloatTypes) {
  EXPECT_EQ(primitive_util::SignificandWidth(F32), 24);
  EXPECT_EQ(primitive_util::SignificandWidth(BF16), 8);
  EXPECT_EQ(primitive_util::ExponentWidth(F32), 8);
  EXPECT_EQ(primitive_util::ExponentWidth(BF16), 8);
  EXPECT_EQ(primitive_util::UnderflowExponent(F32), -125);
  EXPECT_EQ(primitive_util::UnderflowExponent(BF16), -125);
  EXPECT_EQ(primitive_util::OverflowExponent(F32), 128);
  EXPECT_EQ(primitive_util::OverflowExponent(BF16), 128);
}

TEST(PrimitiveUtilTest, CastPreservesValues) {
  bool expecteds[PrimitiveType_ARRAYSIZE][PrimitiveType_ARRAYSIZE] = {};

  auto set_true = [&](PrimitiveType from_type,
                      std::initializer_list<PrimitiveType> to_types) {
    absl::c_for_each(to_types, [&](PrimitiveType to_type) {
      expecteds[from_type][to_type] = true;
    });
  };

  set_true(PRED, {PRED, S1, S2, S4, S8, S16, S32, S64, U1, U2, U4, U8, U16});
  set_true(PRED, {U32, U64, F16, F32, F64, C64, BF16, C128, F8E5M2, F8E4M3});
  set_true(PRED, {F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ, F8E3M4});
  set_true(PRED, {F4E2M1FN});

  set_true(S1, {S1, S2, S4, S8, S16, S32, S64, F16, F32, F64, C64});
  set_true(S1, {BF16, C128, F8E5M2, F8E4M3, F8E4M3FN, F8E4M3B11FNUZ});
  set_true(S1, {F8E5M2FNUZ, F8E4M3FNUZ, F8E3M4, F4E2M1FN});

  set_true(S2, {S2, S4, S8, S16, S32, S64, F16, F32, F64, C64, BF16});
  set_true(S2, {C128, F8E5M2, F8E4M3, F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ});
  set_true(S2, {F8E4M3FNUZ, F8E3M4, F4E2M1FN});

  set_true(S4, {S4, S8, S16, S32, S64, F16, F32, F64, C64, BF16, C128});
  set_true(S4, {F8E5M2, F8E4M3, F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ});
  set_true(S4, {F8E4M3FNUZ, F8E3M4});

  set_true(S8, {S8, S16, S32, S64, F16, F32, F64, C64, BF16, C128});
  set_true(S16, {S16, S32, S64, F32, F64, C64, C128});
  set_true(S32, {S32, S64, F64, C128});
  set_true(S64, {S64});

  set_true(U1, {S2, S4, S8, S16, S32, S64, U1, U2, U4, U8, U16, U32});
  set_true(U1, {U64, F16, F32, F64, C64, BF16, C128, F8E5M2, F8E4M3});
  set_true(U1, {F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ, F8E3M4});
  set_true(U1, {F4E2M1FN});

  set_true(U2, {S4, S8, S16, S32, S64, U2, U4, U8, U16, U32, U64, F16});
  set_true(U2, {F32, F64, C64, BF16, C128, F8E5M2, F8E4M3, F8E4M3FN});
  set_true(U2, {F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ, F8E3M4, F4E2M1FN});

  set_true(U4, {S8, S16, S32, S64, U4, U8, U16, U32, U64, F16, F32});
  set_true(U4, {F64, C64, BF16, C128, F8E4M3, F8E4M3FN, F8E4M3B11FNUZ});
  set_true(U4, {F8E4M3FNUZ, F8E3M4});

  set_true(U8, {S16, S32, S64, U8, U16, U32, U64, F16, F32, F64, C64});
  set_true(U8, {BF16, C128});

  set_true(U16, {S32, S64, U16, U32, U64, F32, F64, C64, C128});
  set_true(U32, {S64, U32, U64, F64, C128});
  set_true(U64, {U64});
  set_true(F16, {F16, F32, F64, C64, C128});
  set_true(F32, {F32, F64, C64, C128});
  set_true(F64, {F64, C128});
  set_true(C64, {C64, C128});
  set_true(BF16, {F32, F64, C64, BF16, C128});
  set_true(C128, {C128});
  set_true(F8E5M2, {F16, F32, F64, C64, BF16, C128, F8E5M2});
  set_true(F8E4M3, {F16, F32, F64, C64, BF16, C128, F8E4M3});
  set_true(F8E4M3FN, {F16, F32, F64, C64, BF16, C128, F8E4M3FN});
  set_true(F8E4M3B11FNUZ, {F16, F32, F64, C64, BF16, C128, F8E4M3B11FNUZ});
  set_true(F8E5M2FNUZ, {F16, F32, F64, C64, BF16, C128, F8E5M2FNUZ});
  set_true(F8E4M3FNUZ, {F16, F32, F64, C64, BF16, C128, F8E4M3FNUZ});
  set_true(F8E3M4, {F16, F32, F64, C64, BF16, C128, F8E3M4});
  set_true(F4E2M1FN, {F16, F32, F64, C64, BF16, C128, F8E5M2, F8E4M3});
  set_true(F4E2M1FN, {F8E4M3FN, F8E3M4, F4E2M1FN});
  set_true(F8E8M0FNU, {F32, F64, C64, BF16, C128, F8E8M0FNU});

  for (int from_type_int = PrimitiveType_MIN;
       from_type_int < PrimitiveType_ARRAYSIZE; ++from_type_int) {
    auto from_type = static_cast<PrimitiveType>(from_type_int);
    if (!primitive_util::IsArrayType(from_type)) {
      continue;
    }
    for (int to_type_int = PrimitiveType_MIN;
         to_type_int < PrimitiveType_ARRAYSIZE; ++to_type_int) {
      auto to_type = static_cast<PrimitiveType>(to_type_int);
      if (!primitive_util::IsArrayType(to_type)) {
        continue;
      }
      bool expected = expecteds[from_type][to_type];
      bool actual = primitive_util::CastPreservesValues(from_type, to_type);
      EXPECT_EQ(expected, actual)
          << primitive_util::LowercasePrimitiveTypeName(from_type) << " -> "
          << primitive_util::LowercasePrimitiveTypeName(to_type);
    }
  }
}

}  // namespace
}  // namespace xla
