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

#include "xla/hlo/parser/hlo_lexer.h"

#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace xla {

std::ostream& operator<<(std::ostream& os, TokKind kind) {
  return os << TokKindToString(kind);
}

namespace {

TEST(HloLexerTest, NonNegativeIntegerCornerCases) {
  const std::vector<uint64_t> values = {0,
                                        1,
                                        0x7fff'ffffULL,
                                        0x8000'0000ULL,
                                        0xffff'ffffULL,
                                        0x1'0000'0000ULL,
                                        0x7fff'ffff'ffff'ffffULL,
                                        0x8000'0000'0000'0000ULL,
                                        0xffff'ffff'ffff'ffffULL};
  const std::string input = absl::StrJoin(values, ",");
  LOG(INFO) << input;
  HloLexer lexer(input);
  bool first_value = true;
  for (uint64_t value : values) {
    if (first_value) {
      first_value = false;
    } else {
      ASSERT_EQ(lexer.Lex(), TokKind::kComma);
    }
    ASSERT_EQ(lexer.Lex(), TokKind::kInt)
        << "Input value was: " << value
        << ", casted to int64_t: " << absl::bit_cast<int64_t>(value);
    EXPECT_EQ(lexer.GetInt64Val(), absl::bit_cast<int64_t>(value))
        << "Input value was: " << value
        << ", casted to int64_t: " << absl::bit_cast<int64_t>(value)
        << ", but lexer returned: " << lexer.GetInt64Val();
  }
  EXPECT_EQ(lexer.Lex(), TokKind::kEof);
}

TEST(HloLexerTest, NegativeIntegerCornerCases) {
  const std::vector<int64_t> values = {-1, -2,
                                       std::numeric_limits<int32_t>::min(),
                                       std::numeric_limits<int64_t>::min() + 1,
                                       std::numeric_limits<int64_t>::min()};
  const std::string input = absl::StrJoin(values, ",");
  LOG(INFO) << input;
  HloLexer lexer(input);
  bool first_value = true;
  for (int64_t value : values) {
    if (first_value) {
      first_value = false;
    } else {
      ASSERT_EQ(lexer.Lex(), TokKind::kComma);
    }
    ASSERT_EQ(lexer.Lex(), TokKind::kInt) << "Input value was: " << value;
    EXPECT_EQ(lexer.GetInt64Val(), value)
        << "Input value was: " << value
        << ", but lexer returned: " << lexer.GetInt64Val();
  }
  EXPECT_EQ(lexer.Lex(), TokKind::kEof);
}

TEST(HloLexerTest, NonNegativeIntegerOverflow1Error) {
  // 1 more than the maximum uint64_t.
  const std::string input = "18446744073709551616";  // 2^64.
  HloLexer lexer(input);
  EXPECT_EQ(lexer.Lex(), TokKind::kError)
      << "Didn't detect overflow during lexing an integer. Lexed value is: "
      << lexer.GetInt64Val() << ", input was: " << input;
}

TEST(HloLexerTest, NonNegativeIntegerOverflow2Error) {
  // No overflow at 20 digits, but overflow at 21 digits.
  const std::string input = "184467440737095516150";  // 10 * (2^64 - 1)
  HloLexer lexer(input);
  EXPECT_EQ(lexer.Lex(), TokKind::kError)
      << "Didn't detect overflow during lexing an integer. Lexed value is: "
      << lexer.GetInt64Val() << ", input was: " << input;
}

// Regression test for correctly detecting overflow on multiplication of 10.
TEST(HloLexerTest, NonNegativeIntegerOverflow3Error) {
  // value is such a number that multiplying it by 10 would overflow. At the
  // same time, this overflown value is greater than the value itself. This test
  // guards against the wrong way (i.e. "value * 10 < value") of checking for
  // overflow.
  for (uint64_t value : {0x1C71C71C71C71C72ULL, (1ULL << 61)}) {
    ASSERT_GT(value, std::numeric_limits<uint64_t>::max() / 10);
    ASSERT_GT(value * 10, value);

    // Input is "value * 10".
    const std::string input = absl::StrCat(value, "0");
    LOG(INFO) << input;
    HloLexer lexer(input);
    EXPECT_EQ(lexer.Lex(), TokKind::kError)
        << "Didn't detect overflow during lexing an integer. Lexed value is: "
        << lexer.GetInt64Val() << ", input was: " << input;
  }
}

TEST(HloLexerTest, NegativeIntegerUnderflow1Error) {
  // 1 less than the minimum int64_t.
  HloLexer lexer("-9223372036854775809");  // -2^63-1.
  EXPECT_EQ(lexer.Lex(), TokKind::kError)
      << "Didn't detect underflow during lexing an integer. Lexed value is: "
      << lexer.GetInt64Val();
}

TEST(HloLexerTest, NegativeIntegerUnderflow2Error) {
  // No underflow at 20 digits, but underflow at 21 digits.
  HloLexer lexer("-92233720368547758080");  // -10 * 2^63.
  EXPECT_EQ(lexer.Lex(), TokKind::kError)
      << "Didn't detect underflow during lexing an integer. Lexed value is: "
      << lexer.GetInt64Val();
}

TEST(HloLexerTest, NegativeButNoDigitsError) {
  HloLexer lexer("-,-1");
  EXPECT_EQ(lexer.Lex(), TokKind::kError);
}

}  // namespace
}  // namespace xla
