/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/scanner.h"

#include "xla/tsl/platform/test.h"

namespace tsl {
namespace strings {

class ScannerTest : public ::testing::Test {
 protected:
  // Returns a string with all chars that are in <clz>, in byte value order.
  string ClassStr(Scanner::CharClass clz) {
    string s;
    for (int i = 0; i < 256; ++i) {
      char ch = i;
      if (Scanner::Matches(clz, ch)) {
        s += ch;
      }
    }
    return s;
  }
};

TEST_F(ScannerTest, Any) {
  absl::string_view remaining, match;
  EXPECT_TRUE(Scanner("   horse0123")
                  .Any(Scanner::SPACE)
                  .Any(Scanner::DIGIT)
                  .Any(Scanner::LETTER)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("   horse", match);
  EXPECT_EQ("0123", remaining);

  EXPECT_TRUE(Scanner("")
                  .Any(Scanner::SPACE)
                  .Any(Scanner::DIGIT)
                  .Any(Scanner::LETTER)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("", match);

  EXPECT_TRUE(Scanner("----")
                  .Any(Scanner::SPACE)
                  .Any(Scanner::DIGIT)
                  .Any(Scanner::LETTER)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("----", remaining);
  EXPECT_EQ("", match);
}

TEST_F(ScannerTest, AnySpace) {
  absl::string_view remaining, match;
  EXPECT_TRUE(Scanner("  a b ")
                  .AnySpace()
                  .One(Scanner::LETTER)
                  .AnySpace()
                  .GetResult(&remaining, &match));
  EXPECT_EQ("  a ", match);
  EXPECT_EQ("b ", remaining);
}

TEST_F(ScannerTest, AnyEscapedNewline) {
  absl::string_view remaining, match;
  EXPECT_TRUE(Scanner("\\\n")
                  .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("\\\n", remaining);
  EXPECT_EQ("", match);
}

TEST_F(ScannerTest, AnyEmptyString) {
  absl::string_view remaining, match;
  EXPECT_TRUE(Scanner("")
                  .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("", match);
}

TEST_F(ScannerTest, Eos) {
  EXPECT_FALSE(Scanner("a").Eos().GetResult());
  EXPECT_TRUE(Scanner("").Eos().GetResult());
  EXPECT_FALSE(Scanner("abc").OneLiteral("ab").Eos().GetResult());
  EXPECT_TRUE(Scanner("abc").OneLiteral("abc").Eos().GetResult());
}

TEST_F(ScannerTest, Many) {
  absl::string_view remaining, match;
  EXPECT_TRUE(Scanner("abc").Many(Scanner::LETTER).GetResult());
  EXPECT_FALSE(Scanner("0").Many(Scanner::LETTER).GetResult());
  EXPECT_FALSE(Scanner("").Many(Scanner::LETTER).GetResult());

  EXPECT_TRUE(
      Scanner("abc ").Many(Scanner::LETTER).GetResult(&remaining, &match));
  EXPECT_EQ(" ", remaining);
  EXPECT_EQ("abc", match);
  EXPECT_TRUE(
      Scanner("abc").Many(Scanner::LETTER).GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("abc", match);
}

TEST_F(ScannerTest, One) {
  absl::string_view remaining, match;
  EXPECT_TRUE(Scanner("abc").One(Scanner::LETTER).GetResult());
  EXPECT_FALSE(Scanner("0").One(Scanner::LETTER).GetResult());
  EXPECT_FALSE(Scanner("").One(Scanner::LETTER).GetResult());

  EXPECT_TRUE(Scanner("abc")
                  .One(Scanner::LETTER)
                  .One(Scanner::LETTER)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("c", remaining);
  EXPECT_EQ("ab", match);
  EXPECT_TRUE(Scanner("a").One(Scanner::LETTER).GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("a", match);
}

TEST_F(ScannerTest, OneLiteral) {
  EXPECT_FALSE(Scanner("abc").OneLiteral("abC").GetResult());
  EXPECT_TRUE(Scanner("abc").OneLiteral("ab").OneLiteral("c").GetResult());
}

TEST_F(ScannerTest, ScanUntil) {
  absl::string_view remaining, match;
  EXPECT_TRUE(Scanner(R"(' \1 \2 \3 \' \\'rest)")
                  .OneLiteral("'")
                  .ScanUntil('\'')
                  .OneLiteral("'")
                  .GetResult(&remaining, &match));
  EXPECT_EQ(R"( \\'rest)", remaining);
  EXPECT_EQ(R"(' \1 \2 \3 \')", match);

  // The "scan until" character is not present.
  remaining = match = "unset";
  EXPECT_FALSE(Scanner(R"(' \1 \2 \3 \\rest)")
                   .OneLiteral("'")
                   .ScanUntil('\'')
                   .GetResult(&remaining, &match));
  EXPECT_EQ("unset", remaining);
  EXPECT_EQ("unset", match);

  // Scan until an escape character.
  remaining = match = "";
  EXPECT_TRUE(
      Scanner(R"(123\456)").ScanUntil('\\').GetResult(&remaining, &match));
  EXPECT_EQ(R"(\456)", remaining);
  EXPECT_EQ("123", match);
}

TEST_F(ScannerTest, ScanEscapedUntil) {
  absl::string_view remaining, match;
  EXPECT_TRUE(Scanner(R"(' \1 \2 \3 \' \\'rest)")
                  .OneLiteral("'")
                  .ScanEscapedUntil('\'')
                  .OneLiteral("'")
                  .GetResult(&remaining, &match));
  EXPECT_EQ("rest", remaining);
  EXPECT_EQ(R"(' \1 \2 \3 \' \\')", match);

  // The "scan until" character is not present.
  remaining = match = "unset";
  EXPECT_FALSE(Scanner(R"(' \1 \2 \3 \' \\rest)")
                   .OneLiteral("'")
                   .ScanEscapedUntil('\'')
                   .GetResult(&remaining, &match));
  EXPECT_EQ("unset", remaining);
  EXPECT_EQ("unset", match);
}

TEST_F(ScannerTest, ZeroOrOneLiteral) {
  absl::string_view remaining, match;
  EXPECT_TRUE(
      Scanner("abc").ZeroOrOneLiteral("abC").GetResult(&remaining, &match));
  EXPECT_EQ("abc", remaining);
  EXPECT_EQ("", match);

  EXPECT_TRUE(
      Scanner("abcd").ZeroOrOneLiteral("ab").ZeroOrOneLiteral("c").GetResult(
          &remaining, &match));
  EXPECT_EQ("d", remaining);
  EXPECT_EQ("abc", match);

  EXPECT_TRUE(
      Scanner("").ZeroOrOneLiteral("abc").GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("", match);
}

// Test output of GetResult (including the forms with optional params),
// and that it can be called multiple times.
TEST_F(ScannerTest, CaptureAndGetResult) {
  absl::string_view remaining, match;

  Scanner scan("  first    second");
  EXPECT_TRUE(scan.Any(Scanner::SPACE)
                  .RestartCapture()
                  .One(Scanner::LETTER)
                  .Any(Scanner::LETTER_DIGIT)
                  .StopCapture()
                  .Any(Scanner::SPACE)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("second", remaining);
  EXPECT_EQ("first", match);
  EXPECT_TRUE(scan.GetResult());
  remaining = "";
  EXPECT_TRUE(scan.GetResult(&remaining));
  EXPECT_EQ("second", remaining);
  remaining = "";
  match = "";
  EXPECT_TRUE(scan.GetResult(&remaining, &match));
  EXPECT_EQ("second", remaining);
  EXPECT_EQ("first", match);

  scan.RestartCapture().One(Scanner::LETTER).One(Scanner::LETTER);
  remaining = "";
  match = "";
  EXPECT_TRUE(scan.GetResult(&remaining, &match));
  EXPECT_EQ("cond", remaining);
  EXPECT_EQ("se", match);
}

// Tests that if StopCapture is not called, then calling GetResult, then
// scanning more, then GetResult again will update the capture.
TEST_F(ScannerTest, MultipleGetResultExtendsCapture) {
  absl::string_view remaining, match;

  Scanner scan("one2three");
  EXPECT_TRUE(scan.Many(Scanner::LETTER).GetResult(&remaining, &match));
  EXPECT_EQ("2three", remaining);
  EXPECT_EQ("one", match);
  EXPECT_TRUE(scan.Many(Scanner::DIGIT).GetResult(&remaining, &match));
  EXPECT_EQ("three", remaining);
  EXPECT_EQ("one2", match);
  EXPECT_TRUE(scan.Many(Scanner::LETTER).GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("one2three", match);
}

TEST_F(ScannerTest, FailedMatchDoesntChangeResult) {
  // A failed match doesn't change pointers passed to GetResult.
  Scanner scan("name");
  absl::string_view remaining = "rem";
  absl::string_view match = "match";
  EXPECT_FALSE(scan.One(Scanner::SPACE).GetResult(&remaining, &match));
  EXPECT_EQ("rem", remaining);
  EXPECT_EQ("match", match);
}

TEST_F(ScannerTest, DefaultCapturesAll) {
  // If RestartCapture() is not called, the whole string is used.
  Scanner scan("a b");
  absl::string_view remaining = "rem";
  absl::string_view match = "match";
  EXPECT_TRUE(scan.Any(Scanner::LETTER)
                  .AnySpace()
                  .Any(Scanner::LETTER)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("a b", match);
}

TEST_F(ScannerTest, AllCharClasses) {
  EXPECT_EQ(256, ClassStr(Scanner::ALL).size());
  EXPECT_EQ("0123456789", ClassStr(Scanner::DIGIT));
  EXPECT_EQ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER));
  EXPECT_EQ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER_DIGIT));
  EXPECT_EQ(
      "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
      "abcdefghijklmnopqrstuvwxyz",
      ClassStr(Scanner::LETTER_DIGIT_DASH_UNDERSCORE));
  EXPECT_EQ(
      "-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz",
      ClassStr(Scanner::LETTER_DIGIT_DASH_DOT_SLASH));
  EXPECT_EQ(
      "-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
      "abcdefghijklmnopqrstuvwxyz",
      ClassStr(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE));
  EXPECT_EQ(".0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER_DIGIT_DOT));
  EXPECT_EQ("+-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER_DIGIT_DOT_PLUS_MINUS));
  EXPECT_EQ(".0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER_DIGIT_DOT_UNDERSCORE));
  EXPECT_EQ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER_DIGIT_UNDERSCORE));
  EXPECT_EQ("abcdefghijklmnopqrstuvwxyz", ClassStr(Scanner::LOWERLETTER));
  EXPECT_EQ("0123456789abcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LOWERLETTER_DIGIT));
  EXPECT_EQ("0123456789_abcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LOWERLETTER_DIGIT_UNDERSCORE));
  EXPECT_EQ("123456789", ClassStr(Scanner::NON_ZERO_DIGIT));
  EXPECT_EQ("\t\n\v\f\r ", ClassStr(Scanner::SPACE));
  EXPECT_EQ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", ClassStr(Scanner::UPPERLETTER));
  EXPECT_EQ(">", ClassStr(Scanner::RANGLE));
}

TEST_F(ScannerTest, Peek) {
  EXPECT_EQ('a', Scanner("abc").Peek());
  EXPECT_EQ('a', Scanner("abc").Peek('b'));
  EXPECT_EQ('\0', Scanner("").Peek());
  EXPECT_EQ('z', Scanner("").Peek('z'));
  EXPECT_EQ('A', Scanner("0123A").Any(Scanner::DIGIT).Peek());
  EXPECT_EQ('\0', Scanner("0123A").Any(Scanner::LETTER_DIGIT).Peek());
}

}  // namespace strings
}  // namespace tsl
