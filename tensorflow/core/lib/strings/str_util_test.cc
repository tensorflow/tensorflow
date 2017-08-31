/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/strings/str_util.h"

#include <vector>
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(CEscape, Basic) {
  EXPECT_EQ(str_util::CEscape("hello"), "hello");
  EXPECT_EQ(str_util::CEscape("hello\n"), "hello\\n");
  EXPECT_EQ(str_util::CEscape("hello\r"), "hello\\r");
  EXPECT_EQ(str_util::CEscape("\t\r\"'"), "\\t\\r\\\"\\'");
  EXPECT_EQ(str_util::CEscape("\320hi\200"), "\\320hi\\200");
}

string ExpectCUnescapeSuccess(StringPiece source) {
  string dest;
  string error;
  EXPECT_TRUE(str_util::CUnescape(source, &dest, &error)) << error;
  return dest;
}

TEST(CUnescape, Basic) {
  EXPECT_EQ("hello", ExpectCUnescapeSuccess("hello"));
  EXPECT_EQ("hello\n", ExpectCUnescapeSuccess("hello\\n"));
  EXPECT_EQ("hello\r", ExpectCUnescapeSuccess("hello\\r"));
  EXPECT_EQ("\t\r\"'", ExpectCUnescapeSuccess("\\t\\r\\\"\\'"));
  EXPECT_EQ("\320hi\200", ExpectCUnescapeSuccess("\\320hi\\200"));
}

TEST(StripTrailingWhitespace, Basic) {
  string test;
  test = "hello";
  str_util::StripTrailingWhitespace(&test);
  EXPECT_EQ(test, "hello");

  test = "foo  ";
  str_util::StripTrailingWhitespace(&test);
  EXPECT_EQ(test, "foo");

  test = "   ";
  str_util::StripTrailingWhitespace(&test);
  EXPECT_EQ(test, "");

  test = "";
  str_util::StripTrailingWhitespace(&test);
  EXPECT_EQ(test, "");

  test = " abc\t";
  str_util::StripTrailingWhitespace(&test);
  EXPECT_EQ(test, " abc");
}

TEST(RemoveLeadingWhitespace, Basic) {
  string text = "  \t   \n  \r Quick\t";
  StringPiece data(text);
  // check that all whitespace is removed
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 11);
  EXPECT_EQ(data, StringPiece("Quick\t"));
  // check that non-whitespace is not removed
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 0);
  EXPECT_EQ(data, StringPiece("Quick\t"));
}

TEST(RemoveLeadingWhitespace, TerminationHandling) {
  // check termination handling
  string text = "\t";
  StringPiece data(text);
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 1);
  EXPECT_EQ(data, StringPiece(""));

  // check termination handling again
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 0);
  EXPECT_EQ(data, StringPiece(""));
}

TEST(RemoveTrailingWhitespace, Basic) {
  string text = "  \t   \n  \r Quick \t";
  StringPiece data(text);
  // check that all whitespace is removed
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 2);
  EXPECT_EQ(data, StringPiece("  \t   \n  \r Quick"));
  // check that non-whitespace is not removed
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 0);
  EXPECT_EQ(data, StringPiece("  \t   \n  \r Quick"));
}

TEST(RemoveTrailingWhitespace, TerminationHandling) {
  // check termination handling
  string text = "\t";
  StringPiece data(text);
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 1);
  EXPECT_EQ(data, StringPiece(""));

  // check termination handling again
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 0);
  EXPECT_EQ(data, StringPiece(""));
}

TEST(RemoveWhitespaceContext, Basic) {
  string text = "  \t   \n  \r Quick \t";
  StringPiece data(text);
  // check that all whitespace is removed
  EXPECT_EQ(str_util::RemoveWhitespaceContext(&data), 13);
  EXPECT_EQ(data, StringPiece("Quick"));
  // check that non-whitespace is not removed
  EXPECT_EQ(str_util::RemoveWhitespaceContext(&data), 0);
  EXPECT_EQ(data, StringPiece("Quick"));

  // Test empty string
  text = "";
  data = text;
  EXPECT_EQ(str_util::RemoveWhitespaceContext(&data), 0);
  EXPECT_EQ(data, StringPiece(""));
}

void TestConsumeLeadingDigits(StringPiece s, int64 expected,
                              StringPiece remaining) {
  uint64 v;
  StringPiece input(s);
  if (str_util::ConsumeLeadingDigits(&input, &v)) {
    EXPECT_EQ(v, static_cast<uint64>(expected));
    EXPECT_EQ(input, remaining);
  } else {
    EXPECT_LT(expected, 0);
    EXPECT_EQ(input, remaining);
  }
}

TEST(ConsumeLeadingDigits, Basic) {
  using str_util::ConsumeLeadingDigits;

  TestConsumeLeadingDigits("123", 123, "");
  TestConsumeLeadingDigits("a123", -1, "a123");
  TestConsumeLeadingDigits("9_", 9, "_");
  TestConsumeLeadingDigits("11111111111xyz", 11111111111ll, "xyz");

  // Overflow case
  TestConsumeLeadingDigits("1111111111111111111111111111111xyz", -1,
                           "1111111111111111111111111111111xyz");

  // 2^64
  TestConsumeLeadingDigits("18446744073709551616xyz", -1,
                           "18446744073709551616xyz");
  // 2^64-1
  TestConsumeLeadingDigits("18446744073709551615xyz", 18446744073709551615ull,
                           "xyz");
  // (2^64-1)*10+9
  TestConsumeLeadingDigits("184467440737095516159yz", -1,
                           "184467440737095516159yz");
}

void TestConsumeNonWhitespace(StringPiece s, StringPiece expected,
                              StringPiece remaining) {
  StringPiece v;
  StringPiece input(s);
  if (str_util::ConsumeNonWhitespace(&input, &v)) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(input, remaining);
  } else {
    EXPECT_EQ(expected, "");
    EXPECT_EQ(input, remaining);
  }
}

TEST(ConsumeNonWhitespace, Basic) {
  TestConsumeNonWhitespace("", "", "");
  TestConsumeNonWhitespace(" ", "", " ");
  TestConsumeNonWhitespace("abc", "abc", "");
  TestConsumeNonWhitespace("abc ", "abc", " ");
}

TEST(ConsumePrefix, Basic) {
  string s("abcdef");
  StringPiece input(s);
  EXPECT_FALSE(str_util::ConsumePrefix(&input, "abcdefg"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_FALSE(str_util::ConsumePrefix(&input, "abce"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_TRUE(str_util::ConsumePrefix(&input, ""));
  EXPECT_EQ(input, "abcdef");

  EXPECT_FALSE(str_util::ConsumePrefix(&input, "abcdeg"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_TRUE(str_util::ConsumePrefix(&input, "abcdef"));
  EXPECT_EQ(input, "");

  input = s;
  EXPECT_TRUE(str_util::ConsumePrefix(&input, "abcde"));
  EXPECT_EQ(input, "f");
}

TEST(JoinStrings, Basic) {
  std::vector<string> s;
  s = {"hi"};
  EXPECT_EQ(str_util::Join(s, " "), "hi");
  s = {"hi", "there", "strings"};
  EXPECT_EQ(str_util::Join(s, " "), "hi there strings");

  std::vector<StringPiece> sp;
  sp = {"hi"};
  EXPECT_EQ(str_util::Join(sp, ",,"), "hi");
  sp = {"hi", "there", "strings"};
  EXPECT_EQ(str_util::Join(sp, "--"), "hi--there--strings");
}

TEST(JoinStrings, Join3) {
  std::vector<string> s;
  s = {"hi"};
  auto l1 = [](string* out, string s) { *out += s; };
  EXPECT_EQ(str_util::Join(s, " ", l1), "hi");
  s = {"hi", "there", "strings"};
  auto l2 = [](string* out, string s) { *out += s[0]; };
  EXPECT_EQ(str_util::Join(s, " ", l2), "h t s");
}

TEST(Split, Basic) {
  EXPECT_TRUE(str_util::Split("", ',').empty());
  EXPECT_EQ(str_util::Join(str_util::Split("a", ','), "|"), "a");
  EXPECT_EQ(str_util::Join(str_util::Split(",", ','), "|"), "|");
  EXPECT_EQ(str_util::Join(str_util::Split("a,b,c", ','), "|"), "a|b|c");
  EXPECT_EQ(str_util::Join(str_util::Split("a,,,b,,c,", ','), "|"),
            "a|||b||c|");
  EXPECT_EQ(str_util::Join(str_util::Split("a!,!b,!c,", ",!"), "|"),
            "a|||b||c|");
  EXPECT_EQ(str_util::Join(
                str_util::Split("a,,,b,,c,", ',', str_util::SkipEmpty()), "|"),
            "a|b|c");
  EXPECT_EQ(
      str_util::Join(
          str_util::Split("a,  ,b,,c,", ',', str_util::SkipWhitespace()), "|"),
      "a|b|c");
  EXPECT_EQ(str_util::Join(str_util::Split("a.  !b,;c,", ".,;!",
                                           str_util::SkipWhitespace()),
                           "|"),
            "a|b|c");
}

TEST(SplitAndParseAsInts, Int32) {
  std::vector<int32> nums;
  EXPECT_TRUE(str_util::SplitAndParseAsInts("", ',', &nums));
  EXPECT_EQ(nums.size(), 0);

  EXPECT_TRUE(str_util::SplitAndParseAsInts("134", ',', &nums));
  EXPECT_EQ(nums.size(), 1);
  EXPECT_EQ(nums[0], 134);

  EXPECT_TRUE(str_util::SplitAndParseAsInts("134,2,13,-5", ',', &nums));
  EXPECT_EQ(nums.size(), 4);
  EXPECT_EQ(nums[0], 134);
  EXPECT_EQ(nums[1], 2);
  EXPECT_EQ(nums[2], 13);
  EXPECT_EQ(nums[3], -5);

  EXPECT_FALSE(str_util::SplitAndParseAsInts("abc", ',', &nums));

  EXPECT_FALSE(str_util::SplitAndParseAsInts("-13,abc", ',', &nums));

  EXPECT_FALSE(str_util::SplitAndParseAsInts("13,abc,5", ',', &nums));
}

TEST(SplitAndParseAsInts, Int64) {
  std::vector<int64> nums;
  EXPECT_TRUE(str_util::SplitAndParseAsInts("", ',', &nums));
  EXPECT_EQ(nums.size(), 0);

  EXPECT_TRUE(str_util::SplitAndParseAsInts("134", ',', &nums));
  EXPECT_EQ(nums.size(), 1);
  EXPECT_EQ(nums[0], 134);

  EXPECT_TRUE(
      str_util::SplitAndParseAsInts("134,2,13,-4000000000", ',', &nums));
  EXPECT_EQ(nums.size(), 4);
  EXPECT_EQ(nums[0], 134);
  EXPECT_EQ(nums[1], 2);
  EXPECT_EQ(nums[2], 13);
  EXPECT_EQ(nums[3], -4000000000);

  EXPECT_FALSE(str_util::SplitAndParseAsInts("abc", ',', &nums));

  EXPECT_FALSE(str_util::SplitAndParseAsInts("-13,abc", ',', &nums));

  EXPECT_FALSE(str_util::SplitAndParseAsInts("13,abc,5", ',', &nums));
}

TEST(SplitAndParseAsFloats, Float) {
  std::vector<float> nums;
  EXPECT_TRUE(str_util::SplitAndParseAsFloats("", ',', &nums));
  EXPECT_EQ(nums.size(), 0);

  EXPECT_TRUE(str_util::SplitAndParseAsFloats("134.2323", ',', &nums));
  ASSERT_EQ(nums.size(), 1);
  EXPECT_NEAR(nums[0], 134.2323f, 1e-5f);

  EXPECT_TRUE(str_util::SplitAndParseAsFloats("134.9,2.123,13.0000,-5.999,1e6",
                                              ',', &nums));
  ASSERT_EQ(nums.size(), 5);
  EXPECT_NEAR(nums[0], 134.9f, 1e-5f);
  EXPECT_NEAR(nums[1], 2.123f, 1e-5f);
  EXPECT_NEAR(nums[2], 13.0f, 1e-5f);
  EXPECT_NEAR(nums[3], -5.999f, 1e-5f);
  EXPECT_NEAR(nums[4], 1e6f, 1e1f);

  EXPECT_FALSE(str_util::SplitAndParseAsFloats("abc", ',', &nums));

  EXPECT_FALSE(str_util::SplitAndParseAsFloats("-13.0,abc", ',', &nums));

  EXPECT_FALSE(str_util::SplitAndParseAsFloats("13.0,abc,-5.999", ',', &nums));
}

TEST(Lowercase, Basic) {
  EXPECT_EQ("", str_util::Lowercase(""));
  EXPECT_EQ("hello", str_util::Lowercase("hello"));
  EXPECT_EQ("hello world", str_util::Lowercase("Hello World"));
}

TEST(Uppercase, Basic) {
  EXPECT_EQ("", str_util::Uppercase(""));
  EXPECT_EQ("HELLO", str_util::Uppercase("hello"));
  EXPECT_EQ("HELLO WORLD", str_util::Uppercase("Hello World"));
}

TEST(TitlecaseString, Basic) {
  string s = "sparse_lookup";
  str_util::TitlecaseString(&s, "_");
  ASSERT_EQ(s, "Sparse_Lookup");

  s = "sparse_lookup";
  str_util::TitlecaseString(&s, " ");
  ASSERT_EQ(s, "Sparse_lookup");

  s = "dense";
  str_util::TitlecaseString(&s, " ");
  ASSERT_EQ(s, "Dense");
}

TEST(StringReplace, Basic) {
  EXPECT_EQ("XYZ_XYZ_XYZ", str_util::StringReplace("ABC_ABC_ABC", "ABC", "XYZ",
                                                   /*replace_all=*/true));
}

TEST(StringReplace, OnlyFirst) {
  EXPECT_EQ("XYZ_ABC_ABC", str_util::StringReplace("ABC_ABC_ABC", "ABC", "XYZ",
                                                   /*replace_all=*/false));
}

TEST(StringReplace, IncreaseLength) {
  EXPECT_EQ("a b c",
            str_util::StringReplace("abc", "b", " b ", /*replace_all=*/true));
}

TEST(StringReplace, IncreaseLengthMultipleMatches) {
  EXPECT_EQ("a b  b c",
            str_util::StringReplace("abbc", "b", " b ", /*replace_all=*/true));
}

TEST(StringReplace, NoChange) {
  EXPECT_EQ("abc",
            str_util::StringReplace("abc", "d", "X", /*replace_all=*/true));
}

TEST(StringReplace, EmptyStringReplaceFirst) {
  EXPECT_EQ("", str_util::StringReplace("", "a", "X", /*replace_all=*/false));
}

TEST(StringReplace, EmptyStringReplaceAll) {
  EXPECT_EQ("", str_util::StringReplace("", "a", "X", /*replace_all=*/true));
}

}  // namespace tensorflow
