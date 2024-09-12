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

#include "tsl/platform/str_util.h"

#include <vector>

#include "tsl/platform/test.h"

namespace tsl {

TEST(CEscape, Basic) {
  EXPECT_EQ(absl::CEscape("hello"), "hello");
  EXPECT_EQ(absl::CEscape("hello\n"), "hello\\n");
  EXPECT_EQ(absl::CEscape("hello\r"), "hello\\r");
  EXPECT_EQ(absl::CEscape("\t\r\"'"), "\\t\\r\\\"\\'");
  EXPECT_EQ(absl::CEscape("\320hi\200"), "\\320hi\\200");
}

string ExpectCUnescapeSuccess(absl::string_view source) {
  string dest;
  string error;
  EXPECT_TRUE(absl::CUnescape(source, &dest, &error)) << error;
  return dest;
}

TEST(CUnescape, Basic) {
  EXPECT_EQ("hello", ExpectCUnescapeSuccess("hello"));
  EXPECT_EQ("hello\n", ExpectCUnescapeSuccess("hello\\n"));
  EXPECT_EQ("hello\r", ExpectCUnescapeSuccess("hello\\r"));
  EXPECT_EQ("\t\r\"'", ExpectCUnescapeSuccess("\\t\\r\\\"\\'"));
  EXPECT_EQ("\320hi\200", ExpectCUnescapeSuccess("\\320hi\\200"));
}

TEST(CUnescape, HandlesCopyOnWriteStrings) {
  string dest = "hello";
  string read = dest;
  // For std::string, read and dest now share the same buffer.

  string error;
  absl::string_view source = "llohe";
  // CUnescape is going to write "llohe" to dest, so dest's buffer will be
  // reallocated, and read's buffer remains untouched.
  EXPECT_TRUE(absl::CUnescape(source, &dest, &error));
  EXPECT_EQ("hello", read);
}

TEST(StripTrailingWhitespace, Basic) {
  string test;
  test = "hello";
  absl::StripTrailingAsciiWhitespace(&test);
  EXPECT_EQ(test, "hello");

  test = "foo  ";
  absl::StripTrailingAsciiWhitespace(&test);
  EXPECT_EQ(test, "foo");

  test = "   ";
  absl::StripTrailingAsciiWhitespace(&test);
  EXPECT_EQ(test, "");

  test = "";
  absl::StripTrailingAsciiWhitespace(&test);
  EXPECT_EQ(test, "");

  test = " abc\t";
  absl::StripTrailingAsciiWhitespace(&test);
  EXPECT_EQ(test, " abc");
}

TEST(RemoveLeadingWhitespace, Basic) {
  string text = "  \t   \n  \r Quick\t";
  absl::string_view data(text);
  // check that all whitespace is removed
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 11);
  EXPECT_EQ(data, absl::string_view("Quick\t"));
  // check that non-whitespace is not removed
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 0);
  EXPECT_EQ(data, absl::string_view("Quick\t"));
}

TEST(RemoveLeadingWhitespace, TerminationHandling) {
  // check termination handling
  string text = "\t";
  absl::string_view data(text);
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 1);
  EXPECT_EQ(data, absl::string_view(""));

  // check termination handling again
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 0);
  EXPECT_EQ(data, absl::string_view(""));
}

TEST(RemoveTrailingWhitespace, Basic) {
  string text = "  \t   \n  \r Quick \t";
  absl::string_view data(text);
  // check that all whitespace is removed
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 2);
  EXPECT_EQ(data, absl::string_view("  \t   \n  \r Quick"));
  // check that non-whitespace is not removed
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 0);
  EXPECT_EQ(data, absl::string_view("  \t   \n  \r Quick"));
}

TEST(RemoveTrailingWhitespace, TerminationHandling) {
  // check termination handling
  string text = "\t";
  absl::string_view data(text);
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 1);
  EXPECT_EQ(data, absl::string_view(""));

  // check termination handling again
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 0);
  EXPECT_EQ(data, absl::string_view(""));
}

TEST(RemoveWhitespaceContext, Basic) {
  string text = "  \t   \n  \r Quick \t";
  absl::string_view data(text);
  // check that all whitespace is removed
  EXPECT_EQ(str_util::RemoveWhitespaceContext(&data), 13);
  EXPECT_EQ(data, absl::string_view("Quick"));
  // check that non-whitespace is not removed
  EXPECT_EQ(str_util::RemoveWhitespaceContext(&data), 0);
  EXPECT_EQ(data, absl::string_view("Quick"));

  // Test empty string
  text = "";
  data = text;
  EXPECT_EQ(str_util::RemoveWhitespaceContext(&data), 0);
  EXPECT_EQ(data, absl::string_view(""));
}

void TestConsumeLeadingDigits(absl::string_view s, int64_t expected,
                              absl::string_view remaining) {
  uint64 v;
  absl::string_view input(s);
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

void TestConsumeNonWhitespace(absl::string_view s, absl::string_view expected,
                              absl::string_view remaining) {
  absl::string_view v;
  absl::string_view input(s);
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
  absl::string_view input(s);
  EXPECT_FALSE(absl::ConsumePrefix(&input, "abcdefg"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_FALSE(absl::ConsumePrefix(&input, "abce"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_TRUE(absl::ConsumePrefix(&input, ""));
  EXPECT_EQ(input, "abcdef");

  EXPECT_FALSE(absl::ConsumePrefix(&input, "abcdeg"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_TRUE(absl::ConsumePrefix(&input, "abcdef"));
  EXPECT_EQ(input, "");

  input = s;
  EXPECT_TRUE(absl::ConsumePrefix(&input, "abcde"));
  EXPECT_EQ(input, "f");
}

TEST(StripPrefix, Basic) {
  EXPECT_EQ(absl::StripPrefix("abcdef", "abcdefg"), "abcdef");
  EXPECT_EQ(absl::StripPrefix("abcdef", "abce"), "abcdef");
  EXPECT_EQ(absl::StripPrefix("abcdef", ""), "abcdef");
  EXPECT_EQ(absl::StripPrefix("abcdef", "abcdeg"), "abcdef");
  EXPECT_EQ(absl::StripPrefix("abcdef", "abcdef"), "");
  EXPECT_EQ(absl::StripPrefix("abcdef", "abcde"), "f");
}

TEST(JoinStrings, Basic) {
  std::vector<string> s;
  s = {"hi"};
  EXPECT_EQ(absl::StrJoin(s, " "), "hi");
  s = {"hi", "there", "strings"};
  EXPECT_EQ(absl::StrJoin(s, " "), "hi there strings");

  std::vector<absl::string_view> sp;
  sp = {"hi"};
  EXPECT_EQ(absl::StrJoin(sp, ",,"), "hi");
  sp = {"hi", "there", "strings"};
  EXPECT_EQ(absl::StrJoin(sp, "--"), "hi--there--strings");
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
  EXPECT_EQ(absl::StrJoin(str_util::Split("a", ','), "|"), "a");
  EXPECT_EQ(absl::StrJoin(str_util::Split(",", ','), "|"), "|");
  EXPECT_EQ(absl::StrJoin(str_util::Split("a,b,c", ','), "|"), "a|b|c");
  EXPECT_EQ(absl::StrJoin(str_util::Split("a,,,b,,c,", ','), "|"), "a|||b||c|");
  EXPECT_EQ(absl::StrJoin(str_util::Split("a!,!b,!c,", ",!"), "|"),
            "a|||b||c|");
  EXPECT_EQ(absl::StrJoin(
                str_util::Split("a,,,b,,c,", ',', str_util::SkipEmpty()), "|"),
            "a|b|c");
  EXPECT_EQ(
      absl::StrJoin(
          str_util::Split("a,  ,b,,c,", ',', str_util::SkipWhitespace()), "|"),
      "a|b|c");
  EXPECT_EQ(absl::StrJoin(str_util::Split("a.  !b,;c,", ".,;!",
                                          str_util::SkipWhitespace()),
                          "|"),
            "a|b|c");
}

TEST(Lowercase, Basic) {
  EXPECT_EQ("", absl::AsciiStrToLower(""));
  EXPECT_EQ("hello", absl::AsciiStrToLower("hello"));
  EXPECT_EQ("hello world", absl::AsciiStrToLower("Hello World"));
}

TEST(Uppercase, Basic) {
  EXPECT_EQ("", absl::AsciiStrToUpper(""));
  EXPECT_EQ("HELLO", absl::AsciiStrToUpper("hello"));
  EXPECT_EQ("HELLO WORLD", absl::AsciiStrToUpper("Hello World"));
}

TEST(SnakeCase, Basic) {
  EXPECT_EQ("", str_util::ArgDefCase(""));
  EXPECT_EQ("", str_util::ArgDefCase("!"));
  EXPECT_EQ("", str_util::ArgDefCase("5"));
  EXPECT_EQ("", str_util::ArgDefCase("!:"));
  EXPECT_EQ("", str_util::ArgDefCase("5-5"));
  EXPECT_EQ("", str_util::ArgDefCase("_!"));
  EXPECT_EQ("", str_util::ArgDefCase("_5"));
  EXPECT_EQ("a", str_util::ArgDefCase("_a"));
  EXPECT_EQ("a", str_util::ArgDefCase("_A"));
  EXPECT_EQ("i", str_util::ArgDefCase("I"));
  EXPECT_EQ("i", str_util::ArgDefCase("i"));
  EXPECT_EQ("i_", str_util::ArgDefCase("I%"));
  EXPECT_EQ("i_", str_util::ArgDefCase("i%"));
  EXPECT_EQ("i", str_util::ArgDefCase("%I"));
  EXPECT_EQ("i", str_util::ArgDefCase("-i"));
  EXPECT_EQ("i", str_util::ArgDefCase("3i"));
  EXPECT_EQ("i", str_util::ArgDefCase("32i"));
  EXPECT_EQ("i3", str_util::ArgDefCase("i3"));
  EXPECT_EQ("i_a3", str_util::ArgDefCase("i_A3"));
  EXPECT_EQ("i_i", str_util::ArgDefCase("II"));
  EXPECT_EQ("i_i", str_util::ArgDefCase("I_I"));
  EXPECT_EQ("i__i", str_util::ArgDefCase("I__I"));
  EXPECT_EQ("i_i_32", str_util::ArgDefCase("II-32"));
  EXPECT_EQ("ii_32", str_util::ArgDefCase("Ii-32"));
  EXPECT_EQ("hi_there", str_util::ArgDefCase("HiThere"));
  EXPECT_EQ("hi_hi", str_util::ArgDefCase("Hi!Hi"));
  EXPECT_EQ("hi_hi", str_util::ArgDefCase("HiHi"));
  EXPECT_EQ("hihi", str_util::ArgDefCase("Hihi"));
  EXPECT_EQ("hi_hi", str_util::ArgDefCase("Hi_Hi"));
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

TEST(Strnlen, Basic) {
  EXPECT_EQ(0, str_util::Strnlen("ab", 0));
  EXPECT_EQ(1, str_util::Strnlen("a", 1));
  EXPECT_EQ(2, str_util::Strnlen("abcd", 2));
  EXPECT_EQ(3, str_util::Strnlen("abc", 10));
  EXPECT_EQ(4, str_util::Strnlen("a \t\n", 10));
}

}  // namespace tsl
