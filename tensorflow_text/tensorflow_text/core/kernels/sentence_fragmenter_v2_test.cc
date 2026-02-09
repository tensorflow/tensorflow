// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_text/core/kernels/sentence_fragmenter_v2.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "icu4c/source/common/unicode/umachine.h"
#include "icu4c/source/common/unicode/unistr.h"

namespace tensorflow {
namespace text {
namespace {

class SentenceBreakingUtilsParamTest : public ::testing::TestWithParam<UChar> {
 protected:
  std::string StringFromUnicodeChar(UChar32 input) {
    std::string result;
    icu::UnicodeString test_unicode_string(input);
    test_unicode_string.toUTF8String(result);
    return result;
  }
};

class SentenceBreakingUtilsStringParamTest
    : public ::testing::TestWithParam<const char*> {};

class IsTerminalPuncParamTest : public SentenceBreakingUtilsParamTest {};

class IsTerminalPuncTest : public ::testing::Test {};

const UChar is_terminal_punc_test_cases[] = {
    0x055C,  // Armenian exclamation mark
    0x055E,  // Armenian question mark
    0x0589,  // Armenian full stop
    0x061F,  // Arabic question mark
    0x06D4,  // Arabic full stop
    0x0700,  // Syriabc end of paragraph
    0x0701,  // Syriac supralinear full stop
    0x0702,  // Syriac sublinear full stop
    0x1362,  // Ethiopic full stop
    0x1367,  // Ethiopic question mark
    0x1368,  // Ethiopic paragraph separator
    0x104A,  // Myanmar sign little section
    0x104B,  // Myanmar sign section
    0x166E,  // Canadian syllabics full stop
    0x17d4,  // Khmer sign khan
    0x1803,  // Mongolian full stop
    0x1809,  // Mongolian Manchu full stop
    0x1944,  // Limbu exclamation mark
    0x1945,  // Limbu question mark
    0x203C,  // double exclamation mark
    0x203D,  // interrobang
    0x2047,  // double question mark
    0x2048,  // question exclamation mark
    0x2049,  // exclamation question mark
    0x3002,  // ideographic full stop
    0x037E,  // Greek question mark
    0xFE52,  // small full stop
    0xFE56,  // small question mark
    0xFE57,  // small exclamation mark
    0xFF01,  // fullwidth exclamation mark
    0xFF0E,  // fullwidth full stop
    0xFF1F,  // fullwidth question mark
    0xFF61,  // halfwidth ideographic full stop
    0x2026,  // ellipsis
    0x0964,
    0x0965,  // Devanagari danda..Devanagari double
};

TEST_P(IsTerminalPuncParamTest, IsTerminalPunc) {
  std::string test_string = StringFromUnicodeChar(GetParam());
  int offset;
  EXPECT_TRUE(IsTerminalPunc(test_string, &offset));
}

INSTANTIATE_TEST_SUITE_P(IsTerminalPuncTest, IsTerminalPuncParamTest,
                         ::testing::ValuesIn(is_terminal_punc_test_cases));

TEST_F(IsTerminalPuncTest, IsMultiCharEllipseTerminalPunc) {
  std::string test_string = "...";
  int offset;
  EXPECT_TRUE(IsTerminalPunc(test_string, &offset));
}

TEST_F(IsTerminalPuncTest, TestMultiUnicodeChars) {
  std::string test_string = "never gonna let you decode";
  int offset;
  EXPECT_FALSE(IsTerminalPunc(test_string, &offset));
}

struct ClosePuncOffsetPairs {
  const UChar close_punc;
  const int offset;
};

class SentenceBreakingUtilsClosePuncPairParamTest
    : public ::testing::TestWithParam<ClosePuncOffsetPairs> {
 protected:
  std::string StringFromUnicodeChar(UChar32 input) {
    std::string result;
    icu::UnicodeString test_unicode_string(input);
    test_unicode_string.toUTF8String(result);
    return result;
  }
};

class ClosePuncParamTest : public SentenceBreakingUtilsClosePuncPairParamTest {
};

const ClosePuncOffsetPairs close_punc_test_cases[] = {
    {0x29, 1},
    {0x5D, 1},
    {0x3E, 1},
    {0x7D, 1},
    {0x207E, 3},  // superscript right parenthesis
    {0x208E, 3},  // subscript right parenthesis
    {0x27E7, 3},  // mathematical right white square bracket
    {0x27E9, 3},  // mathematical right angle bracket
    {0x27EB, 3},  // mathematical right double angle bracket
    {0x2984, 3},  // right white curly bracket
    {0x2986, 3},  // right white parenthesis
    {0x2988, 3},  // Z notation right image bracket
    {0x298A, 3},  // Z notation right binding bracket
    {0x298C, 3},  // right square bracket with underbar
    {0x298E, 3},  // right square bracket with tick in top corner
    {0x2990, 3},  // right square bracket with tick in bottom corner
    {0x2992, 3},  // right angle bracket with dot
    {0x2994, 3},  // right arc greater-than bracket
    {0x2996, 3},  // double right arc less-than bracket
    {0x2998, 3},  // right black tortoise shell bracket
    {0x29D9, 3},  // right wiggly fence
    {0x29DB, 3},  // right double wiggly fence
    {0x29FD, 3},  // right-pointing curved angle bracket
    {0x3009, 3},  // CJK right angle bracket
    {0x300B, 3},  // CJK right double angle bracket
    {0x3011, 3},  // CJK right black lenticular bracket
    {0x3015, 3},  // CJK right tortoise shell bracket
    {0x3017, 3},  // CJK right white lenticular bracket
    {0x3019, 3},  // CJK right white tortoise shell bracket
    {0x301B, 3},  // CJK right white square bracket
    {0xFD3F, 3},  // Ornate right parenthesis
    {0xFE5A, 3},  // small right parenthesis
    {0xFE5C, 3},  // small right curly bracket
    {0xFF09, 3},  // fullwidth right parenthesis
    {0xFF3D, 3},  // fullwidth right square bracket
    {0xFF5D, 3},  // fullwidth right curly bracket
    {0x27, 1},
    {0x60, 1},
    {0x22, 1},
    {0xFF07, 3},  // fullwidth apostrophe
    {0xFF02, 3},  // fullwidth quotation mark
    {0x2019, 3},  // right single quotation mark (English, others)
    {0x201D, 3},  // right double quotation mark (English, others)
    {0x2018, 3},  // left single quotation mark (Czech, German, Slovak)
    {0x201C, 3},  // left double quotation mark (Czech, German, Slovak)
    {0x203A, 3},  // single right-pointing angle quotation mark (French, others)
    {0x00BB, 2},  // right-pointing double angle quotation mark (French, others)
    {0x2039, 3},  // single left-pointing angle quotation mark (Slovenian,
                  // others)
    {0x00AB, 2},  // left-pointing double angle quotation mark (Slovenian,
                  // others)
    {0x300D, 3},  // right corner bracket (East Asian languages)
    {0xfe42, 3},  // presentation form for vertical right corner bracket
    {0xFF63, 3},  // halfwidth right corner bracket (East Asian languages)
    {0x300F, 3},  // right white corner bracket (East Asian languages)
    {0xfe44, 3},  // presentation form for vertical right white corner bracket
    {0x301F, 3},  // low double prime quotation mark (East Asian languages)
    {0x301E, 3}   // close double prime (East Asian languages written
                  // horizontally)
};

TEST_P(ClosePuncParamTest, IsClosePunc) {
  ClosePuncOffsetPairs test_punc = GetParam();
  std::string test_string = StringFromUnicodeChar(test_punc.close_punc);
  int expected_offset = test_punc.offset;
  int offset;
  EXPECT_TRUE(IsClosePunc(test_string, &offset));
  EXPECT_EQ(offset, expected_offset);
}

INSTANTIATE_TEST_SUITE_P(IsClosePuncParamTest, ClosePuncParamTest,
                         ::testing::ValuesIn(close_punc_test_cases));

class OpenParenParamTest : public SentenceBreakingUtilsParamTest {};

const UChar open_paren_test_cases[] = {
    '(',    '[', '<', '{',
    0x207D,  // superscript left parenthesis
    0x208D,  // subscript left parenthesis
    0x27E6,  // mathematical left white square bracket
    0x27E8,  // mathematical left angle bracket
    0x27EA,  // mathematical left double angle bracket
    0x2983,  // left white curly bracket
    0x2985,  // left white parenthesis
    0x2987,  // Z notation left image bracket
    0x2989,  // Z notation left binding bracket
    0x298B,  // left square bracket with underbar
    0x298D,  // left square bracket with tick in top corner
    0x298F,  // left square bracket with tick in bottom corner
    0x2991,  // left angle bracket with dot
    0x2993,  // left arc less-than bracket
    0x2995,  // double left arc greater-than bracket
    0x2997,  // left black tortoise shell bracket
    0x29D8,  // left wiggly fence
    0x29DA,  // left double wiggly fence
    0x29FC,  // left-pointing curved angle bracket
    0x3008,  // CJK left angle bracket
    0x300A,  // CJK left double angle bracket
    0x3010,  // CJK left black lenticular bracket
    0x3014,  // CJK left tortoise shell bracket
    0x3016,  // CJK left white lenticular bracket
    0x3018,  // CJK left white tortoise shell bracket
    0x301A,  // CJK left white square bracket
    0xFD3E,  // Ornate left parenthesis
    0xFE59,  // small left parenthesis
    0xFE5B,  // small left curly bracket
    0xFF08,  // fullwidth left parenthesis
    0xFF3B,  // fullwidth left square bracket
    0xFF5B,  // fullwidth left curly bracket
};

TEST_P(OpenParenParamTest, IsOpenParen) {
  std::string test_string = StringFromUnicodeChar(GetParam());
  EXPECT_TRUE(IsOpenParen(test_string));
}

INSTANTIATE_TEST_SUITE_P(IsOpenParenParamTest, OpenParenParamTest,
                         ::testing::ValuesIn(open_paren_test_cases));

class CloseParenParamTest : public SentenceBreakingUtilsParamTest {};

const UChar close_paren_test_cases[] = {
    ')',    ']', '>', '}',
    0x207E,  // superscript right parenthesis
    0x208E,  // subscript right parenthesis
    0x27E7,  // mathematical right white square bracket
    0x27E9,  // mathematical right angle bracket
    0x27EB,  // mathematical right double angle bracket
    0x2984,  // right white curly bracket
    0x2986,  // right white parenthesis
    0x2988,  // Z notation right image bracket
    0x298A,  // Z notation right binding bracket
    0x298C,  // right square bracket with underbar
    0x298E,  // right square bracket with tick in top corner
    0x2990,  // right square bracket with tick in bottom corner
    0x2992,  // right angle bracket with dot
    0x2994,  // right arc greater-than bracket
    0x2996,  // double right arc less-than bracket
    0x2998,  // right black tortoise shell bracket
    0x29D9,  // right wiggly fence
    0x29DB,  // right double wiggly fence
    0x29FD,  // right-pointing curved angle bracket
    0x3009,  // CJK right angle bracket
    0x300B,  // CJK right double angle bracket
    0x3011,  // CJK right black lenticular bracket
    0x3015,  // CJK right tortoise shell bracket
    0x3017,  // CJK right white lenticular bracket
    0x3019,  // CJK right white tortoise shell bracket
    0x301B,  // CJK right white square bracket
    0xFD3F,  // Ornate right parenthesis
    0xFE5A,  // small right parenthesis
    0xFE5C,  // small right curly bracket
    0xFF09,  // fullwidth right parenthesis
    0xFF3D,  // fullwidth right square bracket
    0xFF5D,  // fullwidth right curly bracket
};

TEST_P(CloseParenParamTest, IsCloseParen) {
  std::string test_string = StringFromUnicodeChar(GetParam());
  EXPECT_TRUE(IsCloseParen(test_string));
}

INSTANTIATE_TEST_SUITE_P(IsCloseParenParamTest, CloseParenParamTest,
                         ::testing::ValuesIn(close_paren_test_cases));

class IsPunctuationWordParamTest : public SentenceBreakingUtilsParamTest {};

const UChar punc_word_test_cases[] = {
    '(',    '[',    '<',    '{',
    0x207D,  // superscript left parenthesis
    0x208D,  // subscript left parenthesis
    0x27E6,  // mathematical left white square bracket
    0x27E8,  // mathematical left angle bracket
    0x27EA,  // mathematical left double angle bracket
    0x2983,  // left white curly bracket
    0x2985,  // left white parenthesis
    0x2987,  // Z notation left image bracket
    0x2989,  // Z notation left binding bracket
    0x298B,  // left square bracket with underbar
    0x298D,  // left square bracket with tick in top corner
    0x298F,  // left square bracket with tick in bottom corner
    0x2991,  // left angle bracket with dot
    0x2993,  // left arc less-than bracket
    0x2995,  // double left arc greater-than bracket
    0x2997,  // left black tortoise shell bracket
    0x29D8,  // left wiggly fence
    0x29DA,  // left double wiggly fence
    0x29FC,  // left-pointing curved angle bracket
    0x3008,  // CJK left angle bracket
    0x300A,  // CJK left double angle bracket
    0x3010,  // CJK left black lenticular bracket
    0x3014,  // CJK left tortoise shell bracket
    0x3016,  // CJK left white lenticular bracket
    0x3018,  // CJK left white tortoise shell bracket
    0x301A,  // CJK left white square bracket
    0xFD3E,  // Ornate left parenthesis
    0xFE59,  // small left parenthesis
    0xFE5B,  // small left curly bracket
    0xFF08,  // fullwidth left parenthesis
    0xFF3B,  // fullwidth left square bracket
    0xFF5B,  // fullwidth left curly bracket
    '"',    '\'',   '`',
    0xFF07,  // fullwidth apostrophe
    0xFF02,  // fullwidth quotation mark
    0x2018,  // left single quotation mark (English, others)
    0x201C,  // left double quotation mark (English, others)
    0x201B,  // single high-reveresed-9 quotation mark (PropList.txt)
    0x201A,  // single low-9 quotation mark (Czech, German, Slovak)
    0x201E,  // double low-9 quotation mark (Czech, German, Slovak)
    0x201F,  // double high-reversed-9 quotation mark (PropList.txt)
    0x2019,  // right single quotation mark (Danish, Finnish, Swedish, Norw.)
    0x201D,  // right double quotation mark (Danish, Finnish, Swedish, Norw.)
    0x2039,  // single left-pointing angle quotation mark (French, others)
    0x00AB,  // left-pointing double angle quotation mark (French, others)
    0x203A,  // single right-pointing angle quotation mark (Slovenian, others)
    0x00BB,  // right-pointing double angle quotation mark (Slovenian, others)
    0x300C,  // left corner bracket (East Asian languages)
    0xFE41,  // presentation form for vertical left corner bracket
    0xFF62,  // halfwidth left corner bracket (East Asian languages)
    0x300E,  // left white corner bracket (East Asian languages)
    0xFE43,  // presentation form for vertical left white corner bracket
    0x301D,  // reversed double prime quotation mark (East Asian langs, horiz.)
    ')',    ']',    '>',    '}',
    0x207E,  // superscript right parenthesis
    0x208E,  // subscript right parenthesis
    0x27E7,  // mathematical right white square bracket
    0x27E9,  // mathematical right angle bracket
    0x27EB,  // mathematical right double angle bracket
    0x2984,  // right white curly bracket
    0x2986,  // right white parenthesis
    0x2988,  // Z notation right image bracket
    0x298A,  // Z notation right binding bracket
    0x298C,  // right square bracket with underbar
    0x298E,  // right square bracket with tick in top corner
    0x2990,  // right square bracket with tick in bottom corner
    0x2992,  // right angle bracket with dot
    0x2994,  // right arc greater-than bracket
    0x2996,  // double right arc less-than bracket
    0x2998,  // right black tortoise shell bracket
    0x29D9,  // right wiggly fence
    0x29DB,  // right double wiggly fence
    0x29FD,  // right-pointing curved angle bracket
    0x3009,  // CJK right angle bracket
    0x300B,  // CJK right double angle bracket
    0x3011,  // CJK right black lenticular bracket
    0x3015,  // CJK right tortoise shell bracket
    0x3017,  // CJK right white lenticular bracket
    0x3019,  // CJK right white tortoise shell bracket
    0x301B,  // CJK right white square bracket
    0xFD3F,  // Ornate right parenthesis
    0xFE5A,  // small right parenthesis
    0xFE5C,  // small right curly bracket
    0xFF09,  // fullwidth right parenthesis
    0xFF3D,  // fullwidth right square bracket
    0xFF5D,  // fullwidth right curly bracket
    '\'',   '"',    '`',
    0xFF07,  // fullwidth apostrophe
    0xFF02,  // fullwidth quotation mark
    0x2019,  // right single quotation mark (English, others)
    0x201D,  // right double quotation mark (English, others)
    0x2018,  // left single quotation mark (Czech, German, Slovak)
    0x201C,  // left double quotation mark (Czech, German, Slovak)
    0x203A,  // single right-pointing angle quotation mark (French, others)
    0x00BB,  // right-pointing double angle quotation mark (French, others)
    0x2039,  // single left-pointing angle quotation mark (Slovenian, others)
    0x00AB,  // left-pointing double angle quotation mark (Slovenian, others)
    0x300D,  // right corner bracket (East Asian languages)
    0xfe42,  // presentation form for vertical right corner bracket
    0xFF63,  // halfwidth right corner bracket (East Asian languages)
    0x300F,  // right white corner bracket (East Asian languages)
    0xfe44,  // presentation form for vertical right white corner bracket
    0x301F,  // low double prime quotation mark (East Asian languages)
    0x301E,  // close double prime (East Asian languages written horizontally)
    0x00A1,  // Spanish inverted exclamation mark
    0x00BF,  // Spanish inverted question mark
    '.',    '!',    '?',
    0x055C,  // Armenian exclamation mark
    0x055E,  // Armenian question mark
    0x0589,  // Armenian full stop
    0x061F,  // Arabic question mark
    0x06D4,  // Arabic full stop
    0x0700,  // Syriac end of paragraph
    0x0701,  // Syriac supralinear full stop
    0x0702,  // Syriac sublinear full stop
    0x0964,  // Devanagari danda..Devanagari double danda
    0x0965,
    0x1362,  // Ethiopic full stop
    0x1367,  // Ethiopic question mark
    0x1368,  // Ethiopic paragraph separator
    0x104A,  // Myanmar sign little section
    0x104B,  // Myanmar sign section
    0x166E,  // Canadian syllabics full stop
    0x17d4,  // Khmer sign khan
    0x1803,  // Mongolian full stop
    0x1809,  // Mongolian Manchu full stop
    0x1944,  // Limbu exclamation mark
    0x1945,  // Limbu question mark
    0x203C,  // double exclamation mark
    0x203D,  // interrobang
    0x2047,  // double question mark
    0x2048,  // question exclamation mark
    0x2049,  // exclamation question mark
    0x3002,  // ideographic full stop
    0x037E,  // Greek question mark
    0xFE52,  // small full stop
    0xFE56,  // small question mark
    0xFE57,  // small exclamation mark
    0xFF01,  // fullwidth exclamation mark
    0xFF0E,  // fullwidth full stop
    0xFF1F,  // fullwidth question mark
    0xFF61,  // halfwidth ideographic full stop
    0x2026,  // ellipsis
    0x30fb,  // Katakana middle dot
    0xff65,  // halfwidth Katakana middle dot
    0x2040,  // character tie
    '-',    '~',
    0x058a,  // Armenian hyphen
    0x1806,  // Mongolian todo soft hyphen
    0x2010,  // hyphen..horizontal bar
    0x2011, 0x2012, 0x2013, 0x2014, 0x2015,
    0x2053,  // swung dash -- from Table 6-3 of Unicode book
    0x207b,  // superscript minus
    0x208b,  // subscript minus
    0x2212,  // minus sign
    0x301c,  // wave dash
    0x3030,  // wavy dash
    0xfe31,  // presentation form for vertical em dash..en dash
    0xfe32,
    0xfe58,  // small em dash
    0xfe63,  // small hyphen-minus
    0xff0d,  // fullwidth hyphen-minus
    ',',    ':',    ';',
    0x00b7,  // middle dot
    0x0387,  // Greek ano teleia
    0x05c3,  // Hebrew punctuation sof pasuq
    0x060c,  // Arabic comma
    0x061b,  // Arabic semicolon
    0x066b,  // Arabic decimal separator
    0x066c,  // Arabic thousands separator
    0x0703,  // Syriac contraction and others
    0x0704, 0x0705, 0x0706, 0x0707, 0x0708, 0x0709, 0x70a,
    0x070c,  // Syric harklean metobelus
    0x0e5a,  // Thai character angkhankhu
    0x0e5b,  // Thai character khomut
    0x0f08,  // Tibetan mark sbrul shad
    0x0f0d,  // Tibetan mark shad..Tibetan mark rgya gram shad
    0x0f0e, 0x0f0f, 0x0f10, 0x0f11, 0x0f12,
    0x1361,  // Ethiopic wordspace
    0x1363,  // other Ethiopic chars
    0x1364, 0x1365, 0x1366,
    0x166d,  // Canadian syllabics chi sign
    0x16eb,  // Runic single punctuation..Runic cross punctuation
    0x16ed,
    0x17d5,  // Khmer sign camnuc pii huuh and other
    0x17d6,
    0x17da,  // Khmer sign koomut
    0x1802,  // Mongolian comma
    0x1804,  // Mongolian four dots and other
    0x1805,
    0x1808,  // Mongolian manchu comma
    0x3001,  // ideographic comma
    0xfe50,  // small comma and others
    0xfe51,
    0xfe54,  // small semicolon and other
    0xfe55,
    0xff0c,  // fullwidth comma
    0xff0e,  // fullwidth stop..fullwidth solidus
    0xff0f,
    0xff1a,  // fullwidth colon..fullwidth semicolon
    0xff1b,
    0xff64,  // halfwidth ideographic comma
    0x2016,  // double vertical line
    0x2032, 0x2033,
    0x2034,  // prime..triple prime
    0xfe61,  // small asterisk
    0xfe68,  // small reverse solidus
    0xff3c,  // fullwidth reverse solidus
};

TEST_P(IsPunctuationWordParamTest, IsPunctuation) {
  std::string test_string = StringFromUnicodeChar(GetParam());
  EXPECT_TRUE(IsPunctuationWord(test_string));
}

INSTANTIATE_TEST_SUITE_P(IsPuncWordParamTest, IsPunctuationWordParamTest,
                         ::testing::ValuesIn(punc_word_test_cases));

class IsEllipsisTest : public ::testing::Test {};

TEST_F(IsEllipsisTest, IsEllipsis) {
  int offset;
  EXPECT_TRUE(IsEllipsis("...", &offset));
  EXPECT_EQ(offset, 3);
  EXPECT_TRUE(IsEllipsis("…", &offset));
  EXPECT_EQ(offset, 3);
  EXPECT_FALSE(IsEllipsis("@", &offset));
  EXPECT_EQ(offset, 1);
}

class IsWhiteSpaceTest : public ::testing::Test {};

TEST_F(IsWhiteSpaceTest, IsWhiteSpace) {
  EXPECT_TRUE(IsWhiteSpace(" "));

  EXPECT_TRUE(IsWhiteSpace("\n"));

  EXPECT_TRUE(IsWhiteSpace("  "));

  EXPECT_FALSE(IsWhiteSpace("@"));

  EXPECT_FALSE(IsWhiteSpace("w"));
}

class IsAcronymTest : public ::testing::Test {};

TEST_F(IsAcronymTest, IsAcronym) {
  int offset = 0;
  EXPECT_TRUE(IsPeriodSeparatedAcronym("U.S.", &offset));
  EXPECT_EQ(offset, 4);

  offset = 0;
  EXPECT_TRUE(IsPeriodSeparatedAcronym("E.A.T.", &offset));
  EXPECT_EQ(offset, 6);

  offset = 0;
  EXPECT_TRUE(IsPeriodSeparatedAcronym("A.B.C.D.E.F.", &offset));
  EXPECT_EQ(offset, 12);

  offset = 0;
  EXPECT_FALSE(IsPeriodSeparatedAcronym("X.", &offset));

  EXPECT_FALSE(IsPeriodSeparatedAcronym("US", &offset));

  EXPECT_FALSE(IsPeriodSeparatedAcronym("U-S", &offset));
}

class EmoticonParamTest : public SentenceBreakingUtilsStringParamTest {};

static const char* const emoticon_test_cases[] = {":(:)",
                                                  ":)",
                                                  ":(",
                                                  ":o)",
                                                  ":]",
                                                  ":3",
                                                  ":>",
                                                  "=]",
                                                  "=)",
                                                  ":}",
                                                  ":^)",
                                                  ":-D",
                                                  ":-)))))",
                                                  ":-))))",
                                                  ":-)))",
                                                  ":-))",
                                                  ":-)",
                                                  ">:[",
                                                  ":-(",
                                                  ":(",
                                                  ":-c",
                                                  ":c",
                                                  ":-<",
                                                  ":<",
                                                  ":-[",
                                                  ":[",
                                                  ":{",
                                                  ";(",
                                                  ":-||",
                                                  ":@",
                                                  ">:(",
                                                  ":'-(",
                                                  ":'(",
                                                  ":'-)",
                                                  ":')",
                                                  "D:<",
                                                  ">:O",
                                                  ":-O",
                                                  ":-o",
                                                  ":*",
                                                  ":-*",
                                                  ":^*",
                                                  ";-)",
                                                  ";)",
                                                  "*-)",
                                                  "*)",
                                                  ";-]",
                                                  ";]",
                                                  ";^)",
                                                  ":-,",
                                                  ">:P",
                                                  ":-P",
                                                  ":p",
                                                  "=p",
                                                  ":-p",
                                                  "=p",
                                                  ":P",
                                                  "=P",
                                                  ";p",
                                                  ";-p",
                                                  ";P",
                                                  ";-P",
                                                  ">:\\",
                                                  ">:/",
                                                  ":-/",
                                                  ":-.",
                                                  ":/",
                                                  ":\\",
                                                  "=/",
                                                  "=\\",
                                                  ":|",
                                                  ":-|",
                                                  ":$",
                                                  ":-#",
                                                  ":#",
                                                  "O:-)",
                                                  "0:-)",
                                                  "0:)",
                                                  "0;^)",
                                                  ">:)",
                                                  ">;)",
                                                  ">:-)",
                                                  "}:-)",
                                                  "}:)",
                                                  "3:-)",
                                                  ">_>^",
                                                  "^<_<",
                                                  "|;-)",
                                                  "|-O",
                                                  ":-J",
                                                  ":-&",
                                                  ":&",
                                                  "#-)",
                                                  "<3",
                                                  "8-)",
                                                  "^_^",
                                                  ":D",
                                                  ":-D",
                                                  "=D",
                                                  "^_^;;",
                                                  "O=)",
                                                  "}=)",
                                                  "B)",
                                                  "B-)",
                                                  "=|",
                                                  "-_-",
                                                  "o_o;",
                                                  "u_u",
                                                  ":-\\",
                                                  ":s",
                                                  ":S",
                                                  ":-s",
                                                  ":-S",
                                                  ";*",
                                                  ";-*"
                                                  "=(",
                                                  ">.<",
                                                  ">:-(",
                                                  ">:(",
                                                  ">=(",
                                                  ";_;",
                                                  "T_T",
                                                  "='(",
                                                  ">_<",
                                                  "D:",
                                                  ":o",
                                                  ":-o",
                                                  "=o",
                                                  "o.o",
                                                  ":O",
                                                  ":-O",
                                                  "=O",
                                                  "O.O",
                                                  "x_x",
                                                  "X-(",
                                                  "X(",
                                                  "X-o",
                                                  "X-O",
                                                  ":X)",
                                                  "(=^.^=)",
                                                  "(=^..^=)",
                                                  "=^_^=",
                                                  "-<@%",
                                                  ":(|)",
                                                  "(]:{",
                                                  "<\\3",
                                                  "~@~",
                                                  "8'(",
                                                  "XD",
                                                  "DX"};

TEST_P(EmoticonParamTest, IsEmoticon) {
  int offset = 0;
  EXPECT_TRUE(IsEmoticon(GetParam(), &offset));
}

INSTANTIATE_TEST_SUITE_P(IsEmoticonParamTest, EmoticonParamTest,
                         ::testing::ValuesIn(emoticon_test_cases));

class IsEmoticonTest : public ::testing::Test {};

TEST_F(IsEmoticonTest, IsEmoticon) {
  int offset = 0;

  EXPECT_TRUE(IsEmoticon(">:-(", &offset));

  EXPECT_FALSE(IsEmoticon("w", &offset));

  EXPECT_FALSE(IsEmoticon(":", &offset));
}

TEST(SentenceFragmenterTest, Basic) {
  //                             1
  //                   012345678901234
  string test_input = "Hello. Foo bar!";
  SentenceFragmenterV2 fragmenter(test_input);
  std::vector<SentenceFragment> fragments;
  EXPECT_TRUE(fragmenter.FindFragments(&fragments).ok());
  EXPECT_EQ(fragments[0].start, 0);
  EXPECT_EQ(fragments[0].limit, 6);
  EXPECT_EQ(fragments[1].start, 7);
  EXPECT_EQ(fragments[1].limit, 15);
}

TEST(SentenceFragmenterTest, BasicEllipsis) {
  //                             1
  //                   012345678901234
  string test_input = "Hello...foo bar";
  SentenceFragmenterV2 fragmenter(test_input);
  std::vector<SentenceFragment> fragments;
  EXPECT_TRUE(fragmenter.FindFragments(&fragments).ok());

  EXPECT_EQ(fragments[0].start, 0);
  EXPECT_EQ(fragments[0].limit, 8);
  EXPECT_EQ(fragments[1].start, 8);
  EXPECT_EQ(fragments[1].limit, 15);
}

TEST(SentenceFragmenterTest, Parentheses) {
  //                             1         2
  //                   012345678901234567890123456789
  string test_input = "Hello (who are you...) foo bar";
  SentenceFragmenterV2 fragmenter(test_input);
  std::vector<SentenceFragment> fragments;
  EXPECT_TRUE(fragmenter.FindFragments(&fragments).ok());
  EXPECT_EQ(fragments[0].start, 0);
  EXPECT_EQ(fragments[0].limit, 22);
  EXPECT_EQ(fragments[1].start, 23);
  EXPECT_EQ(fragments[1].limit, 30);
}

TEST(SentenceFragmenterTest, MidFragmentParentheses) {
  //                             1         2
  //                   012345678901234567890123456789
  string test_input = "Hello (who are you) world? Foo bar";
  SentenceFragmenterV2 fragmenter(test_input);
  std::vector<SentenceFragment> fragments;
  EXPECT_TRUE(fragmenter.FindFragments(&fragments).ok());
  EXPECT_EQ(fragments[0].start, 0);
  EXPECT_EQ(fragments[0].limit, 26);
  EXPECT_EQ(fragments[1].start, 27);
  EXPECT_EQ(fragments[1].limit, 34);
}

TEST(SentenceFragmenterTest, PunctuationAfterParentheses) {
  //                             1         2
  //                   01234567890123456789012345678
  string test_input = "Hello (who are you)? Foo bar!";
  SentenceFragmenterV2 fragmenter(test_input);
  std::vector<SentenceFragment> fragments;
  EXPECT_TRUE(fragmenter.FindFragments(&fragments).ok());
  EXPECT_EQ(fragments[0].start, 0);
  EXPECT_EQ(fragments[0].limit, 20);
  EXPECT_EQ(fragments[1].start, 21);
  EXPECT_EQ(fragments[1].limit, 29);
}

TEST(SentenceFragmenterTest, ManyFinalPunctuations) {
  //                             1         2
  //                   0123456789012345678901234
  string test_input = "Hello!!!!! Who are you??";
  SentenceFragmenterV2 fragmenter(test_input);
  std::vector<SentenceFragment> fragments;
  EXPECT_TRUE(fragmenter.FindFragments(&fragments).ok());
  EXPECT_EQ(fragments[0].start, 0);
  EXPECT_EQ(fragments[0].limit, 10);
  EXPECT_EQ(fragments[1].start, 11);
  EXPECT_EQ(fragments[1].limit, 24);
}

TEST(SentenceFragmenterTest, NewLine) {
  //                             1         2             3
  //                   012345678901234567890 1 23456 7 89012 3 45678
  string test_input = "Who let the dogs out?\r\nWho?\r\nWho?\r\nWho?";
  SentenceFragmenterV2 fragmenter(test_input);
  std::vector<SentenceFragment> fragments;
  EXPECT_TRUE(fragmenter.FindFragments(&fragments).ok());
  EXPECT_EQ(fragments[0].start, 0);
  EXPECT_EQ(fragments[0].limit, 21);
  EXPECT_EQ(fragments[1].start, 23);
  EXPECT_EQ(fragments[1].limit, 27);
  EXPECT_EQ(fragments[2].start, 29);
  EXPECT_EQ(fragments[2].limit, 33);
  EXPECT_EQ(fragments[3].start, 35);
  EXPECT_EQ(fragments[3].limit, 39);
}

TEST(SentenceFragmenterTest, WhiteSpaceInPunctuation) {
  //                             1         2
  //                   0123456789012345678901234
  string test_input = "Hello?? !!! Who are you??";
  SentenceFragmenterV2 fragmenter(test_input);
  std::vector<SentenceFragment> fragments;
  EXPECT_TRUE(fragmenter.FindFragments(&fragments).ok());
  EXPECT_EQ(fragments[0].start, 0);
  EXPECT_EQ(fragments[0].limit, 7);
  EXPECT_EQ(fragments[1].start, 8);
  EXPECT_EQ(fragments[1].limit, 11);
  EXPECT_EQ(fragments[2].start, 12);
  EXPECT_EQ(fragments[2].limit, 25);
}

}  // namespace

TEST(FragmentBoundaryMatchTest, NoStateChange) {
  FragmentBoundaryMatch f;
  //                   ||
  //                   012345678901234
  string test_input = "Hello...foo bar";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_FALSE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), -1);
  EXPECT_EQ(f.first_close_punc_index(), -1);
  EXPECT_EQ(f.limit_index(), 1);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::INITIAL_STATE);
}

TEST(FragmentBoundaryMatchTest, BasicEllipsis) {
  FragmentBoundaryMatch f;
  //                   |  |
  //                   0123456789
  string test_input = "...foo bar";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 3);
  EXPECT_EQ(f.limit_index(), 3);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);
}

TEST(FragmentBoundaryMatchTest, BasicPeriod) {
  FragmentBoundaryMatch f;
  //                   ||
  //                   0123456789
  string test_input = ". Foo bar";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 1);
  EXPECT_EQ(f.limit_index(), 1);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);
}

TEST(FragmentBoundaryMatchTest, BasicAcronym) {
  FragmentBoundaryMatch f;
  //                   |  |
  //                   0123456789
  string test_input = "A.B. xyz";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 4);
  EXPECT_EQ(f.limit_index(), 4);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);
}

TEST(FragmentBoundaryMatchTest, LongerAcronym) {
  FragmentBoundaryMatch f;
  //                   |    |
  //                   0123456789
  string test_input = "I.B.M. yo";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 6);
  EXPECT_EQ(f.limit_index(), 6);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);
}

TEST(FragmentBoundaryMatchTest, Emoticon) {
  FragmentBoundaryMatch f;
  //                   |   |
  //                   0123456789012
  string test_input = ">:-( hello...";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 4);
  EXPECT_EQ(f.limit_index(), 4);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);
}

TEST(FragmentBoundaryMatchTest, ParensWithEllipsis) {
  FragmentBoundaryMatch f;
  //                   ||
  //                   0123456789012345
  string test_input = ".foo...) foo bar";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 1);
  EXPECT_EQ(f.limit_index(), 1);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);
}

TEST(FragmentBoundaryMatchTest, ClosingParenWithEllipsis) {
  FragmentBoundaryMatch f;
  //                   |  |
  //                   012345678901
  string test_input = "...) foo bar";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 3);
  EXPECT_EQ(f.limit_index(), 3);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);
}

TEST(FragmentBoundaryMatchTest, BeginAndEndParenWithEllipsis) {
  FragmentBoundaryMatch f;
  //                   ||
  //                   0123456789012
  string test_input = "(...) foo bar";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_FALSE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), -1);
  EXPECT_EQ(f.first_close_punc_index(), -1);
  EXPECT_EQ(f.limit_index(), 1);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::INITIAL_STATE);

  //            |  |
  //            0123456789012
  test_input = "...) foo bar";
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 3);
  EXPECT_EQ(f.limit_index(), 3);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);
}

TEST(FragmentBoundaryMatchTest, AcronymInSentence) {
  FragmentBoundaryMatch f;
  //                   |   |
  //                   0123456789012
  string test_input = "U.S. don't be surprised.";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 4);
  EXPECT_EQ(f.limit_index(), 4);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);
}

TEST(FragmentBoundaryMatchTest, HelloWithEllipsis) {
  FragmentBoundaryMatch f;
  //                   ||
  //                   01234567890
  string test_input = "o...foo bar";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_FALSE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), -1);
  EXPECT_EQ(f.first_close_punc_index(), -1);
  EXPECT_EQ(f.limit_index(), 1);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::INITIAL_STATE);

  //            |  |
  //            0123456789
  test_input = "...foo bar";
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 3);
  EXPECT_EQ(f.limit_index(), 3);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);
}

TEST(FragmentBoundaryMatchTest, ThreeStatesWithClosigParen) {
  FragmentBoundaryMatch f;
  //                   ||
  //                   0123456789012
  string test_input = "w...) foo bar";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_FALSE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), -1);
  EXPECT_EQ(f.first_close_punc_index(), -1);
  EXPECT_EQ(f.limit_index(), 1);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::INITIAL_STATE);

  //            |  |
  //            0123456789012
  test_input = "...) foo bar";
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 3);
  EXPECT_EQ(f.limit_index(), 3);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);

  //            ||
  //            0123456789012
  test_input = ") foo bar";
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 0);
  EXPECT_EQ(f.limit_index(), 1);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_CLOSE_PUNC);

  //            ||
  //            0123456789012
  test_input = " foo bar";
  EXPECT_FALSE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 0);
  EXPECT_EQ(f.limit_index(), 1);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_CLOSE_PUNC);
}

TEST(FragmentBoundaryMatchTest, NoTransition) {
  FragmentBoundaryMatch f;
  //                   |  |
  //                   0123456789012
  string test_input = "...foo bar";
  int index = 0;
  EXPECT_TRUE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 3);
  EXPECT_EQ(f.limit_index(), 3);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);

  //            ||
  //            0123456789012
  test_input = "foo bar";
  EXPECT_FALSE(f.Advance(index, test_input));
  EXPECT_TRUE(f.GotTerminalPunc());
  EXPECT_EQ(f.first_terminal_punc_index(), 0);
  EXPECT_EQ(f.first_close_punc_index(), 3);
  EXPECT_EQ(f.limit_index(), 3);
  EXPECT_EQ(f.state(), FragmentBoundaryMatch::COLLECTING_TERMINAL_PUNC);
}

}  // namespace text
}  // namespace tensorflow
