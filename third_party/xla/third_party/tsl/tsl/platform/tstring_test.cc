/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/tstring.h"

#include <memory>
#include <string>

#include "tsl/platform/cord.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/stringpiece.h"
#include "tsl/platform/test.h"

using ::tsl::tstring;

static const char kLongString[] =
    "abcdefghij"
    "klmnopqrst"
    "uvwxyz0123"
    "456789ABCD"
    "EFGHIKLMNO";
const size_t kLongStringLen = sizeof(kLongString) / sizeof(char) - sizeof(char);

TEST(TF_TStringTest, Construction) {
  tstring s10;
  tstring s11("a\0a", 3);
  tstring s12(kLongString);
  tstring s13(3, 'b');
  tstring s14(tsl::StringPiece("hi"));
  tstring s15(std::string("bye"));

  EXPECT_EQ("", s10);
  EXPECT_TRUE(s10.empty());
  EXPECT_EQ(tstring::Type::SMALL, s10.type());
  EXPECT_EQ(0, s10.size());
  EXPECT_EQ(0, s10.length());
  EXPECT_EQ(TF_TString_SmallCapacity, s10.capacity());

  EXPECT_EQ(std::string("a\0a", 3), s11);
  EXPECT_FALSE(s11.empty());
  EXPECT_EQ(3, s11.size());
  EXPECT_EQ(3, s11.length());
  EXPECT_EQ(kLongString, s12);
  EXPECT_EQ(kLongStringLen, s12.size());
  EXPECT_EQ(tstring::Type::LARGE, s12.type());
  EXPECT_LT(TF_TString_SmallCapacity, s12.capacity());
  EXPECT_EQ("bbb", s13);
  EXPECT_EQ("hi", s14);
  EXPECT_EQ(tstring::Type::SMALL, s14.type());
  EXPECT_EQ("bye", s15);
}

TEST(TF_TStringTest, CopyMove) {
  tstring s20(kLongString);
  tstring s21(s20);
  tstring s22;

  EXPECT_EQ(s20, s21);

  s22 = std::move(s21);

  EXPECT_EQ(s20, s22);
  EXPECT_EQ("", s21);  // NOLINT
  EXPECT_EQ(tstring::Type::SMALL, s21.type());
}

TEST(TF_TStringTest, Assignment) {
  tstring s30("123456789012345678901234567890");
  tstring s31;
  tstring s32;

  s31 = s30;

  EXPECT_EQ(s30, s31);
  EXPECT_EQ(tstring::Type::LARGE, s31.type());
  EXPECT_EQ(s30.size(), s31.size());

  s32 = std::move(s30);

  EXPECT_EQ(s31, s32);
  EXPECT_EQ("", s30);  // NOLINT
  EXPECT_EQ(tstring::Type::SMALL, s30.type());
  EXPECT_EQ(tstring::Type::LARGE, s32.type());

  s32 = tstring::view(kLongString);

  EXPECT_EQ(kLongString, s32);
  EXPECT_EQ(tstring::Type::VIEW, s32.type());
  EXPECT_EQ(kLongStringLen, s32.size());
  EXPECT_EQ(0, s32.capacity());

  tstring s33(std::move(s32));

  EXPECT_EQ(kLongString, s33);
  EXPECT_EQ(tstring::Type::VIEW, s33.type());
  EXPECT_EQ(kLongStringLen, s33.size());

  s32 = std::string(kLongString);

  EXPECT_EQ(kLongString, s32);
  EXPECT_EQ(tstring::Type::LARGE, s32.type());
  EXPECT_EQ(kLongStringLen, s32.size());

  // LARGE -> SMALL
  s32 = "hello";

  EXPECT_EQ("hello", s32);
  EXPECT_EQ(tstring::Type::SMALL, s32.type());
  EXPECT_EQ(5, s32.size());

  s33 = 'a';

  EXPECT_EQ("a", s33);
  EXPECT_EQ(tstring::Type::SMALL, s33.type());
  EXPECT_EQ(1, s33.size());

  s32 = tsl::StringPiece(kLongString);

  EXPECT_EQ(kLongString, s32);
  EXPECT_EQ(tstring::Type::LARGE, s32.type());
  EXPECT_EQ(kLongStringLen, s32.size());

  // LARGE -> SMALL but still LARGE
  s32.resize(TF_TString_SmallCapacity * 2);

  EXPECT_EQ(tsl::StringPiece(kLongString, TF_TString_SmallCapacity * 2), s32);
  EXPECT_EQ(tstring::Type::LARGE, s32.type());
  EXPECT_EQ(TF_TString_SmallCapacity * 2, s32.size());

  s32 = tstring::view(kLongString, kLongStringLen);

  EXPECT_EQ(kLongString, s32);
  EXPECT_EQ(tstring::Type::VIEW, s32.type());
  EXPECT_EQ(kLongStringLen, s32.size());

  s32.assign("hello1");

  EXPECT_EQ("hello1", s32);

  s32.assign("hello2", 5);

  EXPECT_EQ("hello", s32);

  s30.assign_as_view(kLongString);

  EXPECT_EQ(tstring::Type::VIEW, s30.type());

  s31.assign_as_view(s30);

  EXPECT_EQ(tstring::Type::VIEW, s31.type());

  EXPECT_EQ(kLongString, s30.c_str());
  EXPECT_EQ(kLongString, s31.c_str());

  std::string tmp(kLongString);
  s32.assign_as_view(tmp);

  EXPECT_EQ(tstring::Type::VIEW, s32.type());
  EXPECT_STREQ(kLongString, s32.c_str());

  s33.assign_as_view(kLongString, 2);

  EXPECT_EQ(2, s33.size());

  s32.assign_as_view(tsl::StringPiece(kLongString));

  EXPECT_EQ(tstring::Type::VIEW, s32.type());
  EXPECT_EQ(kLongString, s32.c_str());

#ifdef PLATFORM_GOOGLE
  s33 = absl::Cord(kLongString);

  EXPECT_EQ(kLongString, s33);
  EXPECT_EQ(tstring::Type::LARGE, s33.type());
  EXPECT_EQ(kLongStringLen, s33.size());

  tstring s34((absl::Cord(kLongString)));

  EXPECT_EQ(kLongString, s34);
  EXPECT_EQ(tstring::Type::LARGE, s34.type());
  EXPECT_EQ(kLongStringLen, s34.size());
#endif  // PLATFORM_GOOGLE
}

TEST(TF_TStringTest, Comparison) {
  tstring empty("");
  tstring a("a");
  tstring aa("aa");
  tstring a_("a");
  tstring b("b");
  const char c[] = "c";
  tstring nulla("\0a", 2);
  tstring nullb("\0b", 2);
  tstring nullaa("\0aa", 3);

  EXPECT_TRUE(a < b);
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(a == b);

  EXPECT_TRUE(a < aa);
  EXPECT_TRUE(a != aa);
  EXPECT_FALSE(a > aa);
  EXPECT_FALSE(a == aa);

  EXPECT_TRUE(b > a);
  EXPECT_TRUE(b != a);
  EXPECT_FALSE(b < a);
  EXPECT_FALSE(b == a);
  EXPECT_FALSE(a == b);

  EXPECT_FALSE(b == c);
  EXPECT_TRUE(b != c);

  EXPECT_TRUE(empty < a);
  EXPECT_TRUE(empty != a);
  EXPECT_FALSE(empty > a);
  EXPECT_FALSE(empty == a);

  EXPECT_TRUE(a > empty);
  EXPECT_TRUE(a != empty);
  EXPECT_FALSE(a < empty);
  EXPECT_FALSE(a == empty);

  EXPECT_FALSE(a < a_);
  EXPECT_FALSE(a != a_);
  EXPECT_FALSE(a > a_);
  EXPECT_TRUE(a == a_);

  EXPECT_TRUE(nulla < nullaa);
  EXPECT_TRUE(nulla != nullaa);
  EXPECT_FALSE(nulla > nullaa);
  EXPECT_FALSE(nulla == nullaa);

  EXPECT_TRUE(nulla < nullb);

  EXPECT_TRUE(nullaa > nulla);
  EXPECT_TRUE(nullaa != nulla);
  EXPECT_FALSE(nullaa < nulla);
  EXPECT_FALSE(nullaa == nulla);
}

TEST(TF_TStringTest, Conversion) {
  tstring s50(kLongString);
  std::string s51(s50);
  tsl::StringPiece s52(s50);
  EXPECT_EQ(kLongString, s51);
  EXPECT_EQ(kLongStringLen, s51.size());
  EXPECT_EQ(kLongString, s52);
  EXPECT_EQ(kLongStringLen, s52.size());

#ifdef PLATFORM_GOOGLE
  absl::AlphaNum s53(s50);

  EXPECT_STREQ(kLongString, s53.data());
  EXPECT_EQ(kLongStringLen, s53.size());
#endif  // PLATFORM_GOOGLE
}

TEST(TF_TStringTest, Allocation) {
  tstring s60;

  s60.resize(2);

  EXPECT_EQ(std::string("\0\0", 2), s60);
  EXPECT_EQ(2, s60.size());
  EXPECT_EQ(2, s60.length());

  s60.resize(6, 'a');

  EXPECT_EQ(std::string("\0\0aaaa", 6), s60);
  EXPECT_EQ(6, s60.size());
  EXPECT_EQ(6, s60.length());

  s60.resize(3, 'b');

  EXPECT_EQ(std::string("\0\0a", 3), s60);
  EXPECT_EQ(3, s60.size());
  EXPECT_EQ(3, s60.length());

  s60.clear();
  EXPECT_EQ("", s60);
  EXPECT_TRUE(s60.empty());
  EXPECT_EQ(0, s60.size());
  EXPECT_EQ(0, s60.length());

  s60.reserve(100);
  // 16-byte alignment 7*16-1 = 111
  EXPECT_EQ(111, s60.capacity());
  s60.reserve(100);
}

TEST(TF_TStringTest, ElementAccess) {
  tstring s70(kLongString);

  EXPECT_STREQ(kLongString, s70.data());
  EXPECT_EQ(s70.data(), s70.c_str());

  for (size_t i = 0; i < s70.size(); i++) {
    EXPECT_EQ(kLongString[i], s70.data()[i]);
  }

  tstring::const_iterator i = s70.begin();
  const char* j = kLongString;
  for (; *j != '\0'; i++, j++) {
    EXPECT_EQ(*j, *i);
  }
  EXPECT_EQ('\0', *s70.end());
  EXPECT_EQ(*i, *s70.end());
  EXPECT_EQ(*(i - 1), s70.back());
}

TEST(TF_TStringTest, Modifiers) {
  // Modifiers
  tstring s80("ba");
  tstring s81;
  tstring s82(kLongString);

  s81.append(s80);

  EXPECT_EQ("ba", s81);

  s81.append(s80);

  EXPECT_EQ("baba", s81);

  s81.append("\0c", 2);

  EXPECT_EQ(std::string("baba\0c", 6), s81);

  s81.append("dd");

  EXPECT_EQ(std::string("baba\0cdd", 8), s81);

  s81.append(3, 'z');

  EXPECT_EQ(tstring("baba\0cddzzz", 11), s81);

  s81.append(0, 'z');
  s81.append("dd", 0);
  s81.append("");
  s81.append(tstring());

  EXPECT_EQ(std::string("baba\0cddzzz", 11), s81);

  s81.erase(0, 1);

  EXPECT_EQ(std::string("aba\0cddzzz", 10), s81);

  s81.erase(4, 6);

  EXPECT_EQ(std::string("aba\0", 4), s81);

  s81.insert(1, tstring("\0moo\0", 5), 1, 4);

  EXPECT_EQ(std::string("amoo\0ba\0", 8), s81);

  s81.insert(0, 2, '\0');
  s81.insert(s81.size() - 1, 1, 'q');

  EXPECT_EQ(std::string("\0\0amoo\0baq\0", 11), s81);

  s81.erase(0, s81.size());

  EXPECT_EQ(tstring(), s81);

  s80.swap(s82);

  EXPECT_EQ(kLongString, s80);
  EXPECT_EQ("ba", s82);

  s82.push_back('\0');
  s82.push_back('q');

  EXPECT_EQ(std::string("ba\0q", 4), s82);
}

TEST(TF_TStringTest, Friends) {
  tstring s90("b");
  tstring s91("\0a\0", 3);
  tstring s92;

  EXPECT_EQ("b", s90 + s92);
  EXPECT_EQ("b", s92 + s90);

  EXPECT_EQ(std::string("\0a\0", 3), s92 + s91);
  EXPECT_EQ(std::string("\0a\0", 3), s91 + s92);

  EXPECT_EQ(std::string("b\0a\0", 4), s90 + s91);
  EXPECT_EQ(std::string("\0a\0b", 4), s91 + s90);

  std::stringstream ss;
  ss << s91;

  EXPECT_EQ(std::string("\0a\0", 3), ss.str());
}
