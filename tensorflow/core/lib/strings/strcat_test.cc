#include "tensorflow/core/lib/strings/strcat.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {
namespace strings {

// Test StrCat of ints and longs of various sizes and signdedness.
TEST(StrCat, Ints) {
  const int16 s = -1;
  const uint16 us = 2;
  const int i = -3;
  const unsigned int ui = 4;
  const int32 l = -5;
  const uint32 ul = 6;
  const int64 ll = -7;
  const uint64 ull = 8;
  const ptrdiff_t ptrdiff = -9;
  const size_t size = 10;
  const ssize_t ssize = -11;
  const intptr_t intptr = -12;
  const uintptr_t uintptr = 13;
  string answer;
  answer = StrCat(s, us);
  EXPECT_EQ(answer, "-12");
  answer = StrCat(i, ui);
  EXPECT_EQ(answer, "-34");
  answer = StrCat(l, ul);
  EXPECT_EQ(answer, "-56");
  answer = StrCat(ll, ull);
  EXPECT_EQ(answer, "-78");
  answer = StrCat(ptrdiff, size);
  EXPECT_EQ(answer, "-910");
  answer = StrCat(ssize, intptr);
  EXPECT_EQ(answer, "-11-12");
  answer = StrCat(uintptr, 0);
  EXPECT_EQ(answer, "130");
}

TEST(StrCat, Basics) {
  string result;

  string strs[] = {"Hello", "Cruel", "World"};

  StringPiece pieces[] = {"Hello", "Cruel", "World"};

  const char *c_strs[] = {"Hello", "Cruel", "World"};

  int32 i32s[] = {'H', 'C', 'W'};
  uint64 ui64s[] = {12345678910LL, 10987654321LL};

  result = StrCat(false, true, 2, 3);
  EXPECT_EQ(result, "0123");

  result = StrCat(-1);
  EXPECT_EQ(result, "-1");

  result = StrCat(0.5);
  EXPECT_EQ(result, "0.5");

  result = StrCat(strs[1], pieces[2]);
  EXPECT_EQ(result, "CruelWorld");

  result = StrCat(strs[0], ", ", pieces[2]);
  EXPECT_EQ(result, "Hello, World");

  result = StrCat(strs[0], ", ", strs[1], " ", strs[2], "!");
  EXPECT_EQ(result, "Hello, Cruel World!");

  result = StrCat(pieces[0], ", ", pieces[1], " ", pieces[2]);
  EXPECT_EQ(result, "Hello, Cruel World");

  result = StrCat(c_strs[0], ", ", c_strs[1], " ", c_strs[2]);
  EXPECT_EQ(result, "Hello, Cruel World");

  result = StrCat("ASCII ", i32s[0], ", ", i32s[1], " ", i32s[2], "!");
  EXPECT_EQ(result, "ASCII 72, 67 87!");

  result = StrCat(ui64s[0], ", ", ui64s[1], "!");
  EXPECT_EQ(result, "12345678910, 10987654321!");

  string one = "1";  // Actually, it's the size of this string that we want; a
                     // 64-bit build distinguishes between size_t and uint64,
                     // even though they're both unsigned 64-bit values.
  result = StrCat("And a ", one.size(), " and a ", &result[2] - &result[0],
                  " and a ", one, " 2 3 4", "!");
  EXPECT_EQ(result, "And a 1 and a 2 and a 1 2 3 4!");

  // result = StrCat("Single chars won't compile", '!');
  // result = StrCat("Neither will NULLs", NULL);
  result = StrCat("To output a char by ASCII/numeric value, use +: ", '!' + 0);
  EXPECT_EQ(result, "To output a char by ASCII/numeric value, use +: 33");

  float f = 100000.5;
  result = StrCat("A hundred K and a half is ", f);
  EXPECT_EQ(result, "A hundred K and a half is 100000.5");

  double d = f;
  d *= d;
  result = StrCat("A hundred K and a half squared is ", d);
  EXPECT_EQ(result, "A hundred K and a half squared is 10000100000.25");

  result = StrCat(1, 2, 333, 4444, 55555, 666666, 7777777, 88888888, 999999999);
  EXPECT_EQ(result, "12333444455555666666777777788888888999999999");
}

TEST(StrCat, MaxArgs) {
  string result;
  // Test 10 up to 26 arguments, the current maximum
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a");
  EXPECT_EQ(result, "123456789a");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b");
  EXPECT_EQ(result, "123456789ab");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c");
  EXPECT_EQ(result, "123456789abc");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d");
  EXPECT_EQ(result, "123456789abcd");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e");
  EXPECT_EQ(result, "123456789abcde");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f");
  EXPECT_EQ(result, "123456789abcdef");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g");
  EXPECT_EQ(result, "123456789abcdefg");
  result =
      StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g", "h");
  EXPECT_EQ(result, "123456789abcdefgh");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g",
                  "h", "i");
  EXPECT_EQ(result, "123456789abcdefghi");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g",
                  "h", "i", "j");
  EXPECT_EQ(result, "123456789abcdefghij");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g",
                  "h", "i", "j", "k");
  EXPECT_EQ(result, "123456789abcdefghijk");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g",
                  "h", "i", "j", "k", "l");
  EXPECT_EQ(result, "123456789abcdefghijkl");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g",
                  "h", "i", "j", "k", "l", "m");
  EXPECT_EQ(result, "123456789abcdefghijklm");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g",
                  "h", "i", "j", "k", "l", "m", "n");
  EXPECT_EQ(result, "123456789abcdefghijklmn");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g",
                  "h", "i", "j", "k", "l", "m", "n", "o");
  EXPECT_EQ(result, "123456789abcdefghijklmno");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g",
                  "h", "i", "j", "k", "l", "m", "n", "o", "p");
  EXPECT_EQ(result, "123456789abcdefghijklmnop");
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g",
                  "h", "i", "j", "k", "l", "m", "n", "o", "p", "q");
  EXPECT_EQ(result, "123456789abcdefghijklmnopq");
  // No limit thanks to C++11's variadic templates
  result = StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "a", "b", "c", "d", "e", "f",
                  "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                  "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D",
                  "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
                  "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z");
  EXPECT_EQ(result,
            "12345678910abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
}

TEST(StrAppend, Basics) {
  string result = "existing text";

  string strs[] = {"Hello", "Cruel", "World"};

  StringPiece pieces[] = {"Hello", "Cruel", "World"};

  const char *c_strs[] = {"Hello", "Cruel", "World"};

  int32 i32s[] = {'H', 'C', 'W'};
  uint64 ui64s[] = {12345678910LL, 10987654321LL};

  string::size_type old_size = result.size();
  StrAppend(&result, strs[0]);
  EXPECT_EQ(result.substr(old_size), "Hello");

  old_size = result.size();
  StrAppend(&result, strs[1], pieces[2]);
  EXPECT_EQ(result.substr(old_size), "CruelWorld");

  old_size = result.size();
  StrAppend(&result, strs[0], ", ", pieces[2]);
  EXPECT_EQ(result.substr(old_size), "Hello, World");

  old_size = result.size();
  StrAppend(&result, strs[0], ", ", strs[1], " ", strs[2], "!");
  EXPECT_EQ(result.substr(old_size), "Hello, Cruel World!");

  old_size = result.size();
  StrAppend(&result, pieces[0], ", ", pieces[1], " ", pieces[2]);
  EXPECT_EQ(result.substr(old_size), "Hello, Cruel World");

  old_size = result.size();
  StrAppend(&result, c_strs[0], ", ", c_strs[1], " ", c_strs[2]);
  EXPECT_EQ(result.substr(old_size), "Hello, Cruel World");

  old_size = result.size();
  StrAppend(&result, "ASCII ", i32s[0], ", ", i32s[1], " ", i32s[2], "!");
  EXPECT_EQ(result.substr(old_size), "ASCII 72, 67 87!");

  old_size = result.size();
  StrAppend(&result, ui64s[0], ", ", ui64s[1], "!");
  EXPECT_EQ(result.substr(old_size), "12345678910, 10987654321!");

  string one = "1";  // Actually, it's the size of this string that we want; a
                     // 64-bit build distinguishes between size_t and uint64,
                     // even though they're both unsigned 64-bit values.
  old_size = result.size();
  StrAppend(&result, "And a ", one.size(), " and a ", &result[2] - &result[0],
            " and a ", one, " 2 3 4", "!");
  EXPECT_EQ(result.substr(old_size), "And a 1 and a 2 and a 1 2 3 4!");

  // result = StrCat("Single chars won't compile", '!');
  // result = StrCat("Neither will NULLs", NULL);
  old_size = result.size();
  StrAppend(&result, "To output a char by ASCII/numeric value, use +: ",
            '!' + 0);
  EXPECT_EQ(result.substr(old_size),
            "To output a char by ASCII/numeric value, use +: 33");

  float f = 100000.5;
  old_size = result.size();
  StrAppend(&result, "A hundred K and a half is ", f);
  EXPECT_EQ(result.substr(old_size), "A hundred K and a half is 100000.5");

  double d = f;
  d *= d;
  old_size = result.size();
  StrAppend(&result, "A hundred K and a half squared is ", d);
  EXPECT_EQ(result.substr(old_size),
            "A hundred K and a half squared is 10000100000.25");

  // Test 9 arguments, the old maximum
  old_size = result.size();
  StrAppend(&result, 1, 22, 333, 4444, 55555, 666666, 7777777, 88888888, 9);
  EXPECT_EQ(result.substr(old_size), "1223334444555556666667777777888888889");

  // No limit thanks to C++11's variadic templates
  old_size = result.size();
  StrAppend(&result, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "a", "b", "c", "d", "e",
            "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
            "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E",
            "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
            "S", "T", "U", "V", "W", "X", "Y", "Z",
            "No limit thanks to C++11's variadic templates");
  EXPECT_EQ(result.substr(old_size),
            "12345678910abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "No limit thanks to C++11's variadic templates");
}

TEST(StrAppend, Death) {
  string s = "self";
  EXPECT_DEBUG_DEATH(StrAppend(&s, s.c_str() + 1), "Check failed:");
  EXPECT_DEBUG_DEATH(StrAppend(&s, s), "Check failed:");
}

static void CheckHex64(uint64 v) {
  using tensorflow::strings::Hex;
  string actual = StrCat(Hex(v, tensorflow::strings::ZERO_PAD_16));
  string expected = Printf("%016llx", static_cast<unsigned long long>(v));
  EXPECT_EQ(expected, actual) << " decimal value " << v;

  actual = StrCat(Hex(v, tensorflow::strings::ZERO_PAD_8));
  expected = Printf("%08llx", static_cast<unsigned long long>(v));
  EXPECT_EQ(expected, actual) << " decimal value " << v;

  actual = StrCat(Hex(v));
  expected = Printf("%llx", static_cast<unsigned long long>(v));
  EXPECT_EQ(expected, actual) << " decimal value " << v;
}

static void CheckHex32(uint32 v) {
  using tensorflow::strings::Hex;
  string actual = StrCat(Hex(v, tensorflow::strings::ZERO_PAD_8));
  string expected = Printf("%08x", v);
  EXPECT_EQ(expected, actual) << " decimal value " << v;

  actual = StrCat(Hex(v));
  expected = Printf("%x", v);
  EXPECT_EQ(expected, actual) << " decimal value " << v;
}

static void CheckHexSigned32(int32 v) {
  using tensorflow::strings::Hex;
  string actual = StrCat(Hex(v, tensorflow::strings::ZERO_PAD_8));
  string expected = Printf("%08x", v);
  EXPECT_EQ(expected, actual) << " decimal value " << v;

  actual = StrCat(Hex(v));
  expected = Printf("%x", v);
  EXPECT_EQ(expected, actual) << " decimal value " << v;
}

static void TestFastPrints() {
  using tensorflow::strings::Hex;

  // Test min int to make sure that works
  for (int i = 0; i < 10000; i++) {
    CheckHex64(i);
    CheckHex32(i);
    CheckHexSigned32(i);
    CheckHexSigned32(-i);
  }
  CheckHex64(0x123456789abcdef0ull);
  CheckHex32(0x12345678);

  int8 minus_one_8bit = -1;
  EXPECT_EQ("ff", StrCat(Hex(minus_one_8bit)));

  int16 minus_one_16bit = -1;
  EXPECT_EQ("ffff", StrCat(Hex(minus_one_16bit)));
}

TEST(Numbers, TestFunctionsMovedOverFromNumbersMain) { TestFastPrints(); }

}  // namespace strings
}  // namespace tensorflow
