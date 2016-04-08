/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/strings/numbers.h"

#include <string>
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace strings {

// NOTE: most of the routines in numbers.h are tested indirectly through
// strcat_test.cc in this directory.

// Test StrCat of ints and longs of various sizes and signdedness.
TEST(FpToString, Ints) {
  for (int s = 0; s < 64; s++) {
    for (int delta = -1; delta <= 1; delta++) {
      uint64 fp = (1ull << s) + delta;
      string s = FpToString(fp);
      uint64 fp2;
      EXPECT_TRUE(StringToFp(s, &fp2));
      EXPECT_EQ(fp, fp2);
    }
  }
  Fprint dummy;
  EXPECT_FALSE(StringToFp("", &dummy));
  EXPECT_FALSE(StringToFp("xyz", &dummy));
  EXPECT_FALSE(StringToFp("0000000000000000xyz", &dummy));
}

TEST(Uint64ToHexString, Ints) {
  for (int s = 0; s < 64; s++) {
    for (int delta = -1; delta <= 1; delta++) {
      uint64 fp = (1ull << s) + delta;
      char buf[kFastToBufferSize];
      StringPiece s = Uint64ToHexString(fp, buf);
      uint64 fp2;
      EXPECT_TRUE(HexStringToUint64(s, &fp2));
      EXPECT_EQ(fp, fp2) << s;
    }
  }
  uint64 dummy;
  EXPECT_FALSE(HexStringToUint64("", &dummy));
  EXPECT_FALSE(HexStringToUint64("xyz", &dummy));
  EXPECT_FALSE(HexStringToUint64("0000000000000000xyz", &dummy));
}

TEST(HumanReadableNumBytes, Bytes) {
  EXPECT_EQ("0B", HumanReadableNumBytes(0));
  EXPECT_EQ("4B", HumanReadableNumBytes(4));
  EXPECT_EQ("1023B", HumanReadableNumBytes(1023));

  EXPECT_EQ("1.0KiB", HumanReadableNumBytes(1024));
  EXPECT_EQ("1.0KiB", HumanReadableNumBytes(1025));
  EXPECT_EQ("1.5KiB", HumanReadableNumBytes(1500));
  EXPECT_EQ("1.9KiB", HumanReadableNumBytes(1927));

  EXPECT_EQ("2.0KiB", HumanReadableNumBytes(2048));
  EXPECT_EQ("1.00MiB", HumanReadableNumBytes(1 << 20));
  EXPECT_EQ("11.77MiB", HumanReadableNumBytes(12345678));
  EXPECT_EQ("1.00GiB", HumanReadableNumBytes(1 << 30));

  EXPECT_EQ("1.00TiB", HumanReadableNumBytes(1LL << 40));
  EXPECT_EQ("1.00PiB", HumanReadableNumBytes(1LL << 50));
  EXPECT_EQ("1.00EiB", HumanReadableNumBytes(1LL << 60));

  // Try a few negative numbers
  EXPECT_EQ("-1B", HumanReadableNumBytes(-1));
  EXPECT_EQ("-4B", HumanReadableNumBytes(-4));
  EXPECT_EQ("-1000B", HumanReadableNumBytes(-1000));
  EXPECT_EQ("-11.77MiB", HumanReadableNumBytes(-12345678));
  EXPECT_EQ("-8E", HumanReadableNumBytes(kint64min));
}

TEST(safe_strto32, Int32s) {
  int32 result;

  EXPECT_EQ(true, safe_strto32("1", &result));
  EXPECT_EQ(1, result);
  EXPECT_EQ(true, safe_strto32("123", &result));
  EXPECT_EQ(123, result);
  EXPECT_EQ(true, safe_strto32(" -123 ", &result));
  EXPECT_EQ(-123, result);
  EXPECT_EQ(true, safe_strto32("2147483647", &result));
  EXPECT_EQ(2147483647, result);
  EXPECT_EQ(true, safe_strto32("-2147483648", &result));
  EXPECT_EQ(-2147483648, result);

  // Invalid argument
  EXPECT_EQ(false, safe_strto32(" 132as ", &result));
  EXPECT_EQ(false, safe_strto32(" 132.2 ", &result));
  EXPECT_EQ(false, safe_strto32(" -", &result));
  EXPECT_EQ(false, safe_strto32("", &result));
  EXPECT_EQ(false, safe_strto32("  ", &result));
  EXPECT_EQ(false, safe_strto32("123 a", &result));

  // Overflow
  EXPECT_EQ(false, safe_strto32("2147483648", &result));
  EXPECT_EQ(false, safe_strto32("-2147483649", &result));

  // Check that the StringPiece's length is respected.
  EXPECT_EQ(true, safe_strto32(StringPiece("123", 1), &result));
  EXPECT_EQ(1, result);
  EXPECT_EQ(true, safe_strto32(StringPiece(" -123", 4), &result));
  EXPECT_EQ(-12, result);
  EXPECT_EQ(false, safe_strto32(StringPiece(nullptr, 0), &result));
}

TEST(safe_strtou32, UInt32s) {
  uint32 result;

  EXPECT_TRUE(safe_strtou32("0", &result));
  EXPECT_EQ(0, result);
  EXPECT_TRUE(safe_strtou32("1", &result));
  EXPECT_EQ(1, result);
  EXPECT_TRUE(safe_strtou32("123", &result));
  EXPECT_EQ(123, result);
  EXPECT_TRUE(safe_strtou32("4294967295", &result));
  EXPECT_EQ(4294967295, result);

  // Invalid argument
  EXPECT_FALSE(safe_strtou32(" 132as ", &result));
  EXPECT_FALSE(safe_strtou32(" 132.2 ", &result));
  EXPECT_FALSE(safe_strtou32(" -", &result));
  EXPECT_FALSE(safe_strtou32("", &result));
  EXPECT_FALSE(safe_strtou32("  ", &result));
  EXPECT_FALSE(safe_strtou32("123 a", &result));
  EXPECT_FALSE(safe_strtou32("123 456", &result));

  // Overflow
  EXPECT_FALSE(safe_strtou32("4294967296", &result));
  EXPECT_FALSE(safe_strtou32("-1", &result));

  // Check that the StringPiece's length is respected.
  EXPECT_TRUE(safe_strtou32(StringPiece("123", 1), &result));
  EXPECT_EQ(1, result);
  EXPECT_TRUE(safe_strtou32(StringPiece(" 123", 3), &result));
  EXPECT_EQ(12, result);
  EXPECT_FALSE(safe_strtou32(StringPiece(nullptr, 0), &result));
}

TEST(safe_strto64, Int64s) {
  int64 result;

  EXPECT_EQ(true, safe_strto64("1", &result));
  EXPECT_EQ(1, result);
  EXPECT_EQ(true, safe_strto64("123", &result));
  EXPECT_EQ(123, result);
  EXPECT_EQ(true, safe_strto64(" -123 ", &result));
  EXPECT_EQ(-123, result);
  EXPECT_EQ(true, safe_strto64("9223372036854775807", &result));
  EXPECT_EQ(9223372036854775807, result);
  EXPECT_EQ(true, safe_strto64("-9223372036854775808", &result));
  // kint64min == -9223372036854775808
  // Use -9223372036854775808 directly results in out of range error
  EXPECT_EQ(kint64min, result);

  // Invalid argument
  EXPECT_EQ(false, safe_strto64(" 132as ", &result));
  EXPECT_EQ(false, safe_strto64(" 132.2 ", &result));
  EXPECT_EQ(false, safe_strto64(" -", &result));
  EXPECT_EQ(false, safe_strto64("", &result));
  EXPECT_EQ(false, safe_strto64("  ", &result));
  EXPECT_EQ(false, safe_strto64("123 a", &result));

  // Overflow
  EXPECT_EQ(false, safe_strto64("9223372036854775808", &result));
  EXPECT_EQ(false, safe_strto64("-9223372036854775809", &result));

  // Check that the StringPiece's length is respected.
  EXPECT_EQ(true, safe_strto64(StringPiece("123", 1), &result));
  EXPECT_EQ(1, result);
  EXPECT_EQ(true, safe_strto64(StringPiece(" -123", 4), &result));
  EXPECT_EQ(-12, result);
  EXPECT_EQ(false, safe_strto64(StringPiece(nullptr, 0), &result));
}

TEST(safe_strtou64, UInt64s) {
  uint64 result;

  EXPECT_TRUE(safe_strtou64("0", &result));
  EXPECT_EQ(0, result);
  EXPECT_TRUE(safe_strtou64("1", &result));
  EXPECT_EQ(1, result);
  EXPECT_TRUE(safe_strtou64("123", &result));
  EXPECT_EQ(123, result);
  EXPECT_TRUE(safe_strtou64("  345  ", &result));
  EXPECT_EQ(345, result);
  EXPECT_TRUE(safe_strtou64("18446744073709551615", &result));
  EXPECT_EQ(18446744073709551615UL, result);

  // Invalid argument
  EXPECT_FALSE(safe_strtou64(" 132.2 ", &result));
  EXPECT_FALSE(safe_strtou64(" 132.2 ", &result));
  EXPECT_FALSE(safe_strtou64(" -", &result));
  EXPECT_FALSE(safe_strtou64("", &result));
  EXPECT_FALSE(safe_strtou64("  ", &result));
  EXPECT_FALSE(safe_strtou64("123 a", &result));
  EXPECT_FALSE(safe_strtou64("123 456", &result));

  // Overflow
  EXPECT_FALSE(safe_strtou64("18446744073709551616", &result));
  EXPECT_FALSE(safe_strtou64("-1", &result));

  // Check that the StringPiece's length is respected.
  EXPECT_TRUE(safe_strtou64(StringPiece("123", 1), &result));
  EXPECT_EQ(1, result);
  EXPECT_TRUE(safe_strtou64(StringPiece(" 123", 3), &result));
  EXPECT_EQ(12, result);
  EXPECT_FALSE(safe_strtou64(StringPiece(nullptr, 0), &result));
}

TEST(safe_strtof, Float) {
  float result = 0;

  EXPECT_TRUE(safe_strtof("0.123456", &result));
  EXPECT_EQ(0.123456f, result);
  EXPECT_FALSE(safe_strtof("0.12345abc", &result));

  // Overflow to infinity, underflow to 0.
  EXPECT_TRUE(safe_strtof("1e39", &result));
  EXPECT_EQ(std::numeric_limits<float>::infinity(), result);
  EXPECT_TRUE(safe_strtof("1e-50", &result));
  EXPECT_EQ(0, result);
}

TEST(safe_strtod, Double) {
  double result = 0;

  EXPECT_TRUE(safe_strtod("0.1234567890123", &result));
  EXPECT_EQ(0.1234567890123, result);
  EXPECT_FALSE(safe_strtod("0.1234567890123abc", &result));

  // Overflow to infinity, underflow to 0.
  EXPECT_TRUE(safe_strtod("1e310", &result));
  EXPECT_EQ(std::numeric_limits<double>::infinity(), result);
  EXPECT_TRUE(safe_strtod("1e-325", &result));
  EXPECT_EQ(0, result);
}

}  // namespace strings
}  // namespace tensorflow
