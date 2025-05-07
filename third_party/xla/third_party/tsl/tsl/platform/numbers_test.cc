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

#include "tsl/platform/numbers.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>

#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/types.h"

namespace tsl {
namespace strings {

// NOTE: most of the routines in numbers.h are tested indirectly through
// strcat_test.cc in this directory.

// Test StrCat of ints and longs of various sizes and signedness.
TEST(FpToString, Ints) {
  for (int s = 0; s < 64; s++) {
    for (int delta = -1; delta <= 1; delta++) {
      uint64 fp = (1ull << s) + delta;
      string s = FpToString(fp);
      uint64 fp2;
      EXPECT_TRUE(HexStringToUint64(s, &fp2));
      EXPECT_EQ(fp, fp2);
    }
  }
  Fprint dummy;
  EXPECT_FALSE(HexStringToUint64("", &dummy));
  EXPECT_FALSE(HexStringToUint64("xyz", &dummy));
  EXPECT_FALSE(HexStringToUint64("0000000000000000xyz", &dummy));
}

TEST(Uint64ToHexString, Ints) {
  for (int s = 0; s < 64; s++) {
    for (int delta = -1; delta <= 1; delta++) {
      uint64 fp = (1ull << s) + delta;
      std::string s = absl::StrCat(absl::Hex(fp, absl::kZeroPad16));
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

TEST(HumanReadableNum, Basic) {
  EXPECT_EQ(HumanReadableNum(823), "823");
  EXPECT_EQ(HumanReadableNum(1024), "1.02k");
  EXPECT_EQ(HumanReadableNum(4000), "4.00k");
  EXPECT_EQ(HumanReadableNum(999499), "999.50k");
  EXPECT_EQ(HumanReadableNum(1000000), "1.00M");
  EXPECT_EQ(HumanReadableNum(1048575), "1.05M");
  EXPECT_EQ(HumanReadableNum(1048576), "1.05M");
  EXPECT_EQ(HumanReadableNum(23956812342), "23.96B");
  EXPECT_EQ(HumanReadableNum(123456789012345678), "1.23E+17");
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

TEST(HumanReadableElapsedTime, Basic) {
  EXPECT_EQ(HumanReadableElapsedTime(-10), "-10 s");
  EXPECT_EQ(HumanReadableElapsedTime(-0.001), "-1 ms");
  EXPECT_EQ(HumanReadableElapsedTime(-60.0), "-1 min");
  EXPECT_EQ(HumanReadableElapsedTime(0.00000001), "0.01 us");
  EXPECT_EQ(HumanReadableElapsedTime(0.0000012), "1.2 us");
  EXPECT_EQ(HumanReadableElapsedTime(0.0012), "1.2 ms");
  EXPECT_EQ(HumanReadableElapsedTime(0.12), "120 ms");
  EXPECT_EQ(HumanReadableElapsedTime(1.12), "1.12 s");
  EXPECT_EQ(HumanReadableElapsedTime(90.0), "1.5 min");
  EXPECT_EQ(HumanReadableElapsedTime(600.0), "10 min");
  EXPECT_EQ(HumanReadableElapsedTime(9000.0), "2.5 h");
  EXPECT_EQ(HumanReadableElapsedTime(87480.0), "1.01 days");
  EXPECT_EQ(HumanReadableElapsedTime(7776000.0), "2.96 months");
  EXPECT_EQ(HumanReadableElapsedTime(78840000.0), "2.5 years");
  EXPECT_EQ(HumanReadableElapsedTime(382386614.40), "12.1 years");
  EXPECT_EQ(HumanReadableElapsedTime(DBL_MAX), "5.7e+300 years");
}

TEST(safe_strto32, Int32s) {
  int32 result;

  EXPECT_EQ(true, absl::SimpleAtoi("1", &result));
  EXPECT_EQ(1, result);
  EXPECT_EQ(true, absl::SimpleAtoi("123", &result));
  EXPECT_EQ(123, result);
  EXPECT_EQ(true, absl::SimpleAtoi(" -123 ", &result));
  EXPECT_EQ(-123, result);
  EXPECT_EQ(true, absl::SimpleAtoi("2147483647", &result));
  EXPECT_EQ(2147483647, result);
  EXPECT_EQ(true, absl::SimpleAtoi("-2147483648", &result));
  EXPECT_EQ(-2147483648, result);

  // Invalid argument
  EXPECT_EQ(false, absl::SimpleAtoi(" 132as ", &result));
  EXPECT_EQ(false, absl::SimpleAtoi(" 132.2 ", &result));
  EXPECT_EQ(false, absl::SimpleAtoi(" -", &result));
  EXPECT_EQ(false, absl::SimpleAtoi("", &result));
  EXPECT_EQ(false, absl::SimpleAtoi("  ", &result));
  EXPECT_EQ(false, absl::SimpleAtoi("123 a", &result));

  // Overflow
  EXPECT_EQ(false, absl::SimpleAtoi("2147483648", &result));
  EXPECT_EQ(false, absl::SimpleAtoi("-2147483649", &result));

  // Check that the StringPiece's length is respected.
  EXPECT_EQ(true, absl::SimpleAtoi(absl::string_view("123", 1), &result));
  EXPECT_EQ(1, result);
  EXPECT_EQ(true, absl::SimpleAtoi(absl::string_view(" -123", 4), &result));
  EXPECT_EQ(-12, result);
  EXPECT_EQ(false, absl::SimpleAtoi(absl::string_view(nullptr, 0), &result));
}

TEST(safe_strtou32, UInt32s) {
  uint32 result;

  EXPECT_TRUE(absl::SimpleAtoi("0", &result));
  EXPECT_EQ(0, result);
  EXPECT_TRUE(absl::SimpleAtoi("1", &result));
  EXPECT_EQ(1, result);
  EXPECT_TRUE(absl::SimpleAtoi("123", &result));
  EXPECT_EQ(123, result);
  EXPECT_TRUE(absl::SimpleAtoi("4294967295", &result));
  EXPECT_EQ(4294967295, result);

  // Invalid argument
  EXPECT_FALSE(absl::SimpleAtoi(" 132as ", &result));
  EXPECT_FALSE(absl::SimpleAtoi(" 132.2 ", &result));
  EXPECT_FALSE(absl::SimpleAtoi(" -", &result));
  EXPECT_FALSE(absl::SimpleAtoi("", &result));
  EXPECT_FALSE(absl::SimpleAtoi("  ", &result));
  EXPECT_FALSE(absl::SimpleAtoi("123 a", &result));
  EXPECT_FALSE(absl::SimpleAtoi("123 456", &result));

  // Overflow
  EXPECT_FALSE(absl::SimpleAtoi("4294967296", &result));
  EXPECT_FALSE(absl::SimpleAtoi("-1", &result));

  // Check that the StringPiece's length is respected.
  EXPECT_TRUE(absl::SimpleAtoi(absl::string_view("123", 1), &result));
  EXPECT_EQ(1, result);
  EXPECT_TRUE(absl::SimpleAtoi(absl::string_view(" 123", 3), &result));
  EXPECT_EQ(12, result);
  EXPECT_FALSE(absl::SimpleAtoi(absl::string_view(nullptr, 0), &result));
}

TEST(safe_strto64, Int64s) {
  int64 result;

  EXPECT_EQ(true, absl::SimpleAtoi("1", &result));
  EXPECT_EQ(1, result);
  EXPECT_EQ(true, absl::SimpleAtoi("123", &result));
  EXPECT_EQ(123, result);
  EXPECT_EQ(true, absl::SimpleAtoi(" -123 ", &result));
  EXPECT_EQ(-123, result);
  EXPECT_EQ(true, absl::SimpleAtoi("9223372036854775807", &result));
  EXPECT_EQ(9223372036854775807, result);
  EXPECT_EQ(true, absl::SimpleAtoi("-9223372036854775808", &result));
  // kint64min == -9223372036854775808
  // Use -9223372036854775808 directly results in out of range error
  EXPECT_EQ(kint64min, result);

  // Invalid argument
  EXPECT_EQ(false, absl::SimpleAtoi(" 132as ", &result));
  EXPECT_EQ(false, absl::SimpleAtoi(" 132.2 ", &result));
  EXPECT_EQ(false, absl::SimpleAtoi(" -", &result));
  EXPECT_EQ(false, absl::SimpleAtoi("", &result));
  EXPECT_EQ(false, absl::SimpleAtoi("  ", &result));
  EXPECT_EQ(false, absl::SimpleAtoi("123 a", &result));

  // Overflow
  EXPECT_EQ(false, absl::SimpleAtoi("9223372036854775808", &result));
  EXPECT_EQ(false, absl::SimpleAtoi("-9223372036854775809", &result));

  // Check that the StringPiece's length is respected.
  EXPECT_EQ(true, absl::SimpleAtoi(absl::string_view("123", 1), &result));
  EXPECT_EQ(1, result);
  EXPECT_EQ(true, absl::SimpleAtoi(absl::string_view(" -123", 4), &result));
  EXPECT_EQ(-12, result);
  EXPECT_EQ(false, absl::SimpleAtoi(absl::string_view(nullptr, 0), &result));
}

TEST(safe_strtou64, UInt64s) {
  uint64 result;

  EXPECT_TRUE(absl::SimpleAtoi("0", &result));
  EXPECT_EQ(0, result);
  EXPECT_TRUE(absl::SimpleAtoi("1", &result));
  EXPECT_EQ(1, result);
  EXPECT_TRUE(absl::SimpleAtoi("123", &result));
  EXPECT_EQ(123, result);
  EXPECT_TRUE(absl::SimpleAtoi("  345  ", &result));
  EXPECT_EQ(345, result);
  EXPECT_TRUE(absl::SimpleAtoi("18446744073709551615", &result));
  EXPECT_EQ(18446744073709551615UL, result);

  // Invalid argument
  EXPECT_FALSE(absl::SimpleAtoi(" 132.2 ", &result));
  EXPECT_FALSE(absl::SimpleAtoi(" 132.2 ", &result));
  EXPECT_FALSE(absl::SimpleAtoi(" -", &result));
  EXPECT_FALSE(absl::SimpleAtoi("", &result));
  EXPECT_FALSE(absl::SimpleAtoi("  ", &result));
  EXPECT_FALSE(absl::SimpleAtoi("123 a", &result));
  EXPECT_FALSE(absl::SimpleAtoi("123 456", &result));

  // Overflow
  EXPECT_FALSE(absl::SimpleAtoi("18446744073709551616", &result));
  EXPECT_FALSE(absl::SimpleAtoi("-1", &result));

  // Check that the StringPiece's length is respected.
  EXPECT_TRUE(absl::SimpleAtoi(absl::string_view("123", 1), &result));
  EXPECT_EQ(1, result);
  EXPECT_TRUE(absl::SimpleAtoi(absl::string_view(" 123", 3), &result));
  EXPECT_EQ(12, result);
  EXPECT_FALSE(absl::SimpleAtoi(absl::string_view(nullptr, 0), &result));
}

TEST(safe_strtof, Float) {
  float result = 0;

  EXPECT_TRUE(absl::SimpleAtof("0.123456", &result));
  EXPECT_EQ(0.123456f, result);
  EXPECT_FALSE(absl::SimpleAtof("0.12345abc", &result));

  // Overflow to infinity, underflow to 0.
  EXPECT_TRUE(absl::SimpleAtof("1e39", &result));
  EXPECT_EQ(std::numeric_limits<float>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtof("-1e39", &result));
  EXPECT_EQ(-std::numeric_limits<float>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtof("1e-50", &result));
  EXPECT_EQ(0, result);

  EXPECT_TRUE(absl::SimpleAtof("0xF", &result));
  EXPECT_EQ(0xF, result);

  EXPECT_TRUE(absl::SimpleAtof("-0x2A", &result));
  EXPECT_EQ(-42.0f, result);

  EXPECT_TRUE(absl::SimpleAtof(" -0x2", &result));
  EXPECT_EQ(-2.0f, result);

  EXPECT_TRUE(absl::SimpleAtof("8 \t", &result));
  EXPECT_EQ(8.0f, result);

  EXPECT_TRUE(absl::SimpleAtof("\t20.0\t ", &result));
  EXPECT_EQ(20.0f, result);

  EXPECT_FALSE(absl::SimpleAtof("-infinity is awesome", &result));

  // Make sure we exit cleanly if the string is too long
  char test_str[2 * kFastToBufferSize];
  for (int i = 0; i < 2 * kFastToBufferSize; ++i) test_str[i] = 'a';
  test_str[kFastToBufferSize + 1] = '\0';
  EXPECT_FALSE(absl::SimpleAtof(test_str, &result));

  EXPECT_TRUE(absl::SimpleAtof("-inf", &result));
  EXPECT_EQ(-std::numeric_limits<float>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtof("+inf", &result));
  EXPECT_EQ(std::numeric_limits<float>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtof("InF", &result));
  EXPECT_EQ(std::numeric_limits<float>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtof("-INF", &result));
  EXPECT_EQ(-std::numeric_limits<float>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtof("nan", &result));
  EXPECT_TRUE(std::isnan(result));

  EXPECT_TRUE(absl::SimpleAtof("-nan", &result));
  EXPECT_TRUE(std::isnan(result));

  EXPECT_TRUE(absl::SimpleAtof("-NaN", &result));
  EXPECT_TRUE(std::isnan(result));

  EXPECT_TRUE(absl::SimpleAtof("+NAN", &result));
  EXPECT_TRUE(std::isnan(result));
}

TEST(safe_strtod, Double) {
  double result = 0;

  EXPECT_TRUE(absl::SimpleAtod("0.1234567890123", &result));
  EXPECT_EQ(0.1234567890123, result);
  EXPECT_FALSE(absl::SimpleAtod("0.1234567890123abc", &result));

  // Make sure we exit cleanly if the string is too long
  char test_str[2 * kFastToBufferSize];
  for (int i = 0; i < 2 * kFastToBufferSize; ++i) test_str[i] = 'a';
  test_str[kFastToBufferSize + 1] = '\0';
  EXPECT_FALSE(absl::SimpleAtod(test_str, &result));

  // Overflow to infinity, underflow to 0.
  EXPECT_TRUE(absl::SimpleAtod("1e310", &result));
  EXPECT_EQ(std::numeric_limits<double>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtod("-1e310", &result));
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtod("1e-325", &result));
  EXPECT_EQ(0, result);

  EXPECT_TRUE(absl::SimpleAtod(" -0x1c", &result));
  EXPECT_EQ(-28.0, result);

  EXPECT_TRUE(absl::SimpleAtod("50 \t", &result));
  EXPECT_EQ(50.0, result);

  EXPECT_TRUE(absl::SimpleAtod("\t82.0\t ", &result));
  EXPECT_EQ(82.0, result);

  EXPECT_TRUE(absl::SimpleAtod("infinity", &result));

  EXPECT_TRUE(absl::SimpleAtod("-inf", &result));
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtod("+inf", &result));
  EXPECT_EQ(std::numeric_limits<double>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtod("InF", &result));
  EXPECT_EQ(std::numeric_limits<double>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtod("-INF", &result));
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), result);

  EXPECT_TRUE(absl::SimpleAtod("nan", &result));
  EXPECT_TRUE(std::isnan(result));

  EXPECT_TRUE(absl::SimpleAtod("-nan", &result));
  EXPECT_TRUE(std::isnan(result));

  EXPECT_TRUE(absl::SimpleAtod("-NaN", &result));
  EXPECT_TRUE(std::isnan(result));

  EXPECT_TRUE(absl::SimpleAtod("+NAN", &result));
  EXPECT_TRUE(std::isnan(result));
}

std::string DoubleToString(double d) {
  return absl::StrCat(strings::LegacyPrecision(d));
}

TEST(LegacyPrecision, SimpleDoubleToString) {
  // Make sure that nice, round decimal numbers are printed using few digits,
  // even if they can't be represented exactly in binary.
  EXPECT_EQ("0", DoubleToString(0.0));
  EXPECT_EQ("1", DoubleToString(1.0));
  EXPECT_EQ("-1", DoubleToString(-1.0));
  EXPECT_EQ("0.2", DoubleToString(0.2));
  EXPECT_EQ("1.1", DoubleToString(1.1));
  EXPECT_EQ("1e+23", DoubleToString(1e23));
  EXPECT_EQ("47.8", DoubleToString(47.8));
  EXPECT_EQ("1000.2", DoubleToString(1000.2));
}

void TestDoubleRoundTrip(double value) {
  std::string str = DoubleToString(value);
  double rt = std::strtod(str.c_str(), nullptr);
  if (std::isnan(value)) {
    EXPECT_TRUE(std::isnan(rt));
  } else {
    EXPECT_EQ(value, rt);
  }
}

TEST(LegacyPrecision, DoubleRoundTrip) {
  // Make sure round-trips with strtod() work.  Note that even though we're
  // dealing with floating points, we expect the results to be *exactly*
  // equal, not approximately.
  TestDoubleRoundTrip(1.2345678901234567);
  TestDoubleRoundTrip(1.2345678901234565);
  TestDoubleRoundTrip(1.2345678901234569);
  TestDoubleRoundTrip(47.800000000000001);
  TestDoubleRoundTrip(0.10000000000000005);
  TestDoubleRoundTrip(0.010000000000000005);
  TestDoubleRoundTrip(0.000000010000000000000005);
  TestDoubleRoundTrip(1.0000000000000005);
  TestDoubleRoundTrip(10.000000000000005);
  TestDoubleRoundTrip(100.00000000000005);
  TestDoubleRoundTrip(1000.0000000000005);
  TestDoubleRoundTrip(100000000000000.05);

  // IEEE-754 double finite values furthest from zero.
  TestDoubleRoundTrip(1.7976931348623157e308);
  TestDoubleRoundTrip(-1.7976931348623157e308);

  // IEEE-754 double normalized values closest to zero.
  TestDoubleRoundTrip(2.225073858507202e-308);
  TestDoubleRoundTrip(-2.225073858507202e-308);

  // IEEE-754 double denormalized values closest to zero.
  TestDoubleRoundTrip(5e-324);
  TestDoubleRoundTrip(-5e-324);

  // Biggest and lowest valid numbers.
  TestDoubleRoundTrip(std::numeric_limits<double>::max());
  TestDoubleRoundTrip(std::numeric_limits<double>::lowest());

  // Infinity and NaN.
  TestDoubleRoundTrip(std::numeric_limits<double>::infinity());
  TestDoubleRoundTrip(-std::numeric_limits<double>::infinity());
  TestDoubleRoundTrip(std::numeric_limits<double>::quiet_NaN());
}

std::string FloatToString(float f) {
  return absl::StrCat(strings::LegacyPrecision(f));
}

TEST(LegacyPrecision, SimpleFloatToString) {
  // Make sure that nice, round decimal numbers are printed using few digits,
  // even if they can't be represented exactly in binary.
  EXPECT_EQ("0", FloatToString(0.0f));
  EXPECT_EQ("-0", FloatToString(-0.0f));
  EXPECT_EQ("1", FloatToString(1.0f));
  EXPECT_EQ("-1", FloatToString(-1.0f));
  EXPECT_EQ("0.2", FloatToString(0.2f));
  EXPECT_EQ("1.1", FloatToString(1.1f));
  EXPECT_EQ("1e+23", FloatToString(1e23f));
  EXPECT_EQ("47.8", FloatToString(47.8f));
  EXPECT_EQ("1000.2", FloatToString(1000.2f));
  EXPECT_EQ("1.17549435e-38", FloatToString(1.17549435e-38f));
}

void TestFloatRoundTrip(float value) {
  std::string str = FloatToString(value);
  float rt = std::strtof(str.c_str(), nullptr);
  if (std::isnan(value)) {
    EXPECT_TRUE(std::isnan(rt));
  } else {
    EXPECT_EQ(value, rt);
  }
}

TEST(LegacyPrecision, FloatRoundTrip) {
  // Make sure round-trips with strtod() work.  Note that even though we're
  // dealing with floating points, we expect the results to be *exactly*
  // equal, not approximately.
  FloatToString(1.2345678901234567);
  FloatToString(1.2345678901234565);
  FloatToString(1.2345678901234569);
  FloatToString(47.800005);
  FloatToString(0.10000005);
  FloatToString(0.010000005);
  FloatToString(0.000000010000005);
  FloatToString(1.0000005);
  FloatToString(10.000005);
  FloatToString(100.00005);
  FloatToString(1000.0005);
  FloatToString(100000.05);
  FloatToString(10.0000095);
  FloatToString(10.0000100);
  FloatToString(10.0000105);
  FloatToString(10.0000110);
  FloatToString(10.0000115);

  // IEEE-754 float finite values furthest from zero.
  FloatToString(+3.4028234e38f);
  FloatToString(-3.4028234e38f);

  // IEEE-754 float normalized values closest to zero.
  FloatToString(+1.175494351e-38);
  FloatToString(-1.175494351e-38);

  // IEEE-754 double denormalized values closest to zero.
  FloatToString(+1.4e-45);
  FloatToString(-1.4e-45);

  // Biggest and lowest valid numbers.
  FloatToString(std::numeric_limits<float>::max());
  FloatToString(std::numeric_limits<float>::lowest());

  // Infinity and NaN.
  FloatToString(std::numeric_limits<float>::infinity());
  FloatToString(-std::numeric_limits<float>::infinity());
  FloatToString(std::numeric_limits<float>::quiet_NaN());
}

TEST(LegacyPrecision, NoOpTypes) {
  EXPECT_EQ(absl::StrCat(strings::LegacyPrecision(1)), "1");
  EXPECT_EQ(absl::StrCat(strings::LegacyPrecision("foo")), "foo");
}

}  // namespace strings
}  // namespace tsl
