#include "tensorflow/core/lib/strings/numbers.h"

#include <string>
#include <gtest/gtest.h>

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
}

}  // namespace strings
}  // namespace tensorflow
