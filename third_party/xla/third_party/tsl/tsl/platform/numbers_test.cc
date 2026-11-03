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

#include <cfloat>
#include <cstdint>
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
      uint64_t fp = (1ull << s) + delta;
      std::string s = FpToString(fp);
      uint64_t fp2;
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
      uint64_t fp = (1ull << s) + delta;
      std::string s = absl::StrCat(absl::Hex(fp, absl::kZeroPad16));
      uint64_t fp2;
      EXPECT_TRUE(HexStringToUint64(s, &fp2));
      EXPECT_EQ(fp, fp2) << s;
    }
  }
  uint64_t dummy;
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
  EXPECT_EQ(HumanReadableNum(std::numeric_limits<int64_t>::max()), "9.22E+18");
  EXPECT_EQ(HumanReadableNum(std::numeric_limits<int64_t>::min()), "-9.22E+18");
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
  EXPECT_EQ("-8.00EiB",
            HumanReadableNumBytes(std::numeric_limits<int64_t>::min()));
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

}  // namespace strings
}  // namespace tsl
