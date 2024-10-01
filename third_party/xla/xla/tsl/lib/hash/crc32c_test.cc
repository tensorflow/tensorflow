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

#include "xla/tsl/lib/hash/crc32c.h"

#include <string>

#include "absl/strings/cord.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"
#include "tsl/platform/types.h"

namespace tsl {
namespace crc32c {

TEST(CRC, StandardResults) {
  // From rfc3720 section B.4.
  char buf[32];

  memset(buf, 0, sizeof(buf));
  ASSERT_EQ(0x8a9136aa, Value(buf, sizeof(buf)));

  memset(buf, 0xff, sizeof(buf));
  ASSERT_EQ(0x62a8ab43, Value(buf, sizeof(buf)));

  for (int i = 0; i < 32; i++) {
    buf[i] = i;
  }
  ASSERT_EQ(0x46dd794e, Value(buf, sizeof(buf)));

  for (int i = 0; i < 32; i++) {
    buf[i] = 31 - i;
  }
  ASSERT_EQ(0x113fdb5c, Value(buf, sizeof(buf)));

  unsigned char data[48] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  ASSERT_EQ(0xd9963a56, Value(reinterpret_cast<char*>(data), sizeof(data)));

  // Try unaligned sizes and offsets.
  // Accelerated and unaccelerated code both produce these results.
  ASSERT_EQ(0xdd1b19be, Value(reinterpret_cast<char*>(data), sizeof(data) - 7));
  ASSERT_EQ(0x4930c4b1,
            Value(reinterpret_cast<char*>(data) + 1, sizeof(data) - 4));
}

TEST(CRC, Values) { ASSERT_NE(Value("a", 1), Value("foo", 3)); }

TEST(CRC, Extend) {
  ASSERT_EQ(Value("hello world", 11), Extend(Value("hello ", 6), "world", 5));
}

TEST(CRC, Mask) {
  uint32 crc = Value("foo", 3);
  ASSERT_NE(crc, Mask(crc));
  ASSERT_NE(crc, Mask(Mask(crc)));
  ASSERT_EQ(crc, Unmask(Mask(crc)));
  ASSERT_EQ(crc, Unmask(Unmask(Mask(Mask(crc)))));
}

#if defined(PLATFORM_GOOGLE)
TEST(CRC, ValuesWithCord) {
  ASSERT_NE(Value(absl::Cord("a")), Value(absl::Cord("foo")));
}

TEST(CRC, ExtendWithCord) {
  ASSERT_EQ(Value(absl::Cord("hello world")),
            Extend(Value(absl::Cord("hello ")), absl::Cord("world")));
}
#endif

static void BM_CRC(::testing::benchmark::State& state) {
  int len = state.range(0);
  std::string input(len, 'x');
  uint32 h = 0;
  for (auto s : state) {
    h = Extend(h, input.data() + 1, len - 1);
  }
  state.SetBytesProcessed(state.iterations() * len);
  VLOG(1) << h;
}
BENCHMARK(BM_CRC)->Range(1, 256 * 1024);

}  // namespace crc32c
}  // namespace tsl
