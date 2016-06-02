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

#include "tensorflow/core/lib/core/coding.h"

#include <vector>
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace core {

TEST(Coding, Fixed16) {
  static const uint16 N = 50000;

  string s;
  for (uint16 v = 0; v < N; v++) {
    char buf[sizeof(uint16)];
    EncodeFixed16(buf, v);
    s.append(buf, sizeof(buf));
  }

  const char* p = s.data();
  for (uint16 v = 0; v < N; v++) {
    uint16 actual = DecodeFixed16(p);
    ASSERT_EQ(v, actual);
    p += sizeof(uint16);
  }
}

TEST(Coding, Fixed32) {
  static const uint32 N = 100000;

  string s;
  for (uint32 v = 0; v < N; v++) {
    char buf[sizeof(uint32)];
    EncodeFixed32(buf, v);
    s.append(buf, sizeof(buf));
  }

  const char* p = s.data();
  for (uint32 v = 0; v < N; v++) {
    uint32 actual = DecodeFixed32(p);
    ASSERT_EQ(v, actual);
    p += sizeof(uint32);
  }
}

TEST(Coding, Fixed64) {
  string s;
  for (int power = 0; power <= 63; power++) {
    uint64 v = static_cast<uint64>(1) << power;
    char buf[sizeof(uint64)];
    EncodeFixed64(buf, v - 1);
    s.append(buf, sizeof(buf));
    EncodeFixed64(buf, v + 0);
    s.append(buf, sizeof(buf));
    EncodeFixed64(buf, v + 1);
    s.append(buf, sizeof(buf));
  }

  const char* p = s.data();
  for (int power = 0; power <= 63; power++) {
    uint64 v = static_cast<uint64>(1) << power;
    uint64 actual;
    actual = DecodeFixed64(p);
    ASSERT_EQ(v - 1, actual);
    p += sizeof(uint64);

    actual = DecodeFixed64(p);
    ASSERT_EQ(v + 0, actual);
    p += sizeof(uint64);

    actual = DecodeFixed64(p);
    ASSERT_EQ(v + 1, actual);
    p += sizeof(uint64);
  }
}

// Test that encoding routines generate little-endian encodings
TEST(Coding, EncodingOutput) {
  char dst[8];
  EncodeFixed16(dst, 0x0201);
  ASSERT_EQ(0x01, static_cast<int>(dst[0]));
  ASSERT_EQ(0x02, static_cast<int>(dst[1]));

  EncodeFixed32(dst, 0x04030201);
  ASSERT_EQ(0x01, static_cast<int>(dst[0]));
  ASSERT_EQ(0x02, static_cast<int>(dst[1]));
  ASSERT_EQ(0x03, static_cast<int>(dst[2]));
  ASSERT_EQ(0x04, static_cast<int>(dst[3]));

  EncodeFixed64(dst, 0x0807060504030201ull);
  ASSERT_EQ(0x01, static_cast<int>(dst[0]));
  ASSERT_EQ(0x02, static_cast<int>(dst[1]));
  ASSERT_EQ(0x03, static_cast<int>(dst[2]));
  ASSERT_EQ(0x04, static_cast<int>(dst[3]));
  ASSERT_EQ(0x05, static_cast<int>(dst[4]));
  ASSERT_EQ(0x06, static_cast<int>(dst[5]));
  ASSERT_EQ(0x07, static_cast<int>(dst[6]));
  ASSERT_EQ(0x08, static_cast<int>(dst[7]));
}

TEST(Coding, Varint32) {
  string s;
  for (uint32 i = 0; i < (32 * 32); i++) {
    uint32 v = (i / 32) << (i % 32);
    PutVarint32(&s, v);
  }

  const char* p = s.data();
  const char* limit = p + s.size();
  for (uint32 i = 0; i < (32 * 32); i++) {
    uint32 expected = (i / 32) << (i % 32);
    uint32 actual;
    p = GetVarint32Ptr(p, limit, &actual);
    ASSERT_TRUE(p != NULL);
    ASSERT_EQ(expected, actual);
  }
  ASSERT_EQ(p, s.data() + s.size());
}

TEST(Coding, Varint64) {
  // Construct the list of values to check
  std::vector<uint64> values;
  // Some special values
  values.push_back(0);
  values.push_back(100);
  values.push_back(~static_cast<uint64>(0));
  values.push_back(~static_cast<uint64>(0) - 1);
  for (uint32 k = 0; k < 64; k++) {
    // Test values near powers of two
    const uint64 power = 1ull << k;
    values.push_back(power);
    values.push_back(power - 1);
    values.push_back(power + 1);
  }

  string s;
  for (size_t i = 0; i < values.size(); i++) {
    PutVarint64(&s, values[i]);
  }

  const char* p = s.data();
  const char* limit = p + s.size();
  for (size_t i = 0; i < values.size(); i++) {
    ASSERT_TRUE(p < limit);
    uint64 actual;
    p = GetVarint64Ptr(p, limit, &actual);
    ASSERT_TRUE(p != NULL);
    ASSERT_EQ(values[i], actual);
  }
  ASSERT_EQ(p, limit);
}

TEST(Coding, Varint32Overflow) {
  uint32 result;
  string input("\x81\x82\x83\x84\x85\x11");
  ASSERT_TRUE(GetVarint32Ptr(input.data(), input.data() + input.size(),
                             &result) == NULL);
}

TEST(Coding, Varint32Truncation) {
  uint32 large_value = (1u << 31) + 100;
  string s;
  PutVarint32(&s, large_value);
  uint32 result;
  for (size_t len = 0; len < s.size() - 1; len++) {
    ASSERT_TRUE(GetVarint32Ptr(s.data(), s.data() + len, &result) == NULL);
  }
  ASSERT_TRUE(GetVarint32Ptr(s.data(), s.data() + s.size(), &result) != NULL);
  ASSERT_EQ(large_value, result);
}

TEST(Coding, Varint64Overflow) {
  uint64 result;
  string input("\x81\x82\x83\x84\x85\x81\x82\x83\x84\x85\x11");
  ASSERT_TRUE(GetVarint64Ptr(input.data(), input.data() + input.size(),
                             &result) == NULL);
}

TEST(Coding, Varint64Truncation) {
  uint64 large_value = (1ull << 63) + 100ull;
  string s;
  PutVarint64(&s, large_value);
  uint64 result;
  for (size_t len = 0; len < s.size() - 1; len++) {
    ASSERT_TRUE(GetVarint64Ptr(s.data(), s.data() + len, &result) == NULL);
  }
  ASSERT_TRUE(GetVarint64Ptr(s.data(), s.data() + s.size(), &result) != NULL);
  ASSERT_EQ(large_value, result);
}

}  // namespace core
}  // namespace tensorflow
