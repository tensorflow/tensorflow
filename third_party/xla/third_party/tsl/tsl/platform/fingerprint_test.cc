/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/fingerprint.h"

#include <array>
#include <unordered_set>

#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/types.h"

namespace tsl {
namespace {

TEST(Fingerprint64, IsForeverFrozen) {
  EXPECT_EQ(15404698994557526151ULL, Fingerprint64("Hello"));
  EXPECT_EQ(18308117990299812472ULL, Fingerprint64("World"));
}

TEST(Fingerprint128, IsForeverFrozen) {
  {
    const Fprint128 fingerprint = Fingerprint128("Hello");
    EXPECT_EQ(1163506517679092766ULL, fingerprint.low64);
    EXPECT_EQ(10829806600034513965ULL, fingerprint.high64);
  }

  {
    const Fprint128 fingerprint = Fingerprint128("World");
    EXPECT_EQ(14404540403896557767ULL, fingerprint.low64);
    EXPECT_EQ(4859093245152058524ULL, fingerprint.high64);
  }
}

TEST(Fingerprint128, Fprint128Hasher) {
  // Tests that this compiles:
  const std::unordered_set<Fprint128, Fprint128Hasher> map = {{1, 2}, {3, 4}};
}

TEST(FingerprintCat64, IsForeverFrozen) {
  EXPECT_EQ(16877292868973613377ULL,
            FingerprintCat64(Fingerprint64("Hello"), Fingerprint64("World")));
  // Do not expect commutativity.
  EXPECT_EQ(7158413233176775252ULL,
            FingerprintCat64(Fingerprint64("World"), Fingerprint64("Hello")));
}

// Hashes don't change.
TEST(FingerprintCat64, Idempotence) {
  const uint64_t orig =
      FingerprintCat64(Fingerprint64("Hello"), Fingerprint64("World"));
  EXPECT_EQ(orig,
            FingerprintCat64(Fingerprint64("Hello"), Fingerprint64("World")));
  EXPECT_NE(FingerprintCat64(Fingerprint64("Hello"), Fingerprint64("Hi")),
            FingerprintCat64(Fingerprint64("Hello"), Fingerprint64("World")));

  // Go back to the first test data ('orig') and make sure it hasn't changed.
  EXPECT_EQ(orig,
            FingerprintCat64(Fingerprint64("Hello"), Fingerprint64("World")));
}

TEST(Fprint128ToBytes, WorksCorrectly) {
  const Fprint128 fprint = {0xCAFEF00DDEADBEEF, 0xC0FFEE123456789A};
  constexpr std::array<char, 16> kExpected = {
      '\xca', '\xfe', '\xf0', '\x0d', '\xde', '\xad', '\xbe', '\xef',
      '\xc0', '\xff', '\xee', '\x12', '\x34', '\x56', '\x78', '\x9a',
  };
  EXPECT_EQ(Fprint128ToBytes(fprint), kExpected);
}

}  // namespace
}  // namespace tsl
