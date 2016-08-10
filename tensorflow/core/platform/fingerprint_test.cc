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

#include "tensorflow/core/platform/types.h"

#include <unordered_set>
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
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

}  // namespace
}  // namespace tensorflow
