/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/setround.h"

#include <cmath>

#include "tsl/platform/test.h"

// LLVM does not support <cfenv>. Disable these tests when building with it.
// See b/35384639 for more information.
#if !defined(__clang__) || !defined(__OPTIMIZE__)

namespace tsl {
namespace {

void CheckDownward() {
  EXPECT_EQ(12, std::nearbyint(12.0));
  EXPECT_EQ(12, std::nearbyint(12.1));
  EXPECT_EQ(-13, std::nearbyint(-12.1));
  EXPECT_EQ(12, std::nearbyint(12.5));
  EXPECT_EQ(12, std::nearbyint(12.9));
  EXPECT_EQ(-13, std::nearbyint(-12.9));
  EXPECT_EQ(13, std::nearbyint(13.0));
}

void CheckToNearest() {
  EXPECT_EQ(12, std::nearbyint(12.0));
  EXPECT_EQ(12, std::nearbyint(12.1));
  EXPECT_EQ(-12, std::nearbyint(-12.1));
  EXPECT_EQ(12, std::nearbyint(12.5));
  EXPECT_EQ(13, std::nearbyint(12.9));
  EXPECT_EQ(-13, std::nearbyint(-12.9));
  EXPECT_EQ(13, std::nearbyint(13.0));
}

void CheckTowardZero() {
  EXPECT_EQ(12, std::nearbyint(12.0));
  EXPECT_EQ(12, std::nearbyint(12.1));
  EXPECT_EQ(-12, std::nearbyint(-12.1));
  EXPECT_EQ(12, std::nearbyint(12.5));
  EXPECT_EQ(12, std::nearbyint(12.9));
  EXPECT_EQ(-12, std::nearbyint(-12.9));
  EXPECT_EQ(13, std::nearbyint(13.0));
}

void CheckUpward() {
  EXPECT_EQ(12, std::nearbyint(12.0));
  EXPECT_EQ(13, std::nearbyint(12.1));
  EXPECT_EQ(-12, std::nearbyint(-12.1));
  EXPECT_EQ(13, std::nearbyint(12.5));
  EXPECT_EQ(13, std::nearbyint(12.9));
  EXPECT_EQ(-12, std::nearbyint(-12.9));
  EXPECT_EQ(13, std::nearbyint(13.0));
}

TEST(SetScopedSetRound, Downward) {
  port::ScopedSetRound round(FE_DOWNWARD);
  CheckDownward();
}

TEST(SetScopedSetRound, ToNearest) {
  port::ScopedSetRound round(FE_TONEAREST);
  CheckToNearest();
}

TEST(SetScopedSetRound, TowardZero) {
  port::ScopedSetRound round(FE_TOWARDZERO);
  CheckTowardZero();
}

TEST(SetScopedSetRound, Upward) {
  port::ScopedSetRound round(FE_UPWARD);
  CheckUpward();
}

TEST(SetScopedSetRound, Scoped) {
  std::fesetround(FE_TONEAREST);
  CheckToNearest();
  {
    port::ScopedSetRound round(FE_UPWARD);
    CheckUpward();
  }
  // Check that the rounding mode is reset when round goes out of scope.
  CheckToNearest();
}

}  // namespace
}  // namespace tsl

#endif  // !defined(__clang__) || !defined(__OPTIMIZE__)
