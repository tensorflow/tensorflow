/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace gpu {
namespace {

using testing::Eq;

TEST(UtilTest, DivideRoundUp) {
  EXPECT_THAT(DivideRoundUp(0, 256), Eq(0));
  EXPECT_THAT(DivideRoundUp(2u, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(2, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(255u, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(255, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(256u, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(256, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(257u, 256), Eq(2));
  EXPECT_THAT(DivideRoundUp(257, 256), Eq(2));
}

TEST(UtilTest, AlignByN) {
  EXPECT_THAT(AlignByN(0u, 256), Eq(0));
  EXPECT_THAT(AlignByN(1u, 256), Eq(256));
  EXPECT_THAT(AlignByN(255u, 256), Eq(256));
  EXPECT_THAT(AlignByN(256u, 256), Eq(256));
  EXPECT_THAT(AlignByN(257u, 256), Eq(512));

  EXPECT_THAT(AlignByN(1, 4), Eq(4));
  EXPECT_THAT(AlignByN(80, 4), Eq(80));
  EXPECT_THAT(AlignByN(81, 4), Eq(84));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
