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
#include "tensorflow/lite/testing/split.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace testing {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;

TEST(SplitTest, SplitToPos) {
  EXPECT_THAT(SplitToPos("test;:1-2-3 ;: test", ";:"),
              ElementsAre(Pair(0, 4), Pair(6, 12), Pair(14, 19)));
  EXPECT_THAT(SplitToPos("test;:1-2-3 ;: test", ":"),
              ElementsAre(Pair(0, 5), Pair(6, 13), Pair(14, 19)));
  EXPECT_THAT(SplitToPos("test", ":"), ElementsAre(Pair(0, 4)));
  EXPECT_THAT(SplitToPos("test ", ":"), ElementsAre(Pair(0, 5)));
  EXPECT_THAT(SplitToPos("", ":"), ElementsAre());
  EXPECT_THAT(SplitToPos("test ", ""), ElementsAre(Pair(0, 5)));
  EXPECT_THAT(SplitToPos("::::", ":"), ElementsAre());
}

TEST(SplitTest, SplitString) {
  EXPECT_THAT(Split<string>("A;B;C", ";"), ElementsAre("A", "B", "C"));
}

TEST(SplitTest, SplitFloat) {
  EXPECT_THAT(Split<float>("1.0 B 1e-5", " "), ElementsAre(1.0, 0.0, 1e-5));
}

TEST(SplitTest, SplitInt) {
  EXPECT_THAT(Split<int>("1,-1,258", ","), ElementsAre(1, -1, 258));
}

TEST(SplitTest, SplitUint8) {
  EXPECT_THAT(Split<uint8_t>("1,-1,258", ","), ElementsAre(1, 255, 2));
}

TEST(SplitTest, SplitBool) {
  EXPECT_THAT(Split<bool>("1, 0, 0, 1", ","),
              ElementsAre(true, false, false, true));
}

}  // namespace
}  // namespace testing
}  // namespace tflite
