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
#include "tensorflow/lite/testing/join.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace testing {
namespace {

TEST(JoinTest, JoinInt) {
  std::vector<int> data = {1, 2, 3};
  EXPECT_EQ(Join(data.data(), data.size(), ","), "1,2,3");
}

TEST(JoinTest, JoinFloat) {
  float data[] = {1.0, -3, 2.3, 1e-5};
  EXPECT_EQ(Join(data, 4, " "), "1 -3 2.29999995 9.99999975e-06");
}

TEST(JoinTest, JoinNullData) { EXPECT_THAT(Join<int>(nullptr, 3, ","), ""); }

TEST(JoinTest, JoinZeroData) {
  std::vector<int> data;
  EXPECT_THAT(Join(data.data(), 0, ","), "");
}

}  // namespace
}  // namespace testing
}  // namespace tflite
