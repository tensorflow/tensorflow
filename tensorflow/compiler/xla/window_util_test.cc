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

#include "tensorflow/compiler/xla/window_util.h"

#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

TEST(WindowUtilTest, HasOverlappingWindowTest) {
  // MakeWindow() set a stride of 1 by default.
  EXPECT_FALSE(
      window_util::HasOverlappingWindow(window_util::MakeWindow({1, 1})));
  EXPECT_TRUE(
      window_util::HasOverlappingWindow(window_util::MakeWindow({2, 2, 2, 2})));
}

TEST(WindowUtilTest, MakeWindowStrideTest) {
  // MakeWindow() set a stride of 1 by default.
  Window w = window_util::MakeWindow({1, 2}, {3, 4});
  EXPECT_EQ(w.dimensions()[0].size(), 1);
  EXPECT_EQ(w.dimensions()[1].size(), 2);
  EXPECT_EQ(w.dimensions()[0].stride(), 3);
  EXPECT_EQ(w.dimensions()[1].stride(), 4);
}

}  // namespace
}  // namespace xla
