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
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/toco/toco_types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace toco {
namespace port {
namespace {

#ifdef PLATFORM_GOOGLE
#define TFLITE_PREFIX "third_party/tensorflow/lite/"
#else
#define TFLITE_PREFIX "tensorflow/lite/"
#endif

TEST(TocoPortTest, Exists) {
  EXPECT_TRUE(
      file::Exists(TFLITE_PREFIX "toco/toco_port_test.cc", file::Defaults())
          .ok());

  EXPECT_FALSE(
      file::Exists("non-existent_file_asldjflasdjf", file::Defaults()).ok());
}

TEST(TocoPortTest, Readable) {
  EXPECT_TRUE(
      file::Readable(TFLITE_PREFIX "toco/toco_port_test.cc", file::Defaults())
          .ok());

  EXPECT_FALSE(
      file::Readable("non-existent_file_asldjflasdjf", file::Defaults()).ok());
}

TEST(TocoPortTest, JoinPath) {
  EXPECT_EQ("part1/part2", file::JoinPath("part1", "part2"));
  EXPECT_EQ("part1/part2", file::JoinPath("part1/", "part2"));
  EXPECT_EQ("part1/part2", file::JoinPath("part1", "/part2"));
  EXPECT_EQ("part1/part2", file::JoinPath("part1/", "/part2"));
}

}  // namespace
}  // namespace port
}  // namespace toco

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
