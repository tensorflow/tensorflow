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
#include "tensorflow/lite/tools/evaluation/utils.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace tflite {
namespace evaluation {
namespace {

constexpr char kFilePath[] =
    "tensorflow/lite/tools/evaluation/testdata/labels.txt";

TEST(UtilsTest, ReadFileErrors) {
  std::string correct_path(kFilePath);
  std::string wrong_path("xyz.txt");
  std::vector<std::string> lines;
  EXPECT_FALSE(ReadFileLines(correct_path, nullptr));
  EXPECT_FALSE(ReadFileLines(wrong_path, &lines));
}

TEST(UtilsTest, ReadFileCorrectly) {
  std::string file_path(kFilePath);
  std::vector<std::string> lines;
  EXPECT_TRUE(ReadFileLines(file_path, &lines));

  EXPECT_EQ(lines.size(), 2);
  EXPECT_EQ(lines[0], "label1");
  EXPECT_EQ(lines[1], "label2");
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
