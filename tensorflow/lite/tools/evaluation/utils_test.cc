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
#include "tensorflow/lite/context.h"

namespace tflite {
namespace evaluation {
namespace {

constexpr char kLabelsPath[] =
    "tensorflow/lite/tools/evaluation/testdata/labels.txt";
constexpr char kDirPath[] =
    "tensorflow/lite/tools/evaluation/testdata";
constexpr char kEmptyFilePath[] =
    "tensorflow/lite/tools/evaluation/testdata/empty.txt";

TEST(UtilsTest, StripTrailingSlashesTest) {
  std::string path = "/usr/local/folder/";
  EXPECT_EQ(StripTrailingSlashes(path), "/usr/local/folder");

  path = "/usr/local/folder";
  EXPECT_EQ(StripTrailingSlashes(path), path);

  path = "folder";
  EXPECT_EQ(StripTrailingSlashes(path), path);
}

TEST(UtilsTest, ReadFileErrors) {
  std::string correct_path(kLabelsPath);
  std::string wrong_path("xyz.txt");
  std::vector<std::string> lines;
  EXPECT_FALSE(ReadFileLines(correct_path, nullptr));
  EXPECT_FALSE(ReadFileLines(wrong_path, &lines));
}

TEST(UtilsTest, ReadFileCorrectly) {
  std::string file_path(kLabelsPath);
  std::vector<std::string> lines;
  EXPECT_TRUE(ReadFileLines(file_path, &lines));

  EXPECT_EQ(lines.size(), 2);
  EXPECT_EQ(lines[0], "label1");
  EXPECT_EQ(lines[1], "label2");
}

TEST(UtilsTest, SortedFilenamesTest) {
  std::vector<std::string> files;
  EXPECT_EQ(GetSortedFileNames(kDirPath, &files), kTfLiteOk);

  EXPECT_EQ(files.size(), 2);
  EXPECT_EQ(files[0], kEmptyFilePath);
  EXPECT_EQ(files[1], kLabelsPath);

  EXPECT_EQ(GetSortedFileNames("wrong_path", &files), kTfLiteError);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
