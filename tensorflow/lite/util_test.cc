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

#include "tensorflow/lite/util.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace {

TEST(ConvertVectorToTfLiteIntArray, TestWithVector) {
  std::vector<int> input = {1, 2};
  TfLiteIntArray* output = ConvertVectorToTfLiteIntArray(input);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size, 2);
  EXPECT_EQ(output->data[0], 1);
  EXPECT_EQ(output->data[1], 2);
  TfLiteIntArrayFree(output);
}

TEST(ConvertVectorToTfLiteIntArray, TestWithEmptyVector) {
  std::vector<int> input;
  TfLiteIntArray* output = ConvertVectorToTfLiteIntArray(input);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size, 0);
  TfLiteIntArrayFree(output);
}

TEST(UtilTest, IsFlexOp) {
  EXPECT_TRUE(IsFlexOp("Flex"));
  EXPECT_TRUE(IsFlexOp("FlexOp"));
  EXPECT_FALSE(IsFlexOp("flex"));
  EXPECT_FALSE(IsFlexOp("Fle"));
  EXPECT_FALSE(IsFlexOp("OpFlex"));
  EXPECT_FALSE(IsFlexOp(nullptr));
  EXPECT_FALSE(IsFlexOp(""));
}

TEST(EqualArrayAndTfLiteIntArray, TestWithTFLiteArrayEmpty) {
  int input[] = {1, 2, 3, 4};
  EXPECT_FALSE(EqualArrayAndTfLiteIntArray(nullptr, 4, input));
}

TEST(EqualArrayAndTfLiteIntArray, TestWithTFLiteArrayWrongSize) {
  int input[] = {1, 2, 3, 4};
  TfLiteIntArray* output = ConvertArrayToTfLiteIntArray(4, input);
  EXPECT_FALSE(EqualArrayAndTfLiteIntArray(output, 3, input));
  free(output);
}

TEST(EqualArrayAndTfLiteIntArray, TestMismatch) {
  int input[] = {1, 2, 3, 4};
  TfLiteIntArray* output = ConvertVectorToTfLiteIntArray({1, 2, 2, 4});
  EXPECT_FALSE(EqualArrayAndTfLiteIntArray(output, 4, input));
  free(output);
}

TEST(EqualArrayAndTfLiteIntArray, TestMatch) {
  int input[] = {1, 2, 3, 4};
  TfLiteIntArray* output = ConvertArrayToTfLiteIntArray(4, input);
  EXPECT_TRUE(EqualArrayAndTfLiteIntArray(output, 4, input));
  free(output);
}

TEST(CombineHashes, TestHashOutputsEquals) {
  size_t output1 = CombineHashes({1, 2, 3, 4});
  size_t output2 = CombineHashes({1, 2, 3, 4});
  EXPECT_EQ(output1, output2);
}

TEST(CombineHashes, TestHashOutputsDifferent) {
  size_t output1 = CombineHashes({1, 2, 3, 4});
  size_t output2 = CombineHashes({1, 2, 2, 4});
  EXPECT_NE(output1, output2);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
