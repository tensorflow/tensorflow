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
#include "tensorflow/lite/delegates/utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace delegates {
namespace {

using ::testing::ElementsAreArray;

void ReportError(TfLiteContext* context, const char* format, ...) {}

TEST(UtilsTest, PruneContinuousSubsets_NoSubsets) {
  TfLiteContext context;
  context.ReportError = ReportError;
  std::vector<int> original_indices = {};

  ASSERT_EQ(PruneContinuousSubsets(&context, 5, nullptr), kTfLiteError);

  ASSERT_EQ(PruneContinuousSubsets(&context, 0, &original_indices), kTfLiteOk);
  ASSERT_TRUE(original_indices.empty());

  ASSERT_EQ(PruneContinuousSubsets(&context, 2, &original_indices), kTfLiteOk);
  ASSERT_TRUE(original_indices.empty());
}

TEST(UtilsTest, PruneContinuousSubsets_SingleSubset) {
  TfLiteContext context;
  std::vector<int> original_indices = {0, 1, 2, 3};

  std::vector<int> indices = original_indices;
  ASSERT_EQ(PruneContinuousSubsets(&context, 1, &indices), kTfLiteOk);
  EXPECT_THAT(indices, ElementsAreArray({0, 1, 2, 3}));

  indices = original_indices;
  ASSERT_EQ(PruneContinuousSubsets(&context, 0, &indices), kTfLiteOk);
  ASSERT_TRUE(indices.empty());

  indices = original_indices;
  ASSERT_EQ(PruneContinuousSubsets(&context, 2, &indices), kTfLiteOk);
  EXPECT_THAT(indices, ElementsAreArray({0, 1, 2, 3}));
}

TEST(UtilsTest, PruneContinuousSubsets_MultipleSubsets) {
  TfLiteContext context;
  // 5 subsets: (0, 1), (3, 4, 5), (7), (10, 11), (19).
  std::vector<int> original_indices = {0, 1, 3, 4, 5, 7, 10, 11, 19};

  std::vector<int> indices = original_indices;
  ASSERT_EQ(PruneContinuousSubsets(&context, 4, &indices), kTfLiteOk);
  EXPECT_THAT(indices, ElementsAreArray({0, 1, 3, 4, 5, 7, 10, 11}));

  // Only the longest subset is selected.
  indices = original_indices;
  ASSERT_EQ(PruneContinuousSubsets(&context, 1, &indices), kTfLiteOk);
  EXPECT_THAT(indices, ElementsAreArray({3, 4, 5}));

  indices = original_indices;
  ASSERT_EQ(PruneContinuousSubsets(&context, 0, &indices), kTfLiteOk);
  ASSERT_TRUE(indices.empty());

  indices = original_indices;
  ASSERT_EQ(PruneContinuousSubsets(&context, 1000, &indices), kTfLiteOk);
  EXPECT_THAT(indices, ElementsAreArray({0, 1, 3, 4, 5, 7, 10, 11, 19}));
}

TEST(UtilsTest, PruneContinuousSubsets_UnsortedIndices) {
  TfLiteContext context;
  // 5 subsets: (0, 1), (3, 4, 5), (7), (10, 11), (19).
  std::vector<int> original_indices = {5, 7, 4, 10, 11, 19, 0, 1, 3};

  std::vector<int> indices = original_indices;
  ASSERT_EQ(PruneContinuousSubsets(&context, 4, &indices), kTfLiteOk);
  EXPECT_THAT(indices, ElementsAreArray({0, 1, 3, 4, 5, 7, 10, 11}));

  // Only the longest subset is selected.
  indices = original_indices;
  ASSERT_EQ(PruneContinuousSubsets(&context, 1, &indices), kTfLiteOk);
  EXPECT_THAT(indices, ElementsAreArray({3, 4, 5}));

  indices = original_indices;
  ASSERT_EQ(PruneContinuousSubsets(&context, 0, &indices), kTfLiteOk);
  ASSERT_TRUE(indices.empty());
}

}  // namespace
}  // namespace delegates
}  // namespace tflite
