/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/sparse_core_ops_utils.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

TEST(ConvertSplitsAndBackTest, Split0) {
  const int max_division_level = 6;

  int64 original_split = 0;
  std::vector<int> actual_buckets =
      ConvertBinarySplitsToBucketSplits(original_split, max_division_level);
  std::vector<int> expected_buckets = {};
  int64 re_split =
      ConvertBucketSplitsToBinarySplits(expected_buckets, max_division_level);
  ASSERT_EQ(re_split, original_split);
}

TEST(ConvertSplitsAndBackTest, Split2) {
  const int max_division_level = 6;

  int64 original_split = 2;
  std::vector<int> actual_buckets =
      ConvertBinarySplitsToBucketSplits(original_split, max_division_level);
  std::vector<int> expected_buckets = {16};
  int64 re_split =
      ConvertBucketSplitsToBinarySplits(expected_buckets, max_division_level);
  ASSERT_EQ(re_split, original_split);
}

TEST(ConvertSplitsAndBackTest, Split3) {
  const int max_division_level = 6;

  int64 original_split = 3;
  std::vector<int> actual_buckets =
      ConvertBinarySplitsToBucketSplits(original_split, max_division_level);
  std::vector<int> expected_buckets = {16, 32};
  int64 re_split =
      ConvertBucketSplitsToBinarySplits(expected_buckets, max_division_level);
  ASSERT_EQ(re_split, original_split);
}

}  // namespace
}  // namespace tensorflow
