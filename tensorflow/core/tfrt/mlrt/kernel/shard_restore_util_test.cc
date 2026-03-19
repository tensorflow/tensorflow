/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/mlrt/kernel/shard_restore_util.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"

namespace tensorflow {
namespace tf_mlrt {
namespace {
using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

TEST(ShardRestoreUtilTest, Basic) {
  int num_shards = 2;
  std::vector<int64_t> shard_sizes = {8, 10, 3};

  std::vector<std::vector<int>> shards =
      ShardVariables(num_shards, absl::MakeSpan(shard_sizes));

  EXPECT_EQ(shards.size(), 2);
  EXPECT_THAT(shards[0], ElementsAre(1));
  EXPECT_THAT(shards[1], ElementsAre(0, 2));
}

TEST(ShardRestoreUtilTest, Imbalance) {
  int num_shards = 2;
  std::vector<int64_t> shard_sizes = {3, 3, 10, 3};

  std::vector<std::vector<int>> shards =
      ShardVariables(num_shards, absl::MakeSpan(shard_sizes));

  EXPECT_EQ(shards.size(), 2);
  EXPECT_THAT(shards[0], UnorderedElementsAre(0, 1, 3));
  EXPECT_THAT(shards[1], ElementsAre(2));
}

TEST(ShardRestoreUtilTest, SingleShard) {
  int num_shards = 1;
  std::vector<int64_t> shard_sizes = {10, 2};

  std::vector<std::vector<int>> shards =
      ShardVariables(num_shards, absl::MakeSpan(shard_sizes));

  EXPECT_EQ(shards.size(), 1);
  EXPECT_THAT(shards[0], ElementsAre(0, 1));
}

TEST(ShardRestoreUtilTest, NumVariablesLessThanShard) {
  int num_shards = 2;
  std::vector<int64_t> shard_sizes = {1};

  std::vector<std::vector<int>> shards =
      ShardVariables(num_shards, absl::MakeSpan(shard_sizes));

  EXPECT_EQ(shards.size(), 1);
  EXPECT_THAT(shards[0], ElementsAre(0));
}

}  // namespace
}  // namespace  tf_mlrt
}  // namespace tensorflow
