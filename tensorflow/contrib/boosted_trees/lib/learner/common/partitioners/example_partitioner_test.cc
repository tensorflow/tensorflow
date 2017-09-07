// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/contrib/boosted_trees/lib/learner/common/partitioners/example_partitioner.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace {

class ExamplePartitionerTest : public ::testing::Test {
 protected:
  ExamplePartitionerTest()
      : thread_pool_(tensorflow::Env::Default(), "test_pool", 2),
        batch_features_(2) {
    dense_matrix_ = test::AsTensor<float>({7.0f, -2.0f}, {2, 1});
    TF_EXPECT_OK(
        batch_features_.Initialize({dense_matrix_}, {}, {}, {}, {}, {}, {}));
  }

  thread::ThreadPool thread_pool_;
  Tensor dense_matrix_;
  boosted_trees::utils::BatchFeatures batch_features_;
};

TEST_F(ExamplePartitionerTest, EmptyTree) {
  boosted_trees::trees::DecisionTreeConfig tree_config;
  std::vector<int32> example_partition_ids(2);
  ExamplePartitioner::UpdatePartitions(tree_config, batch_features_, 1,
                                       &thread_pool_,
                                       example_partition_ids.data());
  EXPECT_EQ(0, example_partition_ids[0]);
  EXPECT_EQ(0, example_partition_ids[1]);
}

TEST_F(ExamplePartitionerTest, UpdatePartitions) {
  // Create tree with one split.
  // TODO(salehay): figure out if we can use PARSE_TEXT_PROTO.
  boosted_trees::trees::DecisionTreeConfig tree_config;
  auto* split = tree_config.add_nodes()->mutable_dense_float_binary_split();
  split->set_feature_column(0);
  split->set_threshold(3.0f);
  split->set_left_id(1);
  split->set_right_id(2);
  tree_config.add_nodes()->mutable_leaf();
  tree_config.add_nodes()->mutable_leaf();

  // Partition input:
  // Instance 1 has !(7 <= 3) => go right => leaf 2.
  // Instance 2 has (-2 <= 3) => go left => leaf 1.
  std::vector<int32> example_partition_ids(2);
  ExamplePartitioner::UpdatePartitions(tree_config, batch_features_, 1,
                                       &thread_pool_,
                                       example_partition_ids.data());
  EXPECT_EQ(2, example_partition_ids[0]);
  EXPECT_EQ(1, example_partition_ids[1]);
}

TEST_F(ExamplePartitionerTest, PartitionExamples) {
  // Create tree with one split.
  // TODO(salehay): figure out if we can use PARSE_TEXT_PROTO.
  boosted_trees::trees::DecisionTreeConfig tree_config;
  auto* split = tree_config.add_nodes()->mutable_dense_float_binary_split();
  split->set_feature_column(0);
  split->set_threshold(3.0f);
  split->set_left_id(1);
  split->set_right_id(2);
  tree_config.add_nodes()->mutable_leaf();
  tree_config.add_nodes()->mutable_leaf();

  // Partition input:
  // Instance 1 has !(7 <= 3) => go right => leaf 2.
  // Instance 2 has (-2 <= 3) => go left => leaf 1.
  std::vector<int32> example_partition_ids(2);
  ExamplePartitioner::PartitionExamples(tree_config, batch_features_, 1,
                                        &thread_pool_,
                                        example_partition_ids.data());
  EXPECT_EQ(2, example_partition_ids[0]);
  EXPECT_EQ(1, example_partition_ids[1]);
}

}  // namespace
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow
