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
#include "tensorflow/contrib/boosted_trees/lib/models/multiple_additive_trees.h"

#include "tensorflow/contrib/boosted_trees/lib/testutil/batch_features_testutil.h"
#include "tensorflow/contrib/boosted_trees/lib/testutil/random_tree_gen.h"
#include "tensorflow/contrib/boosted_trees/resources/decision_tree_ensemble_resource.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
using boosted_trees::trees::DecisionTreeEnsembleConfig;
using test::AsTensor;

namespace boosted_trees {
namespace models {
namespace {

const int32 kNumThreadsMultiThreaded = 6;
const int32 kNumThreadsSingleThreaded = 1;

class MultipleAdditiveTreesTest : public ::testing::Test {
 protected:
  MultipleAdditiveTreesTest() : batch_features_(2) {
    // Create a batch of two examples having one dense feature each.
    // The shape of the dense matrix is therefore 2x1 as in one row per example
    // and one column per feature per example.
    auto dense_matrix = test::AsTensor<float>({7.0f, -2.0f}, {2, 1});
    TF_EXPECT_OK(
        batch_features_.Initialize({dense_matrix}, {}, {}, {}, {}, {}, {}));
  }

  boosted_trees::utils::BatchFeatures batch_features_;
};

TEST_F(MultipleAdditiveTreesTest, Empty) {
  // Create empty tree ensemble.
  DecisionTreeEnsembleConfig tree_ensemble_config;
  auto output_tensor = AsTensor<float>({9.0f, 23.0f}, {2, 1});
  auto output_matrix = output_tensor.matrix<float>();

  // Predict for both instances.
  tensorflow::thread::ThreadPool threads(tensorflow::Env::Default(), "test",
                                         kNumThreadsSingleThreaded);
  MultipleAdditiveTrees::Predict(tree_ensemble_config, {}, batch_features_,
                                 &threads, output_matrix);
  EXPECT_EQ(0, output_matrix(0, 0));
  EXPECT_EQ(0, output_matrix(1, 0));
}

TEST_F(MultipleAdditiveTreesTest, SingleClass) {
  // Add one bias and one stump to ensemble for a single class.
  DecisionTreeEnsembleConfig tree_ensemble_config;
  auto* tree1 = tree_ensemble_config.add_trees();
  auto* bias_leaf = tree1->add_nodes()->mutable_leaf()->mutable_sparse_vector();
  bias_leaf->add_index(0);
  bias_leaf->add_value(-0.4f);
  auto* tree2 = tree_ensemble_config.add_trees();
  auto* dense_split = tree2->add_nodes()->mutable_dense_float_binary_split();
  dense_split->set_feature_column(0);
  dense_split->set_threshold(5.0f);
  dense_split->set_left_id(1);
  dense_split->set_right_id(2);
  auto* leaf1 = tree2->add_nodes()->mutable_leaf()->mutable_sparse_vector();
  leaf1->add_index(0);
  leaf1->add_value(0.9f);
  auto* leaf2 = tree2->add_nodes()->mutable_leaf()->mutable_sparse_vector();
  leaf2->add_index(0);
  leaf2->add_value(0.2f);

  tree_ensemble_config.add_tree_weights(1.0);
  tree_ensemble_config.add_tree_weights(1.0);

  auto output_tensor = AsTensor<float>({0.0f, 0.0f}, {2, 1});
  auto output_matrix = output_tensor.matrix<float>();

  tensorflow::thread::ThreadPool threads(tensorflow::Env::Default(), "test",
                                         kNumThreadsSingleThreaded);

  // Normal case.
  {
    MultipleAdditiveTrees::Predict(tree_ensemble_config, {0, 1},
                                   batch_features_, &threads, output_matrix);
    EXPECT_FLOAT_EQ(-0.2f, output_matrix(0, 0));  // -0.4 (bias) + 0.2 (leaf 2).
    EXPECT_FLOAT_EQ(0.5f, output_matrix(1, 0));   // -0.4 (bias) + 0.9 (leaf 1).
  }
  // Weighted case
  {
    DecisionTreeEnsembleConfig weighted = tree_ensemble_config;
    weighted.set_tree_weights(0, 6.0);
    weighted.set_tree_weights(1, 3.2);
    MultipleAdditiveTrees::Predict(weighted, {0, 1}, batch_features_, &threads,
                                   output_matrix);
    // -0.4 (bias) + 0.2 (leaf 2).
    EXPECT_FLOAT_EQ(-0.4f * 6 + 0.2 * 3.2, output_matrix(0, 0));
    // -0.4 (bias) + 0.9 (leaf 1).
    EXPECT_FLOAT_EQ(-0.4f * 6 + 0.9 * 3.2, output_matrix(1, 0));
  }
  // Drop first tree.
  {
    MultipleAdditiveTrees::Predict(tree_ensemble_config, {1}, batch_features_,
                                   &threads, output_matrix);
    EXPECT_FLOAT_EQ(0.2f, output_matrix(0, 0));  // 0.2 (leaf 2).
    EXPECT_FLOAT_EQ(0.9f, output_matrix(1, 0));  // 0.9 (leaf 1).
  }
  // Drop second tree.
  {
    MultipleAdditiveTrees::Predict(tree_ensemble_config, {0}, batch_features_,
                                   &threads, output_matrix);
    EXPECT_FLOAT_EQ(-0.4f, output_matrix(0, 0));  // -0.4 (bias).
    EXPECT_FLOAT_EQ(-0.4f, output_matrix(1, 0));  // -0.4 (bias).
  }
  // Drop all trees.
  {
    MultipleAdditiveTrees::Predict(tree_ensemble_config, {}, batch_features_,
                                   &threads, output_matrix);
    EXPECT_FLOAT_EQ(0.0, output_matrix(0, 0));
    EXPECT_FLOAT_EQ(0.0, output_matrix(1, 0));
  }
}

TEST_F(MultipleAdditiveTreesTest, MultiClass) {
  // Add one bias and one stump to ensemble for two classes.
  DecisionTreeEnsembleConfig tree_ensemble_config;
  auto* tree1 = tree_ensemble_config.add_trees();
  auto* bias_leaf = tree1->add_nodes()->mutable_leaf()->mutable_sparse_vector();
  bias_leaf->add_index(0);
  bias_leaf->add_value(-0.4f);
  bias_leaf->add_index(1);
  bias_leaf->add_value(-0.7f);
  auto* tree2 = tree_ensemble_config.add_trees();
  auto* dense_split = tree2->add_nodes()->mutable_dense_float_binary_split();
  dense_split->set_feature_column(0);
  dense_split->set_threshold(5.0f);
  dense_split->set_left_id(1);
  dense_split->set_right_id(2);
  auto* leaf1 = tree2->add_nodes()->mutable_leaf()->mutable_sparse_vector();
  leaf1->add_index(0);
  leaf1->add_value(0.9f);
  auto* leaf2 = tree2->add_nodes()->mutable_leaf()->mutable_sparse_vector();
  leaf2->add_index(1);
  leaf2->add_value(0.2f);

  tree_ensemble_config.add_tree_weights(1.0);
  tree_ensemble_config.add_tree_weights(1.0);

  // Predict for both instances.
  tensorflow::thread::ThreadPool threads(tensorflow::Env::Default(), "test",
                                         kNumThreadsSingleThreaded);
  auto output_tensor = AsTensor<float>({0.0f, 0.0f, 0.0f, 0.0f}, {2, 2});
  auto output_matrix = output_tensor.matrix<float>();

  // Normal case.
  {
    MultipleAdditiveTrees::Predict(tree_ensemble_config, {0, 1},
                                   batch_features_, &threads, output_matrix);
    EXPECT_FLOAT_EQ(-0.4f, output_matrix(0, 0));  // -0.4 (bias)
    EXPECT_FLOAT_EQ(-0.5f, output_matrix(0, 1));  // -0.7 (bias) + 0.2 (leaf 2)
    EXPECT_FLOAT_EQ(0.5f, output_matrix(1, 0));   // -0.4 (bias) + 0.9 (leaf 1)
    EXPECT_FLOAT_EQ(-0.7f, output_matrix(1, 1));  // -0.7 (bias)
  }
  // Weighted case.
  {
    DecisionTreeEnsembleConfig weighted = tree_ensemble_config;
    weighted.set_tree_weights(0, 6.0);
    weighted.set_tree_weights(1, 3.2);
    MultipleAdditiveTrees::Predict(weighted, {0, 1}, batch_features_, &threads,
                                   output_matrix);
    // bias
    EXPECT_FLOAT_EQ(-0.4f * 6, output_matrix(0, 0));
    // bias + leaf 2
    EXPECT_FLOAT_EQ(-0.7f * 6 + 0.2f * 3.2, output_matrix(0, 1));
    // bias + leaf 2
    EXPECT_FLOAT_EQ(-0.4f * 6 + 0.9f * 3.2f, output_matrix(1, 0));
    // bias
    EXPECT_FLOAT_EQ(-0.7f * 6, output_matrix(1, 1));
  }
  // Dropout first tree.
  {
    MultipleAdditiveTrees::Predict(tree_ensemble_config, {1}, batch_features_,
                                   &threads, output_matrix);
    EXPECT_FLOAT_EQ(0.0, output_matrix(0, 0));
    EXPECT_FLOAT_EQ(0.2f, output_matrix(0, 1));  // 0.2 (leaf 2)
    EXPECT_FLOAT_EQ(0.9f, output_matrix(1, 0));  // 0.9 (leaf 2)
    EXPECT_FLOAT_EQ(0.0f, output_matrix(1, 1));
  }
  // Dropout second tree.
  {
    MultipleAdditiveTrees::Predict(tree_ensemble_config, {0}, batch_features_,
                                   &threads, output_matrix);
    EXPECT_FLOAT_EQ(-0.4f, output_matrix(0, 0));  // -0.4 (bias)
    EXPECT_FLOAT_EQ(-0.7f, output_matrix(0, 1));  // -0.7 (bias)
    EXPECT_FLOAT_EQ(-0.4f, output_matrix(1, 0));  // -0.4 (bias)
    EXPECT_FLOAT_EQ(-0.7f, output_matrix(1, 1));  // -0.7 (bias)
  }
  // Drop both trees.
  {
    MultipleAdditiveTrees::Predict(tree_ensemble_config, {}, batch_features_,
                                   &threads, output_matrix);
    EXPECT_FLOAT_EQ(0.0f, output_matrix(0, 0));
    EXPECT_FLOAT_EQ(0.0f, output_matrix(0, 1));
    EXPECT_FLOAT_EQ(0.0f, output_matrix(1, 0));
    EXPECT_FLOAT_EQ(0.0f, output_matrix(1, 1));
  }
}

TEST_F(MultipleAdditiveTreesTest, DenseLeaves) {
  DecisionTreeEnsembleConfig tree_ensemble_config;
  auto* tree1 = tree_ensemble_config.add_trees();
  auto* bias_leaf = tree1->add_nodes()->mutable_leaf()->mutable_vector();
  bias_leaf->add_value(-0.4f);
  bias_leaf->add_value(-0.7f);
  bias_leaf->add_value(3.0f);
  auto* tree2 = tree_ensemble_config.add_trees();
  auto* dense_split = tree2->add_nodes()->mutable_dense_float_binary_split();
  dense_split->set_feature_column(0);
  dense_split->set_threshold(5.0f);
  dense_split->set_left_id(1);
  dense_split->set_right_id(2);
  auto* leaf1 = tree2->add_nodes()->mutable_leaf()->mutable_vector();
  leaf1->add_value(0.9f);
  leaf1->add_value(0.8f);
  leaf1->add_value(0.7f);
  auto* leaf2 = tree2->add_nodes()->mutable_leaf()->mutable_vector();
  leaf2->add_value(0.2f);
  leaf2->add_value(0.3f);
  leaf2->add_value(0.4f);

  tree_ensemble_config.add_tree_weights(1.0);
  tree_ensemble_config.add_tree_weights(1.0);

  // Predict for both instances.
  tensorflow::thread::ThreadPool threads(tensorflow::Env::Default(), "test",
                                         kNumThreadsSingleThreaded);
  auto output_tensor =
      AsTensor<float>({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {2, 3});
  auto output_matrix = output_tensor.matrix<float>();

  // Normal case.
  {
    MultipleAdditiveTrees::Predict(tree_ensemble_config, {0, 1},
                                   batch_features_, &threads, output_matrix);
    EXPECT_FLOAT_EQ(-0.2f, output_matrix(0, 0));  // -0.4 (tree1) + 0.2 (leaf 2)
    EXPECT_FLOAT_EQ(-0.4f, output_matrix(0, 1));  // -0.7 (tree1) + 0.3 (leaf 2)
    EXPECT_FLOAT_EQ(3.4f, output_matrix(0, 2));   // 3.0 -(tree1) + 0.4 (leaf 2)
    EXPECT_FLOAT_EQ(0.5f, output_matrix(1, 0));   // -0.4 (tree1) + 0.9 (leaf 1)
    EXPECT_FLOAT_EQ(0.1f, output_matrix(1, 1));   // -0.7 (tree1) + 0.8 (leaf 1)
    EXPECT_FLOAT_EQ(3.7f, output_matrix(1, 2));   // 3.0 (tree1) + 0.7 (leaf 1)
  }
}

}  // namespace
}  // namespace models
}  // namespace boosted_trees
}  // namespace tensorflow
