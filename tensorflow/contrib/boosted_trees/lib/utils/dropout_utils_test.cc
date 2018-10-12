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
#include "tensorflow/contrib/boosted_trees/lib/utils/dropout_utils.h"

#include <sys/types.h>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <unordered_set>
#include <utility>

#include "tensorflow/contrib/boosted_trees/proto/tree_config.pb.h"  // NOLINT
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"

using std::unordered_set;
using tensorflow::boosted_trees::learner::LearningRateDropoutDrivenConfig;
using tensorflow::boosted_trees::trees::DecisionTreeEnsembleConfig;

namespace tensorflow {
namespace boosted_trees {
namespace utils {
namespace {

const uint32 kSeed = 123;
const int32 kNumTrees = 1000;

class DropoutUtilsTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Fill an weights.
    for (int i = 0; i < kNumTrees; ++i) {
      weights_.push_back(1.1 + 0.4 * i);
    }
  }

 protected:
  std::vector<float> weights_;
};

TEST_F(DropoutUtilsTest, DropoutProbabilityTest) {
  std::vector<int32> dropped_trees;
  std::vector<float> original_weights;
  std::unordered_set<int32> trees_not_to_drop;

  // Do not drop any trees
  {
    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(0.0);
    config.set_learning_rate(1.0);

    TF_EXPECT_OK(DropoutUtils::DropOutTrees(kSeed, config, trees_not_to_drop,
                                            weights_, &dropped_trees,
                                            &original_weights));

    // Nothing changed
    EXPECT_TRUE(dropped_trees.empty());
    EXPECT_TRUE(original_weights.empty());
  }
  // Drop out all trees
  {
    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(1.0);
    config.set_learning_rate(1.0);

    TF_EXPECT_OK(DropoutUtils::DropOutTrees(kSeed, config, trees_not_to_drop,
                                            weights_, &dropped_trees,
                                            &original_weights));

    // No trees left
    EXPECT_EQ(kNumTrees, dropped_trees.size());
    EXPECT_EQ(kNumTrees, original_weights.size());
    EXPECT_EQ(original_weights, weights_);
  }
  // 50% probability of dropping a tree
  {
    const int32 kNumRuns = 1000;
    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(0.5);
    config.set_learning_rate(1.0);

    int32 total_num_trees = 0;
    for (int i = 0; i < kNumRuns; ++i) {
      // draw random seeds
      uint random_generator_seed =
          static_cast<uint>(Env::Default()->NowMicros());
      uint32 seed = rand_r(&random_generator_seed) % 100 + i;
      TF_EXPECT_OK(DropoutUtils::DropOutTrees(seed, config, trees_not_to_drop,
                                              weights_, &dropped_trees,
                                              &original_weights));

      // We would expect 400-600 trees left
      EXPECT_NEAR(500, kNumTrees - dropped_trees.size(), 100);
      total_num_trees += kNumTrees - dropped_trees.size();

      // Trees dropped are unique
      unordered_set<int32> ids;
      for (const auto& tree : dropped_trees) {
        ids.insert(tree);
      }
      EXPECT_EQ(ids.size(), dropped_trees.size());
    }
    EXPECT_NEAR(500, total_num_trees / kNumRuns, 5);
  }
}

TEST_F(DropoutUtilsTest, DropoutIgnoresNotToDropTest) {
  std::vector<int32> dropped_trees;
  std::vector<float> original_weights;

  // Empty do not drop set.
  {
    std::unordered_set<int32> trees_not_to_drop;

    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(1.0);
    config.set_learning_rate(1.0);

    TF_EXPECT_OK(DropoutUtils::DropOutTrees(kSeed, config, trees_not_to_drop,
                                            weights_, &dropped_trees,
                                            &original_weights));

    // No trees left
    EXPECT_EQ(kNumTrees, dropped_trees.size());
    EXPECT_EQ(kNumTrees, original_weights.size());
    EXPECT_EQ(original_weights, weights_);
  }

  // Do not drop any trees
  {
    std::unordered_set<int32> trees_not_to_drop;
    for (int i = 0; i < kNumTrees; ++i) {
      trees_not_to_drop.insert(i);
    }

    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(1.0);
    config.set_learning_rate(1.0);

    TF_EXPECT_OK(DropoutUtils::DropOutTrees(kSeed, config, trees_not_to_drop,
                                            weights_, &dropped_trees,
                                            &original_weights));

    // No trees were dropped - they all were in do not drop set.
    EXPECT_EQ(0, dropped_trees.size());
    EXPECT_EQ(0, original_weights.size());
  }
  // Do not drop some trees
  {
    std::unordered_set<int32> trees_not_to_drop;
    trees_not_to_drop.insert(0);
    trees_not_to_drop.insert(34);

    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(1.0);
    config.set_learning_rate(1.0);

    TF_EXPECT_OK(DropoutUtils::DropOutTrees(kSeed, config, trees_not_to_drop,
                                            weights_, &dropped_trees,
                                            &original_weights));

    // No trees were dropped - they all were in do not drop set.
    EXPECT_EQ(kNumTrees - 2, dropped_trees.size());
    EXPECT_EQ(kNumTrees - 2, original_weights.size());
    EXPECT_TRUE(std::find(dropped_trees.begin(), dropped_trees.end(), 0) ==
                dropped_trees.end());
    EXPECT_TRUE(std::find(dropped_trees.begin(), dropped_trees.end(), 34) ==
                dropped_trees.end());
  }
}

TEST_F(DropoutUtilsTest, DropoutSeedTest) {
  std::unordered_set<int32> trees_not_to_drop;
  // Different seeds remove different trees
  {
    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(0.5);
    config.set_learning_rate(1.0);

    std::vector<int32> dropped_trees_1;
    std::vector<float> original_weights_1;
    std::vector<int32> dropped_trees_2;
    std::vector<float> original_weights_2;

    DecisionTreeEnsembleConfig new_ensemble_1;
    DecisionTreeEnsembleConfig new_ensemble_2;

    TF_EXPECT_OK(DropoutUtils::DropOutTrees(
        kSeed + 1, config, trees_not_to_drop, weights_, &dropped_trees_1,
        &original_weights_1));
    TF_EXPECT_OK(DropoutUtils::DropOutTrees(
        kSeed + 2, config, trees_not_to_drop, weights_, &dropped_trees_2,
        &original_weights_2));

    EXPECT_FALSE(dropped_trees_1 == dropped_trees_2);
    EXPECT_FALSE(original_weights_1 == original_weights_2);
  }
  //  The same seed produces the same result
  {
    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(0.5);
    config.set_learning_rate(1.0);

    std::vector<int32> dropped_trees_1;
    std::vector<float> original_weights_1;
    std::vector<int32> dropped_trees_2;
    std::vector<float> original_weights_2;

    DecisionTreeEnsembleConfig new_ensemble_1;
    DecisionTreeEnsembleConfig new_ensemble_2;

    TF_EXPECT_OK(DropoutUtils::DropOutTrees(kSeed, config, trees_not_to_drop,
                                            weights_, &dropped_trees_1,
                                            &original_weights_1));
    TF_EXPECT_OK(DropoutUtils::DropOutTrees(kSeed, config, trees_not_to_drop,
                                            weights_, &dropped_trees_2,
                                            &original_weights_2));

    EXPECT_TRUE(dropped_trees_1 == dropped_trees_2);
    EXPECT_TRUE(original_weights_1 == original_weights_2);
  }
}

TEST_F(DropoutUtilsTest, InvalidConfigTest) {
  std::vector<int32> dropped_trees;
  std::vector<float> original_weights;
  std::unordered_set<int32> trees_not_to_drop;
  // Negative prob
  {
    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(-1.34);

    EXPECT_FALSE(DropoutUtils::DropOutTrees(kSeed, config, trees_not_to_drop,
                                            weights_, &dropped_trees,
                                            &original_weights)
                     .ok());
  }
  // Larger than 1 prob of dropping a tree.
  {
    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(1.34);

    EXPECT_FALSE(DropoutUtils::DropOutTrees(kSeed, config, trees_not_to_drop,
                                            weights_, &dropped_trees,
                                            &original_weights)
                     .ok());
  }
  // Negative probability of skipping dropout.
  {
    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(0.5);
    config.set_probability_of_skipping_dropout(-10);

    DecisionTreeEnsembleConfig new_ensemble;
    EXPECT_FALSE(DropoutUtils::DropOutTrees(kSeed, config, trees_not_to_drop,
                                            weights_, &dropped_trees,
                                            &original_weights)
                     .ok());
  }
  // Larger than 1 probability of skipping dropout.
  {
    LearningRateDropoutDrivenConfig config;
    config.set_dropout_probability(0.5);
    config.set_probability_of_skipping_dropout(1.2);

    DecisionTreeEnsembleConfig new_ensemble;
    EXPECT_FALSE(DropoutUtils::DropOutTrees(kSeed, config, trees_not_to_drop,
                                            weights_, &dropped_trees,
                                            &original_weights)
                     .ok());
  }
}
namespace {

void ExpectVecsEquiv(const std::vector<float>& vec1,
                     const std::vector<float>& vec2) {
  EXPECT_EQ(vec1.size(), vec2.size());
  for (int i = 0; i < vec1.size(); ++i) {
    EXPECT_NEAR(vec1[i], vec2[i], 1e-3);
  }
}

std::vector<float> GetWeightsByIndex(const std::vector<float>& weights,
                                     const std::vector<int>& indices) {
  std::vector<float> res;
  res.reserve(indices.size());
  for (const int index : indices) {
    res.push_back(weights[index]);
  }
  return res;
}

void MergeLastElements(const int32 last_n, std::vector<float>* weights) {
  float sum = 0.0;
  for (int i = 0; i < last_n; ++i) {
    sum += weights->back();
    weights->pop_back();
  }
  weights->push_back(sum);
}

}  // namespace

TEST_F(DropoutUtilsTest, GetTreesWeightsForAddingTreesTest) {
  // Adding trees should give the same res in any order
  {
    std::vector<float> weights = {1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<int32> dropped_1 = {0, 3};

    std::vector<int32> dropped_2 = {0};

    std::vector<float> res_1;
    std::vector<float> res_2;
    // Do one order
    {
      std::vector<float> current_weights = weights;
      std::vector<int32> num_updates =
          std::vector<int32>(current_weights.size(), 1);
      DropoutUtils::GetTreesWeightsForAddingTrees(
          dropped_1, GetWeightsByIndex(current_weights, dropped_1),
          current_weights.size(), 1, &current_weights, &num_updates);
      DropoutUtils::GetTreesWeightsForAddingTrees(
          dropped_2, GetWeightsByIndex(current_weights, dropped_2),
          current_weights.size(), 1, &current_weights, &num_updates);
      res_1 = current_weights;
    }
    // Do another order
    {
      std::vector<float> current_weights = weights;
      std::vector<int32> num_updates =
          std::vector<int32>(current_weights.size(), 1);

      DropoutUtils::GetTreesWeightsForAddingTrees(
          dropped_2, GetWeightsByIndex(current_weights, dropped_2),
          current_weights.size(), 1, &current_weights, &num_updates);
      DropoutUtils::GetTreesWeightsForAddingTrees(
          dropped_1, GetWeightsByIndex(current_weights, dropped_1),
          current_weights.size(), 1, &current_weights, &num_updates);
      res_2 = current_weights;
    }
    // The vectors are the same, but the last two elements have the same sum.
    EXPECT_EQ(res_1.size(), 7);
    EXPECT_EQ(res_2.size(), 7);

    MergeLastElements(2, &res_1);
    MergeLastElements(2, &res_2);

    EXPECT_EQ(res_1, res_2);
  }
  // Now when the weights are not all 1s
  {
    std::vector<float> weights = {1.1, 2.1, 3.1, 4.1, 5.1};
    std::vector<int32> dropped_1 = {0, 3};

    std::vector<int32> dropped_2 = {0};

    std::vector<float> res_1;
    std::vector<float> res_2;
    // Do one order
    {
      std::vector<float> current_weights = weights;
      std::vector<int32> num_updates =
          std::vector<int32>(current_weights.size(), 1);
      DropoutUtils::GetTreesWeightsForAddingTrees(
          dropped_1, GetWeightsByIndex(current_weights, dropped_1),
          current_weights.size(), 1, &current_weights, &num_updates);
      DropoutUtils::GetTreesWeightsForAddingTrees(
          dropped_2, GetWeightsByIndex(current_weights, dropped_2),
          current_weights.size(), 1, &current_weights, &num_updates);
      res_1 = current_weights;
    }
    // Do another order
    {
      std::vector<float> current_weights = weights;
      std::vector<int32> num_updates =
          std::vector<int32>(current_weights.size(), 1);
      DropoutUtils::GetTreesWeightsForAddingTrees(
          dropped_2, GetWeightsByIndex(current_weights, dropped_2),
          current_weights.size(), 1, &current_weights, &num_updates);
      DropoutUtils::GetTreesWeightsForAddingTrees(
          dropped_1, GetWeightsByIndex(current_weights, dropped_1),
          current_weights.size(), 1, &current_weights, &num_updates);
      res_2 = current_weights;
    }
    EXPECT_EQ(res_1.size(), 7);
    EXPECT_EQ(res_2.size(), 7);

    // The vectors are the same, but the last two elements have the same sum.
    MergeLastElements(2, &res_1);
    MergeLastElements(2, &res_2);

    ExpectVecsEquiv(res_1, res_2);
  }
}

TEST_F(DropoutUtilsTest, GetTreesWeightsForAddingTreesIndexTest) {
  std::vector<float> weights = {1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<int32> dropped = {0, 3};

  std::vector<float> res;
  std::vector<float> res_2;

  // The tree that is added does not yet have an entry in weights vector.
  {
    std::vector<float> current_weights = weights;
    std::vector<int32> num_updates =
        std::vector<int32>(current_weights.size(), 1);
    DropoutUtils::GetTreesWeightsForAddingTrees(
        dropped, GetWeightsByIndex(current_weights, dropped),
        current_weights.size(), 1, &current_weights, &num_updates);
    EXPECT_EQ(current_weights.size(), weights.size() + 1);
    EXPECT_EQ(num_updates.size(), weights.size() + 1);

    std::vector<int32> expected_num_updates = {2, 1, 1, 2, 1, 1};
    std::vector<float> expected_weights = {2.0 / 3, 1, 1, 2.0 / 3, 1, 2.0 / 3};
    EXPECT_EQ(expected_weights, current_weights);
    EXPECT_EQ(expected_num_updates, num_updates);
  }
  // The tree that is added has already an entry in weights and updates (batch
  // mode).
  {
    std::vector<float> current_weights = weights;
    std::vector<int32> num_updates =
        std::vector<int32>(current_weights.size(), 1);
    DropoutUtils::GetTreesWeightsForAddingTrees(
        dropped, GetWeightsByIndex(current_weights, dropped),
        current_weights.size() - 1, 1, &current_weights, &num_updates);
    EXPECT_EQ(current_weights.size(), weights.size());
    EXPECT_EQ(num_updates.size(), weights.size());

    std::vector<int32> expected_num_updates = {2, 1, 1, 2, 2};
    std::vector<float> expected_weights = {2.0 / 3, 1, 1, 2.0 / 3, 2.0 / 3};
    EXPECT_EQ(expected_weights, current_weights);
    EXPECT_EQ(expected_num_updates, num_updates);
  }
}

}  // namespace
}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
