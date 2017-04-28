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
#include "tensorflow/contrib/boosted_trees/lib/trees/decision_tree.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/batch_features.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/parallel_for.h"

namespace tensorflow {
namespace boosted_trees {
namespace models {

namespace {
void CalculateTreesToKeep(
    const boosted_trees::trees::DecisionTreeEnsembleConfig& config,
    const std::vector<int32>& trees_to_drop, const int32 num_trees,
    const bool only_finalized, std::vector<int32>* trees_to_keep) {
  trees_to_keep->reserve(num_trees - trees_to_drop.size());

  int32 index = 0;
  // This assumes that trees_to_drop is a sorted list of tree ids.
  for (int32 tree = 0; tree < num_trees; ++tree) {
    if ((!trees_to_drop.empty() && index < trees_to_drop.size() &&
         trees_to_drop[index] == tree) ||
        (only_finalized && config.tree_metadata_size() > 0 &&
         !config.tree_metadata(tree).is_finalized())) {
      ++index;
      continue;
    }
    trees_to_keep->push_back(tree);
  }
}

void UpdatePredictions(
    const int32 index_1, const int32 index_2, const float value,
    tensorflow::TTypes<float>::Matrix* output_predictions,
    tensorflow::TTypes<float>::Matrix* additional_output_predictions) {
  (*output_predictions)(index_1, index_2) += value;

  if (additional_output_predictions != nullptr) {
    (*additional_output_predictions)(index_1, index_2) += value;
  }
}

void UpdatePredictionsBasedOnTree(
    const boosted_trees::trees::DecisionTreeEnsembleConfig& config,
    const int32 tree_idx, const boosted_trees::utils::Example& example,
    tensorflow::TTypes<float>::Matrix* output_predictions,
    tensorflow::TTypes<float>::Matrix* additional_output_predictions) {
  const boosted_trees::trees::DecisionTreeConfig& tree = config.trees(tree_idx);
  const float tree_weight = config.tree_weights(tree_idx);
  const int leaf_idx = trees::DecisionTree::Traverse(tree, 0, example);
  QCHECK(leaf_idx >= 0) << "Invalid tree: " << tree.DebugString();
  const auto& leaf_node = tree.nodes(leaf_idx);
  QCHECK(leaf_node.has_leaf())
      << "Invalid leaf node: " << leaf_node.DebugString();
  if (leaf_node.leaf().has_sparse_vector()) {
    const auto& leaf = leaf_node.leaf().sparse_vector();
    QCHECK_EQ(leaf.index_size(), leaf.value_size());
    for (size_t class_idx = 0; class_idx < leaf.index_size(); ++class_idx) {
      const float value = tree_weight * leaf.value(class_idx);

      UpdatePredictions(example.example_idx, leaf.index(class_idx), value,
                        output_predictions, additional_output_predictions);
    }
  } else {
    QCHECK(leaf_node.leaf().has_vector()) << "Unknown leaf type";
    const auto& leaf = leaf_node.leaf().vector();
    for (size_t i = 0; i < leaf.value_size(); ++i) {
      const float value = tree_weight * leaf.value(i);
      UpdatePredictions(example.example_idx, i, value, output_predictions,
                        additional_output_predictions);
    }
  }
}

}  // namespace

void MultipleAdditiveTrees::Predict(
    const boosted_trees::trees::DecisionTreeEnsembleConfig& config,
    const bool only_finalized_trees, const std::vector<int32>& trees_to_drop,
    const boosted_trees::utils::BatchFeatures& features,
    tensorflow::thread::ThreadPool* worker_threads,
    tensorflow::TTypes<float>::Matrix output_predictions,
    tensorflow::TTypes<float>::Matrix no_dropout_predictions) {
  // Zero out predictions as the model is additive.
  output_predictions.setZero();
  no_dropout_predictions.setZero();

  // Get batch size.
  const int64 batch_size = features.batch_size();
  if (batch_size <= 0) {
    return;
  }

  // Prepare the list of trees to keep.
  std::vector<int32> trees_to_keep;
  CalculateTreesToKeep(config, trees_to_drop, config.trees_size(),
                       only_finalized_trees, &trees_to_keep);

  // Lambda for doing a block of work.
  auto update_predictions = [&config, &features, &trees_to_keep, &trees_to_drop,
                             &output_predictions,
                             &no_dropout_predictions](int64 start, int64 end) {
    auto examples_iterable = features.examples_iterable(start, end);
    for (const auto& example : examples_iterable) {
      for (const int32 tree_idx : trees_to_keep) {
        UpdatePredictionsBasedOnTree(config, tree_idx, example,
                                     &output_predictions,
                                     &no_dropout_predictions);
      }

      // Now do predictions for dropped trees
      for (const int32 tree_idx : trees_to_drop) {
        UpdatePredictionsBasedOnTree(config, tree_idx, example,
                                     &no_dropout_predictions, nullptr);
      }
    }
  };

  // TODO(salehay): parallelize this for low latency in serving path where
  // batch size tends to be small but ensemble size tends to be large.
  boosted_trees::utils::ParallelFor(batch_size, worker_threads->NumThreads(),
                                    worker_threads, update_predictions);
}

}  // namespace models
}  // namespace boosted_trees
}  // namespace tensorflow
