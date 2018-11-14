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

void MultipleAdditiveTrees::Predict(
    const boosted_trees::trees::DecisionTreeEnsembleConfig& config,
    const std::vector<int32>& trees_to_include,
    const boosted_trees::utils::BatchFeatures& features,
    tensorflow::thread::ThreadPool* const worker_threads,
    tensorflow::TTypes<float>::Matrix output_predictions,
    Tensor* const output_leaf_index) {
  // Zero out predictions as the model is additive.
  output_predictions.setZero();

  // Get batch size.
  const int64 batch_size = features.batch_size();
  if (batch_size <= 0) {
    return;
  }

  // Lambda for doing a block of work.
  auto update_predictions = [&config, &features, &trees_to_include,
                             &output_predictions,
                             &output_leaf_index](int64 start, int64 end) {
    auto examples_iterable = features.examples_iterable(start, end);
    Tensor dummy_tensor(DT_INT32, TensorShape({1, 1}));
    tensorflow::TTypes<int>::Matrix output_leaf_index_mat =
        output_leaf_index != nullptr ? output_leaf_index->matrix<int>()
                                     : dummy_tensor.matrix<int>();
    for (const auto& example : examples_iterable) {
      for (const int32 tree_idx : trees_to_include) {
        const boosted_trees::trees::DecisionTreeConfig& tree =
            config.trees(tree_idx);
        const float tree_weight = config.tree_weights(tree_idx);
        const int leaf_idx = trees::DecisionTree::Traverse(tree, 0, example);
        QCHECK(leaf_idx >= 0) << "Invalid tree: " << tree.DebugString();
        // Checks if output leaf tree index is required.
        if (output_leaf_index != nullptr) {
          output_leaf_index_mat(example.example_idx, tree_idx) = leaf_idx;
        }
        const auto& leaf_node = tree.nodes(leaf_idx);
        QCHECK(leaf_node.has_leaf())
            << "Invalid leaf node: " << leaf_node.DebugString();
        if (leaf_node.leaf().has_sparse_vector()) {
          const auto& leaf = leaf_node.leaf().sparse_vector();
          QCHECK_EQ(leaf.index_size(), leaf.value_size());
          for (size_t logit_dim = 0; logit_dim < leaf.index_size();
               ++logit_dim) {
            const float value = tree_weight * leaf.value(logit_dim);
            output_predictions(example.example_idx, leaf.index(logit_dim)) +=
                value;
          }
        } else {
          QCHECK(leaf_node.leaf().has_vector()) << "Unknown leaf type";
          const auto& leaf = leaf_node.leaf().vector();
          for (size_t i = 0; i < leaf.value_size(); ++i) {
            const float value = tree_weight * leaf.value(i);
            output_predictions(example.example_idx, i) += value;
          }
        }
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
