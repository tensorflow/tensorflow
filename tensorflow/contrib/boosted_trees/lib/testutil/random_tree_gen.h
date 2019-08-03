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
#ifndef TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_TESTUTIL_RANDOM_TREE_GEN_H_
#define TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_TESTUTIL_RANDOM_TREE_GEN_H_

#include <memory>

#include "tensorflow/contrib/boosted_trees/proto/tree_config.pb.h"  // NOLINT
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace boosted_trees {
namespace testutil {

// Randomly generate a balanced tree, for performance benchmarking purposes,
// that assume all features are sparse float features, for now.
class RandomTreeGen {
 public:
  RandomTreeGen(tensorflow::random::SimplePhilox* rng, int dense_feature_size,
                int sparse_feature_size);

  // Required: depth must be >= 1.
  // If one wants to generate multiple trees with the same depth, see also the
  // overload below.
  boosted_trees::trees::DecisionTreeConfig Generate(int depth);

  // Randomly generate a new tree with the same depth (and tree structure)
  // as the given tree. This is faster.
  boosted_trees::trees::DecisionTreeConfig Generate(
      const boosted_trees::trees::DecisionTreeConfig& tree);

  // Required: depth >= 1; tree_count >= 1.
  boosted_trees::trees::DecisionTreeEnsembleConfig GenerateEnsemble(
      int dept, int tree_count);

 private:
  tensorflow::random::SimplePhilox* rng_;
  const int dense_feature_size_;
  const int sparse_feature_size_;

  // Put together a deeper tree by combining two trees.
  void Combine(boosted_trees::trees::DecisionTreeConfig* root,
               boosted_trees::trees::DecisionTreeConfig* left_branch,
               boosted_trees::trees::DecisionTreeConfig* right_branch);

  // For each node in the provided tree, shift its referenced left/right index
  // by shift.
  void ShiftNodeIndex(boosted_trees::trees::DecisionTreeConfig* tree,
                      int shift);

  // Generate a sparse split in the node.
  void GenerateSplit(boosted_trees::trees::TreeNode* node, int left_id,
                     int right_id);

  TF_DISALLOW_COPY_AND_ASSIGN(RandomTreeGen);
};

}  // namespace testutil
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_TESTUTIL_RANDOM_TREE_GEN_H_
