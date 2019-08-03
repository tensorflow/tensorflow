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
#ifndef TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_TREES_DECISION_TREE_H_
#define TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_TREES_DECISION_TREE_H_

#include "tensorflow/contrib/boosted_trees/lib/utils/example.h"
#include "tensorflow/contrib/boosted_trees/proto/tree_config.pb.h"  // NOLINT

namespace tensorflow {
namespace boosted_trees {
namespace trees {

// Decision tree class to encapsulate tree traversal and mutation logic.
// This class does not hold state and is thread safe.
class DecisionTree {
 public:
  // Traverse given an instance, a sub-root and its set of features
  // and return the leaf index or -1 if the tree is empty or
  // the sub-root is invalid.
  static int Traverse(const DecisionTreeConfig& config, int32 sub_root_id,
                      const utils::Example& example);

  // Links the specified children to the parent, the children must
  // already be added to the decision tree config so this method
  // just ensures nodes are re-linked.
  static void LinkChildren(const std::vector<int32>& children,
                           TreeNode* parent_node);

  // Retrieves node children indices if any.
  static std::vector<int32> GetChildren(const TreeNode& node);
};

}  // namespace trees
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_TREES_DECISION_TREE_H_
