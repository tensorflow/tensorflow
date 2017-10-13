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
#include "tensorflow/contrib/boosted_trees/lib/trees/decision_tree.h"
#include "tensorflow/core/platform/macros.h"

#include <algorithm>

namespace tensorflow {
namespace boosted_trees {
namespace trees {

constexpr int kInvalidLeaf = -1;
int DecisionTree::Traverse(const DecisionTreeConfig& config,
                           const int32 sub_root_id,
                           const utils::Example& example) {
  if (TF_PREDICT_FALSE(config.nodes_size() <= sub_root_id)) {
    return kInvalidLeaf;
  }

  // Traverse tree starting at the provided sub-root.
  int32 node_id = sub_root_id;
  while (true) {
    const auto& current_node = config.nodes(node_id);
    switch (current_node.node_case()) {
      case TreeNode::kLeaf: {
        return node_id;
      }
      case TreeNode::kDenseFloatBinarySplit: {
        const auto& split = current_node.dense_float_binary_split();
        node_id = example.dense_float_features[split.feature_column()] <=
                          split.threshold()
                      ? split.left_id()
                      : split.right_id();
        break;
      }
      case TreeNode::kSparseFloatBinarySplitDefaultLeft: {
        const auto& split =
            current_node.sparse_float_binary_split_default_left().split();
        auto sparse_feature =
            example.sparse_float_features[split.feature_column()];
        node_id = !sparse_feature.has_value() ||
                          sparse_feature.get_value() <= split.threshold()
                      ? split.left_id()
                      : split.right_id();
        break;
      }
      case TreeNode::kSparseFloatBinarySplitDefaultRight: {
        const auto& split =
            current_node.sparse_float_binary_split_default_right().split();
        auto sparse_feature =
            example.sparse_float_features[split.feature_column()];
        node_id = sparse_feature.has_value() &&
                          sparse_feature.get_value() <= split.threshold()
                      ? split.left_id()
                      : split.right_id();
        break;
      }
      case TreeNode::kCategoricalIdBinarySplit: {
        const auto& split = current_node.categorical_id_binary_split();
        const auto& features =
            example.sparse_int_features[split.feature_column()];
        node_id = features.find(split.feature_id()) != features.end()
                      ? split.left_id()
                      : split.right_id();
        break;
      }
      case TreeNode::kCategoricalIdSetMembershipBinarySplit: {
        const auto& split =
            current_node.categorical_id_set_membership_binary_split();
        // The new node_id = left_id if a feature is found, or right_id.
        node_id = split.right_id();
        for (const int64 feature_id :
             example.sparse_int_features[split.feature_column()]) {
          if (std::binary_search(split.feature_ids().begin(),
                                 split.feature_ids().end(), feature_id)) {
            node_id = split.left_id();
            break;
          }
        }
        break;
      }
      case TreeNode::NODE_NOT_SET: {
        LOG(QFATAL) << "Invalid node in tree: " << current_node.DebugString();
        break;
      }
    }
    DCHECK_NE(node_id, 0) << "Malformed tree, cycles found to root:"
                          << current_node.DebugString();
  }
}

void DecisionTree::LinkChildren(const std::vector<int32>& children,
                                TreeNode* parent_node) {
  // Decide how to link children depending on the parent node's type.
  auto children_it = children.begin();
  switch (parent_node->node_case()) {
    case TreeNode::kLeaf: {
      // Essentially no-op.
      QCHECK(children.empty()) << "A leaf node cannot have children.";
      break;
    }
    case TreeNode::kDenseFloatBinarySplit: {
      QCHECK(children.size() == 2)
          << "A binary split node must have exactly two children.";
      auto* split = parent_node->mutable_dense_float_binary_split();
      split->set_left_id(*children_it);
      split->set_right_id(*++children_it);
      break;
    }
    case TreeNode::kSparseFloatBinarySplitDefaultLeft: {
      QCHECK(children.size() == 2)
          << "A binary split node must have exactly two children.";
      auto* split =
          parent_node->mutable_sparse_float_binary_split_default_left()
              ->mutable_split();
      split->set_left_id(*children_it);
      split->set_right_id(*++children_it);
      break;
    }
    case TreeNode::kSparseFloatBinarySplitDefaultRight: {
      QCHECK(children.size() == 2)
          << "A binary split node must have exactly two children.";
      auto* split =
          parent_node->mutable_sparse_float_binary_split_default_right()
              ->mutable_split();
      split->set_left_id(*children_it);
      split->set_right_id(*++children_it);
      break;
    }
    case TreeNode::kCategoricalIdBinarySplit: {
      QCHECK(children.size() == 2)
          << "A binary split node must have exactly two children.";
      auto* split = parent_node->mutable_categorical_id_binary_split();
      split->set_left_id(*children_it);
      split->set_right_id(*++children_it);
      break;
    }
    case TreeNode::kCategoricalIdSetMembershipBinarySplit: {
      QCHECK(children.size() == 2)
          << "A binary split node must have exactly two children.";
      auto* split =
          parent_node->mutable_categorical_id_set_membership_binary_split();
      split->set_left_id(*children_it);
      split->set_right_id(*++children_it);
      break;
    }
    case TreeNode::NODE_NOT_SET: {
      LOG(QFATAL) << "A non-set node cannot have children.";
      break;
    }
  }
}

std::vector<int32> DecisionTree::GetChildren(const TreeNode& node) {
  // A node's children depend on its type.
  switch (node.node_case()) {
    case TreeNode::kLeaf: {
      return {};
    }
    case TreeNode::kDenseFloatBinarySplit: {
      const auto& split = node.dense_float_binary_split();
      return {split.left_id(), split.right_id()};
    }
    case TreeNode::kSparseFloatBinarySplitDefaultLeft: {
      const auto& split = node.sparse_float_binary_split_default_left().split();
      return {split.left_id(), split.right_id()};
    }
    case TreeNode::kSparseFloatBinarySplitDefaultRight: {
      const auto& split =
          node.sparse_float_binary_split_default_right().split();
      return {split.left_id(), split.right_id()};
    }
    case TreeNode::kCategoricalIdBinarySplit: {
      const auto& split = node.categorical_id_binary_split();
      return {split.left_id(), split.right_id()};
    }
    case TreeNode::kCategoricalIdSetMembershipBinarySplit: {
      const auto& split = node.categorical_id_set_membership_binary_split();
      return {split.left_id(), split.right_id()};
    }
    case TreeNode::NODE_NOT_SET: {
      return {};
    }
  }
}

}  // namespace trees
}  // namespace boosted_trees
}  // namespace tensorflow
