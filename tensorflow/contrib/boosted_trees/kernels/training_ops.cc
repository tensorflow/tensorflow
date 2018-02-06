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
#include "tensorflow/contrib/boosted_trees/proto/learner.pb.h"
#include "tensorflow/contrib/boosted_trees/proto/split_info.pb.h"
#include "tensorflow/contrib/boosted_trees/resources/decision_tree_ensemble_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
using tensorflow::boosted_trees::learner::LearningRateDropoutDrivenConfig;

namespace boosted_trees {

namespace {

using boosted_trees::learner::LearningRateConfig;
using boosted_trees::trees::Leaf;
using boosted_trees::trees::TreeNode;
using boosted_trees::trees::TreeNodeMetadata;
using boosted_trees::utils::DropoutUtils;

// SplitCandidate holds the split candidate node along with the stats.
struct SplitCandidate {
  // Id of handler that generated the split candidate.
  int64 handler_id;

  // Split gain.
  float gain;

  // Split info.
  learner::SplitInfo split_info;
};

// Checks that the leaf is not empty.
bool IsLeafWellFormed(const Leaf& leaf) {
  return leaf.has_sparse_vector() || leaf.has_vector();
}

// Helper method to update the best split per partition given
// a current candidate.
void UpdateBestSplit(
    const boosted_trees::learner::LearnerConfig& learner_config,
    int32 partition_id, SplitCandidate* split,
    std::map<int32, SplitCandidate>* best_splits) {
  // Don't consider nodeless splits.
  if (TF_PREDICT_FALSE(split->split_info.split_node().node_case() ==
                       TreeNode::NODE_NOT_SET)) {
    return;
  }

  // Don't consider negative splits if we're pre-pruning the tree.
  // Note that zero-gain splits are acceptable as they're mostly doing as well
  // as what bias centering in that partition would do.
  if (learner_config.pruning_mode() ==
          boosted_trees::learner::LearnerConfig::PRE_PRUNE &&
      split->gain < 0) {
    return;
  }

  // If the current node is pure, one of the leafs will be empty, so the split
  // is meaningless and we should not split.
  if (!(IsLeafWellFormed(split->split_info.right_child()) &&
        IsLeafWellFormed(split->split_info.left_child()))) {
    VLOG(1) << "Split does not actually split anything";
    return;
  }

  // Take the split if we don't have a candidate yet.
  auto best_split_it = best_splits->find(partition_id);
  if (best_split_it == best_splits->end()) {
    best_splits->insert(std::make_pair(partition_id, std::move(*split)));
    return;
  }

  // Determine if best split so far needs to be replaced.
  SplitCandidate& best_split = best_split_it->second;
  if (TF_PREDICT_FALSE(split->gain == best_split.gain)) {
    // Tie break on node case preferring simpler tree node types.
    VLOG(2) << "Attempting to tie break with smaller node case. "
            << "(current split: " << split->split_info.split_node().node_case()
            << ", best split: "
            << best_split.split_info.split_node().node_case() << ")";
    if (split->split_info.split_node().node_case() <
        best_split.split_info.split_node().node_case()) {
      best_split = std::move(*split);
    } else if (split->split_info.split_node().node_case() ==
               best_split.split_info.split_node().node_case()) {
      // Tie break on handler Id.
      VLOG(2) << "Tie breaking with higher handler Id. "
              << "(current split: " << split->handler_id
              << ", best split: " << best_split.handler_id << ")";
      if (split->handler_id > best_split.handler_id) {
        best_split = std::move(*split);
      }
    }
  } else if (split->gain > best_split.gain) {
    best_split = std::move(*split);
  }
}

// Helper method to check whether a node is a terminal node in that it
// only has leaf nodes as children.
bool IsTerminalSplitNode(const size_t node_id,
                         const std::vector<int32>& children,
                         const std::vector<TreeNode>& nodes) {
  for (const int32 child_id : children) {
    const auto& child_node = nodes[child_id];
    CHECK(child_node.node_case() != TreeNode::NODE_NOT_SET);
    if (child_node.node_case() != TreeNode::kLeaf) {
      return false;
    }
  }
  return true;
}

// Helper method to recursively prune the tree in a depth-first fashion.
void RecursivePruneTree(const size_t node_id, std::vector<TreeNode>* nodes) {
  // Base case when we reach a leaf.
  TreeNode& tree_node = (*nodes)[node_id];
  CHECK(tree_node.node_case() != TreeNode::NODE_NOT_SET);
  if (tree_node.node_case() == TreeNode::kLeaf) {
    return;
  }

  // Traverse node children first and recursively prune their sub-trees.
  const std::vector<int32> children =
      boosted_trees::trees::DecisionTree::GetChildren(tree_node);
  for (const int32 child_id : children) {
    RecursivePruneTree(child_id, nodes);
  }

  // Two conditions must be satisfied to prune the node:
  // 1- The split gain is negative.
  // 2- After depth-first pruning, the node only has leaf children.
  TreeNodeMetadata* node_metadata = tree_node.mutable_node_metadata();
  if (node_metadata->gain() < 0 &&
      IsTerminalSplitNode(node_id, children, (*nodes))) {
    // Clear node children.
    for (const int32 child_id : children) {
      auto& child_node = (*nodes)[child_id];
      child_node.Clear();
    }

    // Change node back into leaf.
    (*tree_node.mutable_leaf()) = *node_metadata->mutable_original_leaf();

    // Clear gain for leaf node.
    tree_node.clear_node_metadata();
  } else {
    // Clear original leaf as it's no longer needed for back-track pruning.
    node_metadata->clear_original_leaf();
  }
}

}  // namespace

class CenterTreeEnsembleBiasOp : public OpKernel {
 public:
  explicit CenterTreeEnsembleBiasOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    // Read learner config.
    string serialized_learner_config;
    OP_REQUIRES_OK(context, context->GetAttr("learner_config",
                                             &serialized_learner_config));
    OP_REQUIRES(context,
                learner_config_.ParseFromString(serialized_learner_config),
                errors::InvalidArgument("Unable to parse learner config."));

    // Read centering epsilon.
    OP_REQUIRES_OK(context,
                   context->GetAttr("centering_epsilon", &centering_epsilon_));
  }

  void Compute(OpKernelContext* const context) override {
    // Get decision tree ensemble.
    boosted_trees::models::DecisionTreeEnsembleResource* ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    core::ScopedUnref unref_me(ensemble_resource);
    mutex_lock l(*ensemble_resource->get_mutex());

    // Get the stamp token.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input("stamp_token", &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    // Only the Chief should run this Op and it is guaranteed to be in
    // a consistent state so the stamps must always match.
    CHECK(ensemble_resource->is_stamp_valid(stamp_token));

    // Get the next stamp token.
    const Tensor* next_stamp_token_t;
    OP_REQUIRES_OK(context,
                   context->input("next_stamp_token", &next_stamp_token_t));
    int64 next_stamp_token = next_stamp_token_t->scalar<int64>()();
    CHECK(stamp_token != next_stamp_token);

    // Update the ensemble stamp.
    ensemble_resource->set_stamp(next_stamp_token);

    // Get the delta updates.
    const Tensor* delta_updates_t;
    OP_REQUIRES_OK(context, context->input("delta_updates", &delta_updates_t));
    auto delta_updates = delta_updates_t->vec<float>();
    const int64 logits_dimension = delta_updates_t->dim_size(0);

    // Get the bias.
    boosted_trees::trees::Leaf* const bias =
        RetrieveBias(ensemble_resource, logits_dimension);
    CHECK(bias->has_vector());

    // Update the bias.
    float total_delta = 0;
    auto* bias_vec = bias->mutable_vector();
    for (size_t idx = 0; idx < bias->vector().value_size(); ++idx) {
      float delta = delta_updates(idx);
      bias_vec->set_value(idx, bias_vec->value(idx) + delta);
      total_delta += std::abs(delta);
    }

    // Make a centering continuation decision based on current update.
    bool continue_centering = total_delta > centering_epsilon_;
    if (continue_centering) {
      VLOG(1) << "Continuing to center bias, delta=" << total_delta;
    } else {
      VLOG(1) << "Done centering bias, delta=" << total_delta;
      ensemble_resource->LastTreeMetadata()->set_is_finalized(true);
    }
    Tensor* continue_centering_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("continue_centering", TensorShape({}),
                                          &continue_centering_t));
    continue_centering_t->scalar<bool>()() = continue_centering;
  }

 private:
  // Helper method to retrieve the bias from the tree ensemble.
  boosted_trees::trees::Leaf* RetrieveBias(
      boosted_trees::models::DecisionTreeEnsembleResource* ensemble_resource,
      int64 logits_dimension) {
    const int32 num_trees = ensemble_resource->num_trees();
    if (num_trees <= 0) {
      // Add a new bias leaf.
      ensemble_resource->IncrementAttempts();
      boosted_trees::trees::DecisionTreeConfig* const tree_config =
          ensemble_resource->AddNewTree(1.0);
      auto* const leaf = tree_config->add_nodes()->mutable_leaf();
      for (size_t idx = 0; idx < logits_dimension; ++idx) {
        leaf->mutable_vector()->add_value(0.0);
      }
      return leaf;
    } else if (num_trees == 1) {
      // Confirms that the only tree is a bias and returns its leaf.
      boosted_trees::trees::DecisionTreeConfig* const tree_config =
          ensemble_resource->LastTree();
      CHECK_EQ(tree_config->nodes_size(), 1);
      CHECK_EQ(tree_config->nodes(0).node_case(), TreeNode::kLeaf);
      return tree_config->mutable_nodes(0)->mutable_leaf();
    } else {
      LOG(FATAL) << "Unable to center bias on an already grown ensemble";
    }
  }

  boosted_trees::learner::LearnerConfig learner_config_;
  float centering_epsilon_;
};

REGISTER_KERNEL_BUILDER(Name("CenterTreeEnsembleBias").Device(DEVICE_CPU),
                        CenterTreeEnsembleBiasOp);

class GrowTreeEnsembleOp : public OpKernel {
 public:
  explicit GrowTreeEnsembleOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    // Read number of handlers, note that this is the static number of
    // all handlers but any subset of these handlers may be active at a time.
    OP_REQUIRES_OK(context, context->GetAttr("num_handlers", &num_handlers_));

    OP_REQUIRES_OK(context, context->GetAttr("center_bias", &center_bias_));

    // Read learner config.
    string serialized_learner_config;
    OP_REQUIRES_OK(context, context->GetAttr("learner_config",
                                             &serialized_learner_config));
    OP_REQUIRES(context,
                learner_config_.ParseFromString(serialized_learner_config),
                errors::InvalidArgument("Unable to parse learner config."));

    // Determine whether dropout was used when building this tree.
    if (learner_config_.has_learning_rate_tuner() &&
        learner_config_.learning_rate_tuner().tuner_case() ==
            LearningRateConfig::kDropout) {
      dropout_config_ = learner_config_.learning_rate_tuner().dropout();
      dropout_was_applied_ = true;
    } else {
      dropout_was_applied_ = false;
    }
  }

  void Compute(OpKernelContext* const context) override {
    // Get decision tree ensemble.
    boosted_trees::models::DecisionTreeEnsembleResource* ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    core::ScopedUnref unref_me(ensemble_resource);
    mutex_lock l(*ensemble_resource->get_mutex());

    // Get the stamp token.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input("stamp_token", &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    // Only the Chief should run this Op and it is guaranteed to be in
    // a consistent state so the stamps must always match.
    CHECK(ensemble_resource->is_stamp_valid(stamp_token));

    // Get the next stamp token.
    const Tensor* next_stamp_token_t;
    OP_REQUIRES_OK(context,
                   context->input("next_stamp_token", &next_stamp_token_t));
    int64 next_stamp_token = next_stamp_token_t->scalar<int64>()();
    CHECK(stamp_token != next_stamp_token);

    // Update the ensemble stamp regardless of whether a layer
    // or tree is actually grown.
    ensemble_resource->set_stamp(next_stamp_token);

    // Read the learning_rate.
    const Tensor* learning_rate_t;
    OP_REQUIRES_OK(context, context->input("learning_rate", &learning_rate_t));
    float learning_rate = learning_rate_t->scalar<float>()();

    // Read seed that was used for dropout.
    const Tensor* seed_t;
    OP_REQUIRES_OK(context, context->input("dropout_seed", &seed_t));
    // Cast seed to uint64.
    const uint64 dropout_seed = seed_t->scalar<int64>()();

    // Read partition Ids, gains and split candidates.
    OpInputList partition_ids_list;
    OpInputList gains_list;
    OpInputList splits_list;
    OP_REQUIRES_OK(context,
                   context->input_list("partition_ids", &partition_ids_list));
    OP_REQUIRES_OK(context, context->input_list("gains", &gains_list));
    OP_REQUIRES_OK(context, context->input_list("splits", &splits_list));

    // Increment attempt stats.
    ensemble_resource->IncrementAttempts();

    // In case we want to do feature selection and we have reached the limit,
    // build a list of handlers used so far to avoid adding new features.
    std::vector<int64> allowed_handlers;
    if (learner_config_.constraints().max_number_of_unique_feature_columns() >
        0) {
      allowed_handlers = ensemble_resource->GetUsedHandlers();
      // TODO(soroush): We can disable handlers that are not going to be used to
      // avoid unnecessary computations.
      if (allowed_handlers.size() <
          learner_config_.constraints()
              .max_number_of_unique_feature_columns()) {
        // We have not reached the limit yet. Empty the list of allow features
        // which means we can keep adding new features.
        allowed_handlers.clear();
      }
    }

    // Find best splits for each active partition.
    std::map<int32, SplitCandidate> best_splits;
    FindBestSplitsPerPartition(context, allowed_handlers, partition_ids_list,
                               gains_list, splits_list, &best_splits);

    // No-op if no new splits can be considered.
    if (best_splits.empty()) {
      LOG(WARNING) << "Not growing tree ensemble as no good splits were found.";
      return;
    }

    // Update and retrieve the growable tree.
    // If the tree is fully built and dropout was applied, it also adjusts the
    // weights of dropped and the last tree.
    boosted_trees::trees::DecisionTreeConfig* const tree_config =
        UpdateAndRetrieveGrowableTree(ensemble_resource, learning_rate,
                                      dropout_seed);

    // Split tree nodes.
    for (auto& split_entry : best_splits) {
      SplitTreeNode(split_entry.first, &split_entry.second, tree_config,
                    ensemble_resource);
    }

    // Post-prune finalized tree if needed.
    if (learner_config_.pruning_mode() ==
            boosted_trees::learner::LearnerConfig::POST_PRUNE &&
        ensemble_resource->LastTreeMetadata()->is_finalized()) {
      VLOG(2) << "Post-pruning finalized tree.";
      PruneTree(tree_config);

      // If after post-pruning the whole tree has no gain, remove the tree
      // altogether from the ensemble.
      if (tree_config->nodes_size() <= 0) {
        ensemble_resource->RemoveLastTree();
      }
    }
  }

 private:
  // Helper method which effectively does a reduce over all split candidates
  // and finds the best split for each partition.
  void FindBestSplitsPerPartition(
      OpKernelContext* const context,
      const std::vector<int64>& allowed_handlers,  // Empty means all handlers.
      const OpInputList& partition_ids_list, const OpInputList& gains_list,
      const OpInputList& splits_list,
      std::map<int32, SplitCandidate>* best_splits) {
    // Find best split per partition going through every feature candidate.
    // TODO(salehay): Is this worth parallelizing?
    for (int64 handler_id = 0; handler_id < num_handlers_; ++handler_id) {
      if (!allowed_handlers.empty()) {
        if (!std::binary_search(allowed_handlers.begin(),
                                allowed_handlers.end(), handler_id)) {
          continue;
        }
      }
      const auto& partition_ids = partition_ids_list[handler_id].vec<int32>();
      const auto& gains = gains_list[handler_id].vec<float>();
      const auto& splits = splits_list[handler_id].vec<string>();
      OP_REQUIRES(context, partition_ids.size() == gains.size(),
                  errors::InvalidArgument(
                      "Inconsistent partition Ids and gains tensors: ",
                      partition_ids.size(), " != ", gains.size()));
      OP_REQUIRES(context, partition_ids.size() == splits.size(),
                  errors::InvalidArgument(
                      "Inconsistent partition Ids and splits tensors: ",
                      partition_ids.size(), " != ", splits.size()));
      for (size_t candidate_idx = 0; candidate_idx < splits.size();
           ++candidate_idx) {
        // Get current split candidate.
        const auto& partition_id = partition_ids(candidate_idx);
        const auto& gain = gains(candidate_idx);
        const auto& serialized_split = splits(candidate_idx);
        SplitCandidate split;
        split.handler_id = handler_id;
        split.gain = gain;
        OP_REQUIRES(context, split.split_info.ParseFromString(serialized_split),
                    errors::InvalidArgument("Unable to parse split info."));

        // Update best split for partition based on the current candidate.
        UpdateBestSplit(learner_config_, partition_id, &split, best_splits);
      }
    }
  }

  void UpdateTreeWeightsIfDropout(
      boosted_trees::models::DecisionTreeEnsembleResource* const
          ensemble_resource,
      const uint64 dropout_seed) {
    // It is possible that the tree was built with dropout. If it is the case,
    // we need to adjust the tree weight, or bail out.
    if (!dropout_was_applied_ ||
        !ensemble_resource->LastTreeMetadata()->is_finalized()) {
      return;
    }
    const int32 num_trees = ensemble_resource->num_trees();

    // Based on seed, figure out what trees were dropped before.
    std::unordered_set<int32> trees_not_to_drop;
    if (center_bias_) {
      trees_not_to_drop.insert(0);
    }
    // Last tree is the current tree that is built.
    const int32 current_tree = num_trees - 1;
    trees_not_to_drop.insert(current_tree);

    // Since only chief builds the trees, we are sure that the other tree
    // weights didn't change.
    std::vector<float> weights = ensemble_resource->GetTreeWeights();
    std::vector<int32> dropped_trees;
    std::vector<float> dropped_trees_weights;
    const auto dropout_status = DropoutUtils::DropOutTrees(
        dropout_seed, dropout_config_, trees_not_to_drop, weights,
        &dropped_trees, &dropped_trees_weights);
    CHECK(dropout_status.ok())
        << "Can't figure out what trees were dropped out before, error is "
        << dropout_status.error_message();

    // Now we have dropped trees, update their weights and the current tree
    // weight.
    if (!dropped_trees.empty()) {
      std::vector<int32> increment_num_updates(num_trees, 0);
      DropoutUtils::GetTreesWeightsForAddingTrees(
          dropped_trees, dropped_trees_weights, current_tree,
          1 /* only 1 tree was added */, &weights, &increment_num_updates);

      // Update the weights and num of updates for trees.
      for (int i = 0; i < num_trees; ++i) {
        ensemble_resource->SetTreeWeight(i, weights[i],
                                         increment_num_updates[i]);
      }
    }
  }

  // Helper method to update the growable tree which is by definition the last
  // tree in the ensemble.
  boosted_trees::trees::DecisionTreeConfig* UpdateAndRetrieveGrowableTree(
      boosted_trees::models::DecisionTreeEnsembleResource* const
          ensemble_resource,
      const float learning_rate, const uint64 dropout_seed) {
    const auto num_trees = ensemble_resource->num_trees();
    if (num_trees <= 0 ||
        ensemble_resource->LastTreeMetadata()->is_finalized()) {
      // Create a new tree with a no-op leaf.
      boosted_trees::trees::DecisionTreeConfig* const tree_config =
          ensemble_resource->AddNewTree(learning_rate);
      VLOG(1) << "Adding layer #0 to tree #" << num_trees << " of ensemble of "
              << num_trees + 1 << " trees.";
      tree_config->add_nodes()->mutable_leaf();
      boosted_trees::trees::DecisionTreeMetadata* const tree_metadata =
          ensemble_resource->LastTreeMetadata();
      tree_metadata->set_is_finalized(
          learner_config_.constraints().max_tree_depth() <= 1);
      tree_metadata->set_num_tree_weight_updates(1);
    } else {
      // The growable tree is by definition the last tree in the ensemble.
      boosted_trees::trees::DecisionTreeMetadata* const tree_metadata =
          ensemble_resource->LastTreeMetadata();
      const auto new_num_layers = tree_metadata->num_layers_grown() + 1;
      VLOG(1) << "Adding layer #" << new_num_layers - 1 << " to tree #"
              << num_trees - 1 << " of ensemble of " << num_trees << " trees.";
      // Update growable tree metadata.
      tree_metadata->set_num_layers_grown(new_num_layers);
      tree_metadata->set_is_finalized(
          new_num_layers >= learner_config_.constraints().max_tree_depth());
    }
    UpdateTreeWeightsIfDropout(ensemble_resource, dropout_seed);
    return ensemble_resource->LastTree();
  }

  // Helper method to merge leaf weights as the tree is being grown.
  boosted_trees::trees::Leaf* MergeLeafWeights(
      const boosted_trees::trees::Leaf& source,
      boosted_trees::trees::Leaf* dest) {
    // Resolve leaf merging method based on how the trees are being grown.
    if (learner_config_.growing_mode() ==
        boosted_trees::learner::LearnerConfig::WHOLE_TREE) {
      // No merging occurs when building a whole tree at a time.
      return dest;
    }

    // Handle leaf merging based on type.
    switch (source.leaf_case()) {
      case boosted_trees::trees::Leaf::kVector: {
        // No-op if source is empty
        const auto& src_vec = source.vector();
        if (src_vec.value_size() == 0) {
          break;
        }
        CHECK(source.leaf_case() == dest->leaf_case());

        // Dense add leaf vectors.
        auto* dst_vec = dest->mutable_vector();
        CHECK(src_vec.value_size() == dst_vec->value_size());
        for (size_t idx = 0; idx < source.vector().value_size(); ++idx) {
          (*dst_vec->mutable_value()->Mutable(idx)) += src_vec.value(idx);
        }
        break;
      }
      case boosted_trees::trees::Leaf::kSparseVector: {
        // No-op if source is empty
        const auto& src_vec = source.sparse_vector();
        CHECK(src_vec.value_size() == src_vec.index_size());
        if (src_vec.value_size() == 0) {
          break;
        }
        CHECK(source.leaf_case() == dest->leaf_case());

        // Get mapping of dimension to value for destination.
        std::unordered_map<int32, float> dst_map;
        auto* dst_vec = dest->mutable_sparse_vector();
        CHECK(dst_vec->value_size() == dst_vec->index_size());
        dst_map.reserve(dst_vec->value_size());
        for (size_t idx = 0; idx < dst_vec->value_size(); ++idx) {
          dst_map[dst_vec->index(idx)] = dst_vec->value(idx);
        }
        // Sparse add source vector to destination vector.
        for (size_t idx = 0; idx < src_vec.value_size(); ++idx) {
          dst_map[src_vec.index(idx)] += src_vec.value(idx);
        }
        // Rebuild merged destination leaf.
        dst_vec->clear_index();
        dst_vec->clear_value();
        for (const auto& entry : dst_map) {
          dst_vec->add_index(entry.first);
          dst_vec->add_value(entry.second);
        }
        break;
      }
      case boosted_trees::trees::Leaf::LEAF_NOT_SET: {
        // No-op as there is nothing to merge.
        break;
      }
    }
    return dest;
  }

  // Helper method to split a tree node and append its respective
  // leaf children given the split candidate.
  void SplitTreeNode(
      const int32 node_id, SplitCandidate* split,
      boosted_trees::trees::DecisionTreeConfig* tree_config,
      boosted_trees::models::DecisionTreeEnsembleResource* ensemble_resource) {
    // No-op if we have no real node.
    CHECK(node_id < tree_config->nodes_size())
        << "Invalid node " << node_id << " to split.";
    // Ensure new split node is valid.
    CHECK(split->split_info.split_node().node_case() != TreeNode::NODE_NOT_SET);
    CHECK(tree_config->nodes(node_id).node_case() == TreeNode::kLeaf)
        << "Unexpected node type to split "
        << tree_config->nodes(node_id).node_case() << " for node_id " << node_id
        << ". Tree config: " << tree_config->DebugString();

    // Add left leaf.
    int32 left_id = tree_config->nodes_size();
    (*tree_config->add_nodes()->mutable_leaf()) =
        *MergeLeafWeights(tree_config->nodes(node_id).leaf(),
                          split->split_info.mutable_left_child());

    // Add right leaf.
    int32 right_id = tree_config->nodes_size();
    (*tree_config->add_nodes()->mutable_leaf()) =
        *MergeLeafWeights(tree_config->nodes(node_id).leaf(),
                          split->split_info.mutable_right_child());

    // Link children and add them as new roots.
    boosted_trees::trees::DecisionTree::LinkChildren(
        {left_id, right_id}, split->split_info.mutable_split_node());

    // Add split gain and, if needed, original leaf to node metadata.
    TreeNodeMetadata* node_metadata =
        split->split_info.mutable_split_node()->mutable_node_metadata();
    node_metadata->set_gain(split->gain);
    if (learner_config_.pruning_mode() ==
        boosted_trees::learner::LearnerConfig::POST_PRUNE) {
      (*node_metadata->mutable_original_leaf()) =
          *tree_config->mutable_nodes(node_id)->mutable_leaf();
    }

    // Replace node in tree.
    (*tree_config->mutable_nodes(node_id)) =
        *split->split_info.mutable_split_node();
    if (learner_config_.constraints().max_number_of_unique_feature_columns()) {
      ensemble_resource->MaybeAddUsedHandler(split->handler_id);
    }
  }

  void PruneTree(boosted_trees::trees::DecisionTreeConfig* tree_config) {
    // No-op if tree is empty.
    if (tree_config->nodes_size() <= 0) {
      return;
    }

    // Copy nodes to temp vector and clear original tree.
    std::vector<TreeNode> tree_nodes;
    tree_nodes.reserve(tree_config->nodes_size());
    for (auto& node : (*tree_config->mutable_nodes())) {
      tree_nodes.push_back(node);
      node.Clear();
    }
    tree_config->clear_nodes();

    // Prune the tree recursively starting from the root.
    RecursivePruneTree(0, &tree_nodes);

    // Rebuild compacted tree.
    (*tree_config->add_nodes()) = tree_nodes[0];
    std::unordered_map<size_t, size_t> nodes_map;
    nodes_map[0] = 0;
    for (size_t node_idx = 0; node_idx < tree_nodes.size(); ++node_idx) {
      // Skip pruned nodes.
      auto& original_node = tree_nodes[node_idx];
      if (original_node.node_case() == TreeNode::NODE_NOT_SET) {
        continue;
      }

      // Find node mapped in tree ensemble.
      auto mapped_node_it = nodes_map.find(node_idx);
      CHECK(mapped_node_it != nodes_map.end());
      auto& mapped_node = (*tree_config->mutable_nodes(mapped_node_it->second));

      // Get node children
      auto children =
          boosted_trees::trees::DecisionTree::GetChildren(original_node);
      for (int32& child_idx : children) {
        auto new_idx = tree_config->nodes_size();
        (*tree_config->add_nodes()) = tree_nodes[child_idx];
        nodes_map[child_idx] = new_idx;
        child_idx = new_idx;
      }
      boosted_trees::trees::DecisionTree::LinkChildren(children, &mapped_node);
    }

    // Check if there are any nodes with gain left.
    if (tree_config->nodes_size() == 1 &&
        tree_config->nodes(0).node_metadata().gain() <= 0) {
      // The whole tree should be pruned.
      VLOG(2) << "No useful nodes left after post-pruning tree.";
      tree_config->clear_nodes();
    }
  }

 private:
  boosted_trees::learner::LearnerConfig learner_config_;
  int64 num_handlers_;
  LearningRateDropoutDrivenConfig dropout_config_;
  bool dropout_was_applied_;
  bool center_bias_;
};

REGISTER_KERNEL_BUILDER(Name("GrowTreeEnsemble").Device(DEVICE_CPU),
                        GrowTreeEnsembleOp);

class TreeEnsembleStatsOp : public OpKernel {
 public:
  explicit TreeEnsembleStatsOp(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* const context) override {
    // Get decision tree ensemble.
    boosted_trees::models::DecisionTreeEnsembleResource* ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    core::ScopedUnref unref_me(ensemble_resource);
    tf_shared_lock l(*ensemble_resource->get_mutex());

    // Get the stamp token.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input("stamp_token", &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    // Only the Chief should run this Op and it is guaranteed to be in
    // a consistent state so the stamps must always match.
    CHECK(ensemble_resource->is_stamp_valid(stamp_token));
    const boosted_trees::trees::DecisionTreeEnsembleConfig& ensemble_config =
        ensemble_resource->decision_tree_ensemble();

    // Set tree stats.
    Tensor* num_trees_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "num_trees", TensorShape({}), &num_trees_t));
    Tensor* active_tree_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("active_tree", TensorShape({}),
                                            &active_tree_t));
    Tensor* attempted_tree_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("attempted_trees", TensorShape({}),
                                            &attempted_tree_t));

    const int num_trees = ensemble_resource->num_trees();
    active_tree_t->scalar<int64>()() = num_trees;
    num_trees_t->scalar<int64>()() =
        (num_trees <= 0 ||
         ensemble_resource->LastTreeMetadata()->is_finalized())
            ? num_trees
            : num_trees - 1;
    attempted_tree_t->scalar<int64>()() =
        ensemble_config.growing_metadata().num_trees_attempted();

    // Set layer stats.
    Tensor* num_layers_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "num_layers", TensorShape({}), &num_layers_t));
    Tensor* active_layer_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("active_layer", TensorShape({}),
                                            &active_layer_t));
    Tensor* attempted_layers_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("attempted_layers", TensorShape({}),
                                            &attempted_layers_t));

    int64 num_layers = 0;
    for (const auto& tree_metadata : ensemble_config.tree_metadata()) {
      num_layers += tree_metadata.num_layers_grown();
    }
    num_layers_t->scalar<int64>()() = num_layers;
    int tree_metadata_size = ensemble_config.tree_metadata_size();
    active_layer_t->scalar<int64>()() =
        tree_metadata_size > 0
            ? ensemble_config.tree_metadata(tree_metadata_size - 1)
                  .num_layers_grown()
            : 0;
    attempted_layers_t->scalar<int64>()() =
        ensemble_config.growing_metadata().num_layers_attempted();
  }
};

REGISTER_KERNEL_BUILDER(Name("TreeEnsembleStats").Device(DEVICE_CPU),
                        TreeEnsembleStatsOp);

}  // namespace boosted_trees
}  // namespace tensorflow
