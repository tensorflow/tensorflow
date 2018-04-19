/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_RESOURCES_H_
#define TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_RESOURCES_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/boosted_trees/boosted_trees.pb.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// A StampedResource is a resource that has a stamp token associated with it.
// Before reading from or applying updates to the resource, the stamp should
// be checked to verify that the update is not stale.
class StampedResource : public ResourceBase {
 public:
  StampedResource() : stamp_(-1) {}

  bool is_stamp_valid(int64 stamp) const { return stamp_ == stamp; }

  int64 stamp() const { return stamp_; }
  void set_stamp(int64 stamp) { stamp_ = stamp; }

 private:
  int64 stamp_;
};

// Keep a tree ensemble in memory for efficient evaluation and mutation.
class BoostedTreesEnsembleResource : public StampedResource {
 public:
  // Constructor.
  BoostedTreesEnsembleResource()
      : tree_ensemble_(
            protobuf::Arena::CreateMessage<boosted_trees::TreeEnsemble>(
                &arena_)) {}

  string DebugString() override {
    return strings::StrCat("TreeEnsemble[size=", tree_ensemble_->trees_size(),
                           "]");
  }

  bool InitFromSerialized(const string& serialized, const int64 stamp_token) {
    CHECK_EQ(stamp(), -1) << "Must Reset before Init.";
    if (ParseProtoUnlimited(tree_ensemble_, serialized)) {
      set_stamp(stamp_token);
      return true;
    }
    return false;
  }

  string SerializeAsString() const {
    return tree_ensemble_->SerializeAsString();
  }

  int32 num_trees() const { return tree_ensemble_->trees_size(); }

  // Find the next node to which the example (specified by index_in_batch)
  // traverses down from the current node indicated by tree_id and node_id.
  // Args:
  //   tree_id: the index of the tree in the ensemble.
  //   node_id: the index of the node within the tree.
  //   index_in_batch: the index of the example within the batch (relevant to
  //       the index of the row to read in each bucketized_features).
  //   bucketized_features: vector of feature Vectors.
  int32 next_node(
      const int32 tree_id, const int32 node_id, const int32 index_in_batch,
      const std::vector<TTypes<int32>::ConstVec>& bucketized_features) const;

  float node_value(const int32 tree_id, const int32 node_id) const;

  int32 GetNumLayersGrown(const int32 tree_id) const {
    DCHECK_LT(tree_id, tree_ensemble_->trees_size());
    return tree_ensemble_->tree_metadata(tree_id).num_layers_grown();
  }

  void SetNumLayersGrown(const int32 tree_id, int32 new_num_layers) const {
    DCHECK_LT(tree_id, tree_ensemble_->trees_size());
    tree_ensemble_->mutable_tree_metadata(tree_id)->set_num_layers_grown(
        new_num_layers);
  }

  void UpdateLastLayerNodesRange(const int32 node_range_start,
                                 int32 node_range_end) const {
    tree_ensemble_->mutable_growing_metadata()->set_last_layer_node_start(
        node_range_start);
    tree_ensemble_->mutable_growing_metadata()->set_last_layer_node_end(
        node_range_end);
  }

  void GetLastLayerNodesRange(int32* node_range_start,
                              int32* node_range_end) const {
    *node_range_start =
        tree_ensemble_->growing_metadata().last_layer_node_start();
    *node_range_end = tree_ensemble_->growing_metadata().last_layer_node_end();
  }

  int64 GetNumNodes(const int32 tree_id) {
    DCHECK_LT(tree_id, tree_ensemble_->trees_size());
    return tree_ensemble_->trees(tree_id).nodes_size();
  }

  void UpdateGrowingMetadata() const;

  int32 GetNumLayersAttempted() {
    return tree_ensemble_->growing_metadata().num_layers_attempted();
  }

  bool is_leaf(const int32 tree_id, const int32 node_id) const {
    DCHECK_LT(tree_id, tree_ensemble_->trees_size());
    DCHECK_LT(node_id, tree_ensemble_->trees(tree_id).nodes_size());
    const auto& node = tree_ensemble_->trees(tree_id).nodes(node_id);
    return node.node_case() == boosted_trees::Node::kLeaf;
  }

  int32 feature_id(const int32 tree_id, const int32 node_id) const {
    const auto node = tree_ensemble_->trees(tree_id).nodes(node_id);
    DCHECK_EQ(node.node_case(), boosted_trees::Node::kBucketizedSplit);
    return node.bucketized_split().feature_id();
  }

  int32 bucket_threshold(const int32 tree_id, const int32 node_id) const {
    const auto node = tree_ensemble_->trees(tree_id).nodes(node_id);
    DCHECK_EQ(node.node_case(), boosted_trees::Node::kBucketizedSplit);
    return node.bucketized_split().threshold();
  }

  int32 left_id(const int32 tree_id, const int32 node_id) const {
    const auto node = tree_ensemble_->trees(tree_id).nodes(node_id);
    DCHECK_EQ(node.node_case(), boosted_trees::Node::kBucketizedSplit);
    return node.bucketized_split().left_id();
  }

  int32 right_id(const int32 tree_id, const int32 node_id) const {
    const auto node = tree_ensemble_->trees(tree_id).nodes(node_id);
    DCHECK_EQ(node.node_case(), boosted_trees::Node::kBucketizedSplit);
    return node.bucketized_split().right_id();
  }

  // Add a tree to the ensemble and returns a new tree_id.
  int32 AddNewTree(const float weight);

  // Grows the tree by adding a split and leaves.
  void AddBucketizedSplitNode(const int32 tree_id, const int32 node_id,
                              const int32 feature_id, const int32 threshold,
                              const float gain, const float left_contrib,
                              const float right_contrib, int32* left_node_id,
                              int32* right_node_id);

  // Retrieves tree weights and returns as a vector.
  // It involves a copy, so should be called only sparingly (like once per
  // iteration, not per example).
  std::vector<float> GetTreeWeights() const {
    return {tree_ensemble_->tree_weights().begin(),
            tree_ensemble_->tree_weights().end()};
  }

  float GetTreeWeight(const int32 tree_id) const {
    return tree_ensemble_->tree_weights(tree_id);
  }

  float IsTreeFinalized(const int32 tree_id) const {
    DCHECK_LT(tree_id, tree_ensemble_->trees_size());
    return tree_ensemble_->tree_metadata(tree_id).is_finalized();
  }

  float IsTreePostPruned(const int32 tree_id) const {
    DCHECK_LT(tree_id, tree_ensemble_->trees_size());
    return tree_ensemble_->tree_metadata(tree_id)
               .post_pruned_nodes_meta_size() > 0;
  }

  void SetIsFinalized(const int32 tree_id, const bool is_finalized) {
    DCHECK_LT(tree_id, tree_ensemble_->trees_size());
    return tree_ensemble_->mutable_tree_metadata(tree_id)->set_is_finalized(
        is_finalized);
  }

  // Sets the weight of i'th tree.
  void SetTreeWeight(const int32 tree_id, const float weight) {
    DCHECK_GE(tree_id, 0);
    DCHECK_LT(tree_id, num_trees());
    tree_ensemble_->set_tree_weights(tree_id, weight);
  }

  // Resets the resource and frees the protos in arena.
  // Caller needs to hold the mutex lock while calling this.
  virtual void Reset();

  void PostPruneTree(const int32 current_tree);

  // For a given node, returns the id in a pruned tree, as well as correction
  // to the cached prediction that should be applied. If tree was not
  // post-pruned, current_node_id will be equal to initial_node_id and logit
  // update will be equal to zero.
  void GetPostPruneCorrection(const int32 tree_id, const int32 initial_node_id,
                              int32* current_node_id,
                              float* logit_update) const;
  mutex* get_mutex() { return &mu_; }

 private:
  // Helper method to check whether a node is a terminal node in that it
  // only has leaf nodes as children.
  bool IsTerminalSplitNode(const int32 tree_id, const int32 node_id) const;

  // For each pruned node, finds the leaf where it finally ended up and
  // calculates the total update from that pruned node prediction.
  void CalculateParentAndLogitUpdate(
      const int32 start_node_id,
      const std::vector<std::pair<int32, float>>& nodes_change,
      int32* parent_id, float* change) const;

  // Helper method to collect the information to be used to prune some nodes in
  // the tree.
  void RecursivelyDoPostPrunePreparation(
      const int32 tree_id, const int32 node_id,
      std::vector<int32>* nodes_to_delete,
      std::vector<std::pair<int32, float>>* nodes_meta);

 protected:
  protobuf::Arena arena_;
  mutex mu_;
  boosted_trees::TreeEnsemble* tree_ensemble_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_RESOURCES_H_
