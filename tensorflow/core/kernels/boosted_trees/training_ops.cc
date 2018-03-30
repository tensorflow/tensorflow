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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/boosted_trees/resources.h"

namespace tensorflow {

namespace {
constexpr float kLayerByLayerTreeWeight = 1.0;

// TODO(nponomareva, youngheek): consider using vector.
struct SplitCandidate {
  SplitCandidate() {}

  // Index in the list of the feature ids.
  int64 feature_idx;

  // Index in the tensor of node_ids for the feature with idx feature_idx.
  int64 candidate_idx;

  float gain;
};

enum PruningMode { kNoPruning = 0, kPrePruning = 1, kPostPruning = 2 };

}  // namespace

class BoostedTreesUpdateEnsembleOp : public OpKernel {
 public:
  explicit BoostedTreesUpdateEnsembleOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("max_depth", &max_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("learning_rate", &learning_rate_));
    OP_REQUIRES_OK(context, context->GetAttr("num_features", &num_features_));

    int32 pruning_index;
    OP_REQUIRES_OK(context, context->GetAttr("pruning_mode", &pruning_index));
    pruning_mode_ = static_cast<PruningMode>(pruning_index);
  }

  void Compute(OpKernelContext* const context) override {
    // Get decision tree ensemble.
    BoostedTreesEnsembleResource* ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    core::ScopedUnref unref_me(ensemble_resource);
    mutex_lock l(*ensemble_resource->get_mutex());
    // Increase the ensemble stamp.
    ensemble_resource->set_stamp(ensemble_resource->stamp() + 1);

    // Read node ids, gains, thresholds and node contribs.
    OpInputList node_ids_list;
    OpInputList gains_list;
    OpInputList thresholds_list;
    OpInputList left_node_contribs;
    OpInputList right_node_contribs;
    OP_REQUIRES_OK(context, context->input_list("node_ids", &node_ids_list));
    OP_REQUIRES_OK(context, context->input_list("gains", &gains_list));
    OP_REQUIRES_OK(context,
                   context->input_list("thresholds", &thresholds_list));
    OP_REQUIRES_OK(context, context->input_list("left_node_contribs",
                                                &left_node_contribs));
    OP_REQUIRES_OK(context, context->input_list("right_node_contribs",
                                                &right_node_contribs));

    const Tensor* feature_ids_t;
    OP_REQUIRES_OK(context, context->input("feature_ids", &feature_ids_t));

    auto feature_ids = feature_ids_t->vec<int32>();

    // Find best splits for each active node.
    std::map<int32, SplitCandidate> best_splits;
    FindBestSplitsPerNode(context, node_ids_list, gains_list, &best_splits);

    int32 current_tree =
        UpdateGlobalAttemptsAndRetrieveGrowableTree(ensemble_resource);

    // No-op if no new splits can be considered.
    if (best_splits.empty()) {
      LOG(WARNING) << "Not growing tree ensemble as no good splits were found.";
      return;
    }

    const int32 new_num_layers =
        ensemble_resource->GetNumLayersGrown(current_tree) + 1;
    VLOG(1) << "Adding layer #" << new_num_layers - 1 << " to tree #"
            << current_tree << " of ensemble of " << current_tree + 1
            << " trees.";
    bool split_happened = false;
    // Add the splits to the tree.
    for (auto& split_entry : best_splits) {
      const int32 node_id = split_entry.first;
      const SplitCandidate& candidate = split_entry.second;

      const int64 feature_idx = candidate.feature_idx;
      const int64 candidate_idx = candidate.candidate_idx;

      const int32 feature_id = feature_ids(feature_idx);
      const int32 threshold =
          thresholds_list[feature_idx].vec<int32>()(candidate_idx);
      const float gain = gains_list[feature_idx].vec<float>()(candidate_idx);

      if (pruning_mode_ == kPrePruning) {
        // Don't consider negative splits if we're pre-pruning the tree.
        // Note that zero-gain splits are acceptable.
        if (gain < 0) {
          continue;
        }
      }
      // For now assume that the weights vectors are one dimensional.
      // TODO(nponomareva): change here for multiclass.
      const float left_contrib =
          learning_rate_ *
          left_node_contribs[feature_idx].matrix<float>()(candidate_idx, 0);
      const float right_contrib =
          learning_rate_ *
          right_node_contribs[feature_idx].matrix<float>()(candidate_idx, 0);

      // unused.
      int32 left_node_id;
      int32 right_node_id;

      ensemble_resource->AddBucketizedSplitNode(
          current_tree, node_id, feature_id, threshold, gain, left_contrib,
          right_contrib, &left_node_id, &right_node_id);
      split_happened = true;
    }
    if (split_happened) {
      // Update growable tree metadata.
      ensemble_resource->SetNumLayersGrown(current_tree, new_num_layers);
      // Finalize the tree if needed.
      if (ensemble_resource->GetNumLayersGrown(current_tree) >= max_depth_) {
        ensemble_resource->SetIsFinalized(current_tree, true);
        if (pruning_mode_ == kPostPruning) {
          ensemble_resource->PostPruneTree(current_tree);
        }
        if (ensemble_resource->num_trees() > 0) {
          // Create a dummy new tree with an empty node.
          ensemble_resource->AddNewTree(kLayerByLayerTreeWeight);
        }
      }
    }
  }

 private:
  int32 UpdateGlobalAttemptsAndRetrieveGrowableTree(
      BoostedTreesEnsembleResource* const ensemble_resource) {
    int32 num_trees = ensemble_resource->num_trees();
    int32 current_tree = num_trees - 1;

    // Increment global attempt stats.
    ensemble_resource->UpdateGrowingMetadata();

    // Note we don't set tree weight to be equal to learning rate, since we
    // apply learning rate to leaf weights instead, when doing layer-by-layer
    // boosting.
    if (num_trees <= 0) {
      // Create a new tree with a no-op leaf.
      current_tree = ensemble_resource->AddNewTree(kLayerByLayerTreeWeight);
    }
    return current_tree;
  }

  // Helper method which effectively does a reduce over all split candidates
  // and finds the best split for each node.
  void FindBestSplitsPerNode(
      OpKernelContext* const context, const OpInputList& node_ids_list,
      const OpInputList& gains_list,
      std::map<int32, SplitCandidate>* best_split_per_node) {
    // Find best split per node going through every feature candidate.
    for (int64 feature_idx = 0; feature_idx < num_features_; ++feature_idx) {
      const auto& node_ids = node_ids_list[feature_idx].vec<int32>();
      const auto& gains = gains_list[feature_idx].vec<float>();

      for (size_t candidate_idx = 0; candidate_idx < node_ids.size();
           ++candidate_idx) {
        // Get current split candidate.
        const auto& node_id = node_ids(candidate_idx);
        const auto& gain = gains(candidate_idx);

        auto best_split_it = best_split_per_node->find(node_id);
        SplitCandidate candidate;
        candidate.feature_idx = feature_idx;
        candidate.candidate_idx = candidate_idx;
        candidate.gain = gain;

        if (best_split_it == best_split_per_node->end() ||
            gain > best_split_it->second.gain) {
          (*best_split_per_node)[node_id] = candidate;
        }
      }
    }
  }

 private:
  int32 num_features_;
  float learning_rate_;
  int32 max_depth_;
  PruningMode pruning_mode_;
};

REGISTER_KERNEL_BUILDER(Name("BoostedTreesUpdateEnsemble").Device(DEVICE_CPU),
                        BoostedTreesUpdateEnsembleOp);

}  // namespace tensorflow
