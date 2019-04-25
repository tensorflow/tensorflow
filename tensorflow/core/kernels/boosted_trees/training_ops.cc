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
#include "tensorflow/core/kernels/boosted_trees/tree_helper.h"

namespace tensorflow {

namespace {
constexpr float kLayerByLayerTreeWeight = 1.0;
constexpr float kMinDeltaForCenterBias = 0.01;

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
    const auto feature_ids = feature_ids_t->vec<int32>();

    const Tensor* max_depth_t;
    OP_REQUIRES_OK(context, context->input("max_depth", &max_depth_t));
    const auto max_depth = max_depth_t->scalar<int32>()();

    const Tensor* learning_rate_t;
    OP_REQUIRES_OK(context, context->input("learning_rate", &learning_rate_t));
    const auto learning_rate = learning_rate_t->scalar<float>()();

    // Find best splits for each active node.
    std::map<int32, SplitCandidate> best_splits;
    FindBestSplitsPerNode(context, node_ids_list, gains_list, feature_ids,
                          &best_splits);

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
    int32 node_id_start = ensemble_resource->GetNumNodes(current_tree);
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
          learning_rate *
          left_node_contribs[feature_idx].matrix<float>()(candidate_idx, 0);
      const float right_contrib =
          learning_rate *
          right_node_contribs[feature_idx].matrix<float>()(candidate_idx, 0);

      // unused.
      int32 left_node_id;
      int32 right_node_id;

      ensemble_resource->AddBucketizedSplitNode(
          current_tree, node_id, feature_id, threshold, gain, left_contrib,
          right_contrib, &left_node_id, &right_node_id);
      split_happened = true;
    }
    int32 node_id_end = ensemble_resource->GetNumNodes(current_tree);
    if (split_happened) {
      // Update growable tree metadata.
      ensemble_resource->SetNumLayersGrown(current_tree, new_num_layers);
      // Finalize the tree if needed.
      if (ensemble_resource->GetNumLayersGrown(current_tree) >= max_depth) {
        // If the tree is finalized, next growing will start from node 0;
        node_id_start = 0;
        node_id_end = 1;
        ensemble_resource->SetIsFinalized(current_tree, true);
        if (pruning_mode_ == kPostPruning) {
          // TODO(crawles): change for multi-class.
          ensemble_resource->PostPruneTree(current_tree, 1); /*logit dimension*/
        }
        if (ensemble_resource->num_trees() > 0) {
          // Create a dummy new tree with an empty node.
          ensemble_resource->AddNewTree(kLayerByLayerTreeWeight);
        }
      }
      // If we managed to split, update the node range. If we didn't, don't
      // update as we will try to split the same nodes with new instances.
      ensemble_resource->UpdateLastLayerNodesRange(node_id_start, node_id_end);
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
      const TTypes<const int32>::Vec& feature_ids,
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

        if (TF_PREDICT_FALSE(best_split_it != best_split_per_node->end() &&
                             GainsAreEqual(gain, best_split_it->second.gain))) {
          const auto best_candidate = (*best_split_per_node)[node_id];
          const int32 best_feature_id = feature_ids(best_candidate.feature_idx);
          const int32 feature_id = feature_ids(candidate.feature_idx);
          VLOG(2) << "Breaking ties on feature ids and buckets";
          // Breaking ties deterministically.
          if (feature_id < best_feature_id) {
            (*best_split_per_node)[node_id] = candidate;
          }
        } else if (best_split_it == best_split_per_node->end() ||
                   GainIsLarger(gain, best_split_it->second.gain)) {
          (*best_split_per_node)[node_id] = candidate;
        }
      }
    }
  }

 private:
  int32 num_features_;
  PruningMode pruning_mode_;
};

REGISTER_KERNEL_BUILDER(Name("BoostedTreesUpdateEnsemble").Device(DEVICE_CPU),
                        BoostedTreesUpdateEnsembleOp);

class BoostedTreesCenterBiasOp : public OpKernel {
 public:
  explicit BoostedTreesCenterBiasOp(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* const context) override {
    // Get decision tree ensemble.
    BoostedTreesEnsembleResource* ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    core::ScopedUnref unref_me(ensemble_resource);
    mutex_lock l(*ensemble_resource->get_mutex());
    // Increase the ensemble stamp.
    ensemble_resource->set_stamp(ensemble_resource->stamp() + 1);

    // Read means of hessians and gradients
    const Tensor* mean_gradients_t;
    OP_REQUIRES_OK(context,
                   context->input("mean_gradients", &mean_gradients_t));

    const Tensor* mean_hessians_t;
    OP_REQUIRES_OK(context, context->input("mean_hessians", &mean_hessians_t));

    // Get the regularization options.
    const Tensor* l1_t;
    OP_REQUIRES_OK(context, context->input("l1", &l1_t));
    const auto l1 = l1_t->scalar<float>()();
    const Tensor* l2_t;
    OP_REQUIRES_OK(context, context->input("l2", &l2_t));
    const auto l2 = l2_t->scalar<float>()();

    // For now, assume 1-dimensional weight on leaves.
    float logits;
    float unused_gain;

    // TODO(nponomareva): change this when supporting multiclass.
    const float gradients_mean = mean_gradients_t->flat<float>()(0);
    const float hessians_mean = mean_hessians_t->flat<float>()(0);
    CalculateWeightsAndGains(gradients_mean, hessians_mean, l1, l2, &logits,
                             &unused_gain);

    float current_bias = 0.0;
    bool continue_centering = true;
    if (ensemble_resource->num_trees() == 0) {
      ensemble_resource->AddNewTreeWithLogits(kLayerByLayerTreeWeight, logits);
      current_bias = logits;
    } else {
      const auto& current_biases = ensemble_resource->node_value(0, 0);
      DCHECK_EQ(current_biases.size(), 1);
      current_bias = current_biases[0];
      continue_centering =
          std::abs(logits / current_bias) > kMinDeltaForCenterBias;
      current_bias += logits;
      ensemble_resource->set_node_value(0, 0, current_bias);
    }

    Tensor* continue_centering_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("continue_centering", TensorShape({}),
                                          &continue_centering_t));
    // Check if we need to continue centering bias.
    continue_centering_t->scalar<bool>()() = continue_centering;
  }
};
REGISTER_KERNEL_BUILDER(Name("BoostedTreesCenterBias").Device(DEVICE_CPU),
                        BoostedTreesCenterBiasOp);

}  // namespace tensorflow
