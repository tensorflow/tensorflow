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

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/boosted_trees/boosted_trees.pb.h"
#include "tensorflow/core/kernels/boosted_trees/resources.h"
#include "tensorflow/core/kernels/boosted_trees/tree_helper.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

namespace {
constexpr float kLayerByLayerTreeWeight = 1.0;
constexpr float kMinDeltaForCenterBias = 0.01;

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
    core::RefCountPtr<BoostedTreesEnsembleResource> ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
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
    // Op does not support multi-class, the V2 op below does however.
    int32 logits_dimension = 1;
    // Find best splits for each active node.
    std::map<int32, boosted_trees::SplitCandidate> best_splits;
    FindBestSplitsPerNode(context, learning_rate, node_ids_list, gains_list,
                          thresholds_list, left_node_contribs,
                          right_node_contribs, feature_ids, &best_splits);

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
      const float gain = split_entry.second.gain;
      if (pruning_mode_ == kPrePruning) {
        // Don't consider negative splits if we're pre-pruning the tree.
        // Note that zero-gain splits are acceptable.
        if (gain < 0) {
          continue;
        }
      }

      // unused.
      int32 left_node_id;
      int32 right_node_id;

      ensemble_resource->AddBucketizedSplitNode(current_tree, split_entry,
                                                logits_dimension, &left_node_id,
                                                &right_node_id);
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
          ensemble_resource->PostPruneTree(current_tree, logits_dimension);
        }
        if (ensemble_resource->num_trees() > 0) {
          // Create a dummy new tree with an empty node.
          ensemble_resource->AddNewTree(kLayerByLayerTreeWeight, 1);
        }
      }
      // If we managed to split, update the node range. If we didn't, don't
      // update as we will try to split the same nodes with new instances.
      ensemble_resource->UpdateLastLayerNodesRange(node_id_start, node_id_end);
    }
  }

 private:
  int32 UpdateGlobalAttemptsAndRetrieveGrowableTree(
      const core::RefCountPtr<BoostedTreesEnsembleResource>& resource) {
    int32 num_trees = resource->num_trees();
    int32 current_tree = num_trees - 1;

    // Increment global attempt stats.
    resource->UpdateGrowingMetadata();

    // Note we don't set tree weight to be equal to learning rate, since we
    // apply learning rate to leaf weights instead, when doing layer-by-layer
    // boosting.
    if (num_trees <= 0) {
      // Create a new tree with a no-op leaf.
      current_tree = resource->AddNewTree(kLayerByLayerTreeWeight, 1);
    }
    return current_tree;
  }

  // Helper method which effectively does a reduce over all split candidates
  // and finds the best split for each node.
  void FindBestSplitsPerNode(
      OpKernelContext* const context, const float learning_rate,
      const OpInputList& node_ids_list, const OpInputList& gains_list,
      const OpInputList& thresholds_list,
      const OpInputList& left_node_contribs_list,
      const OpInputList& right_node_contribs_list,
      const TTypes<const int32>::Vec& feature_ids,
      std::map<int32, boosted_trees::SplitCandidate>* best_split_per_node) {
    // Find best split per node going through every feature candidate.
    for (int64 feature_idx = 0; feature_idx < num_features_; ++feature_idx) {
      const auto& node_ids = node_ids_list[feature_idx].vec<int32>();
      const auto& gains = gains_list[feature_idx].vec<float>();
      const auto& thresholds = thresholds_list[feature_idx].vec<int32>();
      const auto& left_node_contribs =
          left_node_contribs_list[feature_idx].matrix<float>();
      const auto& right_node_contribs =
          right_node_contribs_list[feature_idx].matrix<float>();

      for (size_t candidate_idx = 0; candidate_idx < node_ids.size();
           ++candidate_idx) {
        // Get current split candidate.
        const auto& node_id = node_ids(candidate_idx);
        const auto& gain = gains(candidate_idx);

        auto best_split_it = best_split_per_node->find(node_id);
        boosted_trees::SplitCandidate candidate;
        candidate.feature_idx = feature_ids(feature_idx);
        candidate.candidate_idx = candidate_idx;
        candidate.gain = gain;
        candidate.dimension_id = 0;
        candidate.threshold = thresholds(candidate_idx);
        candidate.left_node_contribs.push_back(
            learning_rate * left_node_contribs(candidate_idx, 0));
        candidate.right_node_contribs.push_back(
            learning_rate * right_node_contribs(candidate_idx, 0));
        candidate.split_type = boosted_trees::SplitTypeWithDefault_Name(
            boosted_trees::INEQUALITY_DEFAULT_LEFT);

        if (TF_PREDICT_FALSE(best_split_it != best_split_per_node->end() &&
                             GainsAreEqual(gain, best_split_it->second.gain))) {
          const auto best_candidate = (*best_split_per_node)[node_id];
          const int32 best_feature_id = best_candidate.feature_idx;
          const int32 feature_id = candidate.feature_idx;
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

// V2 of UpdateEnsembleOp that takes in split type and feature dimension id.
class BoostedTreesUpdateEnsembleV2Op : public OpKernel {
 public:
  explicit BoostedTreesUpdateEnsembleV2Op(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_features", &num_features_));
    OP_REQUIRES_OK(context, context->GetAttr("logits_dimension", &logits_dim_));
  }

  void Compute(OpKernelContext* const context) override {
    // Get decision tree ensemble.
    core::RefCountPtr<BoostedTreesEnsembleResource> ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    mutex_lock l(*ensemble_resource->get_mutex());
    // Increase the ensemble stamp.
    ensemble_resource->set_stamp(ensemble_resource->stamp() + 1);

    // Read node ids, gains, thresholds and node contribs.
    OpInputList node_ids_list;
    OpInputList gains_list;
    OpInputList thresholds_list;
    OpInputList dimension_ids_list;
    OpInputList left_node_contribs_list;
    OpInputList right_node_contribs_list;
    OpInputList split_types_list;
    OP_REQUIRES_OK(context, context->input_list("node_ids", &node_ids_list));
    OP_REQUIRES_OK(context, context->input_list("gains", &gains_list));
    OP_REQUIRES_OK(context,
                   context->input_list("thresholds", &thresholds_list));
    OP_REQUIRES_OK(context,
                   context->input_list("dimension_ids", &dimension_ids_list));
    OP_REQUIRES_OK(context, context->input_list("left_node_contribs",
                                                &left_node_contribs_list));
    OP_REQUIRES_OK(context, context->input_list("right_node_contribs",
                                                &right_node_contribs_list));
    OP_REQUIRES_OK(context,
                   context->input_list("split_types", &split_types_list));

    OpInputList feature_ids_list;
    OP_REQUIRES_OK(context,
                   context->input_list("feature_ids", &feature_ids_list));
    // TODO(crawles): Read groups of feature ids and find best splits among all.
    const auto feature_ids = feature_ids_list[0].vec<int32>();

    const Tensor* max_depth_t;
    OP_REQUIRES_OK(context, context->input("max_depth", &max_depth_t));
    const auto max_depth = max_depth_t->scalar<int32>()();

    const Tensor* learning_rate_t;
    OP_REQUIRES_OK(context, context->input("learning_rate", &learning_rate_t));
    const auto learning_rate = learning_rate_t->scalar<float>()();

    const Tensor* pruning_mode_t;
    OP_REQUIRES_OK(context, context->input("pruning_mode", &pruning_mode_t));
    const auto pruning_mode =
        static_cast<PruningMode>(pruning_mode_t->scalar<int32>()());
    // Find best splits for each active node.
    std::map<int32, boosted_trees::SplitCandidate> best_splits;
    FindBestSplitsPerNode(context, learning_rate, node_ids_list, gains_list,
                          thresholds_list, dimension_ids_list,
                          left_node_contribs_list, right_node_contribs_list,
                          split_types_list, feature_ids, &best_splits);

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
      const float gain = split_entry.second.gain;
      const string split_type = split_entry.second.split_type;

      if (pruning_mode == kPrePruning) {
        // Don't consider negative splits if we're pre-pruning the tree.
        // Note that zero-gain splits are acceptable.
        if (gain < 0) {
          continue;
        }
      }

      // unused.
      int32 left_node_id;
      int32 right_node_id;

      boosted_trees::SplitTypeWithDefault split_type_with_default;
      bool parsed = boosted_trees::SplitTypeWithDefault_Parse(
          split_type, &split_type_with_default);
      DCHECK(parsed);
      if (split_type_with_default == boosted_trees::EQUALITY_DEFAULT_RIGHT) {
        // Add equality split to the node.
        ensemble_resource->AddCategoricalSplitNode(current_tree, split_entry,
                                                   logits_dim_, &left_node_id,
                                                   &right_node_id);
      } else {
        // Add inequality split to the node.
        ensemble_resource->AddBucketizedSplitNode(current_tree, split_entry,
                                                  logits_dim_, &left_node_id,
                                                  &right_node_id);
      }
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
        if (pruning_mode == kPostPruning) {
          ensemble_resource->PostPruneTree(current_tree, logits_dim_);
        }
        if (ensemble_resource->num_trees() > 0) {
          // Create a dummy new tree with an empty node.
          ensemble_resource->AddNewTree(kLayerByLayerTreeWeight, logits_dim_);
        }
      }
      // If we managed to split, update the node range. If we didn't, don't
      // update as we will try to split the same nodes with new instances.
      ensemble_resource->UpdateLastLayerNodesRange(node_id_start, node_id_end);
    }
  }

 private:
  int32 UpdateGlobalAttemptsAndRetrieveGrowableTree(
      const core::RefCountPtr<BoostedTreesEnsembleResource>& resource) {
    int32 num_trees = resource->num_trees();
    int32 current_tree = num_trees - 1;

    // Increment global attempt stats.
    resource->UpdateGrowingMetadata();

    // Note we don't set tree weight to be equal to learning rate, since we
    // apply learning rate to leaf weights instead, when doing layer-by-layer
    // boosting.
    if (num_trees <= 0) {
      // Create a new tree with a no-op leaf.
      current_tree = resource->AddNewTree(kLayerByLayerTreeWeight, logits_dim_);
    }
    return current_tree;
  }

  // Helper method which effectively does a reduce over all split candidates
  // and finds the best split for each node.
  void FindBestSplitsPerNode(
      OpKernelContext* const context, const float learning_rate,
      const OpInputList& node_ids_list, const OpInputList& gains_list,
      const OpInputList& thresholds_list, const OpInputList& dimension_ids_list,
      const OpInputList& left_node_contribs_list,
      const OpInputList& right_node_contribs_list,
      const OpInputList& split_types_list,
      const TTypes<const int32>::Vec& feature_ids,
      std::map<int32, boosted_trees::SplitCandidate>* best_split_per_node) {
    // Find best split per node going through every feature candidate.
    for (int64 feature_idx = 0; feature_idx < num_features_; ++feature_idx) {
      const auto& node_ids = node_ids_list[feature_idx].vec<int32>();
      const auto& gains = gains_list[feature_idx].vec<float>();
      const auto& thresholds = thresholds_list[feature_idx].vec<int32>();
      const auto& dimension_ids = dimension_ids_list[feature_idx].vec<int32>();
      const auto& left_node_contribs =
          left_node_contribs_list[feature_idx].matrix<float>();
      const auto& right_node_contribs =
          right_node_contribs_list[feature_idx].matrix<float>();
      const auto& split_types = split_types_list[feature_idx].vec<tstring>();

      for (size_t candidate_idx = 0; candidate_idx < node_ids.size();
           ++candidate_idx) {
        // Get current split candidate.
        const auto& node_id = node_ids(candidate_idx);
        const auto& gain = gains(candidate_idx);
        const auto& threshold = thresholds(candidate_idx);
        const auto& dimension_id = dimension_ids(candidate_idx);
        const auto& split_type = split_types(candidate_idx);

        auto best_split_it = best_split_per_node->find(node_id);
        boosted_trees::SplitCandidate candidate;
        candidate.feature_idx = feature_ids(feature_idx);
        candidate.candidate_idx = candidate_idx;
        candidate.gain = gain;
        candidate.threshold = threshold;
        candidate.dimension_id = dimension_id;
        candidate.split_type = split_type;
        for (int i = 0; i < logits_dim_; ++i) {
          candidate.left_node_contribs.push_back(
              learning_rate * left_node_contribs(candidate_idx, i));
          candidate.right_node_contribs.push_back(
              learning_rate * right_node_contribs(candidate_idx, i));
        }
        if (TF_PREDICT_FALSE(best_split_it != best_split_per_node->end() &&
                             GainsAreEqual(gain, best_split_it->second.gain))) {
          const auto best_candidate = (*best_split_per_node)[node_id];
          const int32 best_feature_id = best_candidate.feature_idx;
          const int32 feature_id = candidate.feature_idx;
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
  int32 logits_dim_;
};

REGISTER_KERNEL_BUILDER(Name("BoostedTreesUpdateEnsembleV2").Device(DEVICE_CPU),
                        BoostedTreesUpdateEnsembleV2Op);

class BoostedTreesCenterBiasOp : public OpKernel {
 public:
  explicit BoostedTreesCenterBiasOp(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* const context) override {
    // Get decision tree ensemble.
    core::RefCountPtr<BoostedTreesEnsembleResource> ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    mutex_lock l(*ensemble_resource->get_mutex());
    // Increase the ensemble stamp.
    ensemble_resource->set_stamp(ensemble_resource->stamp() + 1);

    // Read means of hessians and gradients
    const Tensor* mean_gradients_t;
    OP_REQUIRES_OK(context,
                   context->input("mean_gradients", &mean_gradients_t));
    const int32 logits_dim = mean_gradients_t->dim_size(1);
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
    Eigen::VectorXf logits_vector(1);
    float unused_gain;

    // TODO(crawles): Support multiclass.
    DCHECK_EQ(logits_dim, 1);
    Eigen::VectorXf gradients_mean(1);
    Eigen::VectorXf hessians_mean(1);
    gradients_mean[0] = mean_gradients_t->flat<float>()(0);
    hessians_mean[0] = mean_hessians_t->flat<float>()(0);
    CalculateWeightsAndGains(gradients_mean, hessians_mean, l1, l2,
                             &logits_vector, &unused_gain);
    const float logits = logits_vector[0];

    float current_bias = 0.0;
    bool continue_centering = true;
    if (ensemble_resource->num_trees() == 0) {
      ensemble_resource->AddNewTreeWithLogits(kLayerByLayerTreeWeight, {logits},
                                              1);
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
