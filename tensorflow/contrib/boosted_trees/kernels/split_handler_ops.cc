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
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/contrib/boosted_trees/lib/learner/common/stats/node-stats.h"
#include "tensorflow/contrib/boosted_trees/proto/split_info.pb.h"
#include "tensorflow/contrib/boosted_trees/proto/tree_config.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using boosted_trees::learner::LearnerConfig;
using boosted_trees::learner::LearnerConfig_MultiClassStrategy;
using boosted_trees::learner::ObliviousSplitInfo;
using boosted_trees::learner::SplitInfo;
using boosted_trees::learner::stochastic::GradientStats;
using boosted_trees::learner::stochastic::NodeStats;

namespace {
const int32 DUMMY_FEATURE_DIMENSION = -1;
}  // namespace

class SplitBuilderState {
 public:
  explicit SplitBuilderState(OpKernelContext* const context) {
    const Tensor* l1_regularization_t;
    OP_REQUIRES_OK(context,
                   context->input("l1_regularization", &l1_regularization_t));
    const Tensor* l2_regularization_t;
    OP_REQUIRES_OK(context,
                   context->input("l2_regularization", &l2_regularization_t));
    const Tensor* tree_complexity_regularization_t;
    OP_REQUIRES_OK(context, context->input("tree_complexity_regularization",
                                           &tree_complexity_regularization_t));
    const Tensor* min_node_weight_t;
    OP_REQUIRES_OK(context,
                   context->input("min_node_weight", &min_node_weight_t));

    const Tensor* feature_column_group_id_t;
    OP_REQUIRES_OK(context, context->input("feature_column_group_id",
                                           &feature_column_group_id_t));

    const Tensor* multiclass_strategy_t;
    OP_REQUIRES_OK(
        context, context->input("multiclass_strategy", &multiclass_strategy_t));
    int strategy = multiclass_strategy_t->scalar<int32>()();
    OP_REQUIRES(
        context,
        boosted_trees::learner::LearnerConfig_MultiClassStrategy_IsValid(
            strategy),
        errors::InvalidArgument("Wrong multiclass strategy passed."));

    multiclass_strategy_ = LearnerConfig_MultiClassStrategy(strategy);

    const Tensor* class_id_t;
    OP_REQUIRES_OK(context, context->input("class_id", &class_id_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(class_id_t->shape()),
                errors::InvalidArgument("class_id must be a scalar."));
    class_id_ = class_id_t->scalar<int32>()();

    l1_regularization_ = l1_regularization_t->scalar<float>()();
    l2_regularization_ = l2_regularization_t->scalar<float>()();
    tree_complexity_regularization_ =
        tree_complexity_regularization_t->scalar<float>()();
    min_node_weight_ = min_node_weight_t->scalar<float>()();
    feature_column_group_id_ = feature_column_group_id_t->scalar<int32>()();
  }

  NodeStats ComputeNodeStats(const GradientStats& grad_stats) {
    return NodeStats(l1_regularization_, l2_regularization_, min_node_weight_,
                     multiclass_strategy_, grad_stats);
  }

  void FillLeaf(const NodeStats& best_node_stats,
                boosted_trees::trees::Leaf* leaf) const {
    if (class_id_ == -1) {
      // This would be the case either for TREE_PER_CLASS with only 2 classes,
      // or for other multiclass strategies.
      for (float f : best_node_stats.weight_contribution) {
        leaf->mutable_vector()->add_value(f);
      }
    } else {
      CHECK(best_node_stats.weight_contribution.size() == 1)
          << "Weight contribution size = "
          << best_node_stats.weight_contribution.size();
      leaf->mutable_sparse_vector()->add_index(class_id_);
      leaf->mutable_sparse_vector()->add_value(
          best_node_stats.weight_contribution[0]);
    }
  }

  int32 feature_column_group_id() { return feature_column_group_id_; }
  float tree_complexity_regularization() {
    return tree_complexity_regularization_;
  }

 private:
  LearnerConfig_MultiClassStrategy multiclass_strategy_;
  float l1_regularization_;
  float l2_regularization_;
  float tree_complexity_regularization_;
  float min_node_weight_;
  int32 class_id_;
  int32 feature_column_group_id_;
};

class BuildDenseInequalitySplitsOp : public OpKernel {
 public:
  explicit BuildDenseInequalitySplitsOp(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* const context) override {
    const Tensor* num_minibatches_t;
    OP_REQUIRES_OK(context,
                   context->input("num_minibatches", &num_minibatches_t));
    const int64 num_minibatches = num_minibatches_t->scalar<int64>()();
    const float normalizer_ratio = (1.0f / num_minibatches);

    const Tensor* bucket_boundaries_t;
    OP_REQUIRES_OK(context,
                   context->input("bucket_boundaries", &bucket_boundaries_t));
    const auto& bucket_boundaries = bucket_boundaries_t->vec<float>();

    const Tensor* partition_ids_t;
    OP_REQUIRES_OK(context, context->input("partition_ids", &partition_ids_t));
    const auto& partition_ids = partition_ids_t->vec<int32>();

    const Tensor* bucket_ids_t;
    OP_REQUIRES_OK(context, context->input("bucket_ids", &bucket_ids_t));
    const auto& bucket_ids = bucket_ids_t->matrix<int64>();

    const Tensor* gradients_t;
    OP_REQUIRES_OK(context, context->input("gradients", &gradients_t));

    const Tensor* hessians_t;
    OP_REQUIRES_OK(context, context->input("hessians", &hessians_t));

    const Tensor* weak_learner_type_t;
    OP_REQUIRES_OK(context,
                   context->input("weak_learner_type", &weak_learner_type_t));
    const int32 weak_learner_type = weak_learner_type_t->scalar<int32>()();

    // Find the number of unique partitions before we allocate the output.
    std::vector<int32> partition_boundaries;
    partition_boundaries.push_back(0);
    for (int i = 1; i < partition_ids.size(); ++i) {
      if (partition_ids(i) != partition_ids(i - 1)) {
        // Make sure the input is sorted by partition_ids;
        OP_REQUIRES(context, partition_ids(i) >= partition_ids(i - 1),
                    errors::InvalidArgument("Input should be sorted."));
        partition_boundaries.push_back(i);
      }
    }
    if (partition_ids.size() > 0) {
      partition_boundaries.push_back(partition_ids.size());
    }
    int32 num_elements = partition_boundaries.size() - 1;

    // When the handler is inactive, no bucket boundaries are built for it.
    if (bucket_boundaries.size() == 0) {
      num_elements = 0;
    }

    Tensor* output_partition_ids_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output_partition_ids",
                                            TensorShape({num_elements}),
                                            &output_partition_ids_t));

    tensorflow::TTypes<int32>::Vec output_partition_ids =
        output_partition_ids_t->vec<int32>();

    // For a normal tree, we output a split per partition. For an oblivious
    // tree, we output one split for all partitions of the layer
    int32 size_output = num_elements;
    if (weak_learner_type == LearnerConfig::OBLIVIOUS_DECISION_TREE &&
        num_elements > 0) {
      size_output = 1;
    }

    Tensor* gains_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "gains", TensorShape({size_output}), &gains_t));
    tensorflow::TTypes<float>::Vec gains = gains_t->vec<float>();

    Tensor* output_splits_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("split_infos",
                                                     TensorShape({size_output}),
                                                     &output_splits_t));
    tensorflow::TTypes<string>::Vec output_splits =
        output_splits_t->vec<string>();

    if (num_elements == 0) {
      return;
    }
    SplitBuilderState state(context);
    switch (weak_learner_type) {
      case LearnerConfig::NORMAL_DECISION_TREE: {
        ComputeNormalDecisionTree(
            &state, normalizer_ratio, num_elements, partition_boundaries,
            bucket_boundaries, partition_ids, bucket_ids, gradients_t,
            hessians_t, &output_partition_ids, &gains, &output_splits);
        break;
      }
      case LearnerConfig::OBLIVIOUS_DECISION_TREE: {
        ComputeObliviousDecisionTree(
            &state, normalizer_ratio, num_elements, partition_boundaries,
            bucket_boundaries, partition_ids, bucket_ids, gradients_t,
            hessians_t, &output_partition_ids, &gains, &output_splits);
        break;
      }
    }
  }

 private:
  void ComputeNormalDecisionTree(
      SplitBuilderState* state, const float normalizer_ratio,
      const int num_elements, const std::vector<int32>& partition_boundaries,
      const tensorflow::TTypes<float>::ConstVec& bucket_boundaries,
      const tensorflow::TTypes<int32>::ConstVec& partition_ids,
      const tensorflow::TTypes<int64>::ConstMatrix& bucket_ids,
      const Tensor* gradients_t, const Tensor* hessians_t,
      tensorflow::TTypes<int32>::Vec* output_partition_ids,
      tensorflow::TTypes<float>::Vec* gains,
      tensorflow::TTypes<string>::Vec* output_splits) {
    for (int root_idx = 0; root_idx < num_elements; ++root_idx) {
      float best_gain = std::numeric_limits<float>::lowest();
      int start_index = partition_boundaries[root_idx];
      int end_index = partition_boundaries[root_idx + 1];
      GradientStats root_gradient_stats;
      for (int64 bucket_idx = start_index; bucket_idx < end_index;
           ++bucket_idx) {
        root_gradient_stats +=
            GradientStats(*gradients_t, *hessians_t, bucket_idx);
      }
      root_gradient_stats *= normalizer_ratio;
      NodeStats root_stats = state->ComputeNodeStats(root_gradient_stats);
      int32 best_bucket_idx = 0;
      NodeStats best_right_node_stats(0);
      NodeStats best_left_node_stats(0);
      GradientStats left_gradient_stats;
      for (int64 bucket_idx = start_index; bucket_idx < end_index;
           ++bucket_idx) {
        GradientStats g(*gradients_t, *hessians_t, bucket_idx);
        g *= normalizer_ratio;
        left_gradient_stats += g;
        NodeStats left_stats = state->ComputeNodeStats(left_gradient_stats);
        GradientStats right_gradient_stats =
            root_gradient_stats - left_gradient_stats;
        NodeStats right_stats = state->ComputeNodeStats(right_gradient_stats);
        if (left_stats.gain + right_stats.gain > best_gain) {
          best_gain = left_stats.gain + right_stats.gain;
          best_left_node_stats = left_stats;
          best_right_node_stats = right_stats;
          best_bucket_idx = bucket_idx;
        }
      }
      SplitInfo split_info;
      auto* dense_split =
          split_info.mutable_split_node()->mutable_dense_float_binary_split();
      dense_split->set_feature_column(state->feature_column_group_id());
      dense_split->set_threshold(
          bucket_boundaries(bucket_ids(best_bucket_idx, 0)));

      auto* left_child = split_info.mutable_left_child();
      auto* right_child = split_info.mutable_right_child();

      state->FillLeaf(best_left_node_stats, left_child);
      state->FillLeaf(best_right_node_stats, right_child);
      split_info.SerializeToString(&(*output_splits)(root_idx));
      (*gains)(root_idx) =
          best_gain - root_stats.gain - state->tree_complexity_regularization();
      (*output_partition_ids)(root_idx) = partition_ids(start_index);
    }
  }
  void ComputeObliviousDecisionTree(
      SplitBuilderState* state, const float normalizer_ratio,
      const int num_elements, const std::vector<int32>& partition_boundaries,
      const tensorflow::TTypes<float>::ConstVec& bucket_boundaries,
      const tensorflow::TTypes<int32>::ConstVec& partition_ids,
      const tensorflow::TTypes<int64>::ConstMatrix& bucket_ids,
      const Tensor* gradients_t, const Tensor* hessians_t,
      tensorflow::TTypes<int32>::Vec* output_partition_ids,
      tensorflow::TTypes<float>::Vec* gains,
      tensorflow::TTypes<string>::Vec* output_splits) {
    // Holds the root stats per each node to be split.
    std::vector<GradientStats> current_layer_stats;
    current_layer_stats.reserve(num_elements);
    for (int root_idx = 0; root_idx < num_elements; root_idx++) {
      const int start_index = partition_boundaries[root_idx];
      const int end_index = partition_boundaries[root_idx + 1];
      GradientStats root_gradient_stats;
      for (int64 bucket_idx = start_index; bucket_idx < end_index;
           ++bucket_idx) {
        root_gradient_stats +=
            GradientStats(*gradients_t, *hessians_t, bucket_idx);
      }
      root_gradient_stats *= normalizer_ratio;
      current_layer_stats.push_back(root_gradient_stats);
    }

    float best_gain = std::numeric_limits<float>::lowest();
    int64 best_bucket_id = 0;
    std::vector<NodeStats> best_right_node_stats(num_elements, NodeStats(0));
    std::vector<NodeStats> best_left_node_stats(num_elements, NodeStats(0));
    std::vector<NodeStats> current_left_node_stats(num_elements, NodeStats(0));
    std::vector<NodeStats> current_right_node_stats(num_elements, NodeStats(0));
    int64 current_bucket_id = std::numeric_limits<int64>::max();
    int64 last_bucket_id = -1;
    // Find the lowest bucket id, this is going to be the first bucket id to
    // try.
    for (int root_idx = 0; root_idx < num_elements; root_idx++) {
      const int start_index = partition_boundaries[root_idx];
      if (bucket_ids(start_index, 0) < current_bucket_id) {
        current_bucket_id = bucket_ids(start_index, 0);
      }
    }
    // Indexes offsets for each of the partitions that can be used to access
    // gradients of a partition for a current bucket we consider.
    std::vector<int> current_layer_offsets(num_elements, 0);
    std::vector<GradientStats> left_gradient_stats(num_elements);
    // The idea is to try every bucket id in increasing order. In each iteration
    // we calculate the gain of the layer using the current bucket id as split
    // value, and we also obtain the following bucket id to try.
    while (current_bucket_id > last_bucket_id) {
      last_bucket_id = current_bucket_id;
      int64 next_bucket_id = -1;
      for (int root_idx = 0; root_idx < num_elements; root_idx++) {
        int idx =
            current_layer_offsets[root_idx] + partition_boundaries[root_idx];
        const int end_index = partition_boundaries[root_idx + 1];
        if (idx < end_index && bucket_ids(idx, 0) == current_bucket_id) {
          GradientStats g(*gradients_t, *hessians_t, idx);
          g *= normalizer_ratio;
          left_gradient_stats[root_idx] += g;
          current_layer_offsets[root_idx]++;
          idx++;
        }
        if (idx < end_index &&
            (bucket_ids(idx, 0) < next_bucket_id || next_bucket_id == -1)) {
          next_bucket_id = bucket_ids(idx, 0);
        }
      }
      float gain_of_split = 0.0;
      for (int root_idx = 0; root_idx < num_elements; root_idx++) {
        GradientStats right_gradient_stats =
            current_layer_stats[root_idx] - left_gradient_stats[root_idx];
        NodeStats left_stat =
            state->ComputeNodeStats(left_gradient_stats[root_idx]);
        NodeStats right_stat = state->ComputeNodeStats(right_gradient_stats);
        gain_of_split += left_stat.gain + right_stat.gain;
        current_left_node_stats[root_idx] = left_stat;
        current_right_node_stats[root_idx] = right_stat;
      }
      if (gain_of_split > best_gain) {
        best_gain = gain_of_split;
        best_left_node_stats = current_left_node_stats;
        best_right_node_stats = current_right_node_stats;
        best_bucket_id = current_bucket_id;
      }
      current_bucket_id = next_bucket_id;
    }

    for (int root_idx = 0; root_idx < num_elements; root_idx++) {
      best_gain -= state->ComputeNodeStats(current_layer_stats[root_idx]).gain;
    }
    best_gain -= num_elements * state->tree_complexity_regularization();

    ObliviousSplitInfo oblivious_split_info;
    auto* oblivious_dense_split =
        oblivious_split_info.mutable_split_node()
            ->mutable_oblivious_dense_float_binary_split();
    oblivious_dense_split->set_feature_column(state->feature_column_group_id());
    oblivious_dense_split->set_threshold(bucket_boundaries(best_bucket_id));
    (*gains)(0) = best_gain;

    for (int root_idx = 0; root_idx < num_elements; root_idx++) {
      auto* left_child = oblivious_split_info.add_children();
      auto* right_child = oblivious_split_info.add_children();

      state->FillLeaf(best_left_node_stats[root_idx], left_child);
      state->FillLeaf(best_right_node_stats[root_idx], right_child);

      const int start_index = partition_boundaries[root_idx];
      (*output_partition_ids)(root_idx) = partition_ids(start_index);
      oblivious_split_info.add_children_parent_id(partition_ids(start_index));
    }
    oblivious_split_info.SerializeToString(&(*output_splits)(0));
  }
};
REGISTER_KERNEL_BUILDER(Name("BuildDenseInequalitySplits").Device(DEVICE_CPU),
                        BuildDenseInequalitySplitsOp);

class BuildSparseInequalitySplitsOp : public OpKernel {
 public:
  explicit BuildSparseInequalitySplitsOp(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* const context) override {
    const Tensor* num_minibatches_t;
    OP_REQUIRES_OK(context,
                   context->input("num_minibatches", &num_minibatches_t));
    const int64 num_minibatches = num_minibatches_t->scalar<int64>()();
    const float normalizer_ratio = (1.0f / num_minibatches);

    const Tensor* bucket_boundaries_t;
    OP_REQUIRES_OK(context,
                   context->input("bucket_boundaries", &bucket_boundaries_t));
    const auto& bucket_boundaries = bucket_boundaries_t->vec<float>();

    const Tensor* partition_ids_t;
    OP_REQUIRES_OK(context, context->input("partition_ids", &partition_ids_t));
    const auto& partition_ids = partition_ids_t->vec<int32>();

    const Tensor* bucket_ids_t;
    OP_REQUIRES_OK(context, context->input("bucket_ids", &bucket_ids_t));
    const auto& bucket_ids_and_dimensions = bucket_ids_t->matrix<int64>();

    const int32 tensor_elements = partition_ids.size();

    const Tensor* gradients_t;
    OP_REQUIRES_OK(context, context->input("gradients", &gradients_t));

    const Tensor* hessians_t;
    OP_REQUIRES_OK(context, context->input("hessians", &hessians_t));

    const Tensor* bias_feature_id_t;
    OP_REQUIRES_OK(context,
                   context->input("bias_feature_id", &bias_feature_id_t));
    int64 bias_feature_id = bias_feature_id_t->scalar<int64>()();

    // For each partition (tree node), store starting index for each dimension.
    PartitionAndDimensionBoundaries partition_boundaries;
    // Stores indices in partition_boundaries for those partitions that are
    // not empty (have at least one dimension and a bucket apart from catch-all
    // bucket of -1 bucket id and dimension 0.
    std::vector<int32> non_empty_partitions;
    bool non_empty_partition = false;

    for (int i = 0; i < partition_ids.size(); ++i) {
      // Make sure the input is sorted by partition_ids;
      if (i > 0) {
        CHECK_LE(partition_ids(i - 1), partition_ids(i))
            << "Partition ids should be sorted. Not sorted for " << i;
      }
      const int32 dimension = bucket_ids_and_dimensions(i, 1);

      if (i == 0 || (partition_ids(i) != partition_ids(i - 1))) {
        if (i != 0) {
          // Not the first entry, so partition has changed.
          if (non_empty_partition) {
            // Saves the id of a previous partition in a list of non empty
            // partitions, since it was non empty (had more than just a bias
            // bucket -1.
            non_empty_partitions.push_back(partition_boundaries.size() - 1);
          }
          // Add dummy dimension to signify the end for the previous dimension.
          partition_boundaries.back().emplace_back(DUMMY_FEATURE_DIMENSION, i);
        }
        // Allocate for a new partition.
        partition_boundaries.emplace_back();
        // Save info about the first dimension for a new partition.
        partition_boundaries.back().emplace_back(dimension, i);

        // Each partition has dummy -1 bucket with all gradients and then info
        // for all other dimensions -> if we have >1 elements for a partition,
        // then it is not empty.
        non_empty_partition = (i < partition_ids.size() - 1) &&
                              (partition_ids(i) == partition_ids(i + 1));
      } else if (bucket_ids_and_dimensions(i, 1) !=
                 bucket_ids_and_dimensions(i - 1, 1)) {
        // Dimension changed.
        partition_boundaries.back().emplace_back(dimension, i);
      }
    }
    if (tensor_elements > 0) {
      if (non_empty_partition) {
        non_empty_partitions.push_back(partition_boundaries.size() - 1);
      }
      // Add dummy dimension to signify the end for the previous dimension.
      partition_boundaries.back().emplace_back(DUMMY_FEATURE_DIMENSION,
                                               partition_ids.size());
    }

    int num_elements = non_empty_partitions.size();
    Tensor* output_partition_ids_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output_partition_ids",
                                            TensorShape({num_elements}),
                                            &output_partition_ids_t));

    tensorflow::TTypes<int32>::Vec output_partition_ids =
        output_partition_ids_t->vec<int32>();

    Tensor* gains_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("gains", TensorShape({num_elements}),
                                          &gains_t));

    tensorflow::TTypes<float>::Vec gains = gains_t->vec<float>();

    Tensor* output_splits_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "split_infos", TensorShape({num_elements}),
                                &output_splits_t));
    tensorflow::TTypes<string>::Vec output_splits =
        output_splits_t->vec<string>();
    SplitBuilderState state(context);
    // For each tree node that needs to be split.
    for (int root_idx = 0; root_idx < num_elements; ++root_idx) {
      const auto& dimension_boundaries =
          partition_boundaries[non_empty_partitions[root_idx]];

      float best_gain = std::numeric_limits<float>::lowest();
      int32 best_dimension_idx = 0;
      bool default_right = false;
      int32 best_element_idx = 0;

      NodeStats best_right_node_stats(0);
      NodeStats best_left_node_stats(0);

      // For each partition, the first bucket is dummy catch all.
      int32 bias_start_index = dimension_boundaries[0].start_index;

      OP_REQUIRES(
          context,
          bucket_ids_and_dimensions(bias_start_index, 0) == bias_feature_id,
          errors::InvalidArgument("Bias feature ID missing."));

      // Dimension for bias feature is always 0
      OP_REQUIRES(
          context, bucket_ids_and_dimensions(bias_start_index, 1) == 0,
          errors::InvalidArgument("Bias feature ID must be with dimension 0."));

      // For each root, we do two passes over the quantized feature buckets
      // accumulating gradients on one side and using the root aggregate
      // gradients to get the gradients for the other side.
      // Split gains are evaluated for each pass at every threshold and the best
      // split is picked.
      GradientStats root_gradient_stats(*gradients_t, *hessians_t,
                                        bias_start_index);
      root_gradient_stats *= normalizer_ratio;
      NodeStats root_stats = state.ComputeNodeStats(root_gradient_stats);

      // Iterate through dimensions.
      for (int j = 0; j < dimension_boundaries.size() - 1; ++j) {
        const DimensionBoundary& dimension_and_start = dimension_boundaries[j];
        const int32 dimension_id = dimension_and_start.dimension_id;

        int start_index = dimension_and_start.start_index;
        // Even for the last dimension, we always have additional dummy
        // dimension that we can use to find the end index.
        const int end_index =
            partition_boundaries[non_empty_partitions[root_idx]][j + 1]
                .start_index;
        if (bucket_ids_and_dimensions(start_index, 0) == bias_feature_id) {
          // 0-dimension case which has a first bucket for catch all feature.
          CHECK(bucket_ids_and_dimensions(start_index, 1) == 0)
              << "Dimension of bias feature should be 0";
          ++start_index;
        }

        GradientStats present_gradient_stats;
        for (int64 bucket_idx = start_index; bucket_idx < end_index;
             ++bucket_idx) {
          present_gradient_stats +=
              GradientStats(*gradients_t, *hessians_t, bucket_idx);
        }
        present_gradient_stats *= normalizer_ratio;
        GradientStats not_present =
            root_gradient_stats - present_gradient_stats;
        // If there was (almost) no sparsity, fix the default direction to LEFT.
        bool fixed_default_direction = not_present.IsAlmostZero();

        GradientStats left_gradient_stats;
        for (int64 element_idx = start_index; element_idx < end_index;
             ++element_idx) {
          // Check that bucket ids are sorted.
          if (element_idx != start_index) {
            CHECK(bucket_ids_and_dimensions(element_idx - 1, 0) <
                  bucket_ids_and_dimensions(element_idx, 0))
                << "Bucket ids must be sorted."
                << ", problem on " << element_idx << " and dimension is " << j;
          }

          GradientStats g(*gradients_t, *hessians_t, element_idx);
          g *= normalizer_ratio;
          left_gradient_stats += g;
          // We have the sum of all present gradients. Use that to compute the
          // backward pass gradients.
          GradientStats right_gradient_stats =
              present_gradient_stats - left_gradient_stats;

          {
            NodeStats left_stats_default_left = state.ComputeNodeStats(
                root_gradient_stats - right_gradient_stats);
            NodeStats right_stats_default_left =
                state.ComputeNodeStats(right_gradient_stats);
            if (left_stats_default_left.gain + right_stats_default_left.gain >
                best_gain) {
              best_gain =
                  left_stats_default_left.gain + right_stats_default_left.gain;
              best_left_node_stats = left_stats_default_left;
              best_right_node_stats = right_stats_default_left;
              best_element_idx = element_idx;
              default_right = false;
              best_dimension_idx = dimension_id;
            }
          }
          // Consider calculating the default direction only when there were
          // enough missing examples.
          if (!fixed_default_direction) {
            NodeStats left_stats_default_right =
                state.ComputeNodeStats(left_gradient_stats);
            NodeStats right_stats_default_right = state.ComputeNodeStats(
                root_gradient_stats - left_gradient_stats);
            if (left_stats_default_right.gain + right_stats_default_right.gain >
                best_gain) {
              best_gain = left_stats_default_right.gain +
                          right_stats_default_right.gain;
              best_left_node_stats = left_stats_default_right;
              best_right_node_stats = right_stats_default_right;
              best_element_idx = element_idx;
              default_right = true;
              best_dimension_idx = dimension_id;
            }
          }
        }
      }

      SplitInfo split_info;
      boosted_trees::trees::DenseFloatBinarySplit* dense_split = nullptr;
      if (default_right) {
        dense_split = split_info.mutable_split_node()
                          ->mutable_sparse_float_binary_split_default_right()
                          ->mutable_split();
      } else {
        dense_split = split_info.mutable_split_node()
                          ->mutable_sparse_float_binary_split_default_left()
                          ->mutable_split();
      }
      dense_split->set_feature_column(state.feature_column_group_id());
      // Set the feature index for the best feature column.
      const int64 best_dimension_id =
          bucket_ids_and_dimensions(best_element_idx, 1);
      const int32 best_bucket_id =
          bucket_ids_and_dimensions(best_element_idx, 0);
      dense_split->set_dimension_id(best_dimension_id);
      dense_split->set_threshold(bucket_boundaries(best_bucket_id));

      auto* left_child = split_info.mutable_left_child();
      auto* right_child = split_info.mutable_right_child();
      state.FillLeaf(best_left_node_stats, left_child);
      state.FillLeaf(best_right_node_stats, right_child);
      split_info.SerializeToString(&output_splits(root_idx));
      gains(root_idx) =
          best_gain - root_stats.gain - state.tree_complexity_regularization();
      output_partition_ids(root_idx) = partition_ids(bias_start_index);
    }
  }

 private:
  struct DimensionBoundary {
    DimensionBoundary(const int32 dimension_id, const int32 start_index)
        : dimension_id(dimension_id), start_index(start_index) {}

    int32 dimension_id;
    int32 start_index;
  };

  // For each partition, store start indices of feature column dimensions.
  typedef std::vector<std::vector<DimensionBoundary>>
      PartitionAndDimensionBoundaries;
};
REGISTER_KERNEL_BUILDER(Name("BuildSparseInequalitySplits").Device(DEVICE_CPU),
                        BuildSparseInequalitySplitsOp);

class BuildCategoricalEqualitySplitsOp : public OpKernel {
 public:
  explicit BuildCategoricalEqualitySplitsOp(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* const context) override {
    const Tensor* num_minibatches_t;
    OP_REQUIRES_OK(context,
                   context->input("num_minibatches", &num_minibatches_t));
    const int64 num_minibatches = num_minibatches_t->scalar<int64>()();
    const float normalizer_ratio = (1.0f / num_minibatches);

    const Tensor* partition_ids_t;
    OP_REQUIRES_OK(context, context->input("partition_ids", &partition_ids_t));
    const auto& partition_ids = partition_ids_t->vec<int32>();

    const Tensor* feature_ids_t;
    OP_REQUIRES_OK(context, context->input("feature_ids", &feature_ids_t));
    const auto& feature_ids = feature_ids_t->matrix<int64>();

    const Tensor* gradients_t;
    OP_REQUIRES_OK(context, context->input("gradients", &gradients_t));

    const Tensor* hessians_t;
    OP_REQUIRES_OK(context, context->input("hessians", &hessians_t));

    const Tensor* bias_feature_id_t;
    OP_REQUIRES_OK(context,
                   context->input("bias_feature_id", &bias_feature_id_t));
    int64 bias_feature_id = bias_feature_id_t->scalar<int64>()();

    const Tensor* weak_learner_type_t;
    OP_REQUIRES_OK(context,
                   context->input("weak_learner_type", &weak_learner_type_t));
    const int32 weak_learner_type = weak_learner_type_t->scalar<int32>()();

    // Find the number of unique partitions before we allocate the output.
    std::vector<int32> partition_boundaries;
    partition_boundaries.push_back(0);
    for (int i = 1; i < partition_ids.size(); ++i) {
      // Make sure the input is sorted by partition_ids;
      OP_REQUIRES(context, partition_ids(i - 1) <= partition_ids(i),
                  errors::InvalidArgument("Partition IDs must be sorted."));
      if (partition_ids(i) != partition_ids(i - 1)) {
        partition_boundaries.push_back(i);
      }
    }
    std::vector<int32> non_empty_partitions;
    partition_boundaries.push_back(partition_ids.size());
    for (int i = 0; i < partition_boundaries.size() - 1; ++i) {
      // We want to ignore partitions with only the bias term.
      if (partition_boundaries[i + 1] - partition_boundaries[i] >= 2) {
        non_empty_partitions.push_back(i);
      }
    }
    int num_elements = non_empty_partitions.size();
    Tensor* output_partition_ids_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output_partition_ids",
                                            TensorShape({num_elements}),
                                            &output_partition_ids_t));

    tensorflow::TTypes<int32>::Vec output_partition_ids =
        output_partition_ids_t->vec<int32>();

    // For a normal tree, we output a split per partition. For an oblivious
    // tree, we output one split for all partitions of the layer.
    int size_output = num_elements;
    if (weak_learner_type == LearnerConfig::OBLIVIOUS_DECISION_TREE &&
        num_elements > 0) {
      size_output = 1;
    }

    Tensor* gains_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "gains", TensorShape({size_output}), &gains_t));

    tensorflow::TTypes<float>::Vec gains = gains_t->vec<float>();

    Tensor* output_splits_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("split_infos",
                                                     TensorShape({size_output}),
                                                     &output_splits_t));
    tensorflow::TTypes<string>::Vec output_splits =
        output_splits_t->vec<string>();
    if (num_elements == 0) {
      return;
    }
    SplitBuilderState state(context);
    switch (weak_learner_type) {
      case LearnerConfig::NORMAL_DECISION_TREE: {
        ComputeNormalDecisionTree(
            context, &state, normalizer_ratio, num_elements,
            partition_boundaries, non_empty_partitions, bias_feature_id,
            partition_ids, feature_ids, gradients_t, hessians_t,
            &output_partition_ids, &gains, &output_splits);
        break;
      }
      case LearnerConfig::OBLIVIOUS_DECISION_TREE: {
        ComputeObliviousDecisionTree(
            context, &state, normalizer_ratio, num_elements,
            partition_boundaries, non_empty_partitions, bias_feature_id,
            partition_ids, feature_ids, gradients_t, hessians_t,
            &output_partition_ids, &gains, &output_splits);
        break;
      }
    }
  }

 private:
  void ComputeNormalDecisionTree(
      OpKernelContext* const context, SplitBuilderState* state,
      const float normalizer_ratio, const int num_elements,
      const std::vector<int32>& partition_boundaries,
      const std::vector<int32>& non_empty_partitions,
      const int64 bias_feature_id,
      const tensorflow::TTypes<int32>::ConstVec& partition_ids,
      const tensorflow::TTypes<int64>::ConstMatrix& feature_ids,
      const Tensor* gradients_t, const Tensor* hessians_t,
      tensorflow::TTypes<int32>::Vec* output_partition_ids,
      tensorflow::TTypes<float>::Vec* gains,
      tensorflow::TTypes<string>::Vec* output_splits) {
    for (int root_idx = 0; root_idx < num_elements; ++root_idx) {
      float best_gain = std::numeric_limits<float>::lowest();
      int start_index = partition_boundaries[non_empty_partitions[root_idx]];
      int end_index = partition_boundaries[non_empty_partitions[root_idx] + 1];
      // First feature ID in each partition should be the bias feature.
      OP_REQUIRES(context, feature_ids(start_index, 0) == bias_feature_id,
                  errors::InvalidArgument("Bias feature ID missing."));
      GradientStats root_gradient_stats(*gradients_t, *hessians_t, start_index);
      root_gradient_stats *= normalizer_ratio;
      NodeStats root_stats = state->ComputeNodeStats(root_gradient_stats);
      int32 best_feature_idx = 0;
      bool best_feature_updated = false;
      NodeStats best_right_node_stats(0);
      NodeStats best_left_node_stats(0);
      CHECK(end_index - start_index >= 2)
          << "Partition should have a non bias feature. Start index "
          << start_index << " and end index " << end_index;

      for (int64 feature_idx = start_index + 1; feature_idx < end_index;
           ++feature_idx) {
        GradientStats left_gradient_stats(*gradients_t, *hessians_t,
                                          feature_idx);
        left_gradient_stats *= normalizer_ratio;
        GradientStats right_gradient_stats =
            root_gradient_stats - left_gradient_stats;
        NodeStats left_stats = state->ComputeNodeStats(left_gradient_stats);
        NodeStats right_stats = state->ComputeNodeStats(right_gradient_stats);
        if (!best_feature_updated ||
            left_stats.gain + right_stats.gain > best_gain) {
          best_gain = left_stats.gain + right_stats.gain;
          best_left_node_stats = left_stats;
          best_right_node_stats = right_stats;
          best_feature_idx = feature_idx;
          best_feature_updated = true;
        }
      }
      SplitInfo split_info;
      auto* equality_split = split_info.mutable_split_node()
                                 ->mutable_categorical_id_binary_split();
      equality_split->set_feature_column(state->feature_column_group_id());
      CHECK(feature_ids(best_feature_idx, 0) != bias_feature_id)
          << "Unexpected feature ID selected. "
          << "Start feature ID: [" << start_index << "] "
          << feature_ids(start_index, 0) << ", " << feature_ids(start_index, 1)
          << "\nBest feature ID: [" << best_feature_idx << "] "
          << feature_ids(best_feature_idx, 0) << ", "
          << feature_ids(best_feature_idx, 1)
          << "\nPartition IDS: " << partition_ids(start_index) << "  "
          << partition_ids(best_feature_idx) << " and best gain " << best_gain;
      equality_split->set_feature_id(feature_ids(best_feature_idx, 0));
      auto* left_child = split_info.mutable_left_child();
      auto* right_child = split_info.mutable_right_child();
      state->FillLeaf(best_left_node_stats, left_child);
      state->FillLeaf(best_right_node_stats, right_child);
      split_info.SerializeToString(&(*output_splits)(root_idx));
      (*gains)(root_idx) =
          best_gain - root_stats.gain - state->tree_complexity_regularization();
      (*output_partition_ids)(root_idx) = partition_ids(start_index);
    }
  }

  void ComputeObliviousDecisionTree(
      OpKernelContext* const context, SplitBuilderState* state,
      const float normalizer_ratio, const int num_elements,
      const std::vector<int32>& partition_boundaries,
      const std::vector<int32>& non_empty_partitions,
      const int64 bias_feature_id,
      const tensorflow::TTypes<int32>::ConstVec& partition_ids,
      const tensorflow::TTypes<int64>::ConstMatrix& feature_ids,
      const Tensor* gradients_t, const Tensor* hessians_t,
      tensorflow::TTypes<int32>::Vec* output_partition_ids,
      tensorflow::TTypes<float>::Vec* gains,
      tensorflow::TTypes<string>::Vec* output_splits) {
    // Holds the root stats per each node to be split.
    std::vector<GradientStats> current_layer_stats;
    current_layer_stats.reserve(num_elements);
    for (int root_idx = 0; root_idx < num_elements; root_idx++) {
      const int start_index = partition_boundaries[root_idx];
      // First feature ID in each partition should be the bias feature.
      OP_REQUIRES(context, feature_ids(start_index, 0) == bias_feature_id,
                  errors::InvalidArgument("Bias feature ID missing."));
      GradientStats root_gradient_stats(*gradients_t, *hessians_t, start_index);
      root_gradient_stats *= normalizer_ratio;
      current_layer_stats.push_back(root_gradient_stats);
    }
    float best_gain = std::numeric_limits<float>::lowest();
    int64 best_feature_id = 0;
    std::vector<NodeStats> best_right_node_stats(num_elements, NodeStats(0));
    std::vector<NodeStats> best_left_node_stats(num_elements, NodeStats(0));
    std::vector<NodeStats> current_left_node_stats(num_elements, NodeStats(0));
    std::vector<NodeStats> current_right_node_stats(num_elements, NodeStats(0));
    int64 current_feature_id = std::numeric_limits<int64>::max();
    int64 last_feature_id = -1;
    // Find the lowest feature id, this is going to be the first feature id to
    // try.
    for (int root_idx = 0; root_idx < num_elements; root_idx++) {
      const int start_index = partition_boundaries[root_idx];
      if (feature_ids(start_index + 1, 0) < current_feature_id) {
        current_feature_id = feature_ids(start_index + 1, 0);
      }
    }
    // Indexes offsets for each of the partitions that can be used to access
    // gradients of a partition for a current feature we consider. Start at one
    // beacuse the zero index is for the bias.
    std::vector<int> current_layer_offsets(num_elements, 1);
    // The idea is to try every feature id in increasing order. In each
    // iteration we calculate the gain of the layer using the current feature id
    // as split value, and we also obtain the following feature id to try.
    while (current_feature_id > last_feature_id) {
      last_feature_id = current_feature_id;
      int64 next_feature_id = -1;
      // Left gradient stats per node.
      std::vector<GradientStats> left_gradient_stats(num_elements);
      for (int root_idx = 0; root_idx < num_elements; root_idx++) {
        int idx =
            current_layer_offsets[root_idx] + partition_boundaries[root_idx];
        const int end_index = partition_boundaries[root_idx + 1];
        if (idx < end_index && feature_ids(idx, 0) == current_feature_id) {
          GradientStats g(*gradients_t, *hessians_t, idx);
          g *= normalizer_ratio;
          left_gradient_stats[root_idx] = g;
          current_layer_offsets[root_idx]++;
          idx++;
        }
        if (idx < end_index &&
            (feature_ids(idx, 0) < next_feature_id || next_feature_id == -1)) {
          next_feature_id = feature_ids(idx, 0);
        }
      }
      float gain_of_split = 0.0;
      for (int root_idx = 0; root_idx < num_elements; root_idx++) {
        GradientStats right_gradient_stats =
            current_layer_stats[root_idx] - left_gradient_stats[root_idx];
        NodeStats left_stat =
            state->ComputeNodeStats(left_gradient_stats[root_idx]);
        NodeStats right_stat = state->ComputeNodeStats(right_gradient_stats);
        gain_of_split += left_stat.gain + right_stat.gain;
        current_left_node_stats[root_idx] = left_stat;
        current_right_node_stats[root_idx] = right_stat;
      }
      if (gain_of_split > best_gain) {
        best_gain = gain_of_split;
        best_left_node_stats = current_left_node_stats;
        best_right_node_stats = current_right_node_stats;
        best_feature_id = current_feature_id;
      }
      current_feature_id = next_feature_id;
    }

    for (int root_idx = 0; root_idx < num_elements; root_idx++) {
      best_gain -= state->ComputeNodeStats(current_layer_stats[root_idx]).gain;
    }
    best_gain -= num_elements * state->tree_complexity_regularization();

    ObliviousSplitInfo oblivious_split_info;
    auto* equality_split =
        oblivious_split_info.mutable_split_node()
            ->mutable_oblivious_categorical_id_binary_split();
    equality_split->set_feature_column(state->feature_column_group_id());
    equality_split->set_feature_id(best_feature_id);
    (*gains)(0) = best_gain;

    for (int root_idx = 0; root_idx < num_elements; root_idx++) {
      auto* left_child = oblivious_split_info.add_children();
      auto* right_child = oblivious_split_info.add_children();

      state->FillLeaf(best_left_node_stats[root_idx], left_child);
      state->FillLeaf(best_right_node_stats[root_idx], right_child);

      const int start_index = partition_boundaries[root_idx];
      (*output_partition_ids)(root_idx) = partition_ids(start_index);
      oblivious_split_info.add_children_parent_id(partition_ids(start_index));
    }
    oblivious_split_info.SerializeToString(&(*output_splits)(0));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("BuildCategoricalEqualitySplits").Device(DEVICE_CPU),
    BuildCategoricalEqualitySplitsOp);

}  // namespace tensorflow
