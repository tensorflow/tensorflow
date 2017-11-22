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
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/contrib/boosted_trees/lib/learner/stochastic/stats/node-stats.h"
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

using boosted_trees::learner::SplitInfo;
using boosted_trees::learner::stochastic::GradientStats;
using boosted_trees::learner::stochastic::NodeStats;
using boosted_trees::learner::LearnerConfig_MultiClassStrategy;

namespace {
const int32 DUMMY_FEATURE_DIMENSION = -1;
}  // namespace

class BaseBuildSplitOp : public OpKernel {
 public:
  explicit BaseBuildSplitOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(
        context,
        context->GetAttr("feature_column_group_id", &feature_column_group_id_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("l1_regularization", &l1_regularization_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("l2_regularization", &l2_regularization_));
    OP_REQUIRES_OK(context, context->GetAttr("tree_complexity_regularization",
                                             &tree_complexity_regularization_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("min_node_weight", &min_node_weight_));

    int strategy;
    OP_REQUIRES_OK(context, context->GetAttr("multiclass_strategy", &strategy));
    OP_REQUIRES(
        context,
        boosted_trees::learner::LearnerConfig_MultiClassStrategy_IsValid(
            strategy),
        errors::InvalidArgument("Wrong multiclass strategy passed."));
    multiclass_strategy_ = LearnerConfig_MultiClassStrategy(strategy);
  }

  NodeStats ComputeNodeStats(const GradientStats& grad_stats) {
    return NodeStats(l1_regularization_, l2_regularization_, min_node_weight_,
                     multiclass_strategy_, grad_stats);
  }

  void ReadClassId(OpKernelContext* const context, int32* class_id) {
    const Tensor* class_id_t;
    OP_REQUIRES_OK(context, context->input("class_id", &class_id_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(class_id_t->shape()),
                errors::InvalidArgument("class_id must be a scalar."));
    *class_id = class_id_t->scalar<int32>()();
  }

  void FillLeaf(const int class_id, const NodeStats& best_node_stats,
                boosted_trees::trees::Leaf* leaf) const {
    if (class_id == -1) {
      // This would be the case either for TREE_PER_CLASS with only 2 classes,
      // or for other multiclass strategies.
      for (float f : best_node_stats.weight_contribution) {
        leaf->mutable_vector()->add_value(f);
      }
    } else {
      CHECK(best_node_stats.weight_contribution.size() == 1)
          << "Weight contribution size = "
          << best_node_stats.weight_contribution.size();
      leaf->mutable_sparse_vector()->add_index(class_id);
      leaf->mutable_sparse_vector()->add_value(
          best_node_stats.weight_contribution[0]);
    }
  }

 protected:
  LearnerConfig_MultiClassStrategy multiclass_strategy_;
  int32 feature_column_group_id_;
  float l1_regularization_;
  float l2_regularization_;
  float min_node_weight_;
  float tree_complexity_regularization_;
};

class BuildDenseInequalitySplitsOp : public BaseBuildSplitOp {
 public:
  explicit BuildDenseInequalitySplitsOp(OpKernelConstruction* const context)
      : BaseBuildSplitOp(context) {}

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

    int class_id;
    ReadClassId(context, &class_id);

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
      NodeStats root_stats = ComputeNodeStats(root_gradient_stats);
      int32 best_bucket_idx = 0;
      NodeStats best_right_node_stats(0);
      NodeStats best_left_node_stats(0);
      GradientStats left_gradient_stats;
      for (int64 bucket_idx = start_index; bucket_idx < end_index;
           ++bucket_idx) {
        GradientStats g(*gradients_t, *hessians_t, bucket_idx);
        g *= normalizer_ratio;
        left_gradient_stats += g;
        NodeStats left_stats = ComputeNodeStats(left_gradient_stats);
        GradientStats right_gradient_stats =
            root_gradient_stats - left_gradient_stats;
        NodeStats right_stats = ComputeNodeStats(right_gradient_stats);
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
      dense_split->set_feature_column(feature_column_group_id_);
      dense_split->set_threshold(
          bucket_boundaries(bucket_ids(best_bucket_idx, 0)));

      auto* left_child = split_info.mutable_left_child();
      auto* right_child = split_info.mutable_right_child();

      FillLeaf(class_id, best_left_node_stats, left_child);
      FillLeaf(class_id, best_right_node_stats, right_child);
      split_info.SerializeToString(&output_splits(root_idx));
      gains(root_idx) =
          best_gain - root_stats.gain - tree_complexity_regularization_;
      output_partition_ids(root_idx) = partition_ids(start_index);
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("BuildDenseInequalitySplits").Device(DEVICE_CPU),
                        BuildDenseInequalitySplitsOp);

class BuildSparseInequalitySplitsOp : public BaseBuildSplitOp {
 public:
  explicit BuildSparseInequalitySplitsOp(OpKernelConstruction* const context)
      : BaseBuildSplitOp(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("bias_feature_id", &bias_feature_id_));
  }

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

    int class_id;
    ReadClassId(context, &class_id);

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
          bucket_ids_and_dimensions(bias_start_index, 0) == bias_feature_id_,
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
      NodeStats root_stats = ComputeNodeStats(root_gradient_stats);

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
        CHECK(bucket_ids_and_dimensions(start_index, 1) ==
              bucket_ids_and_dimensions(end_index - 1, 1))
            << "For bucket " << bucket_ids_and_dimensions(start_index, 0)
            << " the dimension was "
            << bucket_ids_and_dimensions(start_index, 1) << " and for "
            << bucket_ids_and_dimensions(end_index - 1, 0) << " "
            << bucket_ids_and_dimensions(end_index - 1, 1);
        if (bucket_ids_and_dimensions(start_index, 0) == bias_feature_id_) {
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
            NodeStats left_stats_default_left =
                ComputeNodeStats(root_gradient_stats - right_gradient_stats);
            NodeStats right_stats_default_left =
                ComputeNodeStats(right_gradient_stats);
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
          {
            NodeStats left_stats_default_right =
                ComputeNodeStats(left_gradient_stats);
            NodeStats right_stats_default_right =
                ComputeNodeStats(root_gradient_stats - left_gradient_stats);
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
      dense_split->set_feature_column(feature_column_group_id_);
      // Set the feature index for the best feature column.
      const int64 best_dimension_id =
          bucket_ids_and_dimensions(best_element_idx, 1);
      const int32 best_bucket_id =
          bucket_ids_and_dimensions(best_element_idx, 0);
      dense_split->set_dimension_id(best_dimension_id);
      dense_split->set_threshold(bucket_boundaries(best_bucket_id));

      auto* left_child = split_info.mutable_left_child();
      auto* right_child = split_info.mutable_right_child();
      FillLeaf(class_id, best_left_node_stats, left_child);
      FillLeaf(class_id, best_right_node_stats, right_child);
      split_info.SerializeToString(&output_splits(root_idx));
      gains(root_idx) =
          best_gain - root_stats.gain - tree_complexity_regularization_;
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

  int64 bias_feature_id_;
};
REGISTER_KERNEL_BUILDER(Name("BuildSparseInequalitySplits").Device(DEVICE_CPU),
                        BuildSparseInequalitySplitsOp);

class BuildCategoricalEqualitySplitsOp : public BaseBuildSplitOp {
 public:
  explicit BuildCategoricalEqualitySplitsOp(OpKernelConstruction* const context)
      : BaseBuildSplitOp(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("bias_feature_id", &bias_feature_id_));
  }

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

    int class_id;
    ReadClassId(context, &class_id);

    // Find the number of unique partitions before we allocate the output.
    std::vector<int32> partition_boundaries;
    std::vector<int32> non_empty_partitions;
    for (int i = 0; i < partition_ids.size() - 1; ++i) {
      // Make sure the input is sorted by partition_ids;
      CHECK_LE(partition_ids(i), partition_ids(i + 1));
      if (i == 0 || partition_ids(i) != partition_ids(i - 1)) {
        partition_boundaries.push_back(i);
        // Some partitions might only have bias feature. We don't want to split
        // those so check that the partition has at least 2 features.
        if (partition_ids(i) == partition_ids(i + 1)) {
          non_empty_partitions.push_back(partition_boundaries.size() - 1);
        }
      }
    }
    if (partition_ids.size() > 0) {
      partition_boundaries.push_back(partition_ids.size());
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
    for (int root_idx = 0; root_idx < num_elements; ++root_idx) {
      float best_gain = std::numeric_limits<float>::lowest();
      int start_index = partition_boundaries[non_empty_partitions[root_idx]];
      int end_index = partition_boundaries[non_empty_partitions[root_idx] + 1];
      // First feature ID in each partition should be the bias feature.
      OP_REQUIRES(context, feature_ids(start_index, 0) == bias_feature_id_,
                  errors::InvalidArgument("Bias feature ID missing."));
      GradientStats root_gradient_stats(*gradients_t, *hessians_t, start_index);
      root_gradient_stats *= normalizer_ratio;
      NodeStats root_stats = ComputeNodeStats(root_gradient_stats);
      int32 best_feature_idx = 0;
      NodeStats best_right_node_stats(0);
      NodeStats best_left_node_stats(0);
      for (int64 feature_idx = start_index + 1; feature_idx < end_index;
           ++feature_idx) {
        GradientStats left_gradient_stats(*gradients_t, *hessians_t,
                                          feature_idx);
        left_gradient_stats *= normalizer_ratio;
        GradientStats right_gradient_stats =
            root_gradient_stats - left_gradient_stats;
        NodeStats left_stats = ComputeNodeStats(left_gradient_stats);
        NodeStats right_stats = ComputeNodeStats(right_gradient_stats);
        if (left_stats.gain + right_stats.gain > best_gain) {
          best_gain = left_stats.gain + right_stats.gain;
          best_left_node_stats = left_stats;
          best_right_node_stats = right_stats;
          best_feature_idx = feature_idx;
        }
      }
      SplitInfo split_info;
      auto* equality_split = split_info.mutable_split_node()
                                 ->mutable_categorical_id_binary_split();
      equality_split->set_feature_column(feature_column_group_id_);
      equality_split->set_feature_id(feature_ids(best_feature_idx, 0));
      auto* left_child = split_info.mutable_left_child();
      auto* right_child = split_info.mutable_right_child();
      FillLeaf(class_id, best_left_node_stats, left_child);
      FillLeaf(class_id, best_right_node_stats, right_child);
      split_info.SerializeToString(&output_splits(root_idx));
      gains(root_idx) =
          best_gain - root_stats.gain - tree_complexity_regularization_;
      output_partition_ids(root_idx) = partition_ids(start_index);
    }
  }

 private:
  int64 bias_feature_id_;
};

REGISTER_KERNEL_BUILDER(
    Name("BuildCategoricalEqualitySplits").Device(DEVICE_CPU),
    BuildCategoricalEqualitySplitsOp);

}  // namespace tensorflow
