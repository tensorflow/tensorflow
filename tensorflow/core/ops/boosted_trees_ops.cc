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

#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_RESOURCE_HANDLE_OP(BoostedTreesEnsembleResource);

REGISTER_OP("IsBoostedTreesEnsembleInitialized")
    .Input("tree_ensemble_handle: resource")
    .Output("is_initialized: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      c->set_output(0, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesCalculateBestGainsPerFeature")
    .Input("node_id_range: int32")
    .Input("stats_summary_list: num_features * float32")
    .Input("l1: float")
    .Input("l2: float")
    .Input("tree_complexity: float")
    .Input("min_node_weight: float")
    .Attr("max_splits: int >= 1")
    .Attr("num_features: int >= 1")  // not passed but populated automatically.
    .Output("node_ids_list: num_features * int32")
    .Output("gains_list: num_features * float32")
    .Output("thresholds_list: num_features * int32")
    .Output("left_node_contribs_list: num_features * float32")
    .Output("right_node_contribs_list: num_features * float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Confirms the rank of the inputs and sets the shape of the outputs.
      int max_splits;
      int num_features;
      TF_RETURN_IF_ERROR(c->GetAttr("max_splits", &max_splits));
      TF_RETURN_IF_ERROR(c->GetAttr("num_features", &num_features));
      shape_inference::ShapeHandle node_id_range_shape;
      shape_inference::ShapeHandle unused_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &node_id_range_shape));
      TF_RETURN_IF_ERROR(
          c->Merge(node_id_range_shape, c->MakeShape({2}), &unused_shape));
      // Checks that all stats summary entries are of the same shape.
      shape_inference::ShapeHandle summary_shape_base;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &summary_shape_base));
      TF_RETURN_IF_ERROR(c->Merge(summary_shape_base,
                                  c->MakeShape({max_splits, -1, 2}),
                                  &unused_shape));
      for (int i = 1; i < num_features; ++i) {
        shape_inference::ShapeHandle summary_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1 + i), 3, &summary_shape));
        TF_RETURN_IF_ERROR(
            c->Merge(summary_shape_base, summary_shape, &unused_shape));
      }
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(num_features + 1), 0, &unused_shape));
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(num_features + 2), 0, &unused_shape));
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(num_features + 3), 0, &unused_shape));
      // Sets the output lists.
      std::vector<shape_inference::ShapeHandle> output_shapes_vec(
          num_features, c->MakeShape({-1}));
      TF_RETURN_IF_ERROR(c->set_output("node_ids_list", output_shapes_vec));
      TF_RETURN_IF_ERROR(c->set_output("gains_list", output_shapes_vec));
      TF_RETURN_IF_ERROR(c->set_output("thresholds_list", output_shapes_vec));
      std::vector<shape_inference::ShapeHandle> output_shapes_contribs(
          num_features, c->MakeShape({-1, 1}));
      TF_RETURN_IF_ERROR(
          c->set_output("left_node_contribs_list", output_shapes_contribs));
      TF_RETURN_IF_ERROR(
          c->set_output("right_node_contribs_list", output_shapes_contribs));
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesCalculateBestFeatureSplit")
    .Input("node_id_range: int32")
    .Input("stats_summary: float32")
    .Input("l1: float")
    .Input("l2: float")
    .Input("tree_complexity: float")
    .Input("min_node_weight: float")
    .Attr("logits_dimension: int >= 1")
    .Attr("split_type: {'inequality', 'equality'} = 'inequality'")
    .Output("node_ids: int32")
    .Output("gains: float32")
    .Output("feature_dimensions: int32")
    .Output("thresholds: int32")
    .Output("left_node_contribs: float32")
    .Output("right_node_contribs: float32")
    .Output("split_with_default_directions: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle node_id_range_shape;
      shape_inference::ShapeHandle unused_shape;
      // node id range is rank 1 with 2 values.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &node_id_range_shape));
      TF_RETURN_IF_ERROR(
          c->Merge(node_id_range_shape, c->MakeShape({2}), &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused_shape));
      ShapeHandle rank_1_output_shape = c->MakeShape({c->UnknownDim()});
      c->set_output(0, rank_1_output_shape);
      c->set_output(1, rank_1_output_shape);
      c->set_output(2, rank_1_output_shape);
      c->set_output(3, rank_1_output_shape);
      c->set_output(6, rank_1_output_shape);
      int logits_dimension;
      TF_RETURN_IF_ERROR(c->GetAttr("logits_dimension", &logits_dimension));
      ShapeHandle contribs_output_shape =
          c->MakeShape({c->UnknownDim(), logits_dimension});
      c->set_output(4, contribs_output_shape);
      c->set_output(5, contribs_output_shape);
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesCalculateBestFeatureSplitV2")
    .Input("node_id_range: int32")
    .Input("stats_summaries_list: num_features * float32")
    .Input("split_types: string")
    .Input("candidate_feature_ids: int32")
    .Input("l1: float")
    .Input("l2: float")
    .Input("tree_complexity: float")
    .Input("min_node_weight: float")
    .Attr("num_features: int >= 1")  // not passed but populated automatically.
    .Attr("logits_dimension: int >= 1")
    .Output("node_ids: int32")
    .Output("gains: float32")
    .Output("feature_ids: int32")
    .Output("feature_dimensions: int32")
    .Output("thresholds: int32")
    .Output("left_node_contribs: float32")
    .Output("right_node_contribs: float32")
    .Output("split_with_default_directions: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Attributes.
      int num_features;
      TF_RETURN_IF_ERROR(c->GetAttr("num_features", &num_features));
      int logits_dimension;
      TF_RETURN_IF_ERROR(c->GetAttr("logits_dimension", &logits_dimension));
      // Inputs.
      shape_inference::ShapeHandle unused_shape;
      // node id range is rank 1 with 2 values.
      shape_inference::ShapeHandle node_id_range_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &node_id_range_shape));
      TF_RETURN_IF_ERROR(
          c->Merge(node_id_range_shape, c->MakeShape({2}), &unused_shape));
      // Stats summary validation.
      shape_inference::ShapeHandle summary_shape_base;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &summary_shape_base));
      // All stats summary entries are of the same shape.
      for (int i = 1; i < num_features; ++i) {
        shape_inference::ShapeHandle summary_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1 + i), 4, &summary_shape));
        TF_RETURN_IF_ERROR(
            c->Merge(summary_shape_base, summary_shape, &unused_shape));
      }
      // Validate rank 1 split_types.
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(1 + num_features), 1, &unused_shape));
      // Validate rank 1 feature_ids.
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(2 + num_features), 1, &unused_shape));
      // Validate rank 0: l1, l2, tree_complexity, min_node_weight.
      for (int i = 0; i < 4; ++i) {
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(3 + num_features + i), 0, &unused_shape));
      }
      // Output shapes.
      ShapeHandle rank_1_output_shape = c->MakeShape({c->UnknownDim()});
      c->set_output(0, rank_1_output_shape);
      c->set_output(1, rank_1_output_shape);
      c->set_output(2, rank_1_output_shape);
      c->set_output(3, rank_1_output_shape);
      c->set_output(4, rank_1_output_shape);
      ShapeHandle contribs_output_shape =
          c->MakeShape({c->UnknownDim(), logits_dimension});
      c->set_output(5, contribs_output_shape);
      c->set_output(6, contribs_output_shape);
      c->set_output(7, rank_1_output_shape);
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesSparseCalculateBestFeatureSplit")
    .Input("node_id_range: int32")
    .Input("stats_summary_indices: int32")
    .Input("stats_summary_values: float")
    .Input("stats_summary_shape: int32")
    .Input("l1: float")
    .Input("l2: float")
    .Input("tree_complexity: float")
    .Input("min_node_weight: float")
    .Attr("logits_dimension: int >= 1")
    .Attr("split_type: {'inequality'} = 'inequality'")
    .Output("node_ids: int32")
    .Output("gains: float32")
    .Output("feature_dimensions: int32")
    .Output("thresholds: int32")
    .Output("left_node_contribs: float32")
    .Output("right_node_contribs: float32")
    .Output("split_with_default_directions: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle node_id_range_shape;
      shape_inference::ShapeHandle unused_shape;
      // node id range is rank 1 with 2 values.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &node_id_range_shape));
      TF_RETURN_IF_ERROR(
          c->Merge(node_id_range_shape, c->MakeShape({2}), &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &unused_shape));
      shape_inference::ShapeHandle summary_shape;
      TF_RETURN_IF_ERROR(
          c->Merge(summary_shape, c->MakeShape({4}), &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused_shape));
      ShapeHandle rank_1_output_shape = c->MakeShape({c->UnknownDim()});
      c->set_output(0, rank_1_output_shape);
      c->set_output(1, rank_1_output_shape);
      c->set_output(2, rank_1_output_shape);
      c->set_output(3, rank_1_output_shape);
      c->set_output(6, rank_1_output_shape);
      int logits_dimension;
      TF_RETURN_IF_ERROR(c->GetAttr("logits_dimension", &logits_dimension));
      ShapeHandle contribs_output_shape =
          c->MakeShape({c->UnknownDim(), logits_dimension});
      c->set_output(4, contribs_output_shape);
      c->set_output(5, contribs_output_shape);
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesCreateEnsemble")
    .Input("tree_ensemble_handle: resource")
    .Input("stamp_token: int64")
    .Input("tree_ensemble_serialized: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_input));
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesDeserializeEnsemble")
    .Input("tree_ensemble_handle: resource")
    .Input("stamp_token: int64")
    .Input("tree_ensemble_serialized: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_input));
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesGetEnsembleStates")
    .Input("tree_ensemble_handle: resource")
    .Output("stamp_token: int64")
    .Output("num_trees: int32")
    .Output("num_finalized_trees: int32")
    .Output("num_attempted_layers: int32")
    .Output("last_layer_nodes_range: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      c->set_output(3, c->Scalar());
      c->set_output(4, c->Vector(2));
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesMakeStatsSummary")
    .Input("node_ids: int32")
    .Input("gradients: float")
    .Input("hessians: float")
    .Input("bucketized_features_list: num_features * int32")
    .Attr("max_splits: int >= 1")
    .Attr("num_buckets: int >= 1")
    .Attr("num_features: int >= 1")
    .Output("stats_summary: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Sets the shape of the output as a Rank 4 Tensor.
      int max_splits;
      int num_buckets;
      int num_features;
      TF_RETURN_IF_ERROR(c->GetAttr("max_splits", &max_splits));
      TF_RETURN_IF_ERROR(c->GetAttr("num_buckets", &num_buckets));
      TF_RETURN_IF_ERROR(c->GetAttr("num_features", &num_features));
      shape_inference::ShapeHandle node_ids_shape;
      shape_inference::ShapeHandle gradients_shape;
      shape_inference::ShapeHandle hessians_shape;
      shape_inference::ShapeHandle bucketized_feature_shape;
      shape_inference::ShapeHandle unused_shape;
      shape_inference::DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &node_ids_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &gradients_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &hessians_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(node_ids_shape, 0),
                                  c->Dim(gradients_shape, 0), &unused_dim));
      TF_RETURN_IF_ERROR(
          c->Merge(gradients_shape, hessians_shape, &unused_shape));
      for (int f = 0; f < num_features; ++f) {
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(3 + f), 1, &bucketized_feature_shape));
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(node_ids_shape, 0),
                                    c->Dim(bucketized_feature_shape, 0),
                                    &unused_dim));
      }
      c->set_output(0,
                    c->MakeShape({num_features, max_splits, num_buckets, 2}));
      return absl::OkStatus();
    });

// V2 of BoostedTreesMakeStatsSummary. Supports multi-dim dense Tensor and
// multi class.
REGISTER_OP("BoostedTreesAggregateStats")
    .Input("node_ids: int32")
    .Input("gradients: float")
    .Input("hessians: float")
    .Input("feature: int32")
    .Attr("max_splits: int >= 1")
    .Attr("num_buckets: int >= 1")
    .Output("stats_summary: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Sets the shape of the output as a Rank 4 Tensor.
      int max_splits;
      int num_buckets;
      TF_RETURN_IF_ERROR(c->GetAttr("max_splits", &max_splits));
      TF_RETURN_IF_ERROR(c->GetAttr("num_buckets", &num_buckets));

      shape_inference::ShapeHandle node_ids_shape;
      shape_inference::ShapeHandle gradients_shape;
      shape_inference::ShapeHandle hessians_shape;
      shape_inference::ShapeHandle feature_shape;

      shape_inference::DimensionHandle batch_size = c->Dim(c->input(0), 0);

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &node_ids_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &gradients_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &hessians_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &feature_shape));

      // Verify all three inputs have same first dimension, i.e., batch_size.
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(gradients_shape, 0),
                                  c->Dim(node_ids_shape, 0), &batch_size));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(hessians_shape, 0),
                                  c->Dim(node_ids_shape, 0), &batch_size));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(feature_shape, 0),
                                  c->Dim(node_ids_shape, 0), &batch_size));

      DimensionHandle logits_dim = c->Dim(c->input(1), 1);
      DimensionHandle hessian_dim = c->Dim(c->input(2), 1);
      DimensionHandle feature_dim = c->Dim(c->input(3), 1);
      DimensionHandle stats_dim;
      TF_RETURN_IF_ERROR(c->Add(logits_dim, hessian_dim, &stats_dim));
      c->set_output(0, c->MakeShape({max_splits, feature_dim,
                                     num_buckets + 1,  // +1 for missing bucket.
                                     stats_dim}));
      return absl::OkStatus();
    });

// Sparse Version of BoostedTreesAggregatesStats.
REGISTER_OP("BoostedTreesSparseAggregateStats")
    .Input("node_ids: int32")
    .Input("gradients: float")
    .Input("hessians: float")
    .Input("feature_indices: int32")
    .Input("feature_values: int32")
    .Input("feature_shape: int32")
    .Attr("max_splits: int >= 1")
    .Attr("num_buckets: int >= 1")
    .Output("stats_summary_indices: int32")
    .Output("stats_summary_values: float")
    .Output("stats_summary_shape: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int max_splits;
      int num_buckets;
      TF_RETURN_IF_ERROR(c->GetAttr("max_splits", &max_splits));
      TF_RETURN_IF_ERROR(c->GetAttr("num_buckets", &num_buckets));

      shape_inference::ShapeHandle node_ids_shape;
      shape_inference::ShapeHandle gradients_shape;
      shape_inference::ShapeHandle hessians_shape;
      shape_inference::ShapeHandle feature_indices_shape;
      shape_inference::ShapeHandle feature_values_shape;
      shape_inference::ShapeHandle feature_shape;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &node_ids_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &gradients_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &hessians_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &feature_indices_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &feature_values_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &feature_shape));

      shape_inference::DimensionHandle batch_size = c->Dim(c->input(0), 0);
      shape_inference::DimensionHandle num_entries;

      // Verify all inputs have same first dimension, i.e., batch_size.
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(gradients_shape, 0),
                                  c->Dim(node_ids_shape, 0), &batch_size));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(hessians_shape, 0),
                                  c->Dim(node_ids_shape, 0), &batch_size));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(feature_indices_shape, 0),
                                  c->Dim(feature_values_shape, 0),
                                  &num_entries));

      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(feature_shape, 0), 2, &unused));

      DimensionHandle logits_dim = c->Dim(c->input(1), 1);
      DimensionHandle hessian_dim = c->Dim(c->input(2), 1);
      DimensionHandle stats_dim;
      TF_RETURN_IF_ERROR(c->Add(logits_dim, hessian_dim, &stats_dim));

      c->set_output(0, c->MakeShape({c->UnknownDim(), 4}));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->MakeShape({4}));
      return absl::OkStatus();
    });

// TODO(nponomareva): when/if creating the new op for unbucketized data, rename
// bucketized_features to features.
REGISTER_OP("BoostedTreesPredict")
    .Input("tree_ensemble_handle: resource")
    .Input("bucketized_features: num_bucketized_features * int32")
    .Attr("num_bucketized_features: int >= 1")  // Inferred.
    .Attr("logits_dimension: int")
    .Output("logits: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle feature_shape;
      int num_bucketized_features;
      TF_RETURN_IF_ERROR(
          c->GetAttr("num_bucketized_features", &num_bucketized_features));
      shape_inference::DimensionHandle batch_size = c->Dim(c->input(1), 0);
      for (int i = 0; i < num_bucketized_features; ++i) {
        TF_RETURN_IF_ERROR(
            c->WithRankAtMost(c->input(i + 1), 2, &feature_shape));
        // Check that all bucketized features have the same batch size.
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(1), 0),
                                    c->Dim(c->input(i + 1), 0), &batch_size));
      }

      int logits_dimension;
      TF_RETURN_IF_ERROR(c->GetAttr("logits_dimension", &logits_dimension));
      auto logits_shape =
          c->MakeShape({c->Dim(feature_shape, 0), logits_dimension});
      // Logits.
      c->set_output(0, logits_shape);
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesExampleDebugOutputs")
    .Input("tree_ensemble_handle: resource")
    .Input("bucketized_features: num_bucketized_features * int32")
    .Attr("num_bucketized_features: int >= 1")  // Inferred.
    .Attr("logits_dimension: int")
    .Output("examples_debug_outputs_serialized: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle feature_shape;
      int num_bucketized_features;
      TF_RETURN_IF_ERROR(
          c->GetAttr("num_bucketized_features", &num_bucketized_features));
      shape_inference::DimensionHandle batch_dim = c->Dim(c->input(1), 0);
      for (int i = 0; i < num_bucketized_features; ++i) {
        TF_RETURN_IF_ERROR(
            c->WithRankAtMost(c->input(i + 1), 2, &feature_shape));
        // Check that all bucketized features have the same batch size.
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(1), 0),
                                    c->Dim(c->input(i + 1), 0), &batch_dim));
      }

      // Multi-class will be supported by modifying the proto.
      auto batch_size = c->MakeShape({c->Dim(feature_shape, 0)});
      c->set_output(0, batch_size);
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesSerializeEnsemble")
    .Input("tree_ensemble_handle: resource")
    .Output("stamp_token: int64")
    .Output("tree_ensemble_serialized: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesTrainingPredict")
    .Input("tree_ensemble_handle: resource")
    .Input("cached_tree_ids: int32")
    .Input("cached_node_ids: int32")
    .Input("bucketized_features: num_bucketized_features * int32")
    .Attr("num_bucketized_features: int >= 1")
    .Attr("logits_dimension: int")
    .Output("partial_logits: float")
    .Output("tree_ids: int32")
    .Output("node_ids: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle feature_shape;
      int num_bucketized_features;
      TF_RETURN_IF_ERROR(
          c->GetAttr("num_bucketized_features", &num_bucketized_features));

      shape_inference::ShapeHandle unused_input;
      shape_inference::DimensionHandle batch_size = c->Dim(c->input(3), 0);
      for (int i = 0; i < num_bucketized_features; ++i) {
        TF_RETURN_IF_ERROR(
            c->WithRankAtMost(c->input(i + 3), 2, &feature_shape));
        TF_RETURN_IF_ERROR(
            c->Merge(c->input(i + 3), feature_shape, &unused_input));
      }
      shape_inference::ShapeHandle tree_ids_shape;
      shape_inference::ShapeHandle node_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &tree_ids_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &node_ids_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(tree_ids_shape, 0),
                                  c->Dim(node_ids_shape, 0), &batch_size));

      int logits_dimension;
      TF_RETURN_IF_ERROR(c->GetAttr("logits_dimension", &logits_dimension));
      auto logits_shape =
          c->MakeShape({c->Dim(feature_shape, 0), logits_dimension});
      // Partial logits.
      c->set_output(0, logits_shape);
      // Tree ids.
      c->set_output(1, c->MakeShape({c->Dim(feature_shape, 0)}));
      // Node ids.
      c->set_output(2, c->MakeShape({c->Dim(feature_shape, 0)}));
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesUpdateEnsemble")
    .Input("tree_ensemble_handle: resource")
    .Input("feature_ids: int32")
    .Input("node_ids: num_features * int32")
    .Input("gains: num_features * float")
    .Input("thresholds: num_features * int32")
    .Input("left_node_contribs: num_features * float")
    .Input("right_node_contribs: num_features * float")
    .Input("max_depth: int32")
    .Input("learning_rate: float")
    .Attr("pruning_mode: int >=0")
    .Attr("num_features: int >= 0")  // Inferred.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape_handle;
      int num_features;
      TF_RETURN_IF_ERROR(c->GetAttr("num_features", &num_features));

      // Feature_ids, should be one for each feature.
      shape_inference::ShapeHandle feature_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &feature_ids_shape));
      TF_RETURN_IF_ERROR(
          c->Merge(c->input(1), c->Vector(num_features), &shape_handle));

      for (int i = 0; i < num_features; ++i) {
        // Node ids.
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i + 2), 1, &shape_handle));
        auto shape_rank_1 = c->MakeShape({c->Dim(shape_handle, 0)});
        auto shape_rank_2 = c->MakeShape({c->Dim(shape_handle, 0), 1});

        // Gains.
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(i + num_features + 2), 1, &shape_handle));
        // TODO(nponomareva): replace this with input("name",vector of shapes).
        TF_RETURN_IF_ERROR(c->Merge(c->input(i + num_features + 2),
                                    shape_rank_1, &shape_handle));
        // Thresholds.
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(i + num_features * 2 + 2), 1, &shape_handle));
        TF_RETURN_IF_ERROR(c->Merge(c->input(i + num_features * 2 + 2),
                                    shape_rank_1, &shape_handle));
        // Left and right node contribs.
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(i + num_features * 3 + 2), 2, &shape_handle));
        TF_RETURN_IF_ERROR(c->Merge(c->input(i + num_features * 3 + 2),
                                    shape_rank_2, &shape_handle));
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(i + num_features * 4 + 2), 2, &shape_handle));
        TF_RETURN_IF_ERROR(c->Merge(c->input(i + num_features * 4 + 2),
                                    shape_rank_2, &shape_handle));
      }
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesUpdateEnsembleV2")
    .Input("tree_ensemble_handle: resource")
    .Input("feature_ids: num_groups * int32")
    .Input("dimension_ids: num_features * int32")
    .Input("node_ids: num_features * int32")
    .Input("gains: num_features * float")
    .Input("thresholds: num_features * int32")
    .Input("left_node_contribs: num_features * float")
    .Input("right_node_contribs: num_features * float")
    .Input("split_types: num_features * string")
    .Input("max_depth: int32")
    .Input("learning_rate: float")
    .Input("pruning_mode: int32")
    .Attr("num_features: int >= 0")  // Inferred.
    .Attr("logits_dimension: int = 1")
    .Attr("num_groups: int = 1")  // Inferred; number of groups to process.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_features;
      int logits_dimension;
      int num_groups;
      TF_RETURN_IF_ERROR(c->GetAttr("num_features", &num_features));
      TF_RETURN_IF_ERROR(c->GetAttr("logits_dimension", &logits_dimension));
      TF_RETURN_IF_ERROR(c->GetAttr("num_groups", &num_groups));
      // num_features was kept for backwards compatibility reasons. It now
      // represents number of groups.
      DCHECK_EQ(num_features, num_groups);
      shape_inference::ShapeHandle shape_handle;
      for (int i = 0; i < num_groups; ++i) {
        int offset = i + 1;

        // Feature ids
        TF_RETURN_IF_ERROR(c->WithRank(c->input(offset), 1, &shape_handle));
        // TODO(nponomareva): replace this with input("name",vector of shapes).
        auto shape_rank_1 = c->MakeShape({c->Dim(shape_handle, 0)});
        TF_RETURN_IF_ERROR(
            c->Merge(c->input(offset), shape_rank_1, &shape_handle));

        // Dimension ids.
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(offset + num_features), 1, &shape_handle));
        TF_RETURN_IF_ERROR(
            c->Merge(c->input(offset), shape_rank_1, &shape_handle));

        // Node ids.
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(offset + num_features * 2), 1, &shape_handle));
        TF_RETURN_IF_ERROR(
            c->Merge(c->input(offset), shape_rank_1, &shape_handle));

        // Gains.
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(offset + num_features * 3), 1, &shape_handle));
        TF_RETURN_IF_ERROR(c->Merge(c->input(offset + num_features * 3),
                                    shape_rank_1, &shape_handle));

        // Thresholds.
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(offset + num_features * 4), 1, &shape_handle));
        TF_RETURN_IF_ERROR(c->Merge(c->input(offset + num_features * 4),
                                    shape_rank_1, &shape_handle));

        // Left and right node contribs.
        auto shape_rank_2 =
            c->MakeShape({c->Dim(shape_handle, 0), logits_dimension});
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(offset + num_features * 5), 2, &shape_handle));
        TF_RETURN_IF_ERROR(c->Merge(c->input(offset + num_features * 5),
                                    shape_rank_2, &shape_handle));
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(offset + num_features * 6), 2, &shape_handle));
        TF_RETURN_IF_ERROR(c->Merge(c->input(offset + num_features * 6),
                                    shape_rank_2, &shape_handle));

        // Split types.
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(offset + num_features * 7), 1, &shape_handle));
        TF_RETURN_IF_ERROR(c->Merge(c->input(offset + num_features * 7),
                                    shape_rank_1, &shape_handle));
      }
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesCenterBias")
    .Input("tree_ensemble_handle: resource")
    .Input("mean_gradients: float")
    .Input("mean_hessians: float")
    // Regularization-related.
    .Input("l1: float")
    .Input("l2: float")
    .Output("continue_centering: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle gradients_shape;
      shape_inference::ShapeHandle hessians_shape;
      shape_inference::ShapeHandle unused_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &gradients_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &hessians_shape));
      TF_RETURN_IF_ERROR(
          c->Merge(gradients_shape, hessians_shape, &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused_shape));

      c->set_output(0, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_RESOURCE_HANDLE_OP(BoostedTreesQuantileStreamResource);

REGISTER_OP("IsBoostedTreesQuantileStreamResourceInitialized")
    .Input("quantile_stream_resource_handle: resource")
    .Output("is_initialized: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      c->set_output(0, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesCreateQuantileStreamResource")
    .Attr("max_elements: int = 1099511627776")  // 1 << 40
    .Input("quantile_stream_resource_handle: resource")
    .Input("epsilon: float")
    .Input("num_streams: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_input));
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesMakeQuantileSummaries")
    .Attr("num_features: int >= 0")
    .Input("float_values: num_features * float")
    .Input("example_weights: float")
    .Input("epsilon: float")
    .Output("summaries: num_features * float")
    .SetShapeFn([](InferenceContext* c) {
      int num_features;
      TF_RETURN_IF_ERROR(c->GetAttr("num_features", &num_features));
      ShapeHandle example_weights_shape;
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(num_features), 1, &example_weights_shape));
      for (int i = 0; i < num_features; ++i) {
        ShapeHandle feature_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &feature_shape));
        // the columns are value, weight, min_rank, max_rank.
        c->set_output(i, c->MakeShape({c->UnknownDim(), 4}));
      }
      // epsilon must be a scalar.
      ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(num_features + 1), 0, &unused_input));
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesFlushQuantileSummaries")
    .Attr("num_features: int >= 0")
    .Input("quantile_stream_resource_handle: resource")
    .Output("summaries: num_features * float")
    .SetShapeFn([](InferenceContext* c) {
      int num_features;
      TF_RETURN_IF_ERROR(c->GetAttr("num_features", &num_features));
      for (int i = 0; i < num_features; ++i) {
        // the columns are value, weight, min_rank, max_rank.
        c->set_output(i, c->MakeShape({c->UnknownDim(), 4}));
      }
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesQuantileStreamResourceAddSummaries")
    .Attr("num_features: int >= 0")
    .Input("quantile_stream_resource_handle: resource")
    .Input("summaries: num_features * float")
    .SetShapeFn([](InferenceContext* c) {
      int num_features;
      TF_RETURN_IF_ERROR(c->GetAttr("num_features", &num_features));
      // resource handle must be a scalar.
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      // each summary must be rank 2.
      for (int i = 1; i < num_features + 1; i++) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 2, &unused_input));
      }
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesQuantileStreamResourceDeserialize")
    .Attr("num_streams: int")
    .Input("quantile_stream_resource_handle: resource")
    .Input("bucket_boundaries: num_streams * float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesQuantileStreamResourceFlush")
    .Attr("generate_quantiles: bool = False")
    .Input("quantile_stream_resource_handle: resource")
    .Input("num_buckets: int64")
    .SetShapeFn([](InferenceContext* c) {
      // All the inputs are scalars.
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesQuantileStreamResourceGetBucketBoundaries")
    .Attr("num_features: int >= 0")
    .Input("quantile_stream_resource_handle: resource")
    .Output("bucket_boundaries: num_features * float")
    .SetShapeFn([](InferenceContext* c) {
      int num_features;
      TF_RETURN_IF_ERROR(c->GetAttr("num_features", &num_features));
      shape_inference::ShapeHandle unused_input;
      // resource handle must be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      for (int i = 0; i < num_features; i++) {
        c->set_output(i, c->Vector(c->UnknownDim()));
      }
      return absl::OkStatus();
    });

REGISTER_OP("BoostedTreesBucketize")
    .Attr("num_features: int >= 0")
    .Input("float_values: num_features * float")
    .Input("bucket_boundaries: num_features * float")
    .Output("buckets: num_features * int32")
    .SetShapeFn([](InferenceContext* c) {
      int num_features;
      TF_RETURN_IF_ERROR(c->GetAttr("num_features", &num_features));
      ShapeHandle feature_shape;
      DimensionHandle unused_dim;
      for (int i = 0; i < num_features; i++) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &feature_shape));
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(feature_shape, 0),
                                    c->Dim(c->input(0), 0), &unused_dim));
      }
      // Bucketized result should have same dimension as input.
      for (int i = 0; i < num_features; i++) {
        c->set_output(i, c->MakeShape({c->Dim(c->input(i), 0)}));
      }
      return absl::OkStatus();
    });

}  // namespace tensorflow
