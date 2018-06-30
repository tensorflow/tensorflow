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

namespace tensorflow {

REGISTER_RESOURCE_HANDLE_OP(BoostedTreesEnsembleResource);

REGISTER_OP("IsBoostedTreesEnsembleInitialized")
    .Input("tree_ensemble_handle: resource")
    .Output("is_initialized: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      c->set_output(0, c->Scalar());
      return Status::OK();
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
      return Status::OK();
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
      return Status::OK();
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
      return Status::OK();
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
      return Status::OK();
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
      return Status::OK();
    });

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
      shape_inference::ShapeHandle unused_input;
      for (int i = 0; i < num_bucketized_features; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i + 1), 1, &feature_shape));
        // Check that the shapes of all bucketized features are the same.
        TF_RETURN_IF_ERROR(c->Merge(c->input(1), feature_shape, &unused_input));
      }

      int logits_dimension;
      TF_RETURN_IF_ERROR(c->GetAttr("logits_dimension", &logits_dimension));
      auto logits_shape =
          c->MakeShape({c->Dim(feature_shape, 0), logits_dimension});
      // Logits.
      c->set_output(0, logits_shape);
      return Status::OK();
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
      shape_inference::ShapeHandle unused_input;
      for (int i = 0; i < num_bucketized_features; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i + 1), 1, &feature_shape));
        // Check that the shapes of all bucketized features are the same.
        TF_RETURN_IF_ERROR(c->Merge(c->input(1), feature_shape, &unused_input));
      }

      // Multi-class will be supported by modifying the proto.
      auto batch_size = c->MakeShape({c->Dim(feature_shape, 0)});
      c->set_output(0, batch_size);
      return Status::OK();
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
      return Status::OK();
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
      for (int i = 0; i < num_bucketized_features; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i + 3), 1, &feature_shape));
        TF_RETURN_IF_ERROR(
            c->Merge(c->input(i + 3), feature_shape, &unused_input));
      }
      // all inputs/outputs except logits should have same shape.
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), feature_shape, &unused_input));
      TF_RETURN_IF_ERROR(c->Merge(c->input(2), feature_shape, &unused_input));

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
      return Status::OK();
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
      return Status::OK();
    });

}  // namespace tensorflow
