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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

REGISTER_OP("BuildDenseInequalitySplits")
    .Attr("feature_column_group_id: int")
    .Attr("l1_regularization: float")
    .Attr("l2_regularization: float")
    .Attr("tree_complexity_regularization: float")
    .Attr("min_node_weight: float")
    .Attr("multiclass_strategy: int")
    .Input("num_minibatches: int64")
    .Input("partition_ids: int32")
    .Input("bucket_ids: int64")
    .Input("gradients: float32")
    .Input("hessians: float32")
    .Input("bucket_boundaries: float32")
    .Input("class_id: int32")
    .Output("output_partition_ids: int32")
    .Output("gains: float32")
    .Output("split_infos: string")
    .SetShapeFn([](InferenceContext* c) {
      DimensionHandle unused_dim;
      ShapeHandle unused_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_shape));

      ShapeHandle partition_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &partition_ids_shape));
      ShapeHandle bucket_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &bucket_ids_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(bucket_ids_shape, 0), &unused_dim));
      ShapeHandle gradients_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 1, &gradients_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(gradients_shape, 0), &unused_dim));
      ShapeHandle hessians_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(4), 1, &hessians_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(hessians_shape, 0), &unused_dim));
      ShapeHandle bucket_boundaries_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &bucket_boundaries_shape));
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Find the split that has the best gain for the accumulated stats.

num_minibatches: A scalar, the number of times per example gradients & hessians
    were accumulated. The stats are divided by this to get per example stats.
partition_ids: A rank 1 tensor of partition IDs.
bucket_ids: A rank 1 tensor of buckets IDs.
gradients: A rank 1 tensor of gradients.
hessians: A rank 1 tensor of hessians.
bucket_boundaries: A rank 1 tensor, thresholds that were used for bucketization.
output_partition_ids: A rank 1 tensor, the partition IDs that we created splits
    for.
gains: A rank 1 tensor, for the computed gain for the created splits.
split_infos: A rank 1 tensor of serialized protos which contains the
    `SplitInfo`s.
)doc");

REGISTER_OP("BuildSparseInequalitySplits")
    .Attr("feature_column_group_id: int")
    .Attr("bias_feature_id: int")
    .Attr("l1_regularization: float")
    .Attr("l2_regularization: float")
    .Attr("tree_complexity_regularization: float")
    .Attr("min_node_weight: float")
    .Attr("multiclass_strategy: int")
    .Input("num_minibatches: int64")
    .Input("partition_ids: int32")
    .Input("bucket_ids: int64")
    .Input("gradients: float32")
    .Input("hessians: float32")
    .Input("bucket_boundaries: float32")
    .Input("class_id: int32")
    .Output("output_partition_ids: int32")
    .Output("gains: float32")
    .Output("split_infos: string")
    .SetShapeFn([](InferenceContext* c) {
      DimensionHandle unused_dim;
      ShapeHandle unused_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_shape));

      ShapeHandle partition_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &partition_ids_shape));
      ShapeHandle bucket_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &bucket_ids_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(bucket_ids_shape, 0), &unused_dim));
      ShapeHandle gradients_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 1, &gradients_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(gradients_shape, 0), &unused_dim));
      ShapeHandle hessians_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(4), 1, &hessians_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(hessians_shape, 0), &unused_dim));
      ShapeHandle bucket_boundaries_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &bucket_boundaries_shape));
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Find the split that has the best gain for the accumulated stats.

num_minibatches: A scalar, the number of times per example gradients & hessians
    were accumulated. The stats are divided by this to get per example stats.
partition_ids: A rank 1 tensor of partition IDs.
bucket_ids: A rank 1 tensor of buckets IDs.
gradients: A rank 1 tensor of gradients.
hessians: A rank 1 tensor of hessians.
bucket_boundaries: A rank 1 tensor, thresholds that were used for bucketization.
output_partition_ids: A rank 1 tensor, the partition IDs that we created splits
    for.
gains: A rank 1 tensor, for the computed gain for the created splits.
split_infos: A rank 1 tensor of serialized protos which contains the
    `SplitInfo`s.
)doc");

REGISTER_OP("BuildCategoricalEqualitySplits")
    .Attr("feature_column_group_id: int")
    .Attr("bias_feature_id: int")
    .Attr("l1_regularization: float")
    .Attr("l2_regularization: float")
    .Attr("tree_complexity_regularization: float")
    .Attr("min_node_weight: float")
    .Attr("multiclass_strategy: int")
    .Input("num_minibatches: int64")
    .Input("partition_ids: int32")
    .Input("feature_ids: int64")
    .Input("gradients: float32")
    .Input("hessians: float32")
    .Input("class_id: int32")
    .Output("output_partition_ids: int32")
    .Output("gains: float32")
    .Output("split_infos: string")
    .SetShapeFn([](InferenceContext* c) {
      DimensionHandle unused_dim;
      ShapeHandle unused_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_shape));

      ShapeHandle partition_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &partition_ids_shape));
      ShapeHandle bucket_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &bucket_ids_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(bucket_ids_shape, 0), &unused_dim));
      ShapeHandle gradients_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 1, &gradients_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(gradients_shape, 0), &unused_dim));
      ShapeHandle hessians_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(4), 1, &hessians_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(hessians_shape, 0), &unused_dim));
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Find the split that has the best gain for the accumulated stats.

num_minibatches: A scalar, the number of times per example gradients & hessians
    were accumulated. The stats are divided by this to get per example stats.
partition_ids: A rank 1 tensor of partition IDs.
feature_ids: A rank 1 tensor of feature IDs.
gradients: A rank 1 tensor of gradients.
hessians: A rank 1 tensor of hessians.
output_partition_ids: A rank 1 tensor, the partition IDs that we created splits
    for.
gains: A rank 1 tensor, for the computed gain for the created splits.
split_infos: A rank 1 tensor of serialized protos which contains the
    `SplitInfo`s.
)doc");

}  // namespace tensorflow
   // namespace tensorflow
