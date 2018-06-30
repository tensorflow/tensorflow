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

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("BuildDenseInequalitySplits")
    .Input("num_minibatches: int64")
    .Input("partition_ids: int32")
    .Input("bucket_ids: int64")
    .Input("gradients: float32")
    .Input("hessians: float32")
    .Input("bucket_boundaries: float32")
    .Input("class_id: int32")
    .Input("feature_column_group_id: int32")
    .Input("l1_regularization: float")
    .Input("l2_regularization: float")
    .Input("tree_complexity_regularization: float")
    .Input("min_node_weight: float")
    .Input("multiclass_strategy: int32")
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
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &bucket_ids_shape));
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
bucket_ids: A rank 2 tensor of buckets IDs and dimensions.
gradients: A rank 1 tensor of gradients.
hessians: A rank 1 tensor of hessians.
bucket_boundaries: A rank 1 tensor, thresholds that were used for bucketization.
class_id: A scalar, the class id for which we're building the splits.
feature_column_group_id: A scalar, the index of the feature we are spiltting on.
l1_regularization: A scalar, which specifies the l1 regularization term.
l2_regularization: A scalar, which specifies the l2 regularization term.
tree_complexity_regularization: A scalar, which specifies the tree complexity
    regularization term.
min_node_weight: A scalar, minimum sum of example hessian needed in a child.
    If a split results in a leaf node with a smaller value, the split will not
    be considered.
multiclass_strategy: A scalar, specifying the multiclass handling strategy.
    See LearnerConfig.MultiClassStrategy for valid values.
output_partition_ids: A rank 1 tensor, the partition IDs that we created splits
    for.
gains: A rank 1 tensor, for the computed gain for the created splits.
split_infos: A rank 1 tensor of serialized protos which contains the
    `SplitInfo`s.
)doc");

REGISTER_OP("BuildSparseInequalitySplits")
    .Input("num_minibatches: int64")
    .Input("partition_ids: int32")
    .Input("bucket_ids: int64")
    .Input("gradients: float32")
    .Input("hessians: float32")
    .Input("bucket_boundaries: float32")
    .Input("class_id: int32")
    .Input("feature_column_group_id: int32")
    .Input("bias_feature_id: int64")
    .Input("l1_regularization: float")
    .Input("l2_regularization: float")
    .Input("tree_complexity_regularization: float")
    .Input("min_node_weight: float")
    .Input("multiclass_strategy: int32")
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
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &bucket_ids_shape));
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
Find the split that has the best gain for the accumulated stats for a particular
feature column.

num_minibatches: A scalar, the number of times per example gradients & hessians
    were accumulated. The stats are divided by this to get per example stats.
partition_ids: A rank 2 tensor of partition IDs for each dimension of feature column.
bucket_ids: A rank 2 tensor of buckets IDs and dimensions.
gradients: A rank 1 tensor of gradients.
hessians: A rank 1 tensor of hessians.
bucket_boundaries: A rank 1 tensor, thresholds that were used for bucketization.
class_id: A scalar, the class id for which we're building the splits.
feature_column_group_id: A scalar, the index of the feature we are spiltting on.
l1_regularization: A scalar, which specifies the l1 regularization term.
l2_regularization: A scalar, which specifies the l2 regularization term.
tree_complexity_regularization: A scalar, which specifies the tree complexity
    regularization term.
min_node_weight: A scalar, minimum sum of example hessian needed in a child.
    If a split results in a leaf node with a smaller value, the split will not
    be considered.
multiclass_strategy: A scalar, specifying the multiclass handling strategy.
    See LearnerConfig.MultiClassStrategy for valid values.
output_partition_ids: A rank 1 tensor, the partition IDs that we created splits
    for.
gains: A rank 1 tensor, for the computed gain for the created splits.
split_infos: A rank 1 tensor of serialized protos which contains the
    `SplitInfo`s.
)doc");

REGISTER_OP("BuildCategoricalEqualitySplits")
    .Input("num_minibatches: int64")
    .Input("partition_ids: int32")
    .Input("feature_ids: int64")
    .Input("gradients: float32")
    .Input("hessians: float32")
    .Input("class_id: int32")
    .Input("feature_column_group_id: int32")
    .Input("bias_feature_id: int64")
    .Input("l1_regularization: float")
    .Input("l2_regularization: float")
    .Input("tree_complexity_regularization: float")
    .Input("min_node_weight: float")
    .Input("multiclass_strategy: int32")
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
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &bucket_ids_shape));
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
feature_ids: A rank 2 tensor of feature IDs and dimensions.
gradients: A rank 1 tensor of gradients.
hessians: A rank 1 tensor of hessians.
class_id: A scalar, the class id for which we're building the splits.
feature_column_group_id: A scalar, the index of the feature we are spiltting on.
l1_regularization: A scalar, which specifies the l1 regularization term.
l2_regularization: A scalar, which specifies the l2 regularization term.
tree_complexity_regularization: A scalar, which specifies the tree complexity
    regularization term.
min_node_weight: A scalar, minimum sum of example hessian needed in a child.
    If a split results in a leaf node with a smaller value, the split will not
    be considered.
multiclass_strategy: A scalar, specifying the multiclass handling strategy.
    See LearnerConfig.MultiClassStrategy for valid values.
output_partition_ids: A rank 1 tensor, the partition IDs that we created splits
    for.
gains: A rank 1 tensor, for the computed gain for the created splits.
split_infos: A rank 1 tensor of serialized protos which contains the
    `SplitInfo`s.
)doc");

}  // namespace tensorflow
