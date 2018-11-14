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
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace boosted_trees {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_RESOURCE_HANDLE_OP(StatsAccumulatorScalarResource);

REGISTER_OP("StatsAccumulatorScalarIsInitialized")
    .Input("stats_accumulator_handle: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a stats accumulator has been initialized.
)doc");

REGISTER_OP("CreateStatsAccumulatorScalar")
    .Input("stats_accumulator_handle: resource")
    .Input("stamp_token: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      // stamp_token is a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      return Status::OK();
    })
    .Doc(R"doc(
Creates a scalar stats accumulator.

stats_accumulator_handle: handle to the stats accumulator.
stamp_token: Token to use as the initial value of the resource stamp.
)doc");

REGISTER_OP("StatsAccumulatorScalarAdd")
    .Attr("num_resource_handles: int >= 1")
    .Input("stats_accumulator_handles: num_resource_handles * resource")
    .Input("stamp_token: int64")
    .Input("partition_ids: num_resource_handles * int32")
    .Input("feature_ids: num_resource_handles * int64")
    .Input("gradients: num_resource_handles * float")
    .Input("hessians: num_resource_handles * float")
    .SetShapeFn([](InferenceContext* c) {
      int num_resource_handles;
      TF_RETURN_IF_ERROR(
          c->GetAttr("num_resource_handles", &num_resource_handles));
      for (int i = 0; i < num_resource_handles; ++i) {
        ShapeHandle unused_input;
        DimensionHandle unused_dim;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused_input));
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(num_resource_handles), 0, &unused_input));
        ShapeHandle partition_ids_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(num_resource_handles + i + 1),
                                       1, &partition_ids_shape));
        ShapeHandle feature_ids_shape;
        TF_RETURN_IF_ERROR(c->WithRank(
            c->input(num_resource_handles * 2 + i + 1), 2, &feature_ids_shape));
        ShapeHandle gradients_shape;
        TF_RETURN_IF_ERROR(c->WithRank(
            c->input(num_resource_handles * 3 + i + 1), 1, &gradients_shape));
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                    c->Dim(gradients_shape, 0), &unused_dim));
        ShapeHandle hessians_shape;
        TF_RETURN_IF_ERROR(c->WithRank(
            c->input(num_resource_handles * 4 + i + 1), 1, &hessians_shape));
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                    c->Dim(hessians_shape, 0), &unused_dim));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Updates the scalar stats accumulator.

stamp_token: Stamp token for Read/Write operations.
             Any operation with a mismatching token will be dropped.
stats_accumulator_handles: A list of handles to the stats accumulator.
partition_ids: A list of vectors of partition_ids.
feature_ids: Rank 2 tensor of feature id and feature dimension ids.
gradients: A list of vectors of gradients for each slot in
    <partition_id, feature_id, feature_dimension_id>.
hessians: A list of vectors of hessians for each slot in
    <partition_id, feature_id, feature_dimension_id>.
)doc");

REGISTER_OP("StatsAccumulatorScalarFlush")
    .Input("stats_accumulator_handle: resource")
    .Input("stamp_token: int64")
    .Input("next_stamp_token: int64")
    .Output("num_updates: int64")
    .Output("output_partition_ids: int32")
    .Output("output_feature_ids: int64")
    .Output("output_gradients: float")
    .Output("output_hessians: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_input));
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->UnknownShape());
      c->set_output(3, c->Vector(c->UnknownDim()));
      c->set_output(4, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Flushes the scalar stats accumulator to output and resets the internal state.

stats_accumulator_handle: handle to the stats accumulator.
stamp_token: Stamp token for Read/Write operations.
             Any operation with a mismatching token will be dropped.
next_stamp_token: Stamp token for the next iteration.
num_updates: Number of times stats were added to this accumulator since last
    flush.
output_partition_ids A vector of partition_ids for the slots.
output_feature_ids: Rank 2 tensor of feature id and feature dimension ids.
output_gradients: A vector of gradients, with a value for each slot
                  in <output_partition_id, output_feature_id>.
output_hessians: A vector of hessians, with a value for each slot
                 in <output_partition_id, output_feature_id>.
)doc");

REGISTER_OP("StatsAccumulatorScalarDeserialize")
    .Input("stats_accumulator_handle: resource")
    .Input("stamp_token: int64")
    .Input("num_updates: int64")
    .Input("partition_ids: int32")
    .Input("feature_ids: int64")
    .Input("gradients: float")
    .Input("hessians: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused_input;
      DimensionHandle unused_dim;
      // stats_accumulator_handle
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      // stamp_token
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      // num_updates
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_input));
      ShapeHandle partition_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &partition_ids_shape));
      ShapeHandle feature_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &feature_ids_shape));
      ShapeHandle gradients_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &gradients_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(gradients_shape, 0), &unused_dim));
      ShapeHandle hessians_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &hessians_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(hessians_shape, 0), &unused_dim));
      return Status::OK();
    })
    .Doc(R"doc(
Resets the scalar stats accumulator with the serialized state.

stats_accumulator_handle: handle to the stats accumulator.
stamp_token: Stamp token for Read/Write operations.
             Any operation with a mismatching token will be dropped.
num_updates: Number of times stats were added to this accumulator since last
    flush.
partition_ids: A vector of partition_ids.
feature_ids: Rank 2 tensor of feature id and feature dimension ids.
gradients: A vector of gradients for each slot in <partition_id, feature_id,
feature_dimension_id>.
hessians: A vector of hessians for each slot in <partition_id, feature_id,
feature_dimension_id>
)doc");

REGISTER_OP("StatsAccumulatorScalarSerialize")
    .Input("stats_accumulator_handle: resource")
    .Output("stamp_token: int64")
    .Output("num_updates: int64")
    .Output("output_partition_ids: int32")
    .Output("output_feature_ids: int64")
    .Output("output_gradients: float")
    .Output("output_hessians: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      // stamp_token
      c->set_output(0, c->Scalar());
      // num_updates
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Vector(c->UnknownDim()));
      c->set_output(3, c->UnknownShape());
      c->set_output(4, c->Vector(c->UnknownDim()));
      c->set_output(5, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Serializes the scalar stats accumulator state.

stats_accumulator_handle: handle to the stats accumulator.
stamp_token: The current stamp token for the resource.
num_updates: Number of times stats were added to this accumulator since last
    flush.
output_partition_ids A vector of partition_ids for the slots.
output_feature_ids: Rank 2 tensor of feature id and feature dimension ids.
output_gradients: A vector of gradients, with a value for each slot
                  in <output_partition_id, output_feature_id>.
output_hessians: A vector of hessians, with a value for each slot
                 in <output_partition_id, output_feature_id>.
)doc");

REGISTER_OP("StatsAccumulatorScalarMakeSummary")
    .Input("partition_ids: int32")
    .Input("feature_ids: int64")
    .Input("gradients: float")
    .Input("hessians: float")
    .Output("output_partition_ids: int32")
    .Output("output_feature_ids: int64")
    .Output("output_gradients: float")
    .Output("output_hessians: float")
    .Doc(R"doc(
)doc");

// Tensor version of the stats accumulator ops.
REGISTER_RESOURCE_HANDLE_OP(StatsAccumulatorTensorResource);

REGISTER_OP("StatsAccumulatorTensorIsInitialized")
    .Input("stats_accumulator_handle: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a tensor stats accumulator has been initialized.
)doc");

REGISTER_OP("CreateStatsAccumulatorTensor")
    .Input("stats_accumulator_handle: resource")
    .Input("stamp_token: int64")
    .Input("per_slot_gradient_shape: int64")
    .Input("per_slot_hessian_shape: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      // stamp_token is a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &unused_input));
      return Status::OK();
    })
    .Doc(R"doc(
Creates a tensor stats accumulator.

stats_accumulator_handle: handle to the tree ensemble resource to be created.
stamp_token: Token to use as the initial value of the resource stamp.
per_slot_gradient_shape: a vector that defines the shape of gradients.
per_slot_hessian_shape:  a vector that defines the shape of hessians.
)doc");

REGISTER_OP("StatsAccumulatorTensorAdd")
    .Attr("num_resource_handles: int >= 1")
    .Input("stats_accumulator_handles: num_resource_handles * resource")
    .Input("stamp_token: int64")
    .Input("partition_ids: num_resource_handles * int32")
    .Input("feature_ids: num_resource_handles * int64")
    .Input("gradients: num_resource_handles * float")
    .Input("hessians: num_resource_handles * float")
    .SetShapeFn([](InferenceContext* c) {
      int num_resource_handles;
      TF_RETURN_IF_ERROR(
          c->GetAttr("num_resource_handles", &num_resource_handles));
      for (int i = 0; i < num_resource_handles; ++i) {
        ShapeHandle unused_input;
        DimensionHandle unused_dim;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused_input));
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(num_resource_handles), 0, &unused_input));
        ShapeHandle partition_ids_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(num_resource_handles + i + 1),
                                       1, &partition_ids_shape));
        ShapeHandle feature_ids_shape;
        TF_RETURN_IF_ERROR(c->WithRank(
            c->input(num_resource_handles * 2 + i + 1), 2, &feature_ids_shape));
        ShapeHandle gradients_shape;
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(
            c->input(num_resource_handles * 3 + i + 1), 2, &gradients_shape));
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                    c->Dim(gradients_shape, 0), &unused_dim));
        ShapeHandle hessians_shape;
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(
            c->input(num_resource_handles * 4 + i + 1), 2, &hessians_shape));
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                    c->Dim(hessians_shape, 0), &unused_dim));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Updates the tensor stats accumulator.

stats_accumulator_handles: A list of handles to the stats accumulator.
stamp_token: Stamp token for Read/Write operations.
             Any operation with a mismatching token will be dropped.
partition_ids: A list of vectors of partition_ids.
feature_ids: Rank 2 tensor of feature id and feature dimension ids.
gradients: A list of vectors of gradients for each slot in
    <partition_id, feature_id, feature_dimension_id>.
hessians: A list of vectors of hessians for each slot in
    <partition_id, feature_id, feature_dimension_id>.
)doc");

REGISTER_OP("StatsAccumulatorTensorFlush")
    .Input("stats_accumulator_handle: resource")
    .Input("stamp_token: int64")
    .Input("next_stamp_token: int64")
    .Output("num_updates: int64")
    .Output("output_partition_ids: int32")
    .Output("output_feature_ids: int64")
    .Output("output_gradients: float")
    .Output("output_hessians: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_input));
      // num_updates
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->UnknownShape());
      c->set_output(3, c->UnknownShape());
      c->set_output(4, c->UnknownShape());
      return Status::OK();
    })
    .Doc(R"doc(
Flushes the stats accumulator to output and resets the internal state.

stats_accumulator_handle: handle to the tree ensemble resource to be created.
stamp_token: Stamp token for Read/Write operations.
             Any operation with a mismatching token will be dropped.
next_stamp_token: Stamp token to be used for the next iteration.
num_updates: Number of times stats were added to this accumulator since last
    flush.
output_partition_ids: A vector of partition_ids for the slots.
output_feature_ids: Rank 2 tensor of feature id and feature dimension ids.
output_gradients: A tensor of gradients, first dimension matches slots
                  in <partition_id, feature_id, feature_dimension_id>.
output_hessians: A tensor of hessians, first dimension matches slots
                 in <partition_id, feature_id, feature_dimension_id>>.
)doc");

REGISTER_OP("StatsAccumulatorTensorDeserialize")
    .Input("stats_accumulator_handle: resource")
    .Input("stamp_token: int64")
    .Input("num_updates: int64")
    .Input("partition_ids: int32")
    .Input("feature_ids: int64")
    .Input("gradients: float")
    .Input("hessians: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused_input;
      DimensionHandle unused_dim;
      // stats_accumulator_handle
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      // stamp_token
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      // num_updates
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_input));
      ShapeHandle partition_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &partition_ids_shape));
      ShapeHandle feature_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &feature_ids_shape));
      ShapeHandle gradients_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(5), 2, &gradients_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(gradients_shape, 0), &unused_dim));
      ShapeHandle hessians_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(6), 2, &hessians_shape));
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(partition_ids_shape, 0),
                                  c->Dim(hessians_shape, 0), &unused_dim));

      return Status::OK();
    })
    .Doc(R"doc(
Resets the tensor stats accumulator with the serialized state.

stats_accumulator_handle: handle to the tree ensemble resource to be created.
stamp_token: Stamp token for Read/Write operations.
             Any operation with a mismatching token will be dropped.
num_updates: Number of times stats were added to this accumulator since last
    flush.
partition_ids: A vector of partition_ids.
feature_ids: Rank 2 tensor of feature id and feature dimension ids.
gradients: A vector of gradients for each slot in <partition_id, feature_id,
feature_dimension_id>
hessians: A vector of hessians for each slot in <partition_id, feature_id,
feature_dimension_id>.
)doc");

REGISTER_OP("StatsAccumulatorTensorSerialize")
    .Input("stats_accumulator_handle: resource")
    .Output("stamp_token: int64")
    .Output("num_updates: int64")
    .Output("output_partition_ids: int32")
    .Output("output_feature_ids: int64")
    .Output("output_gradients: float")
    .Output("output_hessians: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      // stamp_token
      c->set_output(0, c->Scalar());
      // num_updates
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Vector(c->UnknownDim()));
      c->set_output(3, c->UnknownShape());
      c->set_output(4, c->UnknownShape());
      c->set_output(5, c->UnknownShape());
      return Status::OK();
    })
    .Doc(R"doc(
Serializes the scalar stats accumulator state.

stats_accumulator_handle: handle to the tree ensemble resource to be created.
stamp_token: Stamp token for Read/Write operations.
             Any operation with a mismatching token will be dropped.
num_updates: Number of times stats were added to this accumulator since last
    flush.
output_partition_ids: A vector of partition_ids for the slots.
output_feature_ids: Rank 2 tensor of feature id and feature dimension ids.
output_gradients: A tensor of gradients, first dimension matches slots
                  in <partition_id, feature_id, feature_dimension_id>.
output_hessians: A tensor of hessians, first dimension matches slots
                 in <partition_id, feature_id, feature_dimension_id>.
)doc");

REGISTER_OP("StatsAccumulatorTensorMakeSummary")
    .Input("partition_ids: int32")
    .Input("feature_ids: int64")
    .Input("gradients: float")
    .Input("hessians: float")
    .Output("output_partition_ids: int32")
    .Output("output_feature_ids: int64")
    .Output("output_gradients: float")
    .Output("output_hessians: float")
    .Doc(R"doc(
Summarizes the stats by summing the <gradients, hessians> that are for the same
<partition_id, feature_id, feature_dimension_id>.

partition_ids: A vector of partition_ids.
feature_ids: Rank 2 tensor of feature id and feature dimension ids.
gradients: A vector of gradients for each slot in <partition_id, feature_id,
feature_dimension_id>.
hessians: A vector of hessians for each slot in <partition_id, feature_id,
feature_dimension_id>.
output_partition_ids: A vector of partition_ids for the slots.
output_feature_ids: A rank2 tensor of feature_ids and dimensions for the slots.
output_gradients: A tensor of gradients, first dimension matches slots
                  in <partition_id, feature_id, feature_dimension_id>.
output_hessians: A tensor of hessians, first dimension matches slots
                 in <partition_id, feature_id, feature_dimension_id>.
)doc");
}  // namespace boosted_trees
}  // namespace tensorflow
