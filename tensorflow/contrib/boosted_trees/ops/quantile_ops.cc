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

REGISTER_RESOURCE_HANDLE_OP(QuantileStreamResource);

REGISTER_OP("QuantileAccumulatorIsInitialized")
    .Input("quantile_accumulator_handle: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a quantile accumulator has been initialized.
)doc");

REGISTER_OP("CreateQuantileAccumulator")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("max_elements: int = 1099511627776")  // 1 << 40
    .Attr("epsilon: float")
    .Attr("num_quantiles: int")
    .Attr("generate_quantiles: bool=False")
    .Input("quantile_accumulator_handle: resource")
    .Input("stamp_token: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused_input));
      return Status::OK();
    })
    .Doc(R"doc(
Creates a stateful accumulator for quantile summaries.

epsilon: Error bound on the quantile summary.
num_quantiles: Number of buckets that we create from the data.
stamp_token: Token to use as the initial value of the resource stamp.
quantile_accumulator_handle: The handle to the accumulator.
)doc");

REGISTER_OP("QuantileAccumulatorAddSummaries")
    .Attr("num_resource_handles: int >= 1")
    .Input("quantile_accumulator_handles: num_resource_handles * resource")
    .Input("stamp_token: int64")
    .Input("summaries: num_resource_handles * string")
    .SetShapeFn([](InferenceContext* c) {
      int num_resource_handles;
      TF_RETURN_IF_ERROR(
          c->GetAttr("num_resource_handles", &num_resource_handles));
      // All the inputs are scalars.
      shape_inference::ShapeHandle unused_input;
      for (int i = 0; i < 2 * num_resource_handles + 1; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused_input));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Adds each quantile summary to its stream.

quantile_accumulator_handles: The handles to the quantile stream resources.
stamp_token: Stamp token to validate the Read/Write operation.
summaries: A list of serialized QuantileSummaryState.
)doc");

REGISTER_OP("QuantileAccumulatorGetBuckets")
    .Attr("num_resource_handles: int >= 1")
    .Input("quantile_accumulator_handles: num_resource_handles * resource")
    .Input("stamp_token: int64")
    .Output("are_buckets_ready: num_resource_handles * bool")
    .Output("buckets: num_resource_handles * float")
    .SetShapeFn([](InferenceContext* c) {
      int num_resource_handles;
      TF_RETURN_IF_ERROR(
          c->GetAttr("num_resource_handles", &num_resource_handles));
      for (int i = 0; i < num_resource_handles; ++i) {
        c->set_output(i, c->Scalar());
        c->set_output(i + num_resource_handles, c->Vector(c->UnknownDim()));
      }
      return Status::OK();
    })

    .Doc(R"doc(
Returns quantile buckets created during previous flush of the accumulator.

quantile_accumulator_handles: The handles to the quantile stream resources.
stamp_token: Stamp token to validate the Read/Write operation.
are_buckets_ready: Whether the buckets are ready or not.
buckets: Output quantile summary representing boundaries with "num_quantile"
    elements.
)doc");

REGISTER_OP("QuantileAccumulatorFlush")
    .Input("quantile_accumulator_handle: resource")
    .Input("stamp_token: int64")
    .Input("next_stamp_token: int64")
    .Doc(R"doc(
Resets quantile summary streams for each column with a new token.

quantile_accumulator_handle: The handle to the accumulator.
stamp_token: Stamp token for Read/Write operations.
             Any operation with a mismatching token will be dropped.
next_stamp_token: Stamp token to be used for the next iteration.
)doc");

REGISTER_OP("QuantileAccumulatorFlushSummary")
    .Input("quantile_accumulator_handle: resource")
    .Input("stamp_token: int64")
    .Input("next_stamp_token: int64")
    .Output("output: string")
    .Doc(R"doc(
Resets quantile summary stream and returns the summary.

quantile_accumulator_handle: The handle to the accumulator.
stamp_token: Stamp token for Read/Write operations.
             Any operation with a mismatching token will be dropped.
next_stamp_token: Stamp token to be used for the next iteration.
output: A scalar string that is the a summary of the accumulator.
)doc");

REGISTER_OP("QuantileAccumulatorSerialize")
    .Input("quantile_accumulator_handle: resource")
    .Output("stamp_token: int64")
    .Output("stream_state: string")
    .Output("are_buckets_ready: bool")
    .Output("buckets: float")
    .Doc(R"doc(
Serializes the state of the given resource.

quantile_accumulator_handle: The handle to the accumulator.
stamp_token: Stamp token for Read/Write operations.
             Any operation with a mismatching token will be dropped.
stream_state: A serialized QuantileStreamState.
are_buckets_ready: Whether the buckets are ready or not.
buckets: Output quantile buckets representing boundaries with "num_quantile"
    elements.
)doc");

REGISTER_OP("QuantileAccumulatorDeserialize")
    .Input("quantile_accumulator_handle: resource")
    .Input("stamp_token: int64")
    .Input("stream_state: string")
    .Input("are_buckets_ready: bool")
    .Input("buckets: float")
    .Doc(R"doc(
Serializes the state of the given resource.

quantile_accumulator_handle: The handle to the accumulator.
stamp_token: Stamp token for Read/Write operations.
             Any operation with a mismatching token will be dropped.
stream_state: A serialized QuantileStreamState.
are_buckets_ready: Whether the buckets are ready or not.
buckets: Output quantile summary representing boundaries with "num_quantile"
    elements.
)doc");

REGISTER_OP("MakeQuantileSummaries")
    .Attr("num_dense_features: int >= 0")
    .Attr("num_sparse_features: int >= 0")
    .Attr("epsilon: float")
    .Input("dense_float_features: num_dense_features * float")
    .Input("sparse_float_feature_indices: num_sparse_features * int64")
    .Input("sparse_float_feature_values: num_sparse_features * float")
    .Input("sparse_float_feature_shapes: num_sparse_features * int64")
    .Input("example_weights: float")
    .Output("dense_summaries: num_dense_features * string")
    .Output("sparse_summaries: num_sparse_features * string")
    .SetShapeFn([](InferenceContext* c) {
      int num_dense_features;
      TF_RETURN_IF_ERROR(c->GetAttr("num_dense_features", &num_dense_features));
      int num_sparse_features;
      TF_RETURN_IF_ERROR(
          c->GetAttr("num_sparse_features", &num_sparse_features));
      ShapeHandle example_weights_shape;
      int example_weights_index = num_dense_features + num_sparse_features * 3;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(example_weights_index), 2,
                                     &example_weights_shape));
      for (int i = 0; i < num_dense_features; ++i) {
        ShapeHandle dense_feature_shape;
        DimensionHandle unused_dim;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 2, &dense_feature_shape));
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(dense_feature_shape, 0),
                                    c->Dim(example_weights_shape, 0),
                                    &unused_dim));
        c->set_output(i, c->Scalar());
      }
      for (int i = 0; i < num_sparse_features; ++i) {
        c->set_output(i + num_dense_features, c->Scalar());
      }
      return Status::OK();
    })
    .Doc(R"doc(
Creates a summary for the given features.

num_dense_features: Number of dense feature groups to compute quantiles on.
num_sparse_features: Number of sparse feature groups to compute quantiles on.
epsilon: Error bound on the computed summary.
dense_float_features: A list of vectors which contains dense values.
sparse_float_feature_indices: List of rank 2 tensors containing the sparse float
feature indices.
sparse_float_feature_values: List of rank 1 tensors containing the sparse float
feature values.
sparse_float_feature_shapes: List of rank 1 tensors containing the shape of the
float feature.
example_weights: Rank 2 (N, 1) tensor of per-example weights. Should match
    dense and sparse features shape.
dense_summaries: A list of serialized QuantileSummaryState for dense columns.
sparse_summaries: A list of serialized QuantileSummaryState for sparse columns.
)doc");

REGISTER_OP("QuantileBuckets")
    .Attr("num_dense_features: int >= 0")
    .Attr("num_sparse_features: int >= 0")
    .Attr("dense_config: list(string)")
    .Attr("sparse_config: list(string)")
    .Input("dense_float_features: num_dense_features * float")
    .Input("sparse_float_feature_indices: num_sparse_features * int64")
    .Input("sparse_float_feature_values: num_sparse_features * float")
    .Input("sparse_float_feature_shapes: num_sparse_features * int64")
    .Input("example_weights: float")
    .Output("dense_buckets: num_dense_features * float")
    .Output("sparse_buckets: num_sparse_features * float")
    .Doc(R"doc(
Computes quantile buckets for a given list of dense and sparse features with
given example weights.

num_dense_features: Number of dense feature groups to compute quantiles on.
num_sparse_features: Number of sparse feature groups to compute quantiles on.
dense_config: Config for computing buckets for dense values.
Each entry is QuantileConfig proto.
sparse_config: Config for computing buckets for sparse feature values.
Each entry is QuantileConfig proto.
dense_float_features: A list of vectors which contains dense values.
sparse_float_feature_indices: List of rank 2 tensors containing the sparse float
feature indices.
sparse_float_feature_values: List of rank 1 tensors containing the sparse float
feature values.
sparse_float_feature_shapes: List of rank 1 tensors containing the shape of the
float feature.
example_weights: Rank 1 tensor containing the example weight tensor.
dense_buckets: Output quantile summary for each dense float tensor
representing boundaries each with "num_quantile" elements.
sparse_buckets: Output quantile summary for each sparse float value tensor
representing boundaries each with "num_quantile" elements.
)doc");

REGISTER_OP("Quantiles")
    .Attr("num_dense_features: int >= 0")
    .Attr("num_sparse_features: int >= 0")
    .Input("dense_values: num_dense_features * float")
    .Input("sparse_values: num_sparse_features * float")
    .Input("dense_buckets: num_dense_features * float")
    .Input("sparse_buckets: num_sparse_features * float")
    .Input("sparse_indices: num_sparse_features * int64")
    .Output("dense_quantiles: num_dense_features * int32")
    .Output("sparse_quantiles: num_sparse_features * int32")
    .Doc(R"doc(
Computes quantile for each a given list of dense and sparse feature values using
the given buckets.

num_dense_features: Number of dense feature groups to generate quantiles for.
num_sparse_features: Number of sparse feature groups to generate quantiles for.
dense_values: List of rank 1 tensors containing the dense values.
sparse_values: List of rank 1 tensors containing the sparse feature values.
dense_buckets: Quantile summary for each of the dense float tensor.
sparse_buckets: Quantile summary for each of the sparse feature float tensor.
sparse_indices: List of rank 2 tensors with indices for sparse float
tensors.
dense_quantiles: Rank 2 tensors representing associated quantiles for each of
dense float tensors and the dimension.
sparse_quantiles: Rank 2 tensors representing associated quantiles for each of
the sparse feature tensors for each of sparse feature dimensions:
[quantile id, dimension id].
)doc");

REGISTER_OP("BucketizeWithInputBoundaries")
    .Input("input: T")
    .Input("boundaries: float")
    .Output("output: int32")
    .Attr("T: {int32, int64, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Bucketizes 'input' based on 'boundaries'. This function is similar to Bucketize
op in core math_ops, except that boundaries are specified using an input tensor,
as compared with a fixed attribute in Bucketize().

For example, if the inputs are
    boundaries = [0, 10, 100]
    input = [[-5, 10000]
             [150,   10]
             [5,    100]]

then the output will be
    output = [[0, 3]
              [3, 2]
              [1, 3]]

input: Any shape of Tensor contains with numeric type.
boundaries: A vector Tensor of sorted floats specifies the boundaries
of the buckets.
output: Same shape as 'input', where each value of input is replaced with its corresponding bucket index.
)doc");

}  // namespace boosted_trees
}  // namespace tensorflow
