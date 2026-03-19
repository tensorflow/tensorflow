/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// --------------------------------------------------------------------------
static absl::Status ApplySdcaOptimizerShapeFn(InferenceContext* c) {
  std::vector<ShapeHandle> sparse_handles;
  if (c->input("sparse_weights", &sparse_handles).ok()) {
    TF_RETURN_IF_ERROR(
        c->set_output("out_delta_sparse_weights", sparse_handles));
  }
  std::vector<ShapeHandle> dense_handles;
  if (c->input("dense_weights", &dense_handles).ok()) {
    TF_RETURN_IF_ERROR(c->set_output("out_delta_dense_weights", dense_handles));
  }
  return c->set_output(
      "out_example_state_data",
      {c->Matrix(InferenceContext::kUnknownDim, c->MakeDim(4))});
}

REGISTER_OP("SdcaOptimizer")
    .Attr(
        "loss_type: {'logistic_loss', 'squared_loss', 'hinge_loss',"
        "'smooth_hinge_loss', 'poisson_loss'}")
    .Attr("adaptative : bool=false")
    .Attr("num_sparse_features: int >= 0")
    .Attr("num_sparse_features_with_values: int >= 0")
    .Attr("num_dense_features: int >= 0")
    .Attr("l1: float")
    .Attr("l2: float")
    .Attr("num_loss_partitions: int >= 1")
    .Attr("num_inner_iterations: int >= 1")
    .Input("sparse_example_indices: num_sparse_features * int64")
    .Input("sparse_feature_indices: num_sparse_features * int64")
    .Input("sparse_feature_values: num_sparse_features_with_values * float")
    .Input("dense_features: num_dense_features * float")
    .Input("example_weights: float")
    .Input("example_labels: float")
    .Input("sparse_indices: num_sparse_features * int64")
    .Input("sparse_weights: num_sparse_features * float")
    .Input("dense_weights: num_dense_features * float")
    .Input("example_state_data: float")
    .Output("out_example_state_data: float")
    .Output("out_delta_sparse_weights: num_sparse_features * float")
    .Output("out_delta_dense_weights: num_dense_features * float")
    .SetShapeFn(ApplySdcaOptimizerShapeFn);

// The SdcaOptimizerV2 op fixes the "adaptative" typo in v1.
REGISTER_OP("SdcaOptimizerV2")
    .Attr(
        "loss_type: {'logistic_loss', 'squared_loss', 'hinge_loss',"
        "'smooth_hinge_loss', 'poisson_loss'}")
    .Attr("adaptive : bool=false")
    .Attr("num_sparse_features: int >= 0")
    .Attr("num_sparse_features_with_values: int >= 0")
    .Attr("num_dense_features: int >= 0")
    .Attr("l1: float")
    .Attr("l2: float")
    .Attr("num_loss_partitions: int >= 1")
    .Attr("num_inner_iterations: int >= 1")
    .Input("sparse_example_indices: num_sparse_features * int64")
    .Input("sparse_feature_indices: num_sparse_features * int64")
    .Input("sparse_feature_values: num_sparse_features_with_values * float")
    .Input("dense_features: num_dense_features * float")
    .Input("example_weights: float")
    .Input("example_labels: float")
    .Input("sparse_indices: num_sparse_features * int64")
    .Input("sparse_weights: num_sparse_features * float")
    .Input("dense_weights: num_dense_features * float")
    .Input("example_state_data: float")
    .Output("out_example_state_data: float")
    .Output("out_delta_sparse_weights: num_sparse_features * float")
    .Output("out_delta_dense_weights: num_dense_features * float")
    .SetShapeFn(ApplySdcaOptimizerShapeFn);

REGISTER_OP("SdcaShrinkL1")
    .Attr("num_features: int >= 0")
    .Attr("l1: float")
    .Attr("l2: float")
    .Input("weights: Ref(num_features * float)")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("SdcaFprint")
    .Input("input: string")
    .Output("output: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(c->Concatenate(handle, c->Vector(2), &output_shape));
      c->set_output(0, output_shape);
      return absl::OkStatus();
    });

}  // namespace tensorflow
