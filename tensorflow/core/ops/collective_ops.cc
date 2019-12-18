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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("CollectiveReduce")
    .Input("input: T")
    .Output("data: T")
    .Attr("T: {float, float16, float64, int32, int64}")
    .Attr("group_size: int")
    .Attr("group_key: int")
    .Attr("instance_key: int")
    .Attr("merge_op: {'Min', 'Max', 'Mul', 'Add'}")
    .Attr("final_op: {'Id', 'Div'}")
    .Attr("subdiv_offsets: list(int)")
    .Attr("wait_for: list(int) = []")
    .Attr("communication_hint: string = 'auto'")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("CollectiveGather")
    .Input("input: T")
    .Output("data: T")
    .Attr("T: {float, float16, float64, int32, int64}")
    .Attr("group_size: int")
    .Attr("group_key: int")
    .Attr("instance_key: int")
    .Attr("shape: shape")
    .Attr("communication_hint: string = 'auto'")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Scalar input is not supported.
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &unused));

      shape_inference::ShapeHandle in_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 1, &in_subshape));

      auto input_first_dim_value = c->Value(c->Dim(c->input(0), 0));

      // This output should have the same shape as its input except the first
      // dimension should be multiplied by group size.
      shape_inference::ShapeHandle output_first_dim_as_shape;
      if (input_first_dim_value ==
          shape_inference::InferenceContext::kUnknownDim) {
        output_first_dim_as_shape =
            c->Vector(shape_inference::InferenceContext::kUnknownDim);
      } else {
        int group_size;
        TF_CHECK_OK(c->GetAttr("group_size", &group_size));
        std::vector<shape_inference::DimensionHandle> output_first_dim;
        output_first_dim.push_back(
            c->MakeDim(group_size * input_first_dim_value));
        output_first_dim_as_shape = c->MakeShape(output_first_dim);
      }

      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(
          c->Concatenate(output_first_dim_as_shape, in_subshape, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("CollectiveBcastSend")
    .Input("input: T")
    .Output("data: T")
    .Attr("T: {float, float16, float64, int32, int64}")
    .Attr("group_size: int")
    .Attr("group_key: int")
    .Attr("instance_key: int")
    .Attr("shape: shape")
    .Attr("communication_hint: string = 'auto'")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_OP("CollectiveBcastRecv")
    .Output("data: T")
    .Attr("T: {float, float16, float64, int32, int64}")
    .Attr("group_size: int")
    .Attr("group_key: int")
    .Attr("instance_key: int")
    .Attr("shape: shape")
    .Attr("communication_hint: string = 'auto'")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

}  // namespace tensorflow
