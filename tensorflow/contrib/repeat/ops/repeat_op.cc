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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow{

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("Repeat")
    .Input("input: T")
    .Input("repeats: int32")
    .Attr("axis: int")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input = c->input(0);
      ShapeHandle repeats = c->input(1);
      int32 axis;
      TF_RETURN_IF_ERROR(c->GetAttr("axis", &axis));
      
      TF_RETURN_IF_ERROR(c->WithRankAtMost(repeats, 1, &repeats));
      
      int32 rank = c->Rank(input);
      if (rank == 0) {
        rank = 1;
        input = c->MakeShape({1});
      }
      if (axis < -rank || axis >= rank) {
        return errors::InvalidArgument("Expected -", rank, " <= `axis` < ", rank);
      }
      
      int64 repeats_len = c->Value(c->NumElements(repeats));
      if (axis < 0) {
        axis += rank;
      }
      
      DimensionHandle new_dim = c->MakeDim(0);
      if (c->Rank(repeats) == 0) {
        DimensionHandle repeats_scalar;
        TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &repeats_scalar));
        TF_RETURN_IF_ERROR(c->Multiply(c->Dim(input, axis),
                                       repeats_scalar,
                                       &new_dim));
      } else {
        ShapeHandle repeats_data;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &repeats_data));
        if (repeats_len == 1) {
          TF_RETURN_IF_ERROR(c->Multiply(c->Dim(input, axis),
                                         c->Value(c->Dim(repeats_data, 0)),
                                         &new_dim));
        } else {
          DimensionHandle old_dim;
          TF_RETURN_IF_ERROR(c->WithValue(c->Dim(input, axis), repeats_len,
                                          &old_dim));
          for (int i = 0; i < repeats_len; i++) {
            TF_RETURN_IF_ERROR(c->Add(new_dim, c->Dim(repeats_data, i), &new_dim));
          }
        }
      }
      
      ShapeHandle output;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, axis, new_dim, &output));
      c->set_output(0, output);
      
      return Status::OK();
    })
    .Doc(R"doc(
Constructs a tensor by repeating a given tensor.

This operation creates a new tensor by repeating each element of `input`
`repeats` times along `axis`. The output tensor has the same shape as `input`,
except along the given `axis`. For example, repeating `[a b c d]` by
`[2,1,3,4]` along `axis=0` produces `[a a b c c c d d d d]`.
input: 0-D or higher.
repeats: Scalar or 1-D. Length must be 1 or the dimension of `input` along `axis`.
axis: 0 <= axis < rank of `input`.
)doc");

} //end namespace tensorflow
