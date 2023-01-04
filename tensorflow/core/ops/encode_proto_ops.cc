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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("EncodeProto")
    .Input("sizes: int32")
    .Input("values: Tinput_types")
    .Attr("field_names: list(string)")
    .Attr("message_type: string")
    .Attr("descriptor_source: string = 'local://'")
    .Attr("Tinput_types: list(type)")
    .Output("bytes: string")
    .SetShapeFn([](InferenceContext* c) {
      int first_field_index = 1;
      int num_fields = c->num_inputs() - 1;

      ShapeHandle output;
      for (int i = num_fields - 1; i >= 0; --i) {
        ShapeHandle input = c->input(first_field_index + i);
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, 2, &input));
        ShapeHandle inner;
        TF_RETURN_IF_ERROR(c->Subshape(input, 0, -1, &inner));
        TF_RETURN_IF_ERROR(c->Merge(inner, output, &output));
      }

      c->set_output(0, output);
      return OkStatus();
    });

}  // namespace tensorflow
