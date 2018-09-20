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

REGISTER_OP("DecodeProtoV2")
    .Input("bytes: string")
    .Attr("message_type: string")
    .Attr("field_names: list(string)")
    .Attr("output_types: list(type) >= 0")
    .Attr("descriptor_source: string = 'local://'")
    .Attr("message_format: string = 'binary'")
    .Attr("sanitize: bool = false")
    .Output("sizes: int32")
    .Output("values: output_types")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input = c->input(0);

      std::vector<tensorflow::DataType> output_types;
      TF_RETURN_IF_ERROR(c->GetAttr("output_types", &output_types));

      ShapeHandle sizes;
      TF_RETURN_IF_ERROR(
          c->Concatenate(input, c->Vector(output_types.size()), &sizes));
      c->set_output(0, sizes);

      // TODO(nix): to do the best possible job of shape inference, we
      // should examine the proto descriptors here in order to set shape
      // indices to 1 instead of unknown for optional or required fields.
      // Any general-purpose code will have to handle the unknown case,
      // but there might be XLA code that could be sped up with the additional
      // knowledge.
      for (int i = 0; i < output_types.size(); ++i) {
        ShapeHandle values;
        TF_RETURN_IF_ERROR(
            c->Concatenate(input, c->Vector(c->UnknownDim()), &values));
        c->set_output(i + 1, values);
      }

      return Status::OK();
    });

// TODO(nix): Consider adding an additional input argument that truncates
// repeated fields to a maximum count. For now this could be done by passing
// the output through tf.slice.

// TODO(nix): define missing value behavior.

}  // namespace tensorflow
