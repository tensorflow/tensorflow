/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
namespace {

REGISTER_OP("EmptyTensorMap")
//    .Input("element_shape: shape_type")
//    .Input("max_num_elements: int32")
    .Output("handle: variant")
//    .Attr("element_dtype: type")
//    .Attr("shape_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      /*DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle element_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          0, &element_shape));
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{
                 {element_shape, element_dtype}});*/
      return Status::OK();
    });

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      //c->set_output(0, c->Scalar());
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace
}  // namespace tensorflow
