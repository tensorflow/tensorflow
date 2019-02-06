/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

Status StatefulRandomShape(shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &out));
  c->set_output(0, out);
  return Status::OK();
}

REGISTER_OP("StatefulStandardNormal")
    .Input("resource: resource")
    .Input("shape: shape_dtype")
    .Output("output: dtype")
    .Attr("dtype: {half,bfloat16,float,double} = DT_FLOAT")
    .Attr("shape_dtype: {int32, int64} = DT_INT64")
    .SetShapeFn(StatefulRandomShape);

}  // namespace tensorflow
