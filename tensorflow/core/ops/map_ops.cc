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

//TODO(kttian): Support non-scalar values
REGISTER_OP("EmptyTensorMap")
    .Output("handle: variant")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("TensorMapSize")
    .Input("input_handle: variant")
    .Output("size: int32")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TensorMapInsert")
    .Input("input_handle: variant")
    .Input("key: element_dtype")
    .Input("value: element_dtype")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("TensorMapLookup")
    .Input("input_handle: variant")
    .Input("key: element_dtype")
    .Output("value: element_dtype")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("TensorMapErase")
    .Input("input_handle: variant")
    .Input("key: element_dtype")
    .Output("output_handle: variant")
    .Output("tensor: element_dtype")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      c->set_output(1, c->Scalar()); // removed element
      c->set_output(0, c->Scalar()); // map
      return Status::OK();
    });

REGISTER_OP("TensorMapReplace")
    .Input("input_handle: variant")
    .Input("key: element_dtype")
    .Input("value: element_dtype")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

}  // namespace
}  // namespace tensorflow
