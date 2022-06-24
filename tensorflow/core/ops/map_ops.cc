/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// TODO(kttian): Support non-scalar values
REGISTER_OP("EmptyTensorMap")
    .Output("handle: variant")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("TensorMapSize")
    .Input("input_handle: variant")
    .Output("size: int32")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TensorMapLookup")
    .Input("input_handle: variant")
    .Input("key: key_dtype")
    .Output("value: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return OkStatus();
    });

REGISTER_OP("TensorMapInsert")
    .Input("input_handle: variant")
    .Input("key: key_dtype")
    .Input("value: value_dtype")
    .Output("output_handle: variant")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("TensorMapErase")
    .Input("input_handle: variant")
    .Input("key: key_dtype")
    .Output("output_handle: variant")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());  // output map
      return OkStatus();
    });

REGISTER_OP("TensorMapHasKey")
    .Input("input_handle: variant")
    .Input("key: key_dtype")
    .Output("has_key: bool")
    .Attr("key_dtype: type")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TensorMapStackKeys")
    .Input("input_handle: variant")
    .Output("keys: key_dtype")
    .Attr("key_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShape());  // output keys
      return OkStatus();
    });

}  // namespace
}  // namespace tensorflow
