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

using shape_inference::InferenceContext;

REGISTER_OP("LookupTableInsertOrAssignOp")
    .Input("table_int64_args: num_int64_table_args * int64")
    .Input("table_handle: resource")
    .Input("keys: insert_key_tensor_dtype")
    .Input("values: table_value_dtype")
    .Attr("insert_key_tensor_dtype: type")
    .Attr("table_value_dtype: type")
    .Attr("num_int64_table_args: int >= 0")
    .SetShapeFn([](InferenceContext* c) {
      // Note that, by design, shape checks are implementation dependent so they
      // must be deferred until runtime.
      return Status::OK();
    });

REGISTER_OP("LookupTableFindOp")
    .Input("table_int64_args: num_int64_table_args * int64")
    .Input("table_handle: resource")
    .Input("keys: lookup_key_tensor_dtype")
    .Input("num_threads: int64")
    .Output("values: table_value_dtype")
    .Attr("table_value_dtype: type")
    .Attr("lookup_key_tensor_dtype: type")
    .Attr("num_int64_table_args: int >= 0")
    .SetShapeFn([](InferenceContext* c) {
      // The output shape cannot be inferred here because the key size
      // cannot be inferred from the key tensor in general.
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });

REGISTER_OP("ContainerSizeOp")
    .Input("container_handle: resource")
    .Output("size: int64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

}  // namespace tensorflow
