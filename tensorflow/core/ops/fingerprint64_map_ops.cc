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

using shape_inference::InferenceContext;

REGISTER_OP("Fingerprint64Map")
    .Output("table_handle: resource")
    .Attr("heterogeneous_key_dtype: type")
    .Attr("table_value_dtype: type = DT_INT64")
    .Attr("num_oov_buckets: int >= 1")
    .Attr("offset: int >= 0 = 0")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

}  // namespace tensorflow
