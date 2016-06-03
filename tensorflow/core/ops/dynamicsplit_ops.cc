/* Copyright 2016 Google Inc. All Rights Reserved.

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

namespace tensorflow {
REGISTER_OP("DynamicSplit")
    .Input("records: string")
    .Input("record_defaults: OUT_TYPE")
    .Output("output: OUT_TYPE")
    .Attr("OUT_TYPE: {float,int32,int64,string}")
    .Attr("field_delim: string = ','")
    .Doc(R"doc(
Convert string records to tensors. Each column maps to one tensor.

records: a string.
record_defaults: indicate the records decode type.
field_delim: delimiter to separate fields in a record.
output: Each tensor will have the same shape as records.
)doc");

}  // namespace tensorflow
