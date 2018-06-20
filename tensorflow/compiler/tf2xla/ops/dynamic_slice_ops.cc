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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("XlaDynamicUpdateSlice")
    .Input("input: T")
    .Input("update: T")
    .Input("indices: Tindices")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Wraps the XLA DynamicUpdateSlice operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#dynamicupdateslice
.

XlaDynamicUpdateSlice generates a result which is the value of the `input`
operand, with a slice update overwritten at `indices`. The shape of `update`
determines the shape of the sub-array of the result which is updated. The shape
of indices must be rank == 1, with dimension size equal to the rank of `input`.

Handling of out-of-bounds slice indices is implementation-defined.

input: A `Tensor` of type T.
indices: A vector of indices into `input`. Must have length equal to the rank of
  `input`.
update: A `Tensor` of type T. Same rank as `input`.
output: A `Tensor` of type T.
)doc");

}  // namespace tensorflow
