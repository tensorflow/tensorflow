/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("InfeedDequeue")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape)
    .Doc(R"doc(
A placeholder op for a value that will be fed into the computation.

output: A tensor that will be provided using the infeed mechanism.
dtype: The type of elements in the tensor.
shape: The shape of the tensor.
)doc");

REGISTER_OP("InfeedEnqueue")
    .Input("input: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape = {}")
    .Attr("device_ordinal: int = -1")
    .SetIsStateful()
    .Doc(R"doc(
An op which feeds a single Tensor value into the computation.

input: A tensor that will be provided using the infeed mechanism.
dtype: The type of elements in the tensor.
shape: The shape of the tensor.
device_ordinal: The TPU device to use. This should be -1 when the Op
is running on a TPU device, and >= 0 when the Op is running on the CPU
device.
)doc");

REGISTER_OP("InfeedEnqueueTuple")
    .Input("inputs: dtypes")
    .Attr("dtypes: list(type)")
    .Attr("shapes: list(shape)")
    .Attr("device_ordinal: int = -1")
    .SetIsStateful()
    .Doc(R"doc(
An op which feeds multiple Tensor values into the computation as an XLA tuple.

inputs: A list of tensors that will be provided using the infeed mechanism.
dtypes: The element types of each element in `inputs`.
shapes: The shapes of each tensor in `inputs`.
device_ordinal: The TPU device to use. This should be -1 when the Op
is running on a TPU device, and >= 0 when the Op is running on the CPU
device.
)doc");

REGISTER_OP("InfeedDequeueTuple")
    .Output("outputs: dtypes")
    .Attr("dtypes: list(type)")
    .Attr("shapes: list(shape)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      std::vector<PartialTensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      for (int i = 0; i < shapes.size(); ++i) {
        ShapeHandle out;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shapes[i], &out));
        c->set_output(i, out);
      }
      return Status::OK();
    })
    .Doc(R"doc(
A placeholder op for multiple values that will be fed into the computation
simultaneously as an XLA tuple.

outputs: A list of tensors that will be provided using the infeed mechanism.
dtypes: The element types of each element in `outputs`.
shapes: The shapes of each tensor in `outputs`.
)doc");

}  // namespace tensorflow
