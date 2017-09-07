/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("NcclAllReduce")
    .Input("input: T")
    .Output("data: T")
    .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
    .Attr("T: {float, float64, int32, int64}")
    .Attr("num_devices: int")
    .Attr("shared_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Outputs a tensor containing the reduction across all input tensors passed to ops
within the same `shared_name.

The graph should be constructed so if one op runs with shared_name value `c`,
then `num_devices` ops will run with shared_name value `c`.  Failure to do so
will cause the graph execution to fail to complete.

input: the input to the reduction
data: the value of the reduction across all `num_devices` devices.
reduction: the reduction operation to perform.
num_devices: The number of devices participating in this reduction.
shared_name: Identifier that shared between ops of the same reduction.
)doc");

REGISTER_OP("NcclBroadcastSend")
    .Input("input: T")
    .Attr("T: {float, float64, int32, int64}")
    .Attr("num_devices: int")
    .Attr("shared_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Sends `input` to the NcclBroadcastRecv ops registered in the same `shared_name`.

The graph should be constructed so that one device runs `NcclBroadcastSend` and
`num_devices-1` devices run NcclBroadcastRecv ops with shared_name value `c`.
Failure to do so will cause the graph execution to fail to complete.

input: The input to the broadcast
num_devices: The number of devices participating in this reduction.
shared_name: Identifier that is shared between ops of the same broadcast.
    )doc");

REGISTER_OP("NcclBroadcastRecv")
    .Input("shape: int64")
    .Output("output: T")
    .Attr("T: {float, float64, int32, int64}")
    .Attr("num_devices: int")
    .Attr("shared_name: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Sends data of shape `shape` from the NcclBroadcastSend op registered in the
same `shared_name`.

The graph should be constructed so that one device runs `NcclBroadcastSend` and
`num_devices-1` devices run NcclBroadcastRecv ops with shared_name value `c`.
Failure to do so will cause the graph execution to fail to complete.

shape: The shape of the output.
output: The broadcast data received from the NcclBroadcastSend op.
num_devices: The number of devices participating in this reduction.
shared_name: Identifier that is shared between ops of the same broadcast.
    )doc");

}  // namespace tensorflow
