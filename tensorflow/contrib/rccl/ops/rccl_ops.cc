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

REGISTER_OP("RcclAllReduce")
    .Input("input: T")
    .Output("data: T")
    .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
    .Attr("T: {half, float, float64, int32, int64}")
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

// Note: This op has no kernel implementation, but is replaced by
// _RcclReduceSend and _RcclReduceRecv during graph optimization stage.
REGISTER_OP("RcclReduce")
    .Input("input: num_devices * T")
    .Output("data: T")
    .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("num_devices: int")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Reduces `input` from `num_devices` using `reduction` to a single device.

The graph should be constructed so that all inputs have a valid device
assignment, and the op itself is assigned one of these devices.

input: The input to the reduction.
data: the value of the reduction across all `num_devices` devices.
reduction: the reduction operation to perform.
    )doc");

REGISTER_OP("_RcclReduceSend")
    .Input("input: T")
    .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("num_devices: int")
    .Attr("shared_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Replacement node for RcclReduce.

Reduces `input` to the RcclReduceRecv op registered in the same `shared_name`.
The graph should be constructed so that 'num_devices-1' devices run
`_RcclReduceSend` and one device runs _RcclReduceRecv op with shared_name value
`c`. Failure to do so will cause the graph execution to fail to complete.

input: The input to the reduction.
reduction: the reduction operation to perform.
num_devices: The number of devices participating in this reduction.
shared_name: Identifier that is shared between ops of the same reduce.
    )doc");

REGISTER_OP("_RcclReduceRecv")
    .Input("input: T")
    .Output("data: T")
    .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("num_devices: int")
    .Attr("shared_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Replacement node for RcclReduce.

Reduces 'input' from this op and the RcclReduceSend ops registered in the same
`shared_name`.
The graph should be constructed so that 'num_devices-1' devices run
`_RcclReduceSend` and one device runs _RcclReduceRecv op with shared_name value
`c`. Failure to do so will cause the graph execution to fail to complete.

input: The input to the reduction.
data: The reduced data received from this op and the RcclReduceSend op.
reduction: the reduction operation to perform.
num_devices: The number of devices participating in this reduction.
shared_name: Identifier that is shared between ops of the same reduce.
    )doc");

// Note: This op has no kernel implementation, but is replaced by
// _RcclBroadcastSend and _RcclBroadcastRecv during graph optimization stage.
REGISTER_OP("RcclBroadcast")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Sends `input` to all devices that are connected to the output.

The graph should be constructed so that all ops connected to the output have a
valid device assignment, and the op itself is assigned one of these devices.

input: The input to the broadcast.
output: The same as input.
shape: The shape of the input tensor.
    )doc");

REGISTER_OP("_RcclBroadcastSend")
    .Input("input: T")
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("num_devices: int")
    .Attr("shared_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Replacement node for RcclBroadcast.

Sends `input` to the _RcclBroadcastRecv ops registered in the same
`shared_name`.
The graph should be constructed so that one device runs `_RcclBroadcastSend` and
`num_devices-1` devices run _RcclBroadcastRecv ops with shared_name value `c`.
Failure to do so will cause the graph execution to fail to complete.

input: The input to the broadcast.
num_devices: The number of devices participating in this reduction.
shared_name: Identifier that is shared between ops of the same broadcast.
    )doc");

REGISTER_OP("_RcclBroadcastRecv")
    .Input("shape: int32")
    .Output("output: T")
    .Attr("T: {half, float, float64, int32, int64}")
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
Replacement node for RcclBroadcast.

Sends data of shape `shape` from the _RcclBroadcastSend op registered in the
same `shared_name`.
The graph should be constructed so that one device runs `_RcclBroadcastSend` and
`num_devices-1` devices run _RcclBroadcastRecv ops with shared_name value `c`.
Failure to do so will cause the graph execution to fail to complete.

shape: The shape of the output.
output: The broadcast data received from the RcclBroadcastSend op.
num_devices: The number of devices participating in this reduction.
shared_name: Identifier that is shared between ops of the same broadcast.
    )doc");

}  // namespace tensorflow
