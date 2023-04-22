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
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("num_devices: int")
    .Attr("shared_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

// Note: This op has no kernel implementation, but is replaced by
// _NcclReduceSend and _NcclReduceRecv during graph optimization stage.
REGISTER_OP("NcclReduce")
    .Input("input: num_devices * T")
    .Output("data: T")
    .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("num_devices: int")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("_NcclReduceSend")
    .Input("input: T")
    .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("num_devices: int")
    .Attr("shared_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Replacement node for NcclReduce.

Reduces `input` to the NcclReduceRecv op registered in the same `shared_name`.
The graph should be constructed so that 'num_devices-1' devices run
`_NcclReduceSend` and one device runs _NcclReduceRecv op with shared_name value
`c`. Failure to do so will cause the graph execution to fail to complete.

input: The input to the reduction.
reduction: the reduction operation to perform.
num_devices: The number of devices participating in this reduction.
shared_name: Identifier that is shared between ops of the same reduce.
    )doc");

REGISTER_OP("_NcclReduceRecv")
    .Input("input: T")
    .Output("data: T")
    .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("num_devices: int")
    .Attr("shared_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Replacement node for NcclReduce.

Reduces 'input' from this op and the NcclReduceSend ops registered in the same
`shared_name`.
The graph should be constructed so that 'num_devices-1' devices run
`_NcclReduceSend` and one device runs _NcclReduceRecv op with shared_name value
`c`. Failure to do so will cause the graph execution to fail to complete.

input: The input to the reduction.
data: The reduced data received from this op and the NcclReduceSend op.
reduction: the reduction operation to perform.
num_devices: The number of devices participating in this reduction.
shared_name: Identifier that is shared between ops of the same reduce.
    )doc");

// Note: This op has no kernel implementation, but is replaced by
// _NcclBroadcastSend and _NcclBroadcastRecv during graph optimization stage.
REGISTER_OP("NcclBroadcast")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("_NcclBroadcastSend")
    .Input("input: T")
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("num_devices: int")
    .Attr("shared_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Replacement node for NcclBroadcast.

Sends `input` to the _NcclBroadcastRecv ops registered in the same
`shared_name`.
The graph should be constructed so that one device runs `_NcclBroadcastSend` and
`num_devices-1` devices run _NcclBroadcastRecv ops with shared_name value `c`.
Failure to do so will cause the graph execution to fail to complete.

input: The input to the broadcast.
num_devices: The number of devices participating in this reduction.
shared_name: Identifier that is shared between ops of the same broadcast.
    )doc");

REGISTER_OP("_NcclBroadcastRecv")
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
Replacement node for NcclBroadcast.

Sends data of shape `shape` from the _NcclBroadcastSend op registered in the
same `shared_name`.
The graph should be constructed so that one device runs `_NcclBroadcastSend` and
`num_devices-1` devices run _NcclBroadcastRecv ops with shared_name value `c`.
Failure to do so will cause the graph execution to fail to complete.

shape: The shape of the output.
output: The broadcast data received from the NcclBroadcastSend op.
num_devices: The number of devices participating in this reduction.
shared_name: Identifier that is shared between ops of the same broadcast.
    )doc");

}  // namespace tensorflow
