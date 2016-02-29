/* Copyright 2015 Google Inc. All Rights Reserved.

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

REGISTER_OP("_Send")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .Doc(R"doc(
Sends the named tensor from send_device to recv_device.

tensor: The tensor to send.
tensor_name: The name of the tensor to send.
send_device: The name of the device sending the tensor.
send_device_incarnation: The current incarnation of send_device.
recv_device: The name of the device receiving the tensor.
client_terminated: If set to true, this indicates that the node was added
  to the graph as a result of a client-side feed or fetch of Tensor data,
  in which case the corresponding send or recv is expected to be managed
  locally by the caller.
)doc");

REGISTER_OP("_Recv")
    .Output("tensor: tensor_type")
    .Attr("tensor_type: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .Doc(R"doc(
Receives the named tensor from send_device on recv_device.

tensor: The tensor to receive.
tensor_name: The name of the tensor to receive.
send_device: The name of the device sending the tensor.
send_device_incarnation: The current incarnation of send_device.
recv_device: The name of the device receiving the tensor.
client_terminated: If set to true, this indicates that the node was added
  to the graph as a result of a client-side feed or fetch of Tensor data,
  in which case the corresponding send or recv is expected to be managed
  locally by the caller.
)doc");

REGISTER_OP("_HostSend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .Doc(R"doc(
Sends the named tensor from send_device to recv_device.

_HostSend requires its input on host memory whereas _Send requires its
input on device memory.

tensor: The tensor to send.
tensor_name: The name of the tensor to send.
send_device: The name of the device sending the tensor.
send_device_incarnation: The current incarnation of send_device.
recv_device: The name of the device receiving the tensor.
client_terminated: If set to true, this indicates that the node was added
  to the graph as a result of a client-side feed or fetch of Tensor data,
  in which case the corresponding send or recv is expected to be managed
  locally by the caller.
)doc");

REGISTER_OP("_HostRecv")
    .Output("tensor: tensor_type")
    .Attr("tensor_type: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .Doc(R"doc(
Receives the named tensor from send_device on recv_device.

_HostRecv requires its input on host memory whereas _Recv requires its
input on device memory.

tensor: The tensor to receive.
tensor_name: The name of the tensor to receive.
send_device: The name of the device sending the tensor.
send_device_incarnation: The current incarnation of send_device.
recv_device: The name of the device receiving the tensor.
client_terminated: If set to true, this indicates that the node was added
  to the graph as a result of a client-side feed or fetch of Tensor data,
  in which case the corresponding send or recv is expected to be managed
  locally by the caller.
)doc");

}  // end namespace tensorflow
