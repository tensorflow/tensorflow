/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
namespace rpc {

REGISTER_OP("RpcServer")
    .Input("server_address: string")
    .Output("server: resource")
    .SetIsStateful();

REGISTER_OP("RpcClient")
    .Attr("shared_name: string = ''")
    .Input("server_address: string")
    .Attr("list_registered_methods: bool = false")
    .Input("timeout_in_ms: int64")  // 0 indicates no timeout.
                                    // Positive value indicates specified
                                    // timeout.
    .Output("client: resource")
    .Output("method_specs: string")
    .SetIsStateful();

REGISTER_OP("RpcServerStart").Input("server: resource").SetIsStateful();

REGISTER_OP("RpcServerRegister")
    .Input("server: resource")
    .Input("method_name: string")
    .Input("captured_inputs: Tin")
    .Attr("Tin: list(type) >=0 = []")
    .Attr("f: func")
    .Attr("input_specs: string = ''")
    .Attr("output_specs: string")
    .SetIsStateful();

REGISTER_OP("DeleteRpcFutureResource")
    .Input("handle: resource")
    .Input("deleter: variant")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("RpcCall")
    .Input("client: resource")
    .Input("method_name: string")
    .Input("args: Tin")
    .Input("timeout_in_ms: int64")
    .Attr("Tin: list(type) >= 0")
    .Output("future: resource")
    .Output("deleter: variant")
    .SetIsStateful();

REGISTER_OP("RpcCheckStatus")
    .Input("status_or: resource")
    .Output("error_code: int64")
    .Output("error: string")
    .SetIsStateful();

REGISTER_OP("RpcGetValue")
    .Input("status_or: resource")
    .Attr("Tout: list(type) >= 0")
    .Output("output: Tout")
    .SetIsStateful();

}  // namespace rpc
}  // namespace tensorflow
