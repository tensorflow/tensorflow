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

#ifndef TENSORFLOW_DISTRIBUTE_EXPERIMENTAL_RPC_KERNELS_GRPC_RPC_SERVICE_H_
#define TENSORFLOW_DISTRIBUTE_EXPERIMENTAL_RPC_KERNELS_GRPC_RPC_SERVICE_H_

#include "tensorflow/distribute/experimental/rpc/proto/tf_rpc_service.grpc.pb.h"
#include "tensorflow/distribute/experimental/rpc/proto/tf_rpc_service.pb.h"
#include "tensorflow/stream_executor/platform/port.h"

#ifndef PLATFORM_GOOGLE
#ifndef PLATFORM_CHROMIUMOS
namespace tensorflow {
namespace rpc {
namespace grpc {

// Google internal gRPC generates services under namespace "grpc", but
// opensource version does not add any additional namespaces.

// Creating aliases here to make sure we can access services under namespace
// "tensorflow::grpc" both in google internal and open-source.
using ::tensorflow::rpc::RpcService;

}  // namespace grpc
}  // namespace rpc
}  // namespace tensorflow
#endif  // PLATFORM_CHROMIUMOS
#endif  // PLATFORM_GOOGLE
#endif  // TENSORFLOW_DISTRIBUTE_EXPERIMENTAL_RPC_KERNELS_GRPC_RPC_SERVICE_H_
