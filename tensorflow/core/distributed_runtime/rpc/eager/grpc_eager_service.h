/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_H_

#include "tensorflow/core/protobuf/eager_service.grpc.pb.h"
#include "tensorflow/stream_executor/platform/port.h"

#ifndef PLATFORM_GOOGLE

namespace tensorflow {
namespace eager {
namespace grpc {

// Google internal gRPC generates services under namespace "grpc", but
// opensource version does not add any additional namespaces.
// We currently use proto_library BUILD rule with cc_grpc_version and
// has_services arguments. This rule is deprecated but we can't cleanly migrate
// to cc_grpc_library rule yet. The internal version takes service_namespace
// argument, which would have solved the namespace issue, but the external one
// does not.
//
// Creating aliases here to make sure we can access services under namespace
// "tensorflow::grpc" both in google internal and open-source.
using ::tensorflow::eager::EagerService;

}  // namespace grpc
}  // namespace eager
}  // namespace tensorflow
#endif  // PLATFORM_GOOGLE

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_H_
