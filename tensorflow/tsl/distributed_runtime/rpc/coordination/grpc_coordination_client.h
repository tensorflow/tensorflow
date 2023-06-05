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

#ifndef TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_RPC_COORDINATION_GRPC_COORDINATION_CLIENT_H_
#define TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_RPC_COORDINATION_GRPC_COORDINATION_CLIENT_H_

#include <memory>

#include "tensorflow/tsl/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/tsl/distributed_runtime/rpc/grpc_channel.h"

namespace tsl {

CoordinationClientCache* NewGrpcCoordinationClientCache(
    std::shared_ptr<GrpcChannelCache> channel);

CoordinationClient* NewGrpcCoordinationClient(
    std::shared_ptr<::grpc::Channel> channel);

}  // namespace tsl

#endif  // TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_RPC_COORDINATION_GRPC_COORDINATION_CLIENT_H_
