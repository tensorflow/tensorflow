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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_H_

#include "tensorflow/tsl/distributed_runtime/rpc/grpc_channel.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::ChannelCreationFunction;
using tsl::ConvertToChannelCreationFunction;
using tsl::GetChannelArguments;
using tsl::GrpcChannelCache;
using tsl::GrpcChannelSpec;
using tsl::NewGrpcChannelCache;
using tsl::NewHostPortGrpcChannel;
// NOLINTEND(misc-unused-using-decls)
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_H_
