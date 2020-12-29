/* Copyright 2020 The TensorFlow Authors All Rights Reserved.

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
// GRPC utilities

#ifndef TENSORFLOW_CORE_PROFILER_COMMON_GRPC_GRPC_H_
#define TENSORFLOW_CORE_PROFILER_COMMON_GRPC_GRPC_H_

#include <memory>

#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"

namespace tensorflow {
namespace profiler {

// Returns default credentials for use when creating a gRPC server.
std::shared_ptr<::grpc::ServerCredentials> GetDefaultServerCredentials();

// Returns default credentials for use when creating a gRPC channel.
std::shared_ptr<::grpc::ChannelCredentials> GetDefaultChannelCredentials();

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_COMMON_GRPC_GRPC_H_
