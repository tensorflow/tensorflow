/*
 * Copyright 2024 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORFLOW_TSL_PLATFORM_GRPC_CREDENTIALS_H_
#define TENSORFLOW_TSL_PLATFORM_GRPC_CREDENTIALS_H_

#include <memory>

#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"

namespace tsl {

// Get credentials to use in the client gRPC.
// If `verify_secure_credentials`, crash if insecure credentials are used.
std::shared_ptr<::grpc::ChannelCredentials> GetClientCredentials(
    bool verify_secure_credentials = true);

// Get credentials to use in the server gRPC.
// If `verify_secure_credentials`, crash if insecure credentials are used.
std::shared_ptr<::grpc::ServerCredentials> GetServerCredentials(
    bool verify_secure_credentials = true);
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_GRPC_CREDENTIALS_H_
