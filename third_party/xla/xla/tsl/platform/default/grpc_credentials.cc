// Copyright 2024 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tsl/platform/grpc_credentials.h"

#include <memory>

#include "absl/log/check.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "tsl/platform/logging.h"

namespace tsl {

std::shared_ptr<grpc::ChannelCredentials> GetClientCredentials(
    bool verify_secure_credentials) {
  CHECK(!verify_secure_credentials)
      << "Insecure gRPC credentials are unexpectedly used!";
  LOG(INFO) << "gRPC insecure client credentials are used.";
  return grpc::InsecureChannelCredentials();
}

std::shared_ptr<grpc::ServerCredentials> GetServerCredentials(
    bool verify_secure_credentials) {
  CHECK(!verify_secure_credentials)
      << "Insecure gRPC credentials are unexpectedly used!";
  LOG(INFO) << "gRPC insecure server credentials are used.";
  return grpc::InsecureServerCredentials();
}

}  // namespace tsl
