/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ifrt_proxy/common/grpc_credentials_possibly_insecure_wrapper.h"

#include <cstdlib>
#include <cstring>
#include <memory>

#include "absl/log/log.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "xla/python/ifrt_proxy/common/grpc_credentials.h"
#include "tsl/platform/platform.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

bool google_internal_allow_insecure() {
#ifdef IFRT_PROXY_GOOGLE_INTERNAL_ALLOW_INSECURE
  return true;
#endif
  return false;
}

bool use_insecure() {
  const char* env_val = std::getenv("IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS");

  const bool compiled_to_allow_insecure =
      (TSL_IS_IN_OSS == 1 || google_internal_allow_insecure());

  if (!compiled_to_allow_insecure) {
    if (env_val != nullptr) {
      LOG_FIRST_N(INFO, 1) << "Built to use standard GRPC credentials.";
    }
    return false;
  }

  return (env_val != nullptr) && (strcmp("true", env_val) == 0);
}

}  // namespace

// Get credentials to use in the client gRPC.
std::shared_ptr<::grpc::ChannelCredentials>
GetClientCredentialsPossiblyInsecure() {
  if (use_insecure()) {
    LOG(WARNING) << "IFRT proxy using insecure client credentials because of "
                    "environment variable.";
    return ::grpc::InsecureChannelCredentials();  // NOLINT
  }

  return GetClientCredentials();
}

// Get credentials to use in the server gRPC.
std::shared_ptr<::grpc::ServerCredentials>
GetServerCredentialsPossiblyInsecure() {
  if (use_insecure()) {
    LOG(WARNING) << "IFRT proxy using insecure server credentials because of "
                    "environment variable.";
    return ::grpc::InsecureServerCredentials();  // NOLINT
  }

  return GetServerCredentials();
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
