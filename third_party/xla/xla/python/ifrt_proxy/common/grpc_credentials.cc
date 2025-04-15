// Copyright 2023 The OpenXLA Authors.
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

#include "xla/python/ifrt_proxy/common/grpc_credentials.h"

#include <cstdlib>
#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "tsl/platform/platform.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

bool UseInsecureCredentials() {
  // Use insecure only with `bazel test`.
  const bool insecure = (getenv("TEST_UNDECLARED_OUTPUTS_DIR") != nullptr);

  if (insecure) {
    // We should not be getting to this point at all in the google-internal
    // code, but check to be sure.
    CHECK_EQ(TSL_IS_IN_OSS, 1);
  }

  return insecure;
}

}  // namespace

std::shared_ptr<::grpc::ChannelCredentials> GetClientCredentials() {
  if (UseInsecureCredentials()) {
    LOG(WARNING) << "Using insecure client credentials for gRPC.";
    return ::grpc::InsecureChannelCredentials();  // NOLINT
  } else {
    LOG(INFO) << "Using ALTS client credentials for gRPC.";
    return ::grpc::experimental::AltsCredentials(
        ::grpc::experimental::AltsCredentialsOptions());
  }
}

std::shared_ptr<::grpc::ServerCredentials> GetServerCredentials() {
  if (UseInsecureCredentials()) {
    LOG(WARNING) << "Using insecure server credentials for gRPC.";
    return ::grpc::InsecureServerCredentials();  // NOLINT
  } else {
    LOG(INFO) << "Using ALTS server credentials for gRPC.";
    return ::grpc::experimental::AltsServerCredentials(
        ::grpc::experimental::AltsServerCredentialsOptions());
  }
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
