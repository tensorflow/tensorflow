/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/distributed/distributed.h"

#include <string>

#include "grpcpp/grpcpp.h"
#include "tensorflow/compiler/xla/pjrt/distributed/client.h"
#include "tensorflow/compiler/xla/pjrt/distributed/service.h"

namespace xla {

StatusOr<std::unique_ptr<DistributedRuntimeService>>
GetDistributedRuntimeService(
    std::string address,
    const DistributedRuntimeServiceImpl::Options& options) {
  auto credentials = ::grpc::InsecureServerCredentials();
  return DistributedRuntimeService::Get(address, credentials, options);
}

std::shared_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::string address, const DistributedRuntimeClient::Options& options) {
  std::shared_ptr<::grpc::ChannelCredentials> creds =
      ::grpc::InsecureChannelCredentials();
  std::shared_ptr<::grpc::Channel> channel =
      ::grpc::CreateChannel(address, creds);
  return GetDistributedRuntimeClient(channel, options);
}

}  // namespace xla
