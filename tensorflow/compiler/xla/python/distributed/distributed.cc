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

#include "tensorflow/compiler/xla/python/distributed/distributed.h"

#include "grpcpp/grpcpp.h"

namespace xla {

StatusOr<std::unique_ptr<DistributedRuntimeService>>
GetDistributedRuntimeService(std::string address, int num_nodes) {
  auto credentials = ::grpc::InsecureServerCredentials();
  return DistributedRuntimeService::Get(address, credentials, num_nodes);
}

std::shared_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::string address) {
  std::shared_ptr<::grpc::ChannelCredentials> creds =
      ::grpc::InsecureChannelCredentials();
  std::shared_ptr<::grpc::Channel> channel =
      ::grpc::CreateChannel(address, creds);
  return absl::make_unique<DistributedRuntimeClient>(channel);
}

}  // namespace xla
