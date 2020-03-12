/* Copyright 2020 Google LLC

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_DISTRIBUTED_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_DISTRIBUTED_CLIENT_H_

#include <memory>

#include "grpcpp/channel.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/python/distributed/protocol.grpc.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

class DistributedRuntimeClient {
 public:
  explicit DistributedRuntimeClient(std::shared_ptr<::grpc::Channel> channel);
  ~DistributedRuntimeClient();

  xla::Status Connect(const LocalTopologyProto& local_topology,
                      GlobalTopologyProto* global_topology);

  xla::StatusOr<std::string> BlockingKeyValueGet(std::string key,
                                                 absl::Duration timeout);

  xla::Status KeyValueSet(std::string key, std::string value);

 private:
  const std::unique_ptr<grpc::DistributedRuntimeService::Stub> stub_;
  const absl::Duration rpc_timeout_ = absl::Seconds(120);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_DISTRIBUTED_CLIENT_H_
