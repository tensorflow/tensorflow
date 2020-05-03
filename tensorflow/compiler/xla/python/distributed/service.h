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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_DISTRIBUTED_SERVICE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_DISTRIBUTED_SERVICE_H_

#include "absl/time/time.h"
#include "tensorflow/compiler/xla/python/distributed/key_value_store.h"
#include "tensorflow/compiler/xla/python/distributed/protocol.grpc.pb.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

typedef int NodeId;

class DistributedRuntimeServiceImpl final
    : public grpc::DistributedRuntimeService::Service {
 public:
  explicit DistributedRuntimeServiceImpl(int num_nodes);

  DistributedRuntimeServiceImpl(const DistributedRuntimeServiceImpl&) = delete;
  DistributedRuntimeServiceImpl(DistributedRuntimeServiceImpl&&) = delete;
  DistributedRuntimeServiceImpl& operator=(
      const DistributedRuntimeServiceImpl&) = delete;
  DistributedRuntimeServiceImpl&& operator=(DistributedRuntimeServiceImpl&&) =
      delete;

  ::grpc::Status Connect(::grpc::ServerContext* context,
                         const ConnectRequest* request,
                         ConnectResponse* response) override;

  ::grpc::Status KeyValueGet(::grpc::ServerContext* context,
                             const KeyValueGetRequest* request,
                             KeyValueGetResponse* response) override;

  ::grpc::Status KeyValueSet(::grpc::ServerContext* context,
                             const KeyValueSetRequest* request,
                             KeyValueSetResponse* response) override;

 private:
  const absl::Duration kConnectTimeout = absl::Seconds(120);

  absl::Mutex mu_;
  enum class State { kInitializing, kRunning };
  State state_ GUARDED_BY(mu_) = State::kInitializing;

  std::vector<LocalTopologyProto> local_topologies_ GUARDED_BY(mu_);
  GlobalTopologyProto topology_ GUARDED_BY(mu_);
  struct Node {
    bool present = false;
  };
  int num_nodes_present_ GUARDED_BY(mu_) = 0;
  std::vector<Node> nodes_ GUARDED_BY(mu_);

  KeyValueStore key_value_store_;
};

class DistributedRuntimeService {
 public:
  static xla::StatusOr<std::unique_ptr<DistributedRuntimeService>> Get(
      const std::string& address,
      std::shared_ptr<::grpc::ServerCredentials> credentials, int num_nodes);

  explicit DistributedRuntimeService(int num_nodes);
  ~DistributedRuntimeService();

  DistributedRuntimeService(const DistributedRuntimeService&) = delete;
  DistributedRuntimeService(DistributedRuntimeService&&) = delete;
  DistributedRuntimeService& operator=(const DistributedRuntimeService&) =
      delete;
  DistributedRuntimeService& operator=(DistributedRuntimeService&&) = delete;

  ::grpc::Server* server() const { return server_.get(); }

 private:
  DistributedRuntimeServiceImpl impl_;
  std::unique_ptr<::grpc::Server> server_;
};

// Everything below this point is exposed only for tests.

// Given a LocalTopologyProto object from each node, builds a
// GlobalTopologyProto that describes all nodes.
void BuildGlobalTopology(absl::Span<LocalTopologyProto> local_topologies,
                         GlobalTopologyProto* global_topology);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_DISTRIBUTED_SERVICE_H_
