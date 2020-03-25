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

#include "tensorflow/compiler/xla/python/distributed/service.h"

#include "tensorflow/compiler/xla/python/distributed/protocol.h"
#include "tensorflow/compiler/xla/python/distributed/util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

DistributedRuntimeServiceImpl::DistributedRuntimeServiceImpl(int num_nodes) {
  nodes_.resize(num_nodes);
  local_topologies_.resize(num_nodes);
}

// Steals the contents of `local_topologies`.
void BuildGlobalTopology(absl::Span<LocalTopologyProto> local_topologies,
                         GlobalTopologyProto* global_topology) {
  int next_global_device_id = 0;
  for (LocalTopologyProto& local : local_topologies) {
    for (DeviceProto& device : *local.mutable_devices()) {
      device.set_global_device_id(next_global_device_id++);
    }
    global_topology->add_nodes()->Swap(&local);
  }
}

::grpc::Status DistributedRuntimeServiceImpl::Connect(
    ::grpc::ServerContext* context, const ConnectRequest* request,
    ConnectResponse* response) {
  VLOG(10) << "Connect " << request->DebugString();
  if (request->protocol_version() != kDistributedRuntimeProtocolVersion) {
    return ToGrpcStatus(xla::InvalidArgument("Invalid protocol version %d",
                                             request->protocol_version()));
  }
  absl::MutexLock lock(&mu_);
  if (state_ != State::kInitializing) {
    return ToGrpcStatus(xla::FailedPrecondition(
        "Connect() called when system is not initializing."));
  }
  int node_id = request->local_topology().node_id();
  if (node_id < 0 || node_id >= nodes_.size()) {
    return ToGrpcStatus(
        xla::InvalidArgument("Invalid node ID %d, must be in the range [0, %d)",
                             node_id, nodes_.size()));
  }
  if (nodes_[node_id].present) {
    return ToGrpcStatus(xla::InvalidArgument("Duplicate node ID %d", node_id));
  }
  nodes_[node_id].present = true;
  local_topologies_[node_id] = request->local_topology();
  ++num_nodes_present_;

  auto all_nodes_present = [&]() {
    mu_.AssertHeld();
    return num_nodes_present_ == nodes_.size();
  };
  if (!mu_.AwaitWithTimeout(absl::Condition(&all_nodes_present),
                            kConnectTimeout)) {
    return ToGrpcStatus(tensorflow::errors::DeadlineExceeded(
        "Timed out after %s waiting for all nodes to call Connect()",
        absl::FormatDuration(kConnectTimeout)));
  }

  if (node_id == 0) {
    BuildGlobalTopology(absl::Span<LocalTopologyProto>(local_topologies_),
                        &topology_);
    local_topologies_.clear();
    state_ = State::kRunning;
  } else {
    auto running = [&]() {
      mu_.AssertHeld();
      return state_ == State::kRunning;
    };
    mu_.Await(absl::Condition(&running));
  }
  *response->mutable_global_topology() = topology_;
  return ::grpc::Status::OK;
}

::grpc::Status DistributedRuntimeServiceImpl::KeyValueGet(
    ::grpc::ServerContext* context, const KeyValueGetRequest* request,
    KeyValueGetResponse* response) {
  VLOG(10) << "KeyValueGet " << request->DebugString();
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kRunning) {
      return ToGrpcStatus(xla::FailedPrecondition(
          "KeyValueGet() called when system is not running."));
    }
  }
  return key_value_store_.Get(
      request->key(), absl::Milliseconds(request->timeout_milliseconds()),
      response->mutable_value());
}

::grpc::Status DistributedRuntimeServiceImpl::KeyValueSet(
    ::grpc::ServerContext* context, const KeyValueSetRequest* request,
    KeyValueSetResponse* response) {
  VLOG(10) << "KeyValueSet " << request->DebugString();
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kRunning) {
      return ToGrpcStatus(xla::FailedPrecondition(
          "KeyValueSet() called when system is not running; clients must call "
          "Connect() first"));
    }
  }
  return key_value_store_.Set(request->key(), request->value());
}

xla::StatusOr<std::unique_ptr<DistributedRuntimeService>>
DistributedRuntimeService::Get(
    const std::string& address,
    std::shared_ptr<::grpc::ServerCredentials> credentials, int num_nodes) {
  auto service = absl::make_unique<DistributedRuntimeService>(num_nodes);
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(address, credentials);
  VLOG(1) << "Distributed runtmie service address " << address;
  builder.RegisterService(&service->impl_);
  service->server_ = builder.BuildAndStart();
  if (!service->server_) {
    return xla::Unknown("Failed to start RPC server");
  }
  LOG(INFO) << "Jax service listening on " << address;
  return service;
}

DistributedRuntimeService::DistributedRuntimeService(int num_nodes)
    : impl_(num_nodes) {}

DistributedRuntimeService::~DistributedRuntimeService() {
  if (server_) {
    LOG(INFO) << "Jax service shutting down";
    server_->Shutdown();
    server_->Wait();
  }
}

}  // namespace xla
