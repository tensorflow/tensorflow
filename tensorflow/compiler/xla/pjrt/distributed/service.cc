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

#include "tensorflow/compiler/xla/pjrt/distributed/service.h"

#include "absl/time/time.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.h"
#include "tensorflow/compiler/xla/pjrt/distributed/util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/random.h"

namespace xla {

DistributedRuntimeServiceImpl::DistributedRuntimeServiceImpl(
    const Options& options)
    : options_(options), session_id_(tensorflow::random::New64()) {
  nodes_.resize(options.num_nodes);
  local_topologies_.resize(options.num_nodes);
}

DistributedRuntimeServiceImpl::~DistributedRuntimeServiceImpl() {
  {
    absl::MutexLock lock(&mu_);
    state_ = State::kClosed;
    service_status_ =
        tensorflow::errors::FailedPrecondition("Service shutting down.");
    if (!stop_heartbeat_thread_.HasBeenNotified()) {
      stop_heartbeat_thread_.Notify();
    }
  }
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

xla::Status DistributedRuntimeServiceImpl::ValidateNodeId(int node_id) {
  if (node_id < 0) {
    return xla::InvalidArgument("Invalid node ID %d, must be non-negative",
                                node_id);
  }
  if (node_id >= options_.num_nodes) {
    return xla::FailedPrecondition(
        "Invalid node ID %d, must be in the range [0, %d)", node_id,
        options_.num_nodes);
  }
  return xla::Status::OK();
}

xla::Status DistributedRuntimeServiceImpl::ValidateSessionId(
    uint64_t session_id) {
  if (session_id != session_id_) {
    return xla::FailedPrecondition(
        "Session ID of request %llu does not match active session ID %llu",
        session_id, session_id_);
  }
  return xla::Status::OK();
}

::grpc::Status DistributedRuntimeServiceImpl::Connect(
    ::grpc::ServerContext* context, const ConnectRequest* request,
    ConnectResponse* response) {
  VLOG(10) << "Connect " << request->DebugString();
  if (request->protocol_version() != DistributedRuntimeProtocolVersion()) {
    return ToGrpcStatus(xla::InvalidArgument("Invalid protocol version %d",
                                             request->protocol_version()));
  }
  absl::MutexLock lock(&mu_);
  if (state_ != State::kInitializing) {
    // This most likely indicates that a client task was restarted but the
    // old master is still up. Clients should retry on failure.
    return ToGrpcStatus(tensorflow::errors::Aborted(
        "Connect() called when system is not initializing."));
  }
  int node_id = request->node_id();
  xla::Status status = ValidateNodeId(node_id);
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  if (!nodes_[node_id].present) {
    nodes_[node_id].present = true;
    ++num_nodes_present_;
  }
  nodes_[node_id].client_id = request->client_id();

  auto all_nodes_present_or_duplicate_request = [&]() {
    mu_.AssertHeld();
    return num_nodes_present_ == nodes_.size() ||
           nodes_[node_id].client_id != request->client_id();
  };
  auto connect_timeout = absl::Milliseconds(request->timeout_milliseconds());
  if (!mu_.AwaitWithTimeout(
          absl::Condition(&all_nodes_present_or_duplicate_request),
          connect_timeout)) {
    nodes_[node_id].present = false;
    --num_nodes_present_;
    return ToGrpcStatus(tensorflow::errors::DeadlineExceeded(
        "Timed out after ", absl::FormatDuration(connect_timeout),
        " waiting for all nodes to call Connect()"));
  }

  if (nodes_[node_id].client_id != request->client_id()) {
    // This might happen either if two nodes are erroneously configured with the
    // same ID number, or it might happen if a task fails and is restarted
    // while we are waiting for nodes to connect. To elaborate on the second
    // scenario, it would look like this:
    // * a task calls Connect() with a particular node_id and client_id.
    // * the task is killed and restarted, or alternatively the client's RPC
    //   times out and it decides to retry.
    // * the task calls Connect() again with the same node_id and a different
    //   client_id.
    // In this scenario we take whichever client showed up most recently and
    // evict the client with an out-of-date client ID.
    return ToGrpcStatus(
        tensorflow::errors::Aborted("Duplicate node ID ", node_id));
  }

  if (node_id == 0) {
    state_ = State::kRunning;
    heartbeat_thread_.reset(options_.env->StartThread(
        tensorflow::ThreadOptions(), "pjrt_service_heartbeat",
        [this]() { HeartbeatLoop(); }));
  } else {
    auto running = [&]() {
      mu_.AssertHeld();
      return state_ == State::kRunning;
    };
    mu_.Await(absl::Condition(&running));
  }
  nodes_[node_id].last_heartbeat = absl::Now();
  response->set_session_id(session_id_);
  return ::grpc::Status::OK;
}

::grpc::Status DistributedRuntimeServiceImpl::Shutdown(
    ::grpc::ServerContext* context, const ShutdownRequest* request,
    ShutdownResponse* response) {
  VLOG(10) << "Shutdown " << request->DebugString();
  xla::Status status = ValidateSessionId(request->session_id());
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  absl::MutexLock lock(&mu_);
  if (state_ != State::kRunning) {
    if (!service_status_.ok()) {
      return ToGrpcStatus(service_status_);
    }
    return ToGrpcStatus(xla::FailedPrecondition(
        "Shutdown() called when system is not running."));
  }
  int node_id = request->node_id();
  status = ValidateNodeId(node_id);
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  ++num_nodes_shutting_down_;

  auto all_nodes_shutting_down = [&]() {
    mu_.AssertHeld();
    return num_nodes_shutting_down_ == nodes_.size() || !service_status_.ok();
  };
  if (!mu_.AwaitWithTimeout(absl::Condition(&all_nodes_shutting_down),
                            options_.shutdown_timeout)) {
    state_ = State::kClosed;
    return ToGrpcStatus(tensorflow::errors::DeadlineExceeded(
        "Timed out after ", absl::FormatDuration(options_.shutdown_timeout),
        " waiting for all nodes to call Shutdown()"));
  }
  state_ = State::kClosed;
  if (!stop_heartbeat_thread_.HasBeenNotified()) {
    stop_heartbeat_thread_.Notify();
  }
  if (!service_status_.ok()) {
    return ToGrpcStatus(service_status_);
  }
  return ::grpc::Status::OK;
}

::grpc::Status DistributedRuntimeServiceImpl::EnumerateDevices(
    ::grpc::ServerContext* context, const EnumerateDevicesRequest* request,
    EnumerateDevicesResponse* response) {
  VLOG(10) << "EnumerateDevices " << request->DebugString();
  xla::Status status = ValidateSessionId(request->session_id());
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  absl::MutexLock lock(&mu_);
  if (state_ != State::kRunning) {
    if (!service_status_.ok()) {
      return ToGrpcStatus(service_status_);
    }
    return ToGrpcStatus(xla::FailedPrecondition(
        "EnumerateDevices() called when system is not running."));
  }
  int node_id = request->local_topology().node_id();
  status = ValidateNodeId(node_id);
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  local_topologies_[node_id] = request->local_topology();
  ++num_topologies_present_;

  auto all_topologies_present = [&]() {
    mu_.AssertHeld();
    return num_topologies_present_ == nodes_.size() || !service_status_.ok();
  };
  if (!mu_.AwaitWithTimeout(absl::Condition(&all_topologies_present),
                            options_.enumerate_devices_timeout)) {
    return ToGrpcStatus(tensorflow::errors::DeadlineExceeded(
        "Timed out after ",
        absl::FormatDuration(options_.enumerate_devices_timeout),
        " waiting for all nodes to call EnumerateDevices()"));
  }
  if (!service_status_.ok()) {
    return ToGrpcStatus(service_status_);
  }

  if (node_id == 0) {
    topology_.emplace();
    BuildGlobalTopology(absl::Span<LocalTopologyProto>(local_topologies_),
                        &*topology_);
    local_topologies_.clear();
  } else {
    auto topology_ready = [&]() -> bool {
      mu_.AssertHeld();
      return topology_.has_value();
    };
    mu_.Await(absl::Condition(&topology_ready));
  }
  *response->mutable_global_topology() = *topology_;
  return ::grpc::Status::OK;
}

::grpc::Status DistributedRuntimeServiceImpl::Heartbeat(
    ::grpc::ServerContext* context, const HeartbeatRequest* request,
    HeartbeatResponse* response) {
  VLOG(10) << "Heartbeat " << request->DebugString();
  xla::Status status = ValidateSessionId(request->session_id());
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  absl::MutexLock lock(&mu_);
  if (state_ != State::kRunning) {
    if (!service_status_.ok()) {
      return ToGrpcStatus(service_status_);
    }
    return ToGrpcStatus(xla::FailedPrecondition(
        "Heartbeat() called when system is not running."));
  }
  int node_id = request->node_id();
  status = ValidateNodeId(node_id);
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  nodes_[node_id].last_heartbeat = absl::Now();
  return ::grpc::Status::OK;
}

void DistributedRuntimeServiceImpl::HeartbeatLoop() {
  while (true) {
    stop_heartbeat_thread_.WaitForNotificationWithTimeout(
        options_.heartbeat_interval);
    VLOG(10) << "Checking heartbeats";
    if (stop_heartbeat_thread_.HasBeenNotified()) {
      VLOG(10) << "Heartbeat checking stopped.";
      return;
    }
    absl::Time now = absl::Now();
    absl::MutexLock lock(&mu_);
    for (size_t i = 0; i < nodes_.size(); ++i) {
      // If we haven't heard from the node for a number of heartbeat intervals,
      // declare that we are unhealthy.
      VLOG(10) << "Node " << i
               << " last heartbeat: " << nodes_[i].last_heartbeat;
      if (nodes_[i].last_heartbeat +
              options_.max_missing_heartbeats * options_.heartbeat_interval <
          now) {
        LOG(INFO) << "Missed heartbeats from node " << i << ". Shutting down.";
        state_ = State::kClosed;
        service_status_ = tensorflow::errors::Aborted(
            "Shutting down due to missed heartbeat from task ", i);
        return;
      }
    }
  }
}

::grpc::Status DistributedRuntimeServiceImpl::KeyValueGet(
    ::grpc::ServerContext* context, const KeyValueGetRequest* request,
    KeyValueGetResponse* response) {
  VLOG(10) << "KeyValueGet " << request->DebugString();
  xla::Status status = ValidateSessionId(request->session_id());
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kRunning) {
      if (!service_status_.ok()) {
        return ToGrpcStatus(service_status_);
      }
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
  xla::Status status = ValidateSessionId(request->session_id());
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kRunning) {
      if (!service_status_.ok()) {
        return ToGrpcStatus(service_status_);
      }
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
    std::shared_ptr<::grpc::ServerCredentials> credentials,
    const DistributedRuntimeServiceImpl::Options& options) {
  auto service = absl::make_unique<DistributedRuntimeService>(options);
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(address, credentials);
  VLOG(1) << "Distributed runtime service address " << address;
  builder.RegisterService(&service->impl_);
  service->server_ = builder.BuildAndStart();
  if (!service->server_) {
    return xla::Unknown("Failed to start RPC server");
  }
  LOG(INFO) << "Jax service listening on " << address;
  return service;
}

DistributedRuntimeService::DistributedRuntimeService(
    const DistributedRuntimeServiceImpl::Options& options)
    : impl_(options) {}

DistributedRuntimeService::~DistributedRuntimeService() { Shutdown(); }

void DistributedRuntimeService::Shutdown() {
  if (server_) {
    LOG(INFO) << "Jax service shutting down";
    server_->Shutdown();
    server_->Wait();
  }
}

}  // namespace xla
