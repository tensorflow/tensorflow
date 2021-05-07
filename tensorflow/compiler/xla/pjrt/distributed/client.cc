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

#include "tensorflow/compiler/xla/pjrt/distributed/client.h"

#include <chrono>  // NOLINT
#include <random>

#include "absl/time/time.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.h"
#include "tensorflow/compiler/xla/pjrt/distributed/util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/random.h"

namespace xla {

DistributedRuntimeClient::DistributedRuntimeClient(
    std::shared_ptr<::grpc::Channel> channel, const Options& options)
    : stub_(grpc::DistributedRuntimeService::NewStub(std::move(channel))),
      options_(options) {}

DistributedRuntimeClient::~DistributedRuntimeClient() {
  bool connected;
  {
    absl::MutexLock lock(&mu_);
    connected = (state_ == State::kConnected);
  }
  if (connected) {
    if (options_.shutdown_on_destruction) {
      Status status = Shutdown();
      if (!status.ok()) {
        LOG(WARNING) << "PJRT shutdown failed: " << status;
      }
    } else {
      if (!stop_heartbeats_.HasBeenNotified()) {
        stop_heartbeats_.Notify();
      }
    }
  }
}

/*static*/ absl::string_view DistributedRuntimeClient::StateToString(
    State state) {
  switch (state) {
    case State::kNotConnected:
      return "kNotConnected";
    case State::kConnected:
      return "kConnected";
    case State::kShuttingDown:
      return "kShuttingDown";
    case State::kClosed:
      return "kClosed";
  }
}

xla::Status DistributedRuntimeClient::Connect() {
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kNotConnected) {
      return xla::FailedPrecondition("Connect() called when client in state %s",
                                     StateToString(state_));
    }
  }
  ConnectRequest request;
  request.set_protocol_version(kDistributedRuntimeProtocolVersion);
  request.set_timeout_milliseconds(
      absl::ToInt64Milliseconds(options_.rpc_timeout) / 2);
  request.set_node_id(options_.node_id);
  VLOG(10) << "Connect: " << request.DebugString();
  ConnectResponse response;
  ::grpc::Status status;
  absl::Time deadline = absl::Now() + options_.init_timeout;
  int attempt = 0;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  do {
    ::grpc::ClientContext ctx;
    ctx.set_fail_fast(false);
    ctx.set_deadline(absl::ToChronoTime(absl::Now() + options_.rpc_timeout));
    request.set_client_id(tensorflow::random::New64());
    response.Clear();
    status = stub_->Connect(&ctx, request, &response);
    if (!status.ok()) {
      VLOG(1) << "Connect failed() with status: " << FromGrpcStatus(status);
      if (attempt % 10 == 0) {
        LOG(INFO) << "Connect failed() with status: " << FromGrpcStatus(status);
      }
      // Exponential backoff with jitter. Note we will retry for `init_timeout`
      // time in total; the `14` here corresponds to an ~16s maximum interval
      // between connection attempts.
      int backoff = 1 << std::min(14, attempt);
      absl::SleepFor(absl::Milliseconds(backoff * distribution(generator)));
    }
    ++attempt;
  } while (!status.ok() && absl::Now() < deadline);
  if (!status.ok()) {
    LOG(ERROR) << "Connect() failed after " << attempt << " retries in "
               << options_.init_timeout
               << "; most recent failure status: " << FromGrpcStatus(status);
    return tensorflow::errors::DeadlineExceeded(
        absl::StrFormat("Connect() timed out after %s with %d attempts. Most "
                        "recent failure was: %s",
                        absl::FormatDuration(options_.init_timeout), attempt,
                        FromGrpcStatus(status).ToString()));
  }
  VLOG(10) << "Connect() response: " << response.DebugString();
  {
    absl::MutexLock lock(&mu_);
    state_ = State::kConnected;
  }
  session_id_ = response.session_id();

  heartbeat_thread_.reset(options_.env->StartThread(
      tensorflow::ThreadOptions(), "pjrt_distributed_heartbeat",
      [this]() { HeartbeatLoop(); }));
  LOG(INFO) << "Connected to distributed JAX controller";
  return xla::Status::OK();
}

xla::Status DistributedRuntimeClient::EnumerateDevices(
    const LocalTopologyProto& local_topology,
    GlobalTopologyProto* global_topology) {
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kConnected) {
      return xla::FailedPrecondition(
          "EnumerateDevices() called when client not connected.");
    }
  }
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + options_.rpc_timeout));
  EnumerateDevicesRequest request;
  request.set_session_id(session_id_);
  *request.mutable_local_topology() = local_topology;
  request.mutable_local_topology()->set_node_id(options_.node_id);

  VLOG(10) << "EnumerateDevices: " << request.DebugString();
  EnumerateDevicesResponse response;
  ::grpc::Status status = stub_->EnumerateDevices(&ctx, request, &response);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }
  VLOG(10) << "EnumerateDevices() response: " << response.DebugString();
  response.mutable_global_topology()->Swap(global_topology);
  return xla::Status::OK();
}

xla::Status DistributedRuntimeClient::Shutdown() {
  LOG(INFO) << "Waiting for all distributed JAX tasks to shut down.";
  ::grpc::ClientContext ctx;
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kConnected) {
      return xla::FailedPrecondition(
          "Shutdown() called when client not connected.");
    }
    state_ = State::kShuttingDown;
  }
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + options_.shutdown_timeout));
  ShutdownRequest request;
  request.set_session_id(session_id_);
  VLOG(10) << "Shutdown: " << request.DebugString();
  ShutdownResponse response;
  ::grpc::Status status = stub_->Shutdown(&ctx, request, &response);
  LOG(INFO) << "Distributed task shutdown result: " << FromGrpcStatus(status);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }
  if (!stop_heartbeats_.HasBeenNotified()) {
    stop_heartbeats_.Notify();
  }
  VLOG(10) << "Shutdown() response: " << response.DebugString();
  absl::MutexLock lock(&mu_);
  state_ = State::kClosed;
  return xla::Status::OK();
}

xla::StatusOr<std::string> DistributedRuntimeClient::BlockingKeyValueGet(
    std::string key, absl::Duration timeout) {
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kConnected) {
      return xla::FailedPrecondition(
          "BlockingKeyValueGet() called when client not connected.");
    }
  }
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  KeyValueGetRequest request;
  request.set_session_id(session_id_);
  request.set_key(std::move(key));
  timeout = std::min(timeout, absl::Minutes(10));  // Avoid overflow
  request.set_timeout_milliseconds(timeout / absl::Milliseconds(1));
  VLOG(10) << "BlockingKeyValueGet: " << request.DebugString();
  KeyValueGetResponse response;
  ::grpc::Status status = stub_->KeyValueGet(&ctx, request, &response);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }
  return response.value();
}

xla::Status DistributedRuntimeClient::KeyValueSet(std::string key,
                                                  std::string value) {
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kConnected) {
      return xla::FailedPrecondition(
          "KeyValueSet() called when client not connected.");
    }
  }
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + options_.rpc_timeout));
  KeyValueSetRequest request;
  request.set_session_id(session_id_);
  request.set_key(std::move(key));
  request.set_value(std::move(value));
  VLOG(10) << "KeyValueSet: " << request.DebugString();
  KeyValueSetResponse response;
  ::grpc::Status status = stub_->KeyValueSet(&ctx, request, &response);
  return FromGrpcStatus(status);
}

void DistributedRuntimeClient::HeartbeatLoop() {
  int num_missing_heartbeats = 0;
  while (true) {
    stop_heartbeats_.WaitForNotificationWithTimeout(
        options_.heartbeat_interval);
    if (stop_heartbeats_.HasBeenNotified()) {
      return;
    }

    ::grpc::ClientContext ctx;
    ctx.set_fail_fast(false);
    ctx.set_deadline(absl::ToChronoTime(absl::Now() + options_.rpc_timeout));
    HeartbeatRequest request;
    request.set_session_id(session_id_);
    request.set_node_id(options_.node_id);
    VLOG(10) << "Heartbeat: " << request.DebugString();
    HeartbeatResponse response;
    ::grpc::Status status = stub_->Heartbeat(&ctx, request, &response);
    if (status.ok()) {
      num_missing_heartbeats = 0;
    } else {
      ++num_missing_heartbeats;
      bool is_transient_error =
          (status.error_code() == ::grpc::StatusCode::DEADLINE_EXCEEDED ||
           status.error_code() == ::grpc::StatusCode::UNAVAILABLE);
      if (!stop_heartbeats_.HasBeenNotified() &&
          (!is_transient_error ||
           num_missing_heartbeats > options_.max_missing_heartbeats)) {
        // If we are shutting down, missed heartbeats are benign: they may
        // simply mean that the server has shut down already before it saw
        // the heartbeat request.
        absl::MutexLock lock(&mu_);
        if (state_ != State::kShuttingDown) {
          options_.missed_heartbeat_callback(FromGrpcStatus(status),
                                             !is_transient_error);
        }
        return;
      }
    }
  }
}

}  // namespace xla
