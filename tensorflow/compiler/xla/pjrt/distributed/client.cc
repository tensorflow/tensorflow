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

#include <algorithm>
#include <chrono>  // NOLINT
#include <random>
#include <string>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.h"
#include "tensorflow/compiler/xla/pjrt/distributed/util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_error_util.h"
#include "tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"

namespace xla {
class DistributedRuntimeClientImpl : public DistributedRuntimeClient {
 public:
  DistributedRuntimeClientImpl(std::shared_ptr<::grpc::Channel> channel,
                               const Options& options);
  explicit DistributedRuntimeClientImpl(
      std::shared_ptr<::grpc::Channel> channel)
      : DistributedRuntimeClientImpl(channel, Options()) {}
  ~DistributedRuntimeClientImpl() override;

  xla::Status Connect() override;
  xla::Status Shutdown() override;
  xla::Status EnumerateDevices(const LocalTopologyProto& local_topology,
                               GlobalTopologyProto* global_topology) override;
  xla::StatusOr<std::string> BlockingKeyValueGet(
      std::string key, absl::Duration timeout) override;
  xla::Status KeyValueSet(std::string key, std::string value) override;
  xla::Status WaitAtBarrier(std::string barrier_id,
                            absl::Duration timeout) override;

 private:
  // Entry point for the heartbeat thread.
  void HeartbeatLoop();

  const std::unique_ptr<grpc::DistributedRuntimeService::Stub> stub_;
  const DistributedRuntimeClient::Options options_;

  // Possible states of the client.
  // The only legal transitions are downwards in the order below. i.e., there is
  // no way to reopen a closed client.
  enum class State {
    // The client has not yet connected to the server, i.e., had a Connect()
    // RPC succeed.
    kNotConnected,

    // The client is connected to the server and as far as we are aware the
    // connection is healthy.
    kConnected,

    // The client is in the process of shutting down, i.e., Shutdown() has been
    // called.
    kShuttingDown,

    // The client has shut down its server connection, either due to an error
    // or due to an explicit shutdown.
    kClosed,
  };

  static absl::string_view StateToString(State state);

  // state_ is protected by a mutex because the heartbeat thread needs to look
  // at it.
  absl::Mutex mu_;
  State state_ ABSL_GUARDED_BY(mu_) = State::kNotConnected;

  // A unique session ID, assigned by the server during Connect().
  uint64_t session_id_;

  // Notification that tells the heartbeat thread to stop running.
  absl::Notification stop_heartbeats_;

  // Thread responsible for performing heartbeats.
  std::unique_ptr<tensorflow::Thread> heartbeat_thread_;
};

class DistributedRuntimeCoordinationServiceClient
    : public DistributedRuntimeClient {
 public:
  DistributedRuntimeCoordinationServiceClient(
      std::shared_ptr<::grpc::Channel> channel, const Options& options);
  explicit DistributedRuntimeCoordinationServiceClient(
      std::shared_ptr<::grpc::Channel> channel)
      : DistributedRuntimeCoordinationServiceClient(channel, Options()) {}
  ~DistributedRuntimeCoordinationServiceClient() override;

  xla::Status Connect() override;
  xla::Status Shutdown() override;
  xla::Status EnumerateDevices(const LocalTopologyProto& local_topology,
                               GlobalTopologyProto* global_topology) override;
  xla::StatusOr<std::string> BlockingKeyValueGet(
      std::string key, absl::Duration timeout) override;
  xla::Status KeyValueSet(std::string key, std::string value) override;
  xla::Status WaitAtBarrier(std::string barrier_id,
                            absl::Duration timeout) override;

 private:
  std::unique_ptr<tensorflow::CoordinationServiceAgent> coord_agent_;
  tensorflow::CoordinationServiceConfig config_;
  int task_id_;
};

DistributedRuntimeClientImpl::DistributedRuntimeClientImpl(
    std::shared_ptr<::grpc::Channel> channel, const Options& options)
    : stub_(grpc::DistributedRuntimeService::NewStub(std::move(channel))),
      options_(options) {}

DistributedRuntimeClientImpl::~DistributedRuntimeClientImpl() {
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

/*static*/ absl::string_view DistributedRuntimeClientImpl::StateToString(
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

xla::Status DistributedRuntimeClientImpl::Connect() {
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kNotConnected) {
      return xla::FailedPrecondition("Connect() called when client in state %s",
                                     StateToString(state_));
    }
  }
  ConnectRequest request;
  request.set_protocol_version(DistributedRuntimeProtocolVersion());
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
  return OkStatus();
}

xla::Status DistributedRuntimeClientImpl::EnumerateDevices(
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
  return OkStatus();
}

xla::Status DistributedRuntimeClientImpl::Shutdown() {
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
  return OkStatus();
}

xla::StatusOr<std::string> DistributedRuntimeClientImpl::BlockingKeyValueGet(
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
  request.set_timeout_milliseconds(absl::ToInt64Milliseconds(timeout));
  VLOG(10) << "BlockingKeyValueGet: " << request.DebugString();
  KeyValueGetResponse response;
  ::grpc::Status status = stub_->KeyValueGet(&ctx, request, &response);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }
  return response.value();
}

xla::Status DistributedRuntimeClientImpl::KeyValueSet(std::string key,
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

xla::Status DistributedRuntimeClientImpl::WaitAtBarrier(
    std::string barrier_id, absl::Duration timeout) {
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kConnected) {
      return xla::FailedPrecondition(
          "WaitAtBarrier() called when client not connected.");
    }
  }
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  WaitAtBarrierRequest request;
  request.set_session_id(session_id_);
  request.set_barrier_id(std::move(barrier_id));
  request.set_node_id(options_.node_id);
  // TODO(yashkatariya,hanyuangtay): Change timeout_milliseconds to int64 in
  // protocol.proto so that we don't need a minimum timeout here.
  timeout = std::min(timeout, absl::Minutes(10));  // Avoid overflow
  request.set_timeout_milliseconds(absl::ToInt64Milliseconds(timeout));
  VLOG(10) << "WaitAtBarrier: " << request.DebugString();
  WaitAtBarrierResponse response;
  ::grpc::Status status = stub_->WaitAtBarrier(&ctx, request, &response);
  return FromGrpcStatus(status);
}

void DistributedRuntimeClientImpl::HeartbeatLoop() {
  int num_missing_heartbeats = 0;
  while (true) {
    stop_heartbeats_.WaitForNotificationWithTimeout(
        options_.heartbeat_interval);
    if (stop_heartbeats_.HasBeenNotified()) {
      return;
    }

    ::grpc::ClientContext ctx;
    ctx.set_fail_fast(false);
    ctx.set_deadline(
        absl::ToChronoTime(absl::Now() + options_.heartbeat_interval));
    HeartbeatRequest request;
    request.set_session_id(session_id_);
    request.set_node_id(options_.node_id);
    VLOG(10) << "Heartbeat: " << request.DebugString();
    HeartbeatResponse response;
    ::grpc::Status status = stub_->Heartbeat(&ctx, request, &response);
    if (status.ok()) {
      VLOG(10) << "Heartbeat ok";
      num_missing_heartbeats = 0;
    } else {
      ++num_missing_heartbeats;
      VLOG(10) << "Heartbeat error, "
               << options_.max_missing_heartbeats - num_missing_heartbeats
               << " tries left: " << status.error_message();
      bool is_transient_error =
          (status.error_code() == ::grpc::StatusCode::DEADLINE_EXCEEDED ||
           status.error_code() == ::grpc::StatusCode::UNAVAILABLE);
      if (!stop_heartbeats_.HasBeenNotified() &&
          (!is_transient_error ||
           num_missing_heartbeats >= options_.max_missing_heartbeats)) {
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

DistributedRuntimeCoordinationServiceClient::
    DistributedRuntimeCoordinationServiceClient(
        std::shared_ptr<::grpc::Channel> channel, const Options& options) {
  // Convert options to coordination config.
  tensorflow::CoordinationServiceConfig config;
  config.set_service_type("standalone");
  config.set_service_leader("/job:jax_worker/task:0");
  config.set_cluster_register_timeout_in_ms(
      absl::ToInt64Milliseconds(options.init_timeout));
  config.set_heartbeat_timeout_in_ms(absl::ToInt64Milliseconds(
      options.heartbeat_interval * options.max_missing_heartbeats));
  config.set_shutdown_barrier_timeout_in_ms(
      absl::ToInt64Milliseconds(options.shutdown_timeout));
  config.set_agent_destruction_without_shutdown(
      !options.shutdown_on_destruction);
  auto error_fn =
      [timeout_fn = options.missed_heartbeat_callback](const Status& status) {
        LOG(ERROR) << "Coordination service agent in error status: " << status;
        timeout_fn(status, /*coordinator_reported_failure=*/true);
      };

  std::unique_ptr<tensorflow::CoordinationClient> leader_client;
  leader_client.reset(tensorflow::NewGrpcCoordinationClient(channel));
  coord_agent_ = tensorflow::CreateCoordinationServiceAgent();
  const Status status =
      coord_agent_->Initialize(options.env, "jax_worker", options.node_id,
                               config, std::move(leader_client), error_fn);
  if (!status.ok()) {
    LOG(ERROR) << "Coordination agent failed to initialize: " << status;
  }
  task_id_ = options.node_id;
  config_ = config;
}

DistributedRuntimeCoordinationServiceClient::
    ~DistributedRuntimeCoordinationServiceClient() {}

xla::Status DistributedRuntimeCoordinationServiceClient::Connect() {
  Status s = tensorflow::errors::Unknown("Connection not attempted yet.");
  absl::Duration timeout =
      absl::Milliseconds(config_.cluster_register_timeout_in_ms());
  absl::Time deadline = absl::Now() + timeout;
  int attempt = 0;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  do {
    ++attempt;
    s = coord_agent_->Connect();
    if (s.ok()) {
      s = coord_agent_->WaitAtBarrier("PjRT_Client_Connect", timeout,
                                      /*tasks=*/{});
    }
    // Exponential backoff with jitter. Note we will retry for `init_timeout`
    // time in total; the `14` here corresponds to an ~16s maximum interval
    // between connection attempts.

    int backoff = 1 << std::min(14, attempt);
    absl::SleepFor(absl::Milliseconds(backoff * distribution(generator)));
  } while (!s.ok() && absl::Now() < deadline &&
           // Retries are only made for RPC errors. If a valid service error is
           // returned, fail immediately.
           s.GetPayload(tensorflow::CoordinationErrorPayloadKey()) ==
               std::nullopt);
  if (s.ok()) {
    LOG(INFO) << "Connected to distributed JAX controller";
  } else {
    LOG(INFO) << "Failed to connect to distributed JAX controller: " << s;
  }
  return s;
}

xla::Status DistributedRuntimeCoordinationServiceClient::Shutdown() {
  LOG(INFO) << "Distributed task shutdown initiated.";
  Status s = coord_agent_->Shutdown();
  LOG(INFO) << "Distributed task shutdown result: " << s;
  return s;
}

xla::Status DistributedRuntimeCoordinationServiceClient::EnumerateDevices(
    const LocalTopologyProto& local_topology,
    GlobalTopologyProto* global_topology) {
  tensorflow::CoordinationServiceDeviceInfo devices;
  LocalTopologyProto* device =
      devices.mutable_xla()->mutable_devices()->add_nodes();
  *device = local_topology;
  device->set_node_id(task_id_);
  Status s = coord_agent_->WaitForAllTasks(devices);
  if (!s.ok()) return s;
  *global_topology = coord_agent_->GetClusterDeviceInfo().xla().devices();
  return OkStatus();
}

xla::StatusOr<std::string>
DistributedRuntimeCoordinationServiceClient::BlockingKeyValueGet(
    std::string key, absl::Duration timeout) {
  return coord_agent_->GetKeyValue(key, timeout);
}

xla::Status DistributedRuntimeCoordinationServiceClient::KeyValueSet(
    std::string key, std::string value) {
  return coord_agent_->InsertKeyValue(key, value);
}

xla::Status DistributedRuntimeCoordinationServiceClient::WaitAtBarrier(
    std::string barrier_id, absl::Duration timeout) {
  return coord_agent_->WaitAtBarrier(barrier_id, timeout, /*tasks=*/{});
}

std::unique_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::shared_ptr<::grpc::Channel> channel,
    const DistributedRuntimeClient::Options& options,
    bool use_coordination_service) {
  if (use_coordination_service) {
    return std::make_unique<xla::DistributedRuntimeCoordinationServiceClient>(
        channel, options);
  }
  return std::make_unique<xla::DistributedRuntimeClientImpl>(channel, options);
}
}  // namespace xla
