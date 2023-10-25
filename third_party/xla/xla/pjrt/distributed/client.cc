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

#include "xla/pjrt/distributed/client.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "xla/pjrt/distributed/protocol.h"
#include "xla/pjrt/distributed/util.h"
#include "xla/util.h"
#include "tsl/distributed_runtime/coordination/coordination_client.h"
#include "tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "tsl/distributed_runtime/coordination/coordination_service_error_util.h"
#include "tsl/distributed_runtime/rpc/coordination/grpc_coordination_client.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/random.h"
#include "tsl/protobuf/coordination_config.pb.h"
#include "tsl/protobuf/coordination_service.pb.h"

namespace xla {


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
  xla::StatusOr<std::vector<std::pair<std::string, std::string>>>
  KeyValueDirGet(absl::string_view key) override;
  xla::Status KeyValueSet(std::string key, std::string value) override;
  xla::Status KeyValueDelete(std::string key) override;
  xla::Status WaitAtBarrier(std::string barrier_id,
                            absl::Duration timeout) override;
  xla::StatusOr<tsl::CoordinationServiceAgent*> GetCoordinationServiceAgent()
      override;

 private:
  std::unique_ptr<tsl::CoordinationServiceAgent> coord_agent_;
  tensorflow::CoordinationServiceConfig config_;
  absl::Duration min_connect_barrier_timeout_;
  int task_id_;
};

DistributedRuntimeCoordinationServiceClient::
    DistributedRuntimeCoordinationServiceClient(
        std::shared_ptr<::grpc::Channel> channel, const Options& options) {
  // Convert options to coordination config.
  tensorflow::CoordinationServiceConfig config;
  config.set_service_type("standalone");
  config.set_service_leader("/job:jax_worker/task:0");
  config.set_cluster_register_timeout_in_ms(
      absl::ToInt64Milliseconds(options.init_timeout));
  min_connect_barrier_timeout_ = options.rpc_timeout;
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

  std::unique_ptr<tsl::CoordinationClient> leader_client;
  leader_client.reset(tsl::NewGrpcCoordinationClient(channel));
  coord_agent_ = tsl::CreateCoordinationServiceAgent();
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
    ~DistributedRuntimeCoordinationServiceClient() = default;

xla::Status DistributedRuntimeCoordinationServiceClient::Connect() {
  const absl::Time deadline =
      absl::Now() +
      absl::Milliseconds(config_.cluster_register_timeout_in_ms());

  Status s = coord_agent_->Connect();
  if (s.ok()) {
    absl::Duration barrier_timeout = deadline - absl::Now();
    // Note: `init_timeout` in client options may be set to 0 so that the
    // client only attempts to connect once. In that case, we provide some
    // buffer time to wait for all tasks.
    barrier_timeout = std::max(barrier_timeout, min_connect_barrier_timeout_);
    s = coord_agent_->WaitAtBarrier("PjRT_Client_Connect", barrier_timeout,
                                    /*tasks=*/{});
  }
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
  LocalTopologyProto local_device = local_topology;
  local_device.set_node_id(task_id_);
  tensorflow::DeviceInfo devices;
  devices.mutable_device()->Add()->PackFrom(local_device);
  // Client sends LocalTopologyProto.
  Status s = coord_agent_->WaitForAllTasks(devices);
  if (!s.ok()) return s;
  // Server responds with GlobalTopologyProto (refer to service.cc for details).
  tensorflow::DeviceInfo global_devices = coord_agent_->GetClusterDeviceInfo();
  if (global_devices.device_size() != 1) {
    return tsl::errors::Internal(
        "Unexpected cluster device response from EnumerateDevices().");
  }
  global_devices.device().Get(0).UnpackTo(global_topology);
  return OkStatus();
}

xla::StatusOr<std::string>
DistributedRuntimeCoordinationServiceClient::BlockingKeyValueGet(
    std::string key, absl::Duration timeout) {
  return coord_agent_->GetKeyValue(key, timeout);
}

xla::StatusOr<std::vector<std::pair<std::string, std::string>>>
DistributedRuntimeCoordinationServiceClient::KeyValueDirGet(
    absl::string_view key) {
  // TODO(hanyangtay): Migrate to string_view for both client and coordination
  // agent APIs.
  TF_ASSIGN_OR_RETURN(const auto results,
                      coord_agent_->GetKeyValueDir(std::string(key)));

  std::vector<std::pair<std::string, std::string>> kvs;
  kvs.reserve(results.size());

  // Convert tensorflow::KeyValueEntry to std::pair<std::string,
  // string>.
  for (const auto& kv : results) {
    kvs.push_back(std::make_pair(kv.key(), kv.value()));
  }
  return kvs;
}

xla::Status DistributedRuntimeCoordinationServiceClient::KeyValueDelete(
    std::string key) {
  return coord_agent_->DeleteKeyValue(key);
}

xla::Status DistributedRuntimeCoordinationServiceClient::KeyValueSet(
    std::string key, std::string value) {
  return coord_agent_->InsertKeyValue(key, value);
}

xla::Status DistributedRuntimeCoordinationServiceClient::WaitAtBarrier(
    std::string barrier_id, absl::Duration timeout) {
  return coord_agent_->WaitAtBarrier(barrier_id, timeout, /*tasks=*/{});
}

xla::StatusOr<tsl::CoordinationServiceAgent*>
DistributedRuntimeCoordinationServiceClient::GetCoordinationServiceAgent() {
  return coord_agent_.get();
}

std::unique_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::shared_ptr<::grpc::Channel> channel,
    const DistributedRuntimeClient::Options& options) {
  return std::make_unique<xla::DistributedRuntimeCoordinationServiceClient>(
      channel, options);
}
}  // namespace xla
