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
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_error_util.h"
#include "xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_client.h"
#include "tsl/platform/errors.h"
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

  absl::Status Connect() override;
  absl::Status Shutdown() override;
  absl::StatusOr<std::string> BlockingKeyValueGet(
      std::string_view key, absl::Duration timeout) override;
  absl::StatusOr<std::vector<std::pair<std::string, std::string>>>
  KeyValueDirGet(std::string_view key) override;
  absl::Status KeyValueSet(std::string_view key,
                           std::string_view value) override;
  absl::Status KeyValueDelete(std::string_view key) override;
  absl::Status WaitAtBarrier(std::string barrier_id,
                             absl::Duration timeout) override;
  absl::StatusOr<tsl::CoordinationServiceAgent*> GetCoordinationServiceAgent()
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

absl::Status DistributedRuntimeCoordinationServiceClient::Connect() {
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

absl::Status DistributedRuntimeCoordinationServiceClient::Shutdown() {
  LOG(INFO) << "Distributed task shutdown initiated.";
  Status s = coord_agent_->Shutdown();
  LOG(INFO) << "Distributed task shutdown result: " << s;
  return s;
}

absl::StatusOr<std::string>
DistributedRuntimeCoordinationServiceClient::BlockingKeyValueGet(
    std::string_view key, absl::Duration timeout) {
  return coord_agent_->GetKeyValue(key, timeout);
}

absl::StatusOr<std::vector<std::pair<std::string, std::string>>>
DistributedRuntimeCoordinationServiceClient::KeyValueDirGet(
    std::string_view key) {
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

absl::Status DistributedRuntimeCoordinationServiceClient::KeyValueDelete(
    std::string_view key) {
  return coord_agent_->DeleteKeyValue(key);
}

absl::Status DistributedRuntimeCoordinationServiceClient::KeyValueSet(
    std::string_view key, std::string_view value) {
  return coord_agent_->InsertKeyValue(key, value);
}

absl::Status DistributedRuntimeCoordinationServiceClient::WaitAtBarrier(
    std::string barrier_id, absl::Duration timeout) {
  return coord_agent_->WaitAtBarrier(barrier_id, timeout, /*tasks=*/{});
}

absl::StatusOr<tsl::CoordinationServiceAgent*>
DistributedRuntimeCoordinationServiceClient::GetCoordinationServiceAgent() {
  return coord_agent_.get();
}

std::unique_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::shared_ptr<::grpc::Channel> channel,
    const DistributedRuntimeClient::Options& options) {
  return std::make_unique<xla::DistributedRuntimeCoordinationServiceClient>(
      channel, options);
}

namespace {

class DistributedKeyValueStore : public KeyValueStoreInterface {
 public:
  DistributedKeyValueStore(std::shared_ptr<DistributedRuntimeClient> client,
                           std::string prefix)
      : client_(std::move(client)), prefix_(std::move(prefix)) {}

  absl::StatusOr<std::string> Get(std::string_view key,
                                  absl::Duration timeout) override {
    return client_->BlockingKeyValueGet(absl::StrCat(prefix_, key), timeout);
  }

  absl::Status Set(std::string_view key, std::string_view value) override {
    return client_->KeyValueSet(absl::StrCat(prefix_, key), value);
  }

 private:
  std::shared_ptr<DistributedRuntimeClient> client_;
  std::string prefix_;
};

}  // namespace

std::shared_ptr<KeyValueStoreInterface> GetDistributedKeyValueStore(
    std::shared_ptr<DistributedRuntimeClient> client, std::string prefix) {
  return std::make_shared<DistributedKeyValueStore>(std::move(client),
                                                    std::move(prefix));
}

}  // namespace xla
