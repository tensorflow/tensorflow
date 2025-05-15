/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_PJRT_DISTRIBUTED_SERVICE_H_
#define XLA_PJRT_DISTRIBUTED_SERVICE_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"
#include "xla/tsl/distributed_runtime/rpc/async_service_interface.h"
#include "xla/types.h"
#include "tsl/platform/env.h"
#include "tsl/platform/threadpool.h"

namespace xla {

typedef int NodeId;

class CoordinationServiceImpl {
 public:
  struct Options {
    // Number of nodes in the job. Mandatory. Must be non-negative.
    int num_nodes = -1;

    tsl::Env* env = tsl::Env::Default();

    // Interval at which the service should check for missed heartbeat RPCs
    // from the clients.
    absl::Duration heartbeat_interval = absl::Seconds(10);

    // Number of heartbeats that a client may miss in a row before the
    // coordinator concludes that a client has vanished.
    int max_missing_heartbeats = 10;

    // How long should we wait for all clients to call Connect() before
    // giving up?
    absl::Duration cluster_register_timeout = absl::Minutes(60);

    // How long should we wait for all clients to call Shutdown() before giving
    // up and returning a failure?
    absl::Duration shutdown_timeout = absl::Minutes(5);
  };

  CoordinationServiceImpl(const Options& options,
                          ::grpc::ServerBuilder* builder);
  ~CoordinationServiceImpl();

  // Must be called after gRPC server has started.
  void StartRpcThread();

  CoordinationServiceImpl(const CoordinationServiceImpl&) = delete;
  CoordinationServiceImpl(CoordinationServiceImpl&&) = delete;
  CoordinationServiceImpl& operator=(const CoordinationServiceImpl&) = delete;
  CoordinationServiceImpl&& operator=(CoordinationServiceImpl&&) = delete;

 private:
  tsl::Env* env_ = nullptr;  // Not owned.
  std::unique_ptr<tsl::CoordinationService> coord_service_;
  std::unique_ptr<tsl::thread::ThreadPool> coord_compute_pool_;
  std::unique_ptr<tsl::AsyncServiceInterface> coord_rpc_service_;
  std::unique_ptr<tsl::Thread> coord_rpc_thread_;
};

class DistributedRuntimeService {
 public:
  static absl::StatusOr<std::unique_ptr<DistributedRuntimeService>> Get(
      const std::string& address,
      std::shared_ptr<::grpc::ServerCredentials> credentials,
      const CoordinationServiceImpl::Options& options);

  explicit DistributedRuntimeService(
      const CoordinationServiceImpl::Options& options,
      ::grpc::ServerBuilder* builder);
  ~DistributedRuntimeService();

  DistributedRuntimeService(const DistributedRuntimeService&) = delete;
  DistributedRuntimeService(DistributedRuntimeService&&) = delete;
  DistributedRuntimeService& operator=(const DistributedRuntimeService&) =
      delete;
  DistributedRuntimeService& operator=(DistributedRuntimeService&&) = delete;

  void Shutdown();

  ::grpc::Server* server() const { return server_.get(); }

 private:
  std::unique_ptr<CoordinationServiceImpl> coord_impl_;
  std::unique_ptr<::grpc::Server> server_;
};

}  // namespace xla

#endif  // XLA_PJRT_DISTRIBUTED_SERVICE_H_
