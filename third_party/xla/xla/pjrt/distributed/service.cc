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

#include "xla/pjrt/distributed/service.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/server_builder.h"
#include "xla/pjrt/distributed/coordination/coordination_service.h"
#include "xla/pjrt/distributed/coordination/grpc_coordination_service_impl.h"
#include "xla/tsl/distributed_runtime/rpc/async_service_interface.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/threadpool.h"

namespace {

std::unique_ptr<xla::CoordinationService> EnableCoordinationService(
    const xla::CoordinationServiceImpl::Options& options) {
  const std::string job_name = "jax_worker";
  xla::CoordinationService::Config config;
  config.cluster_register_timeout = options.cluster_register_timeout;
  config.cluster_register_with_barrier = true;
  config.heartbeat_timeout = options.heartbeat_timeout;
  config.shutdown_barrier_timeout = options.shutdown_timeout;
  config.job_name = job_name;
  config.num_tasks = options.num_nodes;
  auto service =
      std::make_unique<xla::CoordinationService>(options.env, config);
  return service;
}
}  // namespace

namespace xla {

CoordinationServiceImpl::CoordinationServiceImpl(
    const CoordinationServiceImpl::Options& options,
    ::grpc::ServerBuilder* builder)
    : env_(options.env) {
  coord_service_ = EnableCoordinationService(options);
  coord_compute_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      options.env, "CoordinationServiceRpcHandler",
      /*num_threads=*/4);
  coord_rpc_service_ = std::make_unique<GrpcCoordinationServiceImpl>(
      coord_compute_pool_.get(), builder);
  auto* grpc_coord_service =
      static_cast<GrpcCoordinationServiceImpl*>(coord_rpc_service_.get());
  grpc_coord_service->SetCoordinationServiceInstance(coord_service_.get());
  LOG(INFO) << "Coordination service is enabled.";
}

CoordinationServiceImpl::~CoordinationServiceImpl() {
  // Service object must be destroyed to clear all pending RPCs before shutting
  // down the RPC service.
  coord_service_ = nullptr;
  static_cast<GrpcCoordinationServiceImpl*>(coord_rpc_service_.get())
      ->SetCoordinationServiceInstance(nullptr);
  coord_rpc_service_->Shutdown();
}

void CoordinationServiceImpl::StartRpcThread() {
  coord_rpc_thread_.reset(env_->StartThread(
      tsl::ThreadOptions(), "CoordinationServiceHandleRPCsLoop",
      [service = coord_rpc_service_.get()] { service->HandleRPCsLoop(); }));
}

absl::StatusOr<std::unique_ptr<DistributedRuntimeService>>
DistributedRuntimeService::Get(
    const std::string& address,
    std::shared_ptr<::grpc::ServerCredentials> credentials,
    const CoordinationServiceImpl::Options& options) {
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(address, credentials);
  builder.SetMaxReceiveMessageSize(-1);
  builder.SetMaxSendMessageSize(-1);
  VLOG(1) << "Distributed runtime service address " << address;
  auto service = std::make_unique<DistributedRuntimeService>(options, &builder);
  if (!service->server_) {
    return xla::Unknown("Failed to start RPC server");
  }
  LOG(INFO) << "Jax service listening on " << address;
  return service;
}

DistributedRuntimeService::DistributedRuntimeService(
    const CoordinationServiceImpl::Options& options,
    ::grpc::ServerBuilder* builder) {
  coord_impl_ = std::make_unique<CoordinationServiceImpl>(options, builder);
  server_ = builder->BuildAndStart();
  coord_impl_->StartRpcThread();
}

DistributedRuntimeService::~DistributedRuntimeService() { Shutdown(); }

void DistributedRuntimeService::Shutdown() {
  if (server_) {
    LOG(INFO) << "Jax service shutting down";
    server_->Shutdown(absl::ToChronoTime(absl::Now() + absl::Seconds(5)));
    server_->Wait();
  }

  // Explicitly destroy coordination service before the gRPC server. This clears
  // all pending RPCs before the gRPC server is destroyed.
  coord_impl_ = nullptr;
  server_ = nullptr;
}

}  // namespace xla
