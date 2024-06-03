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
#include <utility>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/server_builder.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"
#include "xla/tsl/distributed_runtime/rpc/async_service_interface.h"
#include "xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_service_impl.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/threadpool.h"
#include "tsl/protobuf/coordination_config.pb.h"

namespace {
constexpr int kBarrierTimedOut = -1000;

std::unique_ptr<tsl::CoordinationServiceInterface> EnableCoordinationService(
    const xla::CoordinationServiceImpl::Options& options) {
  const std::string job_name = "jax_worker";
  tensorflow::CoordinationServiceConfig config;
  config.set_service_type("standalone");
  config.set_service_leader(absl::StrCat("/job:", job_name, "/task:0"));
  config.set_cluster_register_timeout_in_ms(
      absl::ToInt64Milliseconds(options.cluster_register_timeout));
  config.set_heartbeat_timeout_in_ms(absl::ToInt64Milliseconds(
      options.heartbeat_interval * options.max_missing_heartbeats));
  config.set_shutdown_barrier_timeout_in_ms(
      absl::ToInt64Milliseconds(options.shutdown_timeout));
  tensorflow::CoordinatedJob* job =
      config.mutable_coordinated_job_list()->Add();
  job->set_name(job_name);
  job->set_num_tasks(options.num_nodes);
  auto service = tsl::CoordinationServiceInterface::EnableCoordinationService(
      options.env, config, /*cache=*/nullptr);
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
  coord_rpc_service_ = std::make_unique<tsl::GrpcCoordinationServiceImpl>(
      coord_compute_pool_.get(), builder);
  auto* grpc_coord_service =
      static_cast<tsl::GrpcCoordinationServiceImpl*>(coord_rpc_service_.get());
  grpc_coord_service->SetCoordinationServiceInstance(coord_service_.get());
  LOG(INFO) << "Coordination service is enabled.";
}

CoordinationServiceImpl::~CoordinationServiceImpl() {
  // Service object must be destroyed to clear all pending RPCs before shutting
  // down the RPC service.
  coord_service_ = nullptr;
  static_cast<tsl::GrpcCoordinationServiceImpl*>(coord_rpc_service_.get())
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
