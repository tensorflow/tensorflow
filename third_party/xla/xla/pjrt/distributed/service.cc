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

#include "xla/pjrt/distributed/service.h"

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "grpcpp/server_builder.h"
#include "xla/pjrt/distributed/protocol.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/distributed/util.h"
#include "xla/status.h"
#include "xla/util.h"
#include "tsl/distributed_runtime/coordination/coordination_service.h"
#include "tsl/distributed_runtime/rpc/async_service_interface.h"
#include "tsl/distributed_runtime/rpc/coordination/grpc_coordination_service_impl.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/random.h"
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
      absl::ToInt64Milliseconds(options.enumerate_devices_timeout));
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
  // Convert list of local devices to global device message as EnumerateDevies()
  // response.
  service->SetDeviceAggregationFunction(
      [](const tensorflow::DeviceInfo& raw_global_devices) {
        xla::GlobalTopologyProto global_topology;
        int global_device_id = 0;
        // Assign local devices of the same host to the same slice_index.
        int next_slice_index = 0;
        absl::flat_hash_map<std::string, int> boot_id_to_slice_index;
        // Unwrap result to local device proto.
        for (const auto& device : raw_global_devices.device()) {
          xla::LocalTopologyProto local_topology;
          // Note that tensorflow::DeviceInfo.device is xla.LocalTopologyProto!
          device.UnpackTo(&local_topology);
          // Every new boot_id seen is treated as a new host/slice.
          absl::string_view boot_id = local_topology.boot_id();
          auto [it, inserted] =
              boot_id_to_slice_index.try_emplace(boot_id, next_slice_index);
          if (inserted) {
            ++next_slice_index;
          }
          // Set deterministic global ids.
          for (xla::DeviceProto& device : *local_topology.mutable_devices()) {
            device.set_global_device_id(global_device_id++);
            device.set_slice_index(it->second);
          }
          *global_topology.mutable_nodes()->Add() = local_topology;
        }
        if (VLOG_IS_ON(10)) {
          for (auto it = boot_id_to_slice_index.begin();
               it != boot_id_to_slice_index.end(); ++it) {
            LOG(INFO) << "BuildGlobalTopology boot_id_to_slice_index "
                      << it->first << "->" << it->second;
          }
        }
        // Wrap result back in DeviceInfo proto.
        tensorflow::DeviceInfo global_devices;
        global_devices.mutable_device()->Add()->PackFrom(global_topology);
        return global_devices;
      });
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
  LOG(INFO) << "Experimental coordination service is enabled.";
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

xla::StatusOr<std::unique_ptr<DistributedRuntimeService>>
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
    server_->Shutdown();
    server_->Wait();
  }

  // Explicitly destroy coordination service before the gRPC server. This clears
  // all pending RPCs before the gRPC server is destroyed.
  coord_impl_ = nullptr;
  server_ = nullptr;
}

}  // namespace xla
