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
#include "tensorflow/core/tpu/kernels/tpu_pod_state.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "tsl/platform/errors.h"

#if defined(LIBTPU_ON_GCE)
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#else
#include "tensorflow/core/tpu/kernels/tpu_util.h"  // copybara"
#endif

namespace tensorflow {
const char kTpuPodStateResourceName[] = "tpu_pod_state";

namespace {

// Attempt to delete resource_name from resource_manager's default_container.
// Returns OK if the deletion succeeded, or if the resource was not found. Else
// return the deletion error.
template <class ResourceT>
Status DeleteIfExists(ResourceMgr* resource_manager,
                      const char* resource_name) {
  VLOG(1) << "Removing resource " << resource_name << " if it exists";
  Status status = resource_manager->Delete<ResourceT>(
      resource_manager->default_container(), resource_name);
  if (status.ok()) {
    VLOG(1) << "Removed existing resource " << resource_name;
    return OkStatus();
  }
  if (status.code() == error::NOT_FOUND) {
    VLOG(1) << "No resource " << resource_name << " to remove";
    return OkStatus();
  }
  VLOG(1) << "Error removing resource " << resource_name << " : " << status;
  return status;
}

absl::StatusOr<std::unique_ptr<TpuCompilationCacheService>>
ConstructCacheService(ResourceMgr* rmgr, int serving_port,
                      tpu::TpuCompilationCacheInterface* compilation_cache) {
  absl::StatusOr<std::unique_ptr<::grpc::ServerBuilder>> server_builder;
#if defined(LIBTPU_ON_GCE)
  server_builder = tpu::CreateServerBuilder(serving_port);
#else
  server_builder = tpu::CreateServerBuilderGoogle(serving_port);
#endif
  TF_RETURN_IF_ERROR(server_builder.status());

  auto cache_service = std::make_unique<TpuCompilationCacheService>(
      server_builder.value().get(), compilation_cache);
  cache_service->SetMemoryQuota(1ul << 31);  // 2GB
  cache_service->Start();
  return cache_service;
}
}  // namespace

Status GetServerAddressAndPort(std::string* server_address, int* serving_port) {
  char* server_address_output = nullptr;
  auto cleanup = absl::MakeCleanup([&server_address_output]() {
    stream_executor::tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(
        server_address_output);
  });
  size_t server_address_output_size;
  *serving_port = -1;

  TpuConfigurationApi_GetServerAddressAndPort_Params params;
  params.struct_size = TpuConfigurationApi_GetServerAddressAndPort_Params_SIZE;
  params.priv = nullptr;
  params.server_address_output_size = &server_address_output_size;
  params.server_address_output = &server_address_output;
  params.port_output = serving_port;
  StatusHelper status;
  params.status = status.c_status;

  stream_executor::tpu::OpsApiFn()
      ->TpuConfigurationApi_GetServerAddressAndPortFn(&params);
  TF_RETURN_IF_ERROR(status.status());
  *server_address =
      std::string(server_address_output, server_address_output_size);
  CHECK_NE(*serving_port, -1);
  return OkStatus();
}

TpuPodState::TpuPodState(
    int service_port, std::unique_ptr<TpuCompilationCacheService> cache_service)
    : cache_service_(std::move(cache_service)), service_port_(service_port) {}

TpuPodState::~TpuPodState() {
  if (cache_service_) {
    VLOG(1) << "Shutting down Compilation Cache Service.";
    if (cache_service_->Shutdown(20) && service_port_ >= 0) {
      stream_executor::tpu::OpsApiFn()->TpuNetUtil_RecycleUnusedPortFn(
          service_port_);
    } else {
      LOG(ERROR)
          << "Failed to shutdown Compilation Cache Service within timeout.";
    }
  }
  VLOG(1) << "Shutting down Compilation Cache Service done.";
}

string TpuPodState::DebugString() const {
  return "Wrapper for distributed TPU state";
}

Status GetTPUPodState(const ResourceMgr* rmgr, TpuPodState** pod_state) {
  if (!rmgr) {
    return errors::Internal("No resource manager.");
  }
  if (!rmgr->Lookup(rmgr->default_container(), kTpuPodStateResourceName,
                    pod_state)
           .ok()) {
    return errors::FailedPrecondition(
        "The TPU system has not been initialized.");
  }
  return OkStatus();
}

bool HasTPUPodState(const ResourceMgr* rmgr) {
  TpuPodState* pod_state;
  if (!rmgr->Lookup(rmgr->default_container(), kTpuPodStateResourceName,
                    &pod_state)
           .ok()) {
    return false;
  }
  pod_state->Unref();
  return true;
}

Status ConstructTpuPodState(
    ResourceMgr* rmgr, const std::vector<int32_t>& num_devices_per_host,
    tpu::TpuCompilationCacheInterface* compilation_cache,
    std::string* host_config_proto) {
  int serving_port;
  std::string server_address;
  TF_RETURN_IF_ERROR(GetServerAddressAndPort(&server_address, &serving_port));

  char* host_config_output = nullptr;
  auto host_config_cleanup = absl::MakeCleanup([&host_config_output]() {
    stream_executor::tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(
        host_config_output);
  });
  size_t host_config_output_size;

  ConfigureDistributedTpuOp_DoWork_Params params;
  params.struct_size = ConfigureDistributedTpuOp_DoWork_Params_SIZE;
  params.priv = nullptr;
  params.num_cores_per_host_size = num_devices_per_host.size();
  params.num_cores_per_host = num_devices_per_host.data();
  params.server_address_size = server_address.size();
  params.server_address = server_address.data();
  params.host_config_output_size = &host_config_output_size;
  params.host_config_output = &host_config_output;
  StatusHelper status;
  params.status = status.c_status;

  stream_executor::tpu::OpsApiFn()->ConfigureDistributedTpuOp_DoWorkFn(&params);
  TF_RETURN_IF_ERROR(status.status());
  *host_config_proto = std::string(host_config_output, host_config_output_size);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TpuCompilationCacheService> cache_service,
      ConstructCacheService(rmgr, serving_port, compilation_cache));

  // Delete TpuPodState if it exists, and recreate below.
  TF_RETURN_IF_ERROR(
      DeleteIfExists<TpuPodState>(rmgr, kTpuPodStateResourceName));
  return rmgr->Create(rmgr->default_container(), kTpuPodStateResourceName,
                      new TpuPodState(serving_port, std::move(cache_service)));
}
}  // namespace tensorflow
