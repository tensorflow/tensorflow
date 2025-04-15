/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/tools/multihost_hlo_runner/create_client.h"

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/status_macros.h"
#include "xla/xla.pb.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

absl::Status InitDistributedRuntimeInEnv(absl::string_view address, int node_id,
                                         int num_nodes,
                                         absl::Duration init_timeout,
                                         PjRtEnvironment& env) {
  TF_RET_CHECK(!address.empty());
  TF_RET_CHECK(node_id >= 0)
      << "Node id is expected to be in range [0, num_nodes)";
  TF_RET_CHECK(node_id < num_nodes)
      << "Node id is expected to be in range [0, num_nodes)";

  CHECK_GT(address.length(), 0);
  // Multinode. Start service on task 0.
  if (node_id == 0) {
    std::string coordinator_bind_address =
        "[::]:" + std::string(address).substr(address.rfind(':') + 1);
    xla::CoordinationServiceImpl::Options options;
    options.num_nodes = num_nodes;
    TF_ASSIGN_OR_RETURN(env.service, xla::GetDistributedRuntimeService(
                                         coordinator_bind_address, options));
  }
  xla::DistributedRuntimeClient::Options options;
  options.node_id = node_id;
  options.init_timeout = init_timeout;
  env.distributed_client =
      GetDistributedRuntimeClient(std::string(address), options);
  TF_QCHECK_OK(env.distributed_client->Connect());
  env.kv_store = GetDistributedKeyValueStore(env.distributed_client,
                                             /*key_prefix=*/"gpu:");
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<PjRtEnvironment> GetPjRtEnvironmentForHostCpu() {
  PjRtEnvironment env;
  TF_ASSIGN_OR_RETURN(env.client, CreateHostClient());
  return env;
}

absl::StatusOr<PjRtEnvironment> GetPjRtEnvironmentForGpu(
    absl::string_view address, GpuClientOptions gpu_options,
    absl::Duration init_timeout) {
  PjRtEnvironment env;

  // Initialize distributed runtime for a non-mock multi-node environment.
  if (gpu_options.num_nodes > 1 && !gpu_options.enable_mock_nccl) {
    TF_QCHECK_OK(InitDistributedRuntimeInEnv(address, gpu_options.node_id,
                                             gpu_options.num_nodes,
                                             init_timeout, env));
    gpu_options.kv_store = env.kv_store;
  }

  if (gpu_options.enable_mock_nccl) {
    CHECK_GT(gpu_options.num_nodes, 1);
  }

  TF_ASSIGN_OR_RETURN(env.client, CreateGpuClient(gpu_options));
  return env;
}

absl::StatusOr<std::unique_ptr<PjRtClient>> CreateHostClient() {
  xla::CpuClientOptions options;
  return xla::GetXlaPjrtCpuClient(options);
}

absl::StatusOr<std::unique_ptr<PjRtClient>> CreateGpuClient(
    const GpuClientOptions& options) {
  if (options.node_id < 0 || options.node_id >= options.num_nodes) {
    return absl::InvalidArgumentError(
        "Node id is expected to be in range [0, num_nodes)");
  }
  return xla::GetXlaPjrtGpuClient(options);
}

absl::StatusOr<std::unique_ptr<PjRtClient>> CreateMockGpuClient(int num_nodes) {
  GpuClientOptions options;
  options.num_nodes = num_nodes;
  options.enable_mock_nccl = true;
  return CreateGpuClient(options);
}

}  // namespace xla
