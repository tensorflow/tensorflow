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
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/status_macros.h"
#include "xla/xla.pb.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {

static absl::StatusOr<std::unique_ptr<xla::PjRtClient>> GetPjRtClient(
    absl::string_view device_type, absl::string_view address, int node_id,
    int num_nodes, bool enable_mock_nccl, absl::Duration init_timeout,
    std::unique_ptr<xla::DistributedRuntimeService>& service,
    std::shared_ptr<xla::KeyValueStoreInterface>& kv_store,
    std::shared_ptr<xla::DistributedRuntimeClient>& distributed_client) {
  if (device_type == "host") {
    CHECK_EQ(num_nodes, 1);
    return CreateHostClient();
  }

  if (device_type != "gpu") {
    return absl::UnimplementedError(device_type);
  }

  if (enable_mock_nccl) {
    CHECK_GT(num_nodes, 1);
    return CreateMockGpuClient(num_nodes);
  }

  if (num_nodes == 1) {
    return CreateGpuClient({});
  }

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
    TF_ASSIGN_OR_RETURN(service, xla::GetDistributedRuntimeService(
                                     coordinator_bind_address, options));
  }
  xla::DistributedRuntimeClient::Options options;
  options.node_id = node_id;
  options.init_timeout = init_timeout;
  distributed_client =
      GetDistributedRuntimeClient(std::string(address), options);
  TF_QCHECK_OK(distributed_client->Connect());
  kv_store = GetDistributedKeyValueStore(distributed_client,
                                         /*key_prefix=*/"gpu:");
  GpuClientOptions gpu_client_options;
  gpu_client_options.node_id = node_id;
  gpu_client_options.num_nodes = num_nodes;
  gpu_client_options.kv_store = kv_store;
  return CreateGpuClient(std::move(gpu_client_options));
}

absl::StatusOr<PjRtEnvironment> GetPjRtClient(absl::string_view device_type,
                                              absl::string_view address,
                                              int node_id, int num_nodes,
                                              bool enable_mock_nccl,

                                              absl::Duration init_timeout) {
  PjRtEnvironment env;
  TF_ASSIGN_OR_RETURN(env.client,
                      GetPjRtClient(device_type, address, node_id, num_nodes,
                                    enable_mock_nccl, init_timeout, env.service,
                                    env.kv_store, env.distributed_client));
  return env;
}

absl::StatusOr<std::unique_ptr<PjRtClient>> CreateHostClient() {
  return GetTfrtCpuClient(CpuClientOptions());
}

absl::StatusOr<std::unique_ptr<PjRtClient>> CreateGpuClient(
    GpuClientOptions options) {
  if (options.node_id < 0 || options.node_id >= options.num_nodes) {
    return absl::InvalidArgumentError(
        "Node id is expected to be in range [0, num_nodes)");
  }
  return GetStreamExecutorGpuClient(options);
}

absl::StatusOr<std::unique_ptr<PjRtClient>> CreateMockGpuClient(int num_nodes) {
  GpuClientOptions options;
  options.num_nodes = num_nodes;
  options.enable_mock_nccl = true;
  return GetStreamExecutorGpuClient(options);
}

}  // namespace xla
