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

#ifndef XLA_TOOLS_MULTIHOST_HLO_RUNNER_CREATE_CLIENT_H_
#define XLA_TOOLS_MULTIHOST_HLO_RUNNER_CREATE_CLIENT_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"

namespace xla {

struct PjRtEnvironment {
  // Sequence matters here, client should be destroyed before service.
  std::unique_ptr<xla::DistributedRuntimeService> service;
  std::unique_ptr<xla::PjRtClient> client;
  std::shared_ptr<xla::KeyValueStoreInterface> kv_store;
  std::shared_ptr<xla::DistributedRuntimeClient> distributed_client;
};

absl::StatusOr<PjRtEnvironment> GetPjRtClient(absl::string_view device_type,
                                              absl::string_view address,
                                              int node_id, int num_nodes,
                                              bool enable_mock_nccl,
                                              absl::Duration init_timeout);

// Create a PjRtClient which can run HLOs on Host CPU.
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateHostClient();

// Create a PjRtClient which can run HLOs on GPU.
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateGpuClient(
    GpuClientOptions options);

// Create a PjRtClient which mocks multi-hosts GPU run
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateMockGpuClient(
    int num_nodes = 1);

}  // namespace xla

#endif  // XLA_TOOLS_MULTIHOST_HLO_RUNNER_CREATE_CLIENT_H_
