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

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"

namespace xla {

struct PjRtEnvironment {
  // Sequence matters here, client should be destroyed before service.
  std::unique_ptr<xla::DistributedRuntimeService> service;
  std::unique_ptr<xla::PjRtClient> client;
  std::shared_ptr<xla::KeyValueStoreInterface> kv_store;
  std::shared_ptr<xla::DistributedRuntimeClient> distributed_client;
};

// Creates an environment with a PjRtClient for host CPU.
absl::StatusOr<PjRtEnvironment> GetPjRtEnvironmentForHostCpu();

// Creates an environment with a PjRtClient for GPU and potentially distributed
// runtime components if using multiple GPU nodes.
// In GPU options `kv_store` will be initialized separately for the multi-node
// environment.
absl::StatusOr<PjRtEnvironment> GetPjRtEnvironmentForGpu(
    absl::string_view address, GpuClientOptions gpu_options,
    absl::Duration init_timeout);

// Creates a PjRtClient which can run HLOs on Host CPU.
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateHostClient();

// Creates a PjRtClient which can run HLOs on GPU.
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateGpuClient(
    const GpuClientOptions& options);

// Creates a PjRtClient which mocks multi-host GPU runs.
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateMockGpuClient(
    int num_nodes = 1);

}  // namespace xla

#endif  // XLA_TOOLS_MULTIHOST_HLO_RUNNER_CREATE_CLIENT_H_
