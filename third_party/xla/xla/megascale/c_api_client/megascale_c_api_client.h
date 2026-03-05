/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_MEGASCALE_C_API_CLIENT_MEGASCALE_C_API_CLIENT_H_
#define XLA_MEGASCALE_C_API_CLIENT_MEGASCALE_C_API_CLIENT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/megascale/addresses.pb.h"
#include "xla/megascale/c_api_client/megascale_types.h"
#include "xla/megascale/dcn_topology.pb.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/tsl/platform/logging.h"

namespace xla {
namespace megascale {
namespace c_api_client {

struct ProcessesInfo {
  // Mapping of dense process ids to their network addresses.
  std::vector<std::string> addresses;
  // Dense slice index of each process.
  std::optional<std::vector<int32_t>> slice_indexes = std::nullopt;
  // Dense per-slice index of each process.
  std::optional<std::vector<int32_t>> per_slice_indexes = std::nullopt;
  // The number of devices per process.
  int32_t num_devices_per_process = 0;
};

// Returns AoT config for megascale multi slice compilation.
// REQUIRES: num_slices > 1.
absl::StatusOr<std::unique_ptr<xla::MultiSliceConfig>> CreateAoTMegascaleConfig(
    const xla::PjRtTopologyDescription& topology_description, int num_slices);

absl::StatusOr<std::unique_ptr<const xla::MultiSliceConfig>>
CreateMultiSliceMegascaleConfig(
    const xla::PjRtTopologyDescription& topology_description, int num_slices,
    int32_t local_slice_id, int32_t local_host_id,
    const xla::megascale::runtime::EndpointAddresses& endpoint_addresses,
    const xla::megascale::runtime::DCNTopology& dcn_topology,
    std::shared_ptr<CApiPjRtClientContext> megascale_client_ctx);

absl::StatusOr<std::shared_ptr<CApiPjRtClientContext>>
MegaScaleClientContextFromClient(xla::PjRtClient* client);

absl::StatusOr<std::shared_ptr<CApiPjRtClientContext>>
CreateDefaultMegaScaleClientContext();

absl::StatusOr<std::unique_ptr<xla::cpu::CpuCollectives>>
CreateMegascaleCollectives(
    const CApiPjRtClientContext& megascale_client_ctx,
    ProcessesInfo&& processes_info,
    std::optional<xla::megascale::runtime::DCNTopology>&& dcn_topology);

}  // namespace c_api_client
}  // namespace megascale
}  // namespace xla

#endif  // XLA_MEGASCALE_C_API_CLIENT_MEGASCALE_C_API_CLIENT_H_
