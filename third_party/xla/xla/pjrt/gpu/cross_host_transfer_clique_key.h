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

#ifndef XLA_PJRT_GPU_CROSS_HOST_TRANSFER_CLIQUE_KEY_H_
#define XLA_PJRT_GPU_CROSS_HOST_TRANSFER_CLIQUE_KEY_H_

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/runtime/device_id.h"

namespace xla {

// Clique key for one cross-host buffer transfer. Devices stay in source, then
// destination order. A missing incarnation pair matches the historical key.
// Otherwise the source and destination incarnations are sorted and
// deduplicated so both hosts construct the same key.
inline gpu::GpuCliqueKey CrossHostTransferCliqueKey(
    GlobalDeviceId src, GlobalDeviceId dst,
    const std::optional<std::array<IncarnationId, 2>>& incarnations) {
  std::vector<IncarnationId> clique_incarnations;
  if (incarnations.has_value()) {
    clique_incarnations = {(*incarnations)[0], (*incarnations)[1]};
    absl::c_sort(clique_incarnations);
    if (clique_incarnations[0] == clique_incarnations[1]) {
      clique_incarnations.pop_back();
    }
  }
  return gpu::GpuCliqueKey(
      /*devices=*/{src, dst},
      /*num_local_participants=*/1,
      /*communication_id=*/gpu::CommunicationId(0),
      std::move(clique_incarnations));
}

}  // namespace xla

#endif  // XLA_PJRT_GPU_CROSS_HOST_TRANSFER_CLIQUE_KEY_H_
