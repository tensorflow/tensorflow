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

#include "xla/backends/gpu/codegen/emitters/transforms/nontemporal_hint_targets.h"

#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace xla::gpu {

NontemporalHintTargets GetNontemporalHintTargets(
    const stream_executor::GpuComputeCapability& gpu_cc) {
  const stream_executor::RocmComputeCapability* rcc =
      gpu_cc.rocm_compute_capability();
  if (rcc != nullptr && rcc->gfx9_mi300()) {
    // Multi-output stores, and loads in kernels that overwrite an input, slowed
    // MI300X down.
    return {/*loads=*/true, /*stores=*/true, /*single_output_stores=*/true,
            /*out_of_place_loads=*/true};
  }
  return {};
}

}  // namespace xla::gpu
