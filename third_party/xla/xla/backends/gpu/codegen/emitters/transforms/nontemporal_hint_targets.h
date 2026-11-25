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

#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSFORMS_NONTEMPORAL_HINT_TARGETS_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSFORMS_NONTEMPORAL_HINT_TARGETS_H_

#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// gfx942 gets load and store hints from 16x L2. Other targets get none.
struct NontemporalHintTargets {
  bool loads = false;
  bool stores = false;
  // Store hints only if a single output qualifies.
  bool single_output_stores = false;
  // No load hints in kernels that overwrite an input.
  bool out_of_place_loads = false;
  // Buffers below this multiple of L2 stay temporal.
  int stream_l2_multiple = 16;
};
NontemporalHintTargets GetNontemporalHintTargets(
    const stream_executor::GpuComputeCapability& gpu_cc);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSFORMS_NONTEMPORAL_HINT_TARGETS_H_
