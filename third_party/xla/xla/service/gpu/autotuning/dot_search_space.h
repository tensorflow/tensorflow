/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_AUTOTUNING_DOT_SEARCH_SPACE_H_
#define XLA_SERVICE_GPU_AUTOTUNING_DOT_SEARCH_SPACE_H_

#include <string>
#include <vector>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Generates the space of promising Triton configs for a given dot fusion
// and hardware.
//
// Takes into account the properties of the problem (e.g., operand and result
// shapes, fused instructions), and the hardware (e.g., number of cores,
// available registers and memory per core).
//
// Internal doc with rationale: go/xla-gpu-dot-search
class TritonDotFusionSearchSpace {
 public:
  TritonDotFusionSearchSpace(const se::DeviceDescription& device_description,
                             const HloDotInstruction* dot);

  // Generates the list of promising configs in the search space for the
  // autotuner to try.
  std::vector<TritonGemmConfig> GenerateConfigs();

  // Serializes the search space to a human-readable string.
  std::string Serialize();

 private:
  se::DeviceDescription device_description_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_AUTOTUNING_DOT_SEARCH_SPACE_H_
