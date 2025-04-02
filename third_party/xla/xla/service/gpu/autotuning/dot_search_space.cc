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

#include "xla/service/gpu/autotuning/dot_search_space.h"

#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

TritonDotFusionSearchSpace::TritonDotFusionSearchSpace(
    const se::DeviceDescription& device_description,
    const HloDotInstruction* dot)
    :  // Set up basic information about the hardware and the problem.
      device_description_(device_description) {
  // TODO: b/404470821 - Do something based on `dot`.
}

std::vector<TritonGemmConfig> TritonDotFusionSearchSpace::GenerateConfigs() {
  // TODO: b/404470821 - Implement this properly rather than hardcoding the
  // config.
  return {TritonGemmConfig(
      /*block_m=*/64, /*block_n=*/128, /*block_k=*/64,
      /*split_k=*/1, /*num_stages=*/3, /*num_warps=*/4,
      /*num_ctas=*/1)};
}

std::string TritonDotFusionSearchSpace::Serialize() {
  return absl::StrFormat("TODO: b/404470821 - Implement this.");
}

}  // namespace xla::gpu
