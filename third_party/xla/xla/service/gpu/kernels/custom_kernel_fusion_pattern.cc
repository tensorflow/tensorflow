/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CustomKernelFusionPattern::Match
//===----------------------------------------------------------------------===//

CustomKernelFusionPattern::Match::Match(
    CustomFusionConfig config, std::vector<HloInstruction*> instructions,
    int64_t workspace_size_bytes)
    : config_(std::move(config)),
      instructions_(std::move(instructions)),
      workspace_size_bytes_(workspace_size_bytes) {}

void CustomKernelFusionPattern::Match::AddReplacement(HloInstruction* instr,
                                                      Replacement replacement) {
  replacements_[instr] = std::move(replacement);
}

bool CustomKernelFusionPattern::Match::HasReplacement(
    HloInstruction* instr) const {
  return replacements_.contains(instr);
}

absl::StatusOr<HloInstruction*>
CustomKernelFusionPattern::Match::BuildReplacement(
    HloInstruction* instr, HloFusionInstruction* fusion) const {
  if (auto it = replacements_.find(instr); it != replacements_.end()) {
    return it->second(fusion);
  }

  return absl::InvalidArgumentError(
      absl::StrCat("no replacement for instruction: ", instr->name()));
}

//===----------------------------------------------------------------------===//
// CustomKernelFusionPatternRegistry
//===----------------------------------------------------------------------===//

CustomKernelFusionPatternRegistry*
CustomKernelFusionPatternRegistry::Default() {
  static auto* registry = new CustomKernelFusionPatternRegistry();
  return registry;
}

std::vector<CustomKernelFusionPattern::Match>
CustomKernelFusionPatternRegistry::Match(const se::DeviceDescription& device,
                                         HloInstruction* instr) const {
  std::vector<CustomKernelFusionPattern::Match> matches;
  for (auto& pattern : patterns_) {
    if (auto matched = pattern->TryMatch(device, instr); matched.has_value())
      matches.push_back(std::move(*matched));
  }
  return matches;
}

void CustomKernelFusionPatternRegistry::Add(
    std::unique_ptr<CustomKernelFusionPattern> pattern) {
  patterns_.push_back(std::move(pattern));
}

}  // namespace xla::gpu
