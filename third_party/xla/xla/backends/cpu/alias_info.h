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

#ifndef XLA_BACKENDS_CPU_ALIAS_INFO_H_
#define XLA_BACKENDS_CPU_ALIAS_INFO_H_

#include <optional>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/custom_fusion_configs.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/shape_util.h"

namespace xla::cpu {

class CpuAliasInfo : public AliasInfo {
 public:
  std::optional<bool> MayAlias(const HloInstruction* operand,
                               const ShapeIndex& operand_index,
                               const HloInstruction* user,
                               const ShapeIndex& user_index) const override {
    if (user->IsCustomFusion()) {
      absl::StatusOr<BackendConfig> config =
          user->backend_config<BackendConfig>();
      if (config.ok() && config->fusion_config().kind() == kYnnFusionKind) {
        // YNNPACK supports external-output aliasing. Keep the usual HLO
        // shape and use checks rather than unconditionally allowing reuse.
        return std::nullopt;
      }
      // Elementwise HLO uses do not guarantee that a library's compiled
      // partition can safely overwrite an input. Require explicit opt-in.
      return false;
    }
    return std::nullopt;
  }
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_ALIAS_INFO_H_
