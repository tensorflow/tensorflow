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

#ifndef XLA_SERVICE_GPU_ALIAS_INFO_H_
#define XLA_SERVICE_GPU_ALIAS_INFO_H_

#include <optional>

#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
class GpuAliasInfo : public AliasInfo {
 public:
  explicit GpuAliasInfo(const se::DeviceDescription* device_description)
      : device_description_(device_description) {}

  // Backend-specific may-alias hint. If an empty optional is returned, the
  // default rules in HloDataflowAnalysis are used. `operand` should be an
  // operand of `user`. `operand_index` should be the output index of `operand`,
  // `user_index` should be the output index of `user`.
  std::optional<bool> MayAlias(const HloInstruction* operand,
                               const ShapeIndex& operand_index,
                               const HloInstruction* user,
                               const ShapeIndex& user_index) const override;

 protected:
  const se::DeviceDescription* device_description_;
};
}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_ALIAS_INFO_H_
