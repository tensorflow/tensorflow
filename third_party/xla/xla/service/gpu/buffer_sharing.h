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

#ifndef XLA_SERVICE_GPU_BUFFER_SHARING_H_
#define XLA_SERVICE_GPU_BUFFER_SHARING_H_

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
std::optional<bool> FusionCanShareBufferHint(
    const HloInstruction* user, const HloInstruction* operand,
    const ShapeIndex& user_index,
    const se::DeviceDescription& device_description);

std::optional<bool> CanShareBufferHint(
    const HloInstruction* user, const HloInstruction* operand,
    const ShapeIndex& user_index,
    const se::DeviceDescription& device_description);
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_BUFFER_SHARING_H_
