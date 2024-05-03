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
#include "xla/service/gpu/fusions/copy.h"

#include <memory>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/runtime/copy_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"

namespace xla {
namespace gpu {

absl::StatusOr<FusionEmissionResult> MemcpyFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  FusionEmissionResult result;
  for (int i = 0; i < src_buffers_.size(); ++i) {
    if (src_buffers_[i] != dst_buffers_[i]) {
      result.thunks.emplace_back(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(&fusion),
          /*source_buffer=*/src_buffers_[i],
          /*destination_buffer=*/dst_buffers_[i],
          /*mem_size=*/src_buffers_[i].size()));
    }
  }
  return result;
}

}  // namespace gpu
}  // namespace xla
