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

#include "xla/backends/gpu/runtime/thunk_buffer_debug_pass.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_checksum.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_float_check.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_saver_inserter.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

absl::StatusOr<std::unique_ptr<ThunkBufferDebugPass>>
ThunkBufferDebugPass::Create(Mode mode,
                             std::vector<ShapedSlice> module_output_slices) {
  for (const ShapedSlice& slice : module_output_slices) {
    if (slice.shape.IsTuple()) {
      return absl::InvalidArgumentError(
          "Module output slices must not contain tuple shapes");
    }
  }

  return absl::WrapUnique(
      new ThunkBufferDebugPass(mode, std::move(module_output_slices)));
}

absl::StatusOr<bool> ThunkBufferDebugPass::Run(
    ThunkSequence* thunk_sequence, const DebugOptions& debug_options,
    const HloModule* absl_nullable hlo_module,
    const se::DeviceDescription& device_info,
    ThunkPassBufferAllocator& allocator) {
  VLOG(1) << "ThunkBufferDebugPass running";

  if (hlo_module == nullptr) {
    // We need the HLO module to dump the buffer debug log proto to a file. If
    // it's not available, there's no point in doing extra work.
    VLOG(1) << "HLO module is null, skip buffer checksumming";
    return false;
  }

  switch (mode_) {
    case Mode::kChecksum:
      RETURN_IF_ERROR(RunChecksumPassInternal(thunk_sequence, debug_options,
                                              hlo_module, module_output_slices_,
                                              allocator));
      break;
    case Mode::kFloatChecker:
      RETURN_IF_ERROR(
          RunFloatCheckPassInternal(thunk_sequence, debug_options, hlo_module,
                                    module_output_slices_, allocator));
      break;
    case Mode::kBufferSaver:
      RETURN_IF_ERROR(RunDebugSaverInserter(
          thunk_sequence, debug_options, *hlo_module, module_output_slices_));
      break;
  }

  return true;
}

absl::StatusOr<std::vector<ShapedSlice>> GetOutputShapedBuffers(
    const HloModule* hlo_module, const BufferAssignment* buffer_assignment) {
  std::vector<ShapedSlice> buffers_to_check;
  if (hlo_module == nullptr || buffer_assignment == nullptr) {
    return buffers_to_check;
  }
  const HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      root->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsArray()) {
          ASSIGN_OR_RETURN(auto slice,
                           buffer_assignment->GetUniqueSlice(root, index));
          buffers_to_check.push_back(ShapedSlice{slice, subshape});
        }
        return absl::OkStatus();
      }));
  // It is already sorted by construction because ShapeUtil::ForEachSubshape
  // visits in a deterministic order.
  return buffers_to_check;
}

}  // namespace xla::gpu
