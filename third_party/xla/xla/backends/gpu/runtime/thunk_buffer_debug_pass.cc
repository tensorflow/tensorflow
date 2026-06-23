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

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
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
                                              hlo_module, buffer_assignment_,
                                              allocator));
      break;
    case Mode::kFloatChecker:
      RETURN_IF_ERROR(RunFloatCheckPassInternal(thunk_sequence, debug_options,
                                                hlo_module, buffer_assignment_,
                                                allocator));
      break;
    case Mode::kBufferSaver:
      RETURN_IF_ERROR(RunDebugSaverInserter(thunk_sequence, debug_options,
                                            *hlo_module, buffer_assignment_));
      break;
  }

  return true;
}

absl::StatusOr<absl::flat_hash_map<size_t, ShapedSlice>> GetOutputShapedBuffers(
    const HloModule* hlo_module, const BufferAssignment* buffer_assignment) {
  absl::flat_hash_map<size_t, ShapedSlice> buffers_to_check;
  if (hlo_module == nullptr || buffer_assignment == nullptr) {
    return buffers_to_check;
  }
  const HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  size_t buffer_idx = 0;
  RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      root->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsArray()) {
          size_t current_idx = buffer_idx++;
          ASSIGN_OR_RETURN(auto slice,
                           buffer_assignment->GetUniqueSlice(root, index));
          buffers_to_check.emplace(current_idx, ShapedSlice{slice, subshape});
        }
        return absl::OkStatus();
      }));
  // It is already sorted by construction because ShapeUtil::ForEachSubshape
  // visits in a deterministic order and we use an incrementing counter.
  return buffers_to_check;
}

absl::StatusOr<absl::flat_hash_map<size_t, BufferAllocation::Slice>>
GetOutputBuffers(const HloModule* hlo_module,
                 const BufferAssignment* buffer_assignment) {
  absl::flat_hash_map<size_t, BufferAllocation::Slice> buffers_to_check;
  ASSIGN_OR_RETURN(auto shaped_buffers,
                   GetOutputShapedBuffers(hlo_module, buffer_assignment));
  buffers_to_check.reserve(shaped_buffers.size());
  std::vector<std::pair<size_t, ShapedSlice>> sorted_shaped_buffers;
  sorted_shaped_buffers.reserve(shaped_buffers.size());
  sorted_shaped_buffers.assign(shaped_buffers.begin(), shaped_buffers.end());
  absl::c_sort(sorted_shaped_buffers,
               [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& [idx, shaped_slice] : sorted_shaped_buffers) {
    buffers_to_check.emplace(idx, shaped_slice.slice);
  }
  return buffers_to_check;
}

}  // namespace xla::gpu
