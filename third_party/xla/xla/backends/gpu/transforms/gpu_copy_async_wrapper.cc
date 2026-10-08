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

#include "xla/backends/gpu/transforms/gpu_copy_async_wrapper.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// Returns true if `shape` carries a host-memory-space annotation.
bool HasHostMemorySpace(const Shape& shape) {
  if (!shape.has_layout()) {
    return false;
  }
  return shape.layout().memory_space() ==
         static_cast<int64_t>(stream_executor::MemorySpace::kHost);
}

// Returns true if `instr` is eligible for async D2D wrapping.
bool IsEligibleD2DCopy(const HloInstruction* instr, int64_t min_copy_bytes) {
  if (instr->opcode() != HloOpcode::kCopy) {
    return false;
  }
  const Shape& shape = instr->shape();
  // Only wrap array shapes; tuples are not directly copyable via D2D memcpy.
  if (!shape.IsArray()) {
    return false;
  }
  // Skip copies involving host memory — those are handled by other paths.
  if (HasHostMemorySpace(shape) ||
      HasHostMemorySpace(instr->operand(0)->shape())) {
    return false;
  }
  // Only wrap copies that are plain memcpys: a layout-changing copy is a
  // transpose in disguise and must stay on the kernel emission path.
  if (!LayoutUtil::LayoutsInShapesEqual(shape, instr->operand(0)->shape(),
                                        Layout::Equal().MinorToMajorOnly())) {
    return false;
  }
  // Enforce the minimum size threshold to amortize sync overhead.
  if (ShapeUtil::ByteSizeOf(shape) < min_copy_bytes) {
    return false;
  }
  return true;
}

}  // namespace

absl::StatusOr<bool> GpuCopyAsyncWrapper::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const auto& debug_options = module->config().debug_options();
  if (!debug_options.has_xla_gpu_async_copy_min_bytes()) {
    return false;
  }
  const int64_t min_bytes = debug_options.xla_gpu_async_copy_min_bytes();
  if (min_bytes == -1) {
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (computation->IsAsyncComputation()) {
      continue;
    }

    std::vector<HloInstruction*> instructions =
        computation->MakeInstructionPostOrder();
    for (HloInstruction* instr : instructions) {
      if (!IsEligibleD2DCopy(instr, min_bytes)) {
        continue;
      }

      ABSL_RETURN_IF_ERROR(computation
                          ->CreateAsyncInstructions(
                              instr, {ShapeUtil::MakeScalarShape(U32)})
                          .status());
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla::gpu
