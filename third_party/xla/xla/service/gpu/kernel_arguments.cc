/*Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/kernel_arguments.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<KernelArguments> KernelArguments::Create(
    const BufferAssignment& buffer_assignment,
    const HloFusionInstruction* fusion) {
  std::vector<KernelArgument> kernel_arguments;
  for (const HloInstruction* operand : fusion->operands()) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        buffer_assignment.GetUniqueSlice(operand, {}));
    kernel_arguments.emplace_back(
        KernelArgument(operand->shape(), slice, /*written=*/false));
  }

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      fusion->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                            buffer_assignment.GetUniqueSlice(fusion, index));
        kernel_arguments.emplace_back(
            KernelArgument(subshape, slice, /*written=*/true));
        return absl::OkStatus();
      }));

  return KernelArguments{std::move(kernel_arguments), /*dedup=*/true};
}

std::vector<KernelArgument> KernelArguments::ProcessArguments(
    std::vector<KernelArgument> kernel_arguments, bool dedup) {
  absl::flat_hash_set<BufferAllocation::Slice> buffers_written;
  for (const KernelArgument& kernel_argument : kernel_arguments) {
    if (kernel_argument.written()) {
      buffers_written.insert(kernel_argument.slice());
    }
  }

  absl::flat_hash_map<BufferAllocation::Slice, std::optional<int64_t>>
      first_indices_for_slices;
  int next_llvm_arg_index = 0;
  for (int i = 0; i < static_cast<int>(kernel_arguments.size()); ++i) {
    KernelArgument& kernel_argument = kernel_arguments[i];

    auto& first_index = first_indices_for_slices[kernel_argument.slice_];
    if (dedup && first_index) {
      const KernelArgument& same = kernel_arguments[*first_index];
      kernel_argument.first_with_same_slice_ = first_index;
      kernel_argument.alignment_ = same.alignment_;
      kernel_argument.aliased_ = same.aliased_;
      kernel_argument.written_ = same.written_;
      kernel_argument.llvm_arg_index_ = same.llvm_arg_index_;
      continue;
    } else {
      first_index = i;
      kernel_argument.llvm_arg_index_ = next_llvm_arg_index++;
    }

    const BufferAllocation* alloc = kernel_argument.slice().allocation();
    if (alloc->is_entry_computation_parameter()) {
      kernel_argument.alignment_ = kEntryParameterAlignBytes;
    } else if (alloc->is_constant()) {
      kernel_argument.alignment_ = kConstantBufferAlignBytes;
    } else {
      kernel_argument.alignment_ = kXlaAllocatedBufferAlignBytes;
    }

    // Note: This code here doesn't check if any partially overlapping buffers
    // are written. Our investigation shows that HloDataflowAnalysis only
    // aliases input and output buffers if they are exactly the same size and
    // location and it aliases one output with at most one input. If that
    // changes then we will have to modify this to something like:
    //
    // kernel_argument.written =
    //   OverlapsAny(buffers_written, kernel_argument.slice);
    kernel_argument.written_ = buffers_written.contains(kernel_argument.slice_);

    kernel_argument.aliased_ = kernel_argument.written_ && [&] {
      for (size_t j = 0; j < kernel_arguments.size(); ++j) {
        const KernelArgument& other_kernel_argument = kernel_arguments[j];
        if (i != j && kernel_argument.slice_ != other_kernel_argument.slice_ &&
            kernel_argument.slice_.OverlapsWith(other_kernel_argument.slice_)) {
          return true;
        }
      }
      return false;
    }();
  }
  return kernel_arguments;
}

absl::StatusOr<KernelArguments> KernelArguments::Create(
    const BufferAssignment& buffer_assignment,
    const HloInstruction* non_fusion_hlo,
    absl::Span<const HloInstruction* const> needed_operands, bool dedup) {
  std::vector<KernelArgument> kernel_arguments;
  for (const HloInstruction* operand : needed_operands) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        buffer_assignment.GetUniqueSlice(operand, {}));
    kernel_arguments.emplace_back(
        KernelArgument(operand->shape(), slice, /*written=*/false));
  }

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      non_fusion_hlo->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) return absl::OkStatus();

        TF_ASSIGN_OR_RETURN(
            BufferAllocation::Slice slice,
            buffer_assignment.GetUniqueSlice(non_fusion_hlo, index));

        kernel_arguments.emplace_back(
            KernelArgument(subshape, slice, /*written=*/true));
        return absl::OkStatus();
      }));

  return KernelArguments{std::move(kernel_arguments), dedup};
}

}  // namespace gpu
}  // namespace xla
