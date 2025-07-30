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
#include "xla/codegen/emitters/kernel_arguments.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::emitters {

namespace {

int64_t GetAlignment(const BufferAllocation* alloc,
                     const KernelArguments::BufferAlignment& buffer_alignment) {
  if (alloc->is_entry_computation_parameter()) {
    return buffer_alignment.entry_parameter_align_bytes;
  }

  if (alloc->is_constant()) {
    return buffer_alignment.constant_buffer_align_bytes;
  }

  return buffer_alignment.xla_allocated_buffer_align_bytes;
}

void FillKernelArgumentAttributes(
    std::vector<KernelArgument>& kernel_arguments,
    const KernelArguments::BufferAlignment& buffer_alignment,
    const absl::flat_hash_set<BufferAllocation::Slice>& buffers_written) {
  absl::flat_hash_map<BufferAllocation::Slice, std::optional<int64_t>>
      first_indices_for_slices;
  int64_t next_slice_index = 0;

  for (int64_t i = 0; i < kernel_arguments.size(); ++i) {
    KernelArgument& kernel_argument = kernel_arguments[i];

    auto& first_index = first_indices_for_slices[kernel_argument.slice()];
    if (first_index.has_value()) {
      KernelArgument& first_with_same_slice = kernel_arguments[*first_index];

      kernel_argument.set_slice_index(first_with_same_slice.slice_index());
      kernel_argument.set_alignment(first_with_same_slice.alignment());
      kernel_argument.set_written(first_with_same_slice.written());
      kernel_argument.set_aliased(first_with_same_slice.aliased());
      continue;
    }

    first_index = i;
    kernel_argument.set_slice_index(next_slice_index);
    next_slice_index++;

    kernel_argument.set_alignment(
        GetAlignment(kernel_argument.slice().allocation(), buffer_alignment));

    // Note: This code here doesn't check if any partially overlapping buffers
    // are written. Our investigation shows that HloDataflowAnalysis only
    // aliases input and output buffers if they are exactly the same size and
    // location and it aliases one output with at most one input. If that
    // changes then we will have to modify this to something like:
    //
    // kernel_argument.written =
    //   OverlapsAny(buffers_written, kernel_argument.slice);
    kernel_argument.set_written(
        buffers_written.contains(kernel_argument.slice()));

    kernel_argument.set_aliased(kernel_argument.written() && [&] {
      for (size_t j = 0; j < kernel_arguments.size(); ++j) {
        if (i == j) {
          continue;
        }

        const KernelArgument& other_kernel_argument = kernel_arguments[j];
        if (kernel_argument.slice() == other_kernel_argument.slice()) {
          continue;
        }

        if (kernel_argument.slice().OverlapsWith(
                other_kernel_argument.slice())) {
          return true;
        }
      }
      return false;
    }());
  }
}

}  // namespace

absl::StatusOr<KernelArguments> KernelArguments::Create(
    const BufferAssignment& buffer_assignment,
    const BufferAlignment& buffer_alignment,
    const HloInstruction* hlo_instruction) {
  std::vector<KernelArgument> kernel_arguments;
  for (const HloInstruction* operand : hlo_instruction->operands()) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        buffer_assignment.GetUniqueSlice(operand, {}));
    kernel_arguments.emplace_back(KernelArgument(operand->shape(), slice));
  }

  absl::flat_hash_set<BufferAllocation::Slice> buffers_written;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      hlo_instruction->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) return absl::OkStatus();

        TF_ASSIGN_OR_RETURN(
            BufferAllocation::Slice slice,
            buffer_assignment.GetUniqueSlice(hlo_instruction, index));

        kernel_arguments.emplace_back(KernelArgument(subshape, slice));
        buffers_written.insert(slice);
        return absl::OkStatus();
      }));

  FillKernelArgumentAttributes(kernel_arguments, buffer_alignment,
                               buffers_written);

  return KernelArguments{std::move(kernel_arguments)};
}

}  // namespace xla::emitters
