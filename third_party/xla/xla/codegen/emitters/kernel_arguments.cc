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
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
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
    if (kernel_argument.kind() == KernelArgument::Kind::kUnmanaged) {
      if (kernel_argument.shape().dimensions().empty()) {
        kernel_argument.set_alignment(0);  // scalars have no alignment
      }
      continue;
    }

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
        if (i == j ||
            kernel_arguments[j].kind() == KernelArgument::Kind::kUnmanaged) {
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

struct OutputArguments {
  std::vector<KernelArgument> output_arguments;
  absl::flat_hash_set<BufferAllocation::Slice> buffers_written;
};

// Extract output arguments from an instruction's shape and return both
// the arguments and the set of written buffer slices
absl::StatusOr<OutputArguments> ExtractOutputArguments(
    const BufferAssignment& buffer_assignment,
    const HloInstruction* hlo_instruction) {
  OutputArguments result;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      hlo_instruction->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) return absl::OkStatus();

        TF_ASSIGN_OR_RETURN(
            BufferAllocation::Slice slice,
            buffer_assignment.GetUniqueSlice(hlo_instruction, index));

        result.output_arguments.emplace_back(KernelArgument(subshape, slice));
        result.buffers_written.insert(slice);
        return absl::OkStatus();
      }));
  return result;
}
absl::StatusOr<KernelArguments> CreateKernelArguments(
    const BufferAssignment& buffer_assignment,
    const KernelArguments::BufferAlignment& buffer_alignment,
    const HloInstruction* hlo_instruction,
    absl::Span<const Shape> unmanaged_arguments) {
  std::vector<KernelArgument> kernel_arguments;
  for (const HloInstruction* operand : hlo_instruction->operands()) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        buffer_assignment.GetUniqueSlice(operand, {}));
    kernel_arguments.emplace_back(KernelArgument(operand->shape(), slice));
  }

  TF_ASSIGN_OR_RETURN(
      OutputArguments output_result,
      ExtractOutputArguments(buffer_assignment, hlo_instruction));

  absl::c_move(output_result.output_arguments,
               std::back_inserter(kernel_arguments));
  for (const Shape& unmanaged_argument : unmanaged_arguments) {
    kernel_arguments.emplace_back(unmanaged_argument);
  }
  FillKernelArgumentAttributes(kernel_arguments, buffer_alignment,
                               output_result.buffers_written);
  return KernelArguments(std::move(kernel_arguments));
}

}  // namespace

absl::StatusOr<KernelArguments> KernelArguments::Create(
    const BufferAssignment& buffer_assignment,
    const BufferAlignment& buffer_alignment,
    const HloInstruction* hlo_instruction) {
  return CreateKernelArguments(buffer_assignment, buffer_alignment,
                               hlo_instruction, {});
}

absl::StatusOr<KernelArguments> KernelArguments::Create(
    const BufferAssignment& buffer_assignment,
    const BufferAlignment& buffer_alignment,
    const HloInstruction* hlo_instruction,
    absl::Span<const Shape> unmanaged_arguments) {
  return CreateKernelArguments(buffer_assignment, buffer_alignment,
                               hlo_instruction, unmanaged_arguments);
}

absl::StatusOr<KernelArguments> KernelArguments::Create(
    const BufferAssignment& buffer_assignment,
    const BufferAlignment& buffer_alignment,
    const HloInstruction* hlo_instruction,
    absl::Span<const int32_t> interleaved_output_indices) {
  if (interleaved_output_indices.empty()) {
    // Fall back to regular Create method when no interleaving is requested
    return CreateKernelArguments(buffer_assignment, buffer_alignment,
                                 hlo_instruction, {});
  }

  const auto& operands = hlo_instruction->operands();

  TF_ASSIGN_OR_RETURN(
      OutputArguments output_result,
      ExtractOutputArguments(buffer_assignment, hlo_instruction));
  auto& [output_arguments, buffers_written] = output_result;

  // Check bounds: all output indices must be valid positions
  size_t total_positions = operands.size() + output_arguments.size();
  for (int32_t idx : interleaved_output_indices) {
    if (idx < 0 || static_cast<size_t>(idx) >= total_positions) {
      return absl::InvalidArgumentError("Output index out of bounds");
    }
  }

  std::vector<KernelArgument> kernel_arguments;
  kernel_arguments.reserve(total_positions);

  // Interleave the inputs and outputs according to the indices
  size_t arg_idx = 0;
  size_t output_pos = 0;

  for (size_t i = 0; i < total_positions; ++i) {
    if (output_pos < interleaved_output_indices.size() &&
        interleaved_output_indices[output_pos] == static_cast<int32_t>(i)) {
      // Place output at this position
      if (output_pos >= output_arguments.size()) {
        return absl::InvalidArgumentError("Invalid output position index");
      }
      kernel_arguments.emplace_back(output_arguments[output_pos]);
      ++output_pos;
    } else {
      // Place input at this position
      if (arg_idx >= operands.size()) {
        return absl::InvalidArgumentError(
            "Not enough inputs for remaining positions");
      }
      TF_ASSIGN_OR_RETURN(
          BufferAllocation::Slice slice,
          buffer_assignment.GetUniqueSlice(operands[arg_idx], {}));
      kernel_arguments.emplace_back(
          KernelArgument(operands[arg_idx]->shape(), slice));
      ++arg_idx;
    }
  }

  // Verify we used all inputs and outputs
  if (arg_idx != operands.size() || output_pos != output_arguments.size()) {
    return absl::InvalidArgumentError("Did not use all inputs/outputs");
  }

  FillKernelArgumentAttributes(kernel_arguments, buffer_alignment,
                               buffers_written);

  return KernelArguments(std::move(kernel_arguments));
}

}  // namespace xla::emitters
