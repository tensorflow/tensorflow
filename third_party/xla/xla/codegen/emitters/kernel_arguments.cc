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
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/frontend_attributes.h"
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
  // Shape index of each entry of `output_arguments`, in the same order.
  std::vector<ShapeIndex> output_shape_indices;
  absl::flat_hash_set<BufferAllocation::Slice> buffers_written;
};

// Extract output arguments from an instruction's shape and return both
// the arguments and the set of written buffer slices
absl::StatusOr<OutputArguments> ExtractOutputArguments(
    KernelArguments::SliceProvider slice_provider,
    const HloInstruction* hlo_instruction) {
  OutputArguments result;
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      hlo_instruction->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) return absl::OkStatus();

        ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                         slice_provider(*hlo_instruction, index));

        result.output_arguments.emplace_back(KernelArgument(subshape, slice));
        result.output_shape_indices.push_back(index);
        result.buffers_written.insert(slice);
        return absl::OkStatus();
      }));
  return result;
}

// Validates `interleaved_output_indices` and returns the position that each
// result must take in the argument list, in result order.
//
// An empty `interleaved_output_indices` means "no interleaving": the results
// follow the operands.
absl::StatusOr<std::vector<int64_t>> ResultPositions(
    absl::Span<const int32_t> interleaved_output_indices, size_t num_operands,
    size_t num_results) {
  std::vector<int64_t> result_positions;
  result_positions.reserve(num_results);

  if (interleaved_output_indices.empty()) {
    for (size_t i = 0; i < num_results; ++i) {
      result_positions.push_back(num_operands + i);
    }
    return result_positions;
  }

  if (interleaved_output_indices.size() != num_results) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected an interleaving index for each of the ", num_results,
        " results, got ", interleaved_output_indices.size(), " indices"));
  }

  const size_t num_arguments = num_operands + num_results;
  int64_t previous_index = -1;
  for (const int32_t index : interleaved_output_indices) {
    if (index < 0 || static_cast<size_t>(index) >= num_arguments) {
      return absl::InvalidArgumentError("Output index out of bounds");
    }
    if (index <= previous_index) {
      return absl::InvalidArgumentError(
          absl::StrCat("Output indices must be strictly increasing, got ",
                       index, " after ", previous_index));
    }
    previous_index = index;
    result_positions.push_back(index);
  }
  return result_positions;
}

}  // namespace

absl::StatusOr<KernelArguments> KernelArguments::CreateInternal(
    SliceProvider slice_provider, const BufferAlignment& buffer_alignment,
    const HloInstruction* hlo_instruction,
    absl::Span<const Shape> unmanaged_arguments,
    absl::Span<const int32_t> interleaved_output_indices) {
  absl::flat_hash_set<int> no_invariant_operands =
      NonInvariantOperands(*hlo_instruction);

  std::vector<KernelArgument> operand_arguments;
  operand_arguments.reserve(hlo_instruction->operand_count());
  for (auto [op_idx, operand] : llvm::enumerate(hlo_instruction->operands())) {
    ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                     slice_provider(*operand, {}));
    KernelArgument arg(operand->shape(), slice);
    if (no_invariant_operands.contains(op_idx)) {
      arg.set_invariant(false);
    }
    operand_arguments.push_back(std::move(arg));
  }

  ABSL_ASSIGN_OR_RETURN(OutputArguments output_result,
                   ExtractOutputArguments(slice_provider, hlo_instruction));

  ABSL_ASSIGN_OR_RETURN(
      std::vector<int64_t> result_positions,
      ResultPositions(interleaved_output_indices, operand_arguments.size(),
                      output_result.output_arguments.size()));

  const size_t num_managed_arguments =
      operand_arguments.size() + output_result.output_arguments.size();

  std::vector<KernelArgument> kernel_arguments;
  kernel_arguments.reserve(num_managed_arguments + unmanaged_arguments.size());
  ArgumentPositions positions;
  positions.operands.reserve(operand_arguments.size());
  positions.results.reserve(output_result.output_arguments.size());

  // `result_positions` is strictly increasing and holds exactly one position
  // per result, so the remaining positions hold exactly the operands, in
  // operand order.
  size_t next_operand = 0;
  size_t next_result = 0;
  for (size_t position = 0; position < num_managed_arguments; ++position) {
    if (next_result < result_positions.size() &&
        result_positions[next_result] == static_cast<int64_t>(position)) {
      positions.results.emplace_back(
          output_result.output_shape_indices[next_result], position);
      kernel_arguments.push_back(
          std::move(output_result.output_arguments[next_result]));
      ++next_result;
      continue;
    }
    positions.operands.push_back(position);
    kernel_arguments.push_back(std::move(operand_arguments[next_operand]));
    ++next_operand;
  }

  for (const Shape& unmanaged_argument : unmanaged_arguments) {
    kernel_arguments.emplace_back(unmanaged_argument);
  }
  FillKernelArgumentAttributes(kernel_arguments, buffer_alignment,
                               output_result.buffers_written);
  return KernelArguments(std::move(kernel_arguments), std::move(positions));
}

absl::StatusOr<KernelArguments> KernelArguments::Create(
    SliceProvider slice_provider, const BufferAlignment& buffer_alignment,
    const HloInstruction* hlo_instruction,
    absl::Span<const Shape> unmanaged_arguments) {
  return CreateInternal(slice_provider, buffer_alignment, hlo_instruction,
                        unmanaged_arguments,
                        /*interleaved_output_indices=*/{});
}

absl::StatusOr<KernelArguments> KernelArguments::Create(
    const BufferAssignment& buffer_assignment,
    const BufferAlignment& buffer_alignment,
    const HloInstruction* hlo_instruction) {
  auto slice_provider = [&buffer_assignment](const HloInstruction& instruction,
                                             const ShapeIndex& index) {
    return buffer_assignment.GetUniqueSlice(&instruction, index);
  };
  return CreateInternal(slice_provider, buffer_alignment, hlo_instruction, {},
                        /*interleaved_output_indices=*/{});
}

absl::StatusOr<KernelArguments> KernelArguments::Create(
    const BufferAssignment& buffer_assignment,
    const BufferAlignment& buffer_alignment,
    const HloInstruction* hlo_instruction,
    absl::Span<const Shape> unmanaged_arguments) {
  auto slice_provider = [&buffer_assignment](const HloInstruction& instruction,
                                             const ShapeIndex& index) {
    return buffer_assignment.GetUniqueSlice(&instruction, index);
  };
  return CreateInternal(slice_provider, buffer_alignment, hlo_instruction,
                        unmanaged_arguments,
                        /*interleaved_output_indices=*/{});
}

absl::StatusOr<KernelArguments> KernelArguments::Create(
    const BufferAssignment& buffer_assignment,
    const BufferAlignment& buffer_alignment,
    const HloInstruction* hlo_instruction,
    absl::Span<const int32_t> interleaved_output_indices) {
  auto slice_provider = [&buffer_assignment](const HloInstruction& instruction,
                                             const ShapeIndex& index) {
    return buffer_assignment.GetUniqueSlice(&instruction, index);
  };
  return CreateInternal(slice_provider, buffer_alignment, hlo_instruction, {},
                        interleaved_output_indices);
}

absl::StatusOr<int64_t> KernelArguments::PositionOfOperand(
    int64_t operand_number_in_hlo) const {
  if (!positions_.has_value()) {
    return absl::FailedPreconditionError(
        "KernelArguments were not created from an HLO instruction, so operand "
        "positions are unknown");
  }
  if (operand_number_in_hlo < 0 ||
      operand_number_in_hlo >=
          static_cast<int64_t>(positions_->operands.size())) {
    return absl::OutOfRangeError(
        absl::StrCat("No such operand: ", operand_number_in_hlo));
  }
  return positions_->operands[operand_number_in_hlo];
}

absl::StatusOr<int64_t> KernelArguments::PositionOfResult(
    const ShapeIndex& shape_index) const {
  if (!positions_.has_value()) {
    return absl::FailedPreconditionError(
        "KernelArguments were not created from an HLO instruction, so result "
        "positions are unknown");
  }
  for (const auto& [index, position] : positions_->results) {
    if (index == shape_index) {
      return position;
    }
  }
  return absl::OutOfRangeError(
      absl::StrCat("No such result: ", shape_index.ToString()));
}

}  // namespace xla::emitters
