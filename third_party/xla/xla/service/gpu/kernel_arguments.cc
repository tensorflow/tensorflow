/*Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

StatusOr<KernelArgument> KernelArgument::Create(
    absl::Span<const BufferAllocation> allocations, mlir::Value value,
    bool is_written) {
  TF_ASSIGN_OR_RETURN(
      auto slice, xla::gpu::GetAllocationSlice(value, allocations, nullptr));
  return KernelArgument(value, GetShape(value), slice, is_written);
}

StatusOr<KernelArguments> KernelArguments::Create(
    absl::Span<const BufferAllocation> allocations,
    mlir::lmhlo::FusionOp fusion) {
  auto operands = GetHloOperands(fusion);
  auto outputs = GetHloOutputs(fusion);
  std::vector<KernelArgument> kernel_arguments;
  kernel_arguments.reserve(operands.size() + outputs.size());

  for (auto value : operands) {
    TF_ASSIGN_OR_RETURN(auto arg, KernelArgument::Create(allocations, value,
                                                         /*is_written=*/false));
    kernel_arguments.emplace_back(std::move(arg));
  }
  for (auto value : outputs) {
    TF_ASSIGN_OR_RETURN(auto arg, KernelArgument::Create(allocations, value,
                                                         /*is_written=*/true));
    kernel_arguments.emplace_back(std::move(arg));
  }

  return KernelArguments{std::move(kernel_arguments)};
}

StatusOr<KernelArguments> KernelArguments::Create(
    const BufferAssignment& buffer_assignment,
    const HloFusionInstruction* fusion) {
  std::vector<KernelArgument> kernel_arguments;
  for (const HloInstruction* operand : fusion->operands()) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        buffer_assignment.GetUniqueSlice(operand, {}));
    kernel_arguments.emplace_back(
        KernelArgument(nullptr, operand->shape(), slice, false));
  }
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      buffer_assignment.GetUniqueSlice(fusion, {}));
  kernel_arguments.emplace_back(
      KernelArgument(nullptr, fusion->shape(), slice, true));
  return KernelArguments{std::move(kernel_arguments)};
}

std::vector<KernelArgument> KernelArguments::ProcessArguments(
    std::vector<KernelArgument> kernel_arguments) {
  absl::flat_hash_set<BufferAllocation::Slice> buffers_written;
  for (const KernelArgument& kernel_argument : kernel_arguments) {
    if (kernel_argument.written()) {
      buffers_written.insert(kernel_argument.slice());
    }
  }

  absl::flat_hash_map<BufferAllocation::Slice, std::optional<int64_t>>
      first_indices_for_slices;
  for (int i = 0; i < static_cast<int>(kernel_arguments.size()); ++i) {
    KernelArgument& kernel_argument = kernel_arguments[i];

    auto& first_index = first_indices_for_slices[kernel_argument.slice_];
    if (first_index) {
      const KernelArgument& same = kernel_arguments[*first_index];
      kernel_argument.first_with_same_slice_ = first_index;
      kernel_argument.alignment_ = same.alignment_;
      kernel_argument.aliased_ = same.aliased_;
      kernel_argument.written_ = same.written_;
      continue;
    } else {
      first_index = i;
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

StatusOr<KernelArguments> KernelArguments::Create(
    absl::Span<const BufferAllocation> allocations,
    mlir::Operation* non_fusion_op, mlir::ValueRange needed_operands) {
  std::vector<KernelArgument> kernel_arguments;
  kernel_arguments.reserve(needed_operands.size());
  for (const auto& [i, value] : llvm::enumerate(needed_operands)) {
    bool written = WritesMlirBuffer(non_fusion_op, value);
    TF_ASSIGN_OR_RETURN(auto arg,
                        KernelArgument::Create(allocations, value, written));
    kernel_arguments.emplace_back(std::move(arg));
  }
  return KernelArguments{std::move(kernel_arguments)};
}

}  // namespace gpu
}  // namespace xla
