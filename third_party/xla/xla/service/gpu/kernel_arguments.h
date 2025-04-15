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
#ifndef XLA_SERVICE_GPU_KERNEL_ARGUMENTS_H_
#define XLA_SERVICE_GPU_KERNEL_ARGUMENTS_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

// An argument descriptor for kernels.
// Thread-safe.
class KernelArgument {
 public:
  const Shape& shape() const { return shape_; }
  const BufferAllocation::Slice& slice() const { return slice_; }
  bool written() const { return written_; }
  int64_t alignment() const { return alignment_; }
  std::optional<int> first_with_same_slice() const {
    return first_with_same_slice_;
  }
  bool aliased() const { return aliased_; }
  int llvm_arg_index() const { return llvm_arg_index_; }

 private:
  KernelArgument(Shape shape, BufferAllocation::Slice slice, bool written)
      : shape_(shape), slice_(slice), written_(written) {}

  Shape shape_;
  BufferAllocation::Slice slice_;
  bool aliased_ = true;
  int64_t alignment_ = 1;
  bool written_ = true;
  int llvm_arg_index_;
  // Holds the index of the first argument which has the same slice as this,
  // if this is not the first such argument.
  std::optional<int> first_with_same_slice_;

  friend class KernelArguments;
};

class KernelArguments {
 public:
  static absl::StatusOr<KernelArguments> Create(
      const BufferAssignment& buffer_assignment,
      const HloFusionInstruction* fusion);

  static absl::StatusOr<KernelArguments> Create(
      const BufferAssignment& buffer_assignment,
      const HloInstruction* hlo_instruction,
      absl::Span<const HloInstruction* const> needed_operands,
      bool dedup = true);

  const std::vector<KernelArgument>& args() const { return args_; }

 private:
  explicit KernelArguments(std::vector<KernelArgument> args, bool dedup = true)
      : args_(ProcessArguments(std::move(args), dedup)) {}

  static std::vector<KernelArgument> ProcessArguments(
      std::vector<KernelArgument> kernel_arguments, bool dedup);

  std::vector<KernelArgument> args_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_KERNEL_ARGUMENTS_H_
