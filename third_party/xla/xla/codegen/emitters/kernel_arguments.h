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
#ifndef XLA_CODEGEN_EMITTERS_KERNEL_ARGUMENTS_H_
#define XLA_CODEGEN_EMITTERS_KERNEL_ARGUMENTS_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"

namespace xla::emitters {

// An argument descriptor for kernels.
// Thread-safe.
class KernelArgument {
 public:
  // Managed arguments are those that are assigned slices during the buffer
  // assignment pass.
  // Unmanaged arguments can be scalars of tensors. Scalars are simply passed
  // by value. For buffers the memory is managed by the runtime thunk.
  // The[KernelThunk] currently assumes that all arguments are managed.
  // The [CollectiveKernelThunk] distinguishes between the two types of
  // arguments because it manages scratch buffers for the collectives itself.
  enum class Kind { kManaged, kUnmanaged };

  KernelArgument(Shape shape, BufferAllocation::Slice slice)
      : kind_(Kind::kManaged), shape_(shape), slice_(slice) {}
  // Constructor for arguments that don't have an associated slice.
  explicit KernelArgument(Shape shape)
      : kind_(Kind::kUnmanaged), shape_(shape) {}

  Kind kind() const { return kind_; }
  const Shape& shape() const { return shape_; }
  const BufferAllocation::Slice& slice() const { return slice_; }

  bool written() const { return written_; }
  void set_written(bool written) { written_ = written; }

  // An alignment of 0 means that the alignment attribute shouldn't be set.
  int64_t alignment() const { return alignment_; }
  void set_alignment(int64_t alignment) { alignment_ = alignment; }

  bool aliased() const { return aliased_; }
  void set_aliased(bool aliased) { aliased_ = aliased; }

  int64_t slice_index() const { return slice_index_; }
  void set_slice_index(int64_t slice_index) { slice_index_ = slice_index; }

 private:
  Kind kind_;
  Shape shape_;
  BufferAllocation::Slice slice_;
  bool aliased_ = true;
  int64_t alignment_ = 1;
  bool written_ = true;

  // The index of the unique slice in the kernel argument list. When the kernel
  // is called, runtime will pass the same buffer to arguments with the same
  // slice index.
  //
  // This index is used as a hint to XLA emitters to merge uses of arguments
  // with the same slice index into a single argument.
  int64_t slice_index_;

  friend class KernelArguments;
};

class KernelArguments {
 public:
  struct BufferAlignment {
    // Minimum alignment for buffers passed as incoming arguments by users.
    int64_t entry_parameter_align_bytes;

    // Minimum alignment for buffers allocated by XLA: the temp buffers and the
    // live out (result) buffers.
    int64_t xla_allocated_buffer_align_bytes;

    // Minimum alignment for constant buffers.
    int64_t constant_buffer_align_bytes;
  };

  // Creates a KernelArguments object for the given HLO instruction.
  // The unmanaged_arguments are added to the end of the list of input/output
  // arguments.
  static absl::StatusOr<KernelArguments> Create(
      const BufferAssignment& buffer_assignment,
      const BufferAlignment& buffer_alignment,
      const HloInstruction* hlo_instruction,
      absl::Span<const Shape> unmanaged_arguments);

  static absl::StatusOr<KernelArguments> Create(
      const BufferAssignment& buffer_assignment,
      const BufferAlignment& buffer_alignment,
      const HloInstruction* hlo_instruction);

  // Certain kernels require output arguments to be interleaved with input
  // arguments. This function creates a KernelArguments object where the output
  // arguments are interleaved with the input arguments according to the
  // provided indices.
  // Example: If hlo_instruction->operands() has 3 elements and hlo_instruction
  // shape yields 2 output arguments, and interleaved_output_indices = {1, 4}:
  // - Final argument order will be: input0, output0, input1, input2, output1
  static absl::StatusOr<KernelArguments> Create(
      const BufferAssignment& buffer_assignment,
      const BufferAlignment& buffer_alignment,
      const HloInstruction* hlo_instruction,
      absl::Span<const int32_t> interleaved_output_indices);

  explicit KernelArguments(std::vector<KernelArgument>&& args)
      : args_(std::move(args)) {}

  const std::vector<KernelArgument>& args() const { return args_; }

  std::vector<ShapedSlice> GetArgumentShapedSlices() const {
    std::vector<ShapedSlice> arg_slices;
    arg_slices.reserve(args_.size());
    for (const KernelArgument& arg : args_) {
      arg_slices.push_back({arg.slice(), arg.shape()});
    }
    return arg_slices;
  }

  std::vector<BufferAllocation::Slice> GetArgumentBufferSlices() const {
    std::vector<BufferAllocation::Slice> arg_slices;
    arg_slices.reserve(args_.size());
    for (const KernelArgument& arg : args_) {
      arg_slices.push_back(arg.slice());
    }
    return arg_slices;
  }

  std::vector<Shape> GetArgumentBufferShapes() const {
    std::vector<Shape> arg_shapes;
    arg_shapes.reserve(args_.size());
    for (const KernelArgument& arg : args_) {
      arg_shapes.push_back(arg.shape());
    }
    return arg_shapes;
  }

  std::vector<bool> GetArgumentOutputFlags() const {
    std::vector<bool> output_flags;
    output_flags.reserve(args_.size());
    for (const KernelArgument& arg : args_) {
      output_flags.push_back(arg.written());
    }
    return output_flags;
  }

  std::vector<KernelArgument::Kind> GetArgumentKinds() const {
    std::vector<KernelArgument::Kind> kinds;
    kinds.reserve(args_.size());
    for (const KernelArgument& arg : args_) {
      kinds.push_back(arg.kind());
    }
    return kinds;
  }

 private:
  std::vector<KernelArgument> args_;
};

}  // namespace xla::emitters

#endif  // XLA_CODEGEN_EMITTERS_KERNEL_ARGUMENTS_H_
