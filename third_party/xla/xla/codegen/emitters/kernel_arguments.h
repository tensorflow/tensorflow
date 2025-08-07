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
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"

namespace xla::emitters {

// An argument descriptor for kernels.
// Thread-safe.
class KernelArgument {
 public:
  KernelArgument(Shape shape, BufferAllocation::Slice slice)
      : shape_(shape), slice_(slice) {}
  const Shape& shape() const { return shape_; }
  const BufferAllocation::Slice& slice() const { return slice_; }

  bool written() const { return written_; }
  void set_written(bool written) { written_ = written; }

  int64_t alignment() const { return alignment_; }
  void set_alignment(int64_t alignment) { alignment_ = alignment; }

  bool aliased() const { return aliased_; }
  void set_aliased(bool aliased) { aliased_ = aliased; }

  int64_t slice_index() const { return slice_index_; }
  void set_slice_index(int64_t slice_index) { slice_index_ = slice_index; }

 private:
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

  static absl::StatusOr<KernelArguments> Create(
      const BufferAssignment& buffer_assignment,
      const BufferAlignment& buffer_alignment,
      const HloInstruction* hlo_instruction);

  explicit KernelArguments(std::vector<KernelArgument>&& args)
      : args_(std::move(args)) {}

  const std::vector<KernelArgument>& args() const { return args_; }

  std::vector<BufferAllocation::Slice> GetArgumentBufferSlices() const {
    std::vector<BufferAllocation::Slice> arg_slices;
    arg_slices.reserve(args_.size());
    for (const KernelArgument& arg : args_) {
      arg_slices.push_back(arg.slice());
    }
    return arg_slices;
  }

  std::vector<bool> GetArgumentOutputFlags() const {
    std::vector<bool> output_flags;
    output_flags.reserve(args_.size());
    for (const KernelArgument& arg : args_) {
      output_flags.push_back(arg.written());
    }
    return output_flags;
  }

 private:
  std::vector<KernelArgument> args_;
};

}  // namespace xla::emitters

#endif  // XLA_CODEGEN_EMITTERS_KERNEL_ARGUMENTS_H_
