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
#ifndef XLA_SERVICE_GPU_KERNEL_ARGUMENTS_H_
#define XLA_SERVICE_GPU_KERNEL_ARGUMENTS_H_

#include <optional>
#include <utility>
#include <vector>

#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

// An argument descriptor for kernels.
// Thread-safe.
class KernelArgument {
 public:
  static StatusOr<KernelArgument> Create(
      absl::Span<const BufferAllocation* const> allocations, mlir::Value value,
      bool is_written);

  mlir::Value value() const { return value_; }
  const Shape& shape() const { return shape_; }
  const BufferAllocation::Slice& slice() const { return slice_; }
  bool written() const { return written_; }
  int64_t alignment() const { return alignment_; }
  std::optional<int> first_with_same_slice() const {
    return first_with_same_slice_;
  }
  bool aliased() const { return aliased_; }

 private:
  KernelArgument(mlir::Value value, Shape shape, BufferAllocation::Slice slice,
                 bool written)
      : value_(value), shape_(shape), slice_(slice), written_(written) {}

  mlir::Value value_;
  Shape shape_;
  BufferAllocation::Slice slice_;
  bool aliased_ = true;
  int64_t alignment_ = 1;
  bool written_ = true;
  // Holds the index of the first argument which has the same slice as this,
  // if this is not the first such argument.
  std::optional<int> first_with_same_slice_;

  friend class KernelArguments;
};

class KernelArguments {
 public:
  static StatusOr<KernelArguments> Create(
      absl::Span<const BufferAllocation* const> allocations,
      mlir::lmhlo::FusionOp fusion);

  static StatusOr<KernelArguments> Create(
      const BufferAssignment& buffer_assignment,
      const HloFusionInstruction* fusion);

  static StatusOr<KernelArguments> Create(
      absl::Span<const BufferAllocation* const> allocations,
      mlir::Operation* non_fusion_op, mlir::ValueRange needed_operands);

  const std::vector<KernelArgument>& args() const { return args_; }

 private:
  explicit KernelArguments(std::vector<KernelArgument> args)
      : args_(ProcessArguments(std::move(args))) {}

  static std::vector<KernelArgument> ProcessArguments(
      std::vector<KernelArgument> kernel_arguments);

  std::vector<KernelArgument> args_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_KERNEL_ARGUMENTS_H_
