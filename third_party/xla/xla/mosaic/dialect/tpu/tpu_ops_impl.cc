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

#include "xla/mosaic/dialect/tpu/tpu_ops_impl.h"

#include <cstdint>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tpu::impl {

LogicalResult verifyGather(Operation* op, ArrayRef<int64_t> operand_shape,
                           ArrayRef<int64_t> offsets_shape,
                           ArrayRef<int64_t> result_shape) {
  // Expected shapes:
  //   Slice shape   : [s0, ..., sm]
  //
  //   1D offsets:
  //     Operand shape : [z, s0, ..., sm]
  //     Offsets shape : [o]
  //     Result shape  : [o, s0, ..., sm]
  //
  //   2D offsets:
  //     Operand shape : [1, z, s0, ..., sm]
  //     Offsets shape : [1, o]
  //     Result shape  : [1, o, s0, ..., sm]

  uint64_t offsets_rank = offsets_shape.size();

  // We require offsets shape and operand sample shape to be 1D or (1, N), and
  // their ranks must match.
  // Offsets shape : [o] or [1, o]
  // Operand sample shape : [z] or [1, z]
  if ((offsets_rank != 1 && offsets_rank != 2) ||
      (offsets_rank == 2 && offsets_shape[0] != 1)) {
    return op->emitOpError("Offsets shape must be 1D or (1, N), got (")
           << absl::StrJoin(offsets_shape, ", ") << ")";
  }

  if (result_shape.size() < offsets_rank) {
    return op->emitOpError(
               "Target (gather result) rank must be >= offsets rank, ")
           << "got target rank: " << result_shape.size()
           << ", offsets rank: " << offsets_rank;
  }
  uint64_t slice_rank = result_shape.size() - offsets_rank;
  if (operand_shape.size() <= slice_rank) {
    return op->emitOpError(
               "Source (gather operand) rank must be > slice rank, ")
           << "got source rank: " << operand_shape.size()
           << ", slice rank: " << slice_rank;
  }
  uint64_t operand_sample_rank = operand_shape.size() - slice_rank;
  ArrayRef<int64_t> result_offset_dims = result_shape.take_front(offsets_rank);
  ArrayRef<int64_t> result_slice_dims = result_shape.take_back(slice_rank);
  ArrayRef<int64_t> operand_slice_dims = operand_shape.take_back(slice_rank);
  ArrayRef<int64_t> operand_sample_dims =
      operand_shape.take_front(operand_sample_rank);

  if (operand_sample_rank > 2 ||
      (operand_sample_rank == 2 && operand_sample_dims[0] != 1)) {
    return op->emitOpError("Source (gather operand) sample shape must be ")
           << "1D or (1, N), got (" << absl::StrJoin(operand_sample_dims, ", ")
           << ")";
  }
  if (operand_sample_rank != offsets_rank) {
    return op->emitOpError("Source (gather operand) sample rank must match ")
           << "offsets rank, got " << operand_sample_rank << " vs "
           << offsets_rank;
  }

  const std::string result_shape_str = absl::StrJoin(result_shape, ", ");

  // Make sure that there is one output slice per offset.
  // Offsets shape : [o] or [1, o]
  // Result shape  : [o'0, .., o'p, s0, .., sm]
  // [o] or [1, o] == [o'0, .., o'p]
  if (!absl::c_equal(offsets_shape, result_offset_dims)) {
    return op->emitOpError("Offsets shape (")
           << absl::StrJoin(offsets_shape, ", ")
           << ") must match the majormost dimensions of the target (gather "
              "result) shape ("
           << result_shape_str << ")";
  }

  // At each offset, we are copying an ND slice of data. Make sure that the
  // slice shape is the same in the operand and the output for the gather.
  // Operand shape : [z, s0, .., sm] or [1, z, s0, .., sm]
  // Result shape :  [o, s'0, .., s'm] or [1, o, s'0, .., s'm]
  // [s0, .., sm] == [s'0, .., s'm]
  if (!absl::c_equal(operand_slice_dims, result_slice_dims)) {
    const std::string plural = slice_rank == 1 ? "" : "s";
    return op->emitOpError(absl::StrFormat(
        "%d minormost dimension%s of the source (gather operand) shape (%s) "
        "must match the minormost dimension%s of the target (gather result) "
        "shape (%s)",
        slice_rank, plural, absl::StrJoin(operand_shape, ", "), plural,
        result_shape_str));
  }
  return success();
}

LogicalResult verifyScatter(Operation* op, ArrayRef<int64_t> updates_shape,
                            ArrayRef<int64_t> offsets_shape,
                            ArrayRef<int64_t> operand_shape) {
  // Expected shapes:
  //   Slice shape   : [s0, ..., sm]
  //
  //   1D offsets:
  //     Operand shape : [z, s0, ..., sm]
  //     Offsets shape : [o]
  //     Updates shape : [o, s0, ..., sm]
  //
  //   2D offsets:
  //     Operand shape : [1, z, s0, ..., sm]
  //     Offsets shape : [1, o]
  //     Updates shape : [1, o, s0, ..., sm]

  uint64_t offsets_rank = offsets_shape.size();

  // We require offsets shape and operand sample shape to be 1D or (1, N), and
  // their ranks must match.
  // Offsets shape : [o] or [1, o]
  // Operand sample shape : [z] or [1, z]
  if ((offsets_rank != 1 && offsets_rank != 2) ||
      (offsets_rank == 2 && offsets_shape[0] != 1)) {
    return op->emitOpError("Offsets shape must be 1D or (1, N), got (")
           << absl::StrJoin(offsets_shape, ", ") << ")";
  }

  if (updates_shape.size() < offsets_rank) {
    return op->emitOpError(
               "Source (scatter updates) rank must be >= offsets rank, ")
           << "got source rank: " << updates_shape.size()
           << ", offsets rank: " << offsets_rank;
  }
  uint64_t slice_rank = updates_shape.size() - offsets_rank;
  if (operand_shape.size() <= slice_rank) {
    return op->emitOpError(
               "Target (scatter operand) rank must be > slice rank, ")
           << "got target rank: " << operand_shape.size()
           << ", slice rank: " << slice_rank;
  }
  uint64_t operand_sample_rank = operand_shape.size() - slice_rank;
  ArrayRef<int64_t> updates_offset_dims =
      updates_shape.take_front(offsets_rank);
  ArrayRef<int64_t> updates_slice_dims = updates_shape.take_back(slice_rank);
  ArrayRef<int64_t> operand_slice_dims = operand_shape.take_back(slice_rank);
  ArrayRef<int64_t> operand_sample_dims =
      operand_shape.take_front(operand_sample_rank);
  if (operand_sample_rank > 2 ||
      (operand_sample_rank == 2 && operand_sample_dims[0] != 1)) {
    return op->emitOpError("Target (scatter operand) sample shape must be ")
           << "1D or (1, N), got (" << absl::StrJoin(operand_sample_dims, ", ")
           << ")";
  }
  if (operand_sample_rank != offsets_rank) {
    return op->emitOpError("Target (scatter operand) sample rank must match ")
           << "offsets rank, got " << operand_sample_rank << " vs "
           << offsets_rank;
  }

  const std::string updates_shape_str = absl::StrJoin(updates_shape, ", ");

  // Make sure that there is one slice of updates per offset.
  // Offsets shape : [o] or [1, o]
  // Updates shape : [o'0, .., o'p, s0, .., sm]
  // [o] or [1, o] == [o'0, .., o'p]
  if (!absl::c_equal(offsets_shape, updates_offset_dims)) {
    return op->emitOpError("Offsets shape (")
           << absl::StrJoin(offsets_shape, ", ")
           << ") must match the majormost dimensions of the source "
              "(scatter updates) shape ("
           << updates_shape_str << ")";
  }

  // At each offset, we are copying an ND slice of data. Make sure that the
  // slice shape is the same in the updates and the operand for the scatter.
  // Updates shape : [o, s0, .., sm] or [1, o, s0, .., sm]
  // Operand shape : [z, s'0, .., s'm] or [1, z, s'0, .., s'm]
  // [s0, .., sm] == [s'0, .., s'm]
  if (!absl::c_equal(operand_slice_dims, updates_slice_dims)) {
    const std::string plural = slice_rank == 1 ? "" : "s";
    return op->emitOpError(absl::StrFormat(
        "%d minormost dimension%s of the source (scatter updates) shape (%s) "
        "must match the minormost dimension%s of the target (scatter operand) "
        "shape (%s)",
        slice_rank, plural, updates_shape_str, plural,
        absl::StrJoin(operand_shape, ", ")));
  }
  return success();
}
}  // namespace mlir::tpu::impl
