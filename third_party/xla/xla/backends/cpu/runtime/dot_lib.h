/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_DOT_LIB_H_
#define XLA_BACKENDS_CPU_RUNTIME_DOT_LIB_H_

#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Allocation slices of the dot operation.
struct DotSlices {
  BufferAllocation::Slice lhs_buffer;
  Shape lhs_shape;

  BufferAllocation::Slice rhs_buffer;
  Shape rhs_shape;

  BufferAllocation::Slice out_buffer;
  Shape out_shape;
};

// Shape of the batched dot operation supported by the XLA:CPU runtime.
struct DotShape {
  // Product of batch dimensions.
  int64_t batch_size;

  // Shapes of the non-batch matrix-multiplication for the dot operation
  Shape lhs_matmul_shape;
  Shape rhs_matmul_shape;
  Shape out_matmul_shape;
};

// Dot operation is implemented as a matrix-matrix multiply (row-major x
// rowm-major or col-major x col-major). For batched dot operations, it is
// implemented as multiple matrix multiplications repeated for each batch
// element.
struct DotCanonicalDims {
  // The number of rows in the LHS.
  int64_t m;

  // The number of columns in the LHS, which also must be equal to the
  // number of rows in the RHS.
  int64_t k;

  // The number of columns in the RHS.
  int64_t n;

  // True if the LHS matrix is column major.
  bool lhs_column_major;

  // True if the LHS contraction dimension is 1.
  bool lhs_canonical;

  // True if the RHS matrix is column major.
  bool rhs_column_major;

  // True if the RHS contraction dimension is 0.
  bool rhs_canonical;

  // True if the output matrix is column major.
  bool output_column_major;
};

// Returns buffer uses of the dot operation.
absl::InlinedVector<BufferUse, 4> DotBufferUses(const DotSlices& slices);

// Verifies dot dimensions and shapes and returns the shape of the dot operation
// in a form that is convenient for the runtime implementation.
absl::StatusOr<DotShape> GetDotShape(const DotDimensionNumbers& dot_dimensions,
                                     const Shape& lhs_shape,
                                     const Shape& rhs_shape,
                                     const Shape& out_shape);

// Get canonical dot dimensions for the given dot shape.
absl::StatusOr<DotCanonicalDims> GetDotCanonicalDims(
    const DotDimensionNumbers& dot_dimensions, const DotShape& dot_shape);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_DOT_LIB_H_
