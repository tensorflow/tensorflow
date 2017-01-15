/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"

#include <functional>

#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace gpu {

using Index = BufferAllocation::Index;

namespace {

// This struct contains the metadata of a matrix, e.g., its base address and
// dimensions.
struct MatrixDescriptor {
  MatrixDescriptor(se::DeviceMemoryBase matrix_data, bool needs_transpose,
                   int64 matrix_num_rows, int64 matrix_num_cols)
      : data(matrix_data),
        transpose(needs_transpose),
        num_rows(matrix_num_rows),
        num_cols(matrix_num_cols) {}

  se::DeviceMemoryBase data;
  bool transpose;  // Whether this matrix needs to be transposed.
  int64 num_rows;
  int64 num_cols;
};

// Performs a gemm call on lhs_matrix and rhs_matrix and stores the result to
// output_matrix.
template <typename Element>
tensorflow::Status DoGemm(MatrixDescriptor lhs_matrix,
                          MatrixDescriptor rhs_matrix,
                          MatrixDescriptor output_matrix, se::Stream* stream) {
  DCHECK(!output_matrix.transpose);

  se::DeviceMemory<Element> lhs_data(lhs_matrix.data);
  se::DeviceMemory<Element> rhs_data(rhs_matrix.data);
  se::DeviceMemory<Element> output_data(output_matrix.data);

  bool launch_ok =
      stream
          ->ThenBlasGemm(
              lhs_matrix.transpose ? se::blas::Transpose::kTranspose
                                   : se::blas::Transpose::kNoTranspose,
              rhs_matrix.transpose ? se::blas::Transpose::kTranspose
                                   : se::blas::Transpose::kNoTranspose,
              output_matrix.num_rows, output_matrix.num_cols,
              lhs_matrix.transpose
                  ? lhs_matrix.num_rows
                  : lhs_matrix.num_cols,  // Size of the reduce dimension.
              /*alpha=*/1.0,
              lhs_data,
              lhs_matrix.num_rows,  // The leading dimension of LHS.
              rhs_data,
              rhs_matrix.num_rows,  // The leading dimension of RHS.
              /*beta=*/0.0, &output_data,
              output_matrix
                  .num_rows)  // The leading dimension of the output matrix.
          .ok();
  if (!launch_ok) {
    return InternalError("Unable to launch cuBLAS gemm on stream %p", stream);
  }
  return tensorflow::Status::OK();
}

// Return, if the given type is a valid Gemm elemental type, the executor for
// that type, else null.
// TODO(b/27202055): consider more element types.
std::function<tensorflow::Status(MatrixDescriptor, MatrixDescriptor,
                                 MatrixDescriptor, se::Stream*)>
FindGemmExecutor(PrimitiveType type) {
  switch (type) {
    case F32:
      return &DoGemm<float>;
    case F64:
      return &DoGemm<double>;
    default:
      return nullptr;
  }
}

}  // namespace

GemmThunk::GemmThunk(Index lhs_buffer, Index rhs_buffer, Index output_buffer,
                     const Shape& lhs_shape, const Shape& rhs_shape,
                     const Shape& output_shape, bool transpose_lhs,
                     bool transpose_rhs, const HloInstruction* hlo_instruction)
    : Thunk(Kind::kGemm, hlo_instruction),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      lhs_shape_(lhs_shape),
      rhs_shape_(rhs_shape),
      output_shape_(output_shape),
      transpose_lhs_(transpose_lhs),
      transpose_rhs_(transpose_rhs) {}

tensorflow::Status GemmThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream) {
  VLOG(2) << "Executing a GemmThunk";
  auto executor = FindGemmExecutor(output_shape_.element_type());
  DCHECK(executor != nullptr);

  se::DeviceMemoryBase lhs_data =
      buffer_allocations.GetDeviceAddress(lhs_buffer_);
  se::DeviceMemoryBase rhs_data =
      buffer_allocations.GetDeviceAddress(rhs_buffer_);
  se::DeviceMemoryBase output_data =
      buffer_allocations.GetDeviceAddress(output_buffer_);

  // BLAS gemm reduces rows of LHS and columns of RHS. The Dot operator between
  // matrices reduces dimension 1 of LHS and dimension 0 of RHS regardless of
  // their layout. Therefore, we should treat dimension 0 as row and dimension 1
  // as column when mapping a matrix Dot to BLAS gemm.
  int64 output_num_rows = output_shape_.dimensions(0);
  int64 output_num_cols = output_shape_.dimensions(1);

  // BLAS gemm expects the inputs and the output are in column-major order.
  // Therefore, we need to convert dot between row-major matrices to that
  // between column-major matrices. The key insight for the conversion is that,
  // in linear storage, matrix M in column-major order is identical to the
  // tranpose of M in row-major order. In other words,
  //
  //   column-major(M) = row-major(M^T).
  //
  // Leveraging this insight, we can perform dot between row-major matrices as
  // follows.
  //
  // row-major(C)
  //   = row-major(A x B) = column-major((A x B)^T) = column-major(B^T x A^T)
  //   = gemm(column-major(B^T), column-major(A^T))
  //   = gemm(row-major(B), row-major(A))
  //
  // Although we do not modify the content of A and B in linear memory, we
  // should use the dimensions of B^T and A^T when calling gemm. For example,
  // the leading dimension of the LHS matrix of gemm is the number of rows in
  // B^T and thus the number of columns in B.

  auto make_descriptor = [this](se::DeviceMemoryBase data, const Shape& shape,
                                bool transpose) -> MatrixDescriptor {
    bool is_row_major = shape.layout().minor_to_major(0) != 0;
    bool layout_mismatch = shape.layout().minor_to_major(0) !=
                           output_shape_.layout().minor_to_major(0);
    return MatrixDescriptor(data, transpose ^ layout_mismatch,
                            shape.dimensions(is_row_major),
                            shape.dimensions(!is_row_major));
  };

  const MatrixDescriptor lhs_descriptor =
      make_descriptor(lhs_data, lhs_shape_, transpose_lhs_);
  const MatrixDescriptor rhs_descriptor =
      make_descriptor(rhs_data, rhs_shape_, transpose_rhs_);
  if (output_shape_.layout().minor_to_major(0) == 0) {
    return executor(
        lhs_descriptor, rhs_descriptor,
        MatrixDescriptor(output_data, false, output_num_rows, output_num_cols),
        stream);
  } else {
    return executor(
        rhs_descriptor, lhs_descriptor,
        MatrixDescriptor(output_data, false, output_num_cols, output_num_rows),
        stream);
  }
}

}  // namespace gpu
}  // namespace xla
