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

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

namespace {

// This struct contains the metadata of a matrix, e.g., its base address and
// dimensions.
struct MatrixDescriptor {
  MatrixDescriptor(se::DeviceMemoryBase matrix_data, bool needs_transpose,
                   int64 matrix_num_rows, int64 matrix_num_cols,
                   int64 matrix_batch_size)
      : data(matrix_data),
        transpose(needs_transpose),
        num_rows(matrix_num_rows),
        num_cols(matrix_num_cols),
        batch_size(matrix_batch_size) {}

  se::DeviceMemoryBase data;
  bool transpose;  // Whether this matrix needs to be transposed.
  int64 num_rows;
  int64 num_cols;
  int64 batch_size;
};

// Performs a gemm call without an explicit algorithm on lhs_matrix and
// rhs_matrix, and stores the result to output_matrix.
template <typename Element>
bool DoGemm(MatrixDescriptor lhs_matrix, MatrixDescriptor rhs_matrix,
            MatrixDescriptor output_matrix, double alpha, double beta,
            se::Stream* stream) {
  DCHECK(!output_matrix.transpose);

  const int64 batch_size = lhs_matrix.batch_size;
  CHECK_EQ(batch_size, rhs_matrix.batch_size);
  CHECK_EQ(batch_size, output_matrix.batch_size);
  se::DeviceMemory<Element> lhs_data(lhs_matrix.data);
  se::DeviceMemory<Element> rhs_data(rhs_matrix.data);
  se::DeviceMemory<Element> output_data(output_matrix.data);

  auto lhs_transpose = lhs_matrix.transpose ? se::blas::Transpose::kTranspose
                                            : se::blas::Transpose::kNoTranspose;
  auto rhs_transpose = rhs_matrix.transpose ? se::blas::Transpose::kTranspose
                                            : se::blas::Transpose::kNoTranspose;
  auto k = lhs_matrix.transpose ? lhs_matrix.num_rows : lhs_matrix.num_cols;

  if (batch_size == 1) {
    return stream
        ->ThenBlasGemm(
            lhs_transpose, rhs_transpose, output_matrix.num_rows,
            output_matrix.num_cols, /*size of reduce dim=*/k, /*alpha=*/alpha,
            lhs_data, /*leading dim of LHS=*/lhs_matrix.num_rows, rhs_data,
            /*leading dim of RHS=*/rhs_matrix.num_rows, /*beta=*/beta,
            &output_data, /*leading dim of output=*/output_matrix.num_rows)
        .ok();
  }

  int64 lhs_stride = lhs_matrix.num_rows * lhs_matrix.num_cols;
  int64 rhs_stride = rhs_matrix.num_rows * rhs_matrix.num_cols;
  int64 output_stride = output_matrix.num_rows * output_matrix.num_cols;
  return stream
      ->ThenBlasGemmStridedBatched(
          lhs_transpose, rhs_transpose, output_matrix.num_rows,
          output_matrix.num_cols, /*size of reduce dim=*/k,
          /*alpha=*/alpha, lhs_data,
          /*leading dim of LHS=*/lhs_matrix.num_rows, lhs_stride, rhs_data,
          /*leading dim of RHS=*/rhs_matrix.num_rows, rhs_stride,
          /*beta=*/beta, &output_data,
          /*leading dim of output=*/output_matrix.num_rows, output_stride,
          batch_size)
      .ok();
}

// Like DoGemm, but takes an explicit computation type and algorithm.
// computation_type specifies the type of intermediate values generated during
// the matmul (e.g. your input/output matricies could be f16s but you could do
// computations with f32s).  algorithm is an opaque identifier which functions
// as a hint to cublas.
//
// Not all algorithms are valid for all matrix sizes, and not all CUDA versions
// and GPUs even support gemm-with-algorithm.  So expect that this may fail
// unless you've already checked that it works for this particular GPU + input
// size.
//
// If you pass a non-null ProfileResult, this will always return true (assuming
// the Stream was valid to begin with); check the is_valid property of the
// ProfileResult to see whether the call actually succeeded.
template <typename Element>
bool DoGemmWithAlgorithm(MatrixDescriptor lhs_matrix,
                         MatrixDescriptor rhs_matrix,
                         MatrixDescriptor output_matrix, double alpha,
                         double beta,
                         se::blas::ComputationType computation_type,
                         se::blas::AlgorithmType algorithm, se::Stream* stream,
                         se::blas::ProfileResult* output_profile_result) {
  DCHECK(!output_matrix.transpose);

  CHECK_EQ(1, lhs_matrix.batch_size);
  CHECK_EQ(1, rhs_matrix.batch_size);
  CHECK_EQ(1, output_matrix.batch_size);

  se::DeviceMemory<Element> lhs_data(lhs_matrix.data);
  se::DeviceMemory<Element> rhs_data(rhs_matrix.data);
  se::DeviceMemory<Element> output_data(output_matrix.data);

  auto lhs_transpose = lhs_matrix.transpose ? se::blas::Transpose::kTranspose
                                            : se::blas::Transpose::kNoTranspose;
  auto rhs_transpose = rhs_matrix.transpose ? se::blas::Transpose::kTranspose
                                            : se::blas::Transpose::kNoTranspose;
  auto k = lhs_matrix.transpose ? lhs_matrix.num_rows : lhs_matrix.num_cols;

  return stream
      ->ThenBlasGemmWithAlgorithm(
          lhs_transpose, rhs_transpose, output_matrix.num_rows,
          output_matrix.num_cols, /*size of reduce dim=*/k,
          /*alpha=*/static_cast<Element>(alpha), lhs_data,
          /*leading dim of LHS=*/lhs_matrix.num_rows, rhs_data,
          /*leading dim of RHS=*/rhs_matrix.num_rows,
          /*beta=*/static_cast<Element>(beta), &output_data,
          /*leading dim of output=*/output_matrix.num_rows, computation_type,
          algorithm, output_profile_result)
      .ok();
}

// Experimentally tries to pick the best algorithm for the given gemm.
//
// This may fail under perfectly normal circumstances.  In particular, it will
// fail if the program was built with < CUDA 8 or if we're using a gpu older
// than sm_50 -- in both cases, cublas doesn't support gemm-with-algorithm at
// all.
template <typename Element>
StatusOr<se::blas::AlgorithmType> DoGemmAutotune(
    MatrixDescriptor lhs_matrix, MatrixDescriptor rhs_matrix,
    MatrixDescriptor output_matrix, double alpha, double beta,
    se::blas::ComputationType computation_type, se::Stream* stream) {
  std::vector<se::blas::AlgorithmType> algorithms;
  CHECK(stream->parent()->GetBlasGemmAlgorithms(&algorithms));

  se::blas::ProfileResult best_result;
  for (auto algorithm : algorithms) {
    se::blas::ProfileResult profile_result;
    // We expect GemmWithAlgorithm to fail sometimes -- in fact, it will fail
    // for all algorithms if we're targeting < sm_50.  But because we pass a
    // non-null ProfileResult, DoGemmWithAlgorithm should always return true,
    // and the actual success-ness is returned in ProfileResult::is_valid.
    CHECK(DoGemmWithAlgorithm<Element>(lhs_matrix, rhs_matrix, output_matrix,
                                       alpha, beta, computation_type, algorithm,
                                       stream, &profile_result));

    if (profile_result.is_valid()) {
      VLOG(3) << "cublas gemm algorithm " << algorithm << " took "
              << profile_result.elapsed_time_in_ms() << "ms";
      if (profile_result.elapsed_time_in_ms() <
          best_result.elapsed_time_in_ms()) {
        best_result = profile_result;
      }
    } else {
      VLOG(4) << "cublas gemm algorithm " << algorithm << " failed.";
    }
  }

  if (best_result.is_valid()) {
    return best_result.algorithm();
  }

  return InternalError(
      "Unable to autotune cuBLAS gemm on stream %p; none of the %u algorithms "
      "ran successfully",
      stream, algorithms.size());
}

// Helper functions to go from a PrimitiveType to a templated version of
// DoGemm/DoGemmWithAlgorithm/DoGemmAutotune.
auto GetGemmFn(PrimitiveType type) -> decltype(&DoGemm<float>) {
  switch (type) {
    case F16:
      return &DoGemm<Eigen::half>;
    case F32:
      return &DoGemm<float>;
    case F64:
      return &DoGemm<double>;
    case C64:
      return &DoGemm<std::complex<float>>;
    default:
      LOG(FATAL) << "Unsupported type.";
  }
}
auto GetGemmWithAlgorithmFn(PrimitiveType type)
    -> decltype(&DoGemmWithAlgorithm<float>) {
  switch (type) {
    case F16:
      return &DoGemmWithAlgorithm<Eigen::half>;
    case F32:
      return &DoGemmWithAlgorithm<float>;
    case F64:
      return &DoGemmWithAlgorithm<double>;
    case C64:
      return &DoGemmWithAlgorithm<std::complex<float>>;
    default:
      LOG(FATAL) << "Unsupported type.";
  }
}
auto GetGemmAutotuneFn(PrimitiveType type) -> decltype(&DoGemmAutotune<float>) {
  switch (type) {
    case F16:
      return &DoGemmAutotune<Eigen::half>;
    case F32:
      return &DoGemmAutotune<float>;
    case F64:
      return &DoGemmAutotune<double>;
    case C64:
      return &DoGemmAutotune<std::complex<float>>;
    default:
      LOG(FATAL) << "Unsupported type.";
  }
}

// Converts from an XLA PrimitiveType to a blas::ComputationType, which is used
// to specify the precision with which matmul computations should be performed,
// separately from the precision of the inputs and result.
se::blas::ComputationType GetBlasComputationType(PrimitiveType type) {
  switch (type) {
    case F16:
      // Use F32 as computation type for F16 as we currently only implement the
      // cuDNN pseudo half configuration for half precision.
      return se::blas::ComputationType::kF32;
    case F32:
      return se::blas::ComputationType::kF32;
    case F64:
      return se::blas::ComputationType::kF64;
    case C64:
      return se::blas::ComputationType::kComplexF32;
    default:
      LOG(FATAL) << "Unsupported type.";
  }
}

DotDimensionNumbers GetDimensionNumbers(const HloInstruction& hlo_instruction) {
  if (hlo_instruction.opcode() == HloOpcode::kDot) {
    return hlo_instruction.dot_dimension_numbers();
  }
  CHECK_EQ(hlo_instruction.opcode(), HloOpcode::kFusion);
  CHECK_EQ(hlo_instruction.fusion_kind(), HloInstruction::FusionKind::kOutput);
  CHECK(hlo_instruction.fused_expression_root()->opcode() == HloOpcode::kAdd ||
        hlo_instruction.fused_expression_root()->opcode() ==
            HloOpcode::kMultiply);
  // Try to find the dot inside the output fusion node.
  const HloInstruction* dot =
      hlo_instruction.fused_expression_root()->operand(0);
  if (dot->opcode() != HloOpcode::kDot) {
    dot = hlo_instruction.fused_expression_root()->operand(1);
  }
  CHECK_EQ(dot->opcode(), HloOpcode::kDot);

  return dot->dot_dimension_numbers();
}

}  // namespace

GemmThunk::GemmThunk(const BufferAllocation::Slice& lhs_buffer,
                     const BufferAllocation::Slice& rhs_buffer,
                     const BufferAllocation::Slice& output_buffer,
                     const Shape& lhs_shape, const Shape& rhs_shape,
                     const Shape& output_shape, double alpha, double beta,
                     const HloInstruction* hlo_instruction,
                     bool implements_whole_instruction)
    : Thunk(Kind::kGemm, hlo_instruction),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      lhs_shape_(lhs_shape),
      rhs_shape_(rhs_shape),
      output_shape_(output_shape),
      alpha_(alpha),
      beta_(beta),
      implements_whole_instruction_(implements_whole_instruction) {}

Status GemmThunk::ExecuteOnStream(const BufferAllocations& buffer_allocations,
                                  se::Stream* stream,
                                  HloExecutionProfiler* profiler) {
  VLOG(2) << "Executing a GemmThunk";

  se::DeviceMemoryBase lhs_data =
      buffer_allocations.GetDeviceAddress(lhs_buffer_);
  se::DeviceMemoryBase rhs_data =
      buffer_allocations.GetDeviceAddress(rhs_buffer_);
  se::DeviceMemoryBase output_data =
      buffer_allocations.GetDeviceAddress(output_buffer_);

  DotDimensionNumbers dim_nums = GetDimensionNumbers(*hlo_instruction());
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size(),
           dim_nums.rhs_batch_dimensions_size());
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size() + 2,
           ShapeUtil::Rank(output_shape_));

  int64 row_dim = dim_nums.lhs_batch_dimensions_size();
  int64 col_dim = dim_nums.lhs_batch_dimensions_size() + 1;
  int64 batch_size = std::accumulate(output_shape_.dimensions().begin(),
                                     output_shape_.dimensions().end() - 2, 1,
                                     std::multiplies<int64>());

  // Check that the batch dims don't cover the last two dims.
  for (int64 batch_dim : dim_nums.lhs_batch_dimensions()) {
    CHECK_NE(row_dim, batch_dim);
    CHECK_NE(col_dim, batch_dim);
  }

  // Verify that the non-batch dimensions are minor-most. This is required for
  // efficient access.
  for (const auto* shape : {&lhs_shape_, &rhs_shape_, &output_shape_}) {
    CHECK_LT(shape->layout().minor_to_major(row_dim), 2);
    CHECK_LT(shape->layout().minor_to_major(col_dim), 2);
  }

  // BLAS gemm reduces rows of LHS and columns of RHS. The Dot operator between
  // matrices reduces dimension 1 of LHS and dimension 0 of RHS regardless of
  // their layout. Therefore, we should treat dimension 0 as row and dimension 1
  // as column when mapping a matrix Dot to BLAS gemm.
  int64 output_num_rows = output_shape_.dimensions(row_dim);
  int64 output_num_cols = output_shape_.dimensions(col_dim);

  // BLAS gemm expects the inputs and the output are in column-major order.
  // Therefore, we need to convert dot between row-major matrices to that
  // between column-major matrices. The key insight for the conversion is that,
  // in linear storage, matrix M in column-major order is identical to the
  // transpose of M in row-major order. In other words,
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

  auto make_descriptor = [&](se::DeviceMemoryBase data, const Shape& shape,
                             bool transpose) -> MatrixDescriptor {
    bool is_row_major = LayoutUtil::Minor(shape.layout(), row_dim) != 0;
    bool layout_mismatch = LayoutUtil::Minor(shape.layout(), row_dim) !=
                           LayoutUtil::Minor(output_shape_.layout(), row_dim);
    return MatrixDescriptor(
        data, transpose ^ layout_mismatch,
        shape.dimensions(row_dim + static_cast<int64>(is_row_major)),
        shape.dimensions(row_dim + static_cast<int64>(!is_row_major)),
        batch_size);
  };

  const MatrixDescriptor lhs_descriptor = make_descriptor(
      lhs_data, lhs_shape_, dim_nums.lhs_contracting_dimensions(0) == row_dim);
  const MatrixDescriptor rhs_descriptor = make_descriptor(
      rhs_data, rhs_shape_, dim_nums.rhs_contracting_dimensions(0) == col_dim);

  // Dispatches to a regular cublas gemm, a gemm-with-algorithm, or attempts to
  // autotune this gemm to figure out the best algorithm.
  auto launch = [&](MatrixDescriptor lhs_matrix, MatrixDescriptor rhs_matrix,
                    MatrixDescriptor output_matrix, se::Stream* stream) {
    PrimitiveType element_type = output_shape_.element_type();
    se::blas::ComputationType computation_type =
        GetBlasComputationType(element_type);

    // TODO(b/112111608): Implement auto tune for batched gemm.
    if (batch_size != 1) {
      return GetGemmFn(element_type)(lhs_matrix, rhs_matrix, output_matrix,
                                     alpha_, beta_, stream);
    }

    auto thunk_name = [&] {
      return hlo_instruction() != nullptr ? hlo_instruction()->ToString()
                                          : "<null>";
    };

    const string& device_name = stream->parent()->GetDeviceDescription().name();
    auto autotune_it = autotune_results_.find(device_name);
    if (autotune_it == autotune_results_.end()) {
      VLOG(3) << "Starting autotune of GemmThunk " << thunk_name();

      // If the output buffer already contains a bias then autotune into a
      // scratch buffer. This avoids overwriting the bias buffer. The scratch
      // buffer may contain arbitrary garbage values.
      se::DeviceMemoryBase scratch_data = output_data;
      std::unique_ptr<se::TemporaryDeviceMemory<char>> scratch_mem;
      if (beta_ != 0.0) {
        auto temp_status = stream->AllocateTemporaryArray<char>(
            ShapeUtil::ByteSizeOf(output_shape_));
        if (!temp_status.ok()) {
          return false;
        }
        scratch_mem = std::move(temp_status).ValueOrDie();
        scratch_data = scratch_mem->device_memory();
      }
      const MatrixDescriptor scratch_descriptor(
          scratch_data, false, output_num_cols, output_num_rows, batch_size);

      StatusOr<se::blas::AlgorithmType> best_algorithm = GetGemmAutotuneFn(
          element_type)(lhs_matrix, rhs_matrix, scratch_descriptor, alpha_,
                        beta_, computation_type, stream);
      autotune_it =
          autotune_results_.insert({device_name, best_algorithm}).first;

      if (autotune_it->second.ok()) {
        VLOG(2) << "Autotune on GemmThunk " << thunk_name()
                << " successful; best algorithm is "
                << best_algorithm.ValueOrDie();
      } else {
        VLOG(2) << "Autotune on GemmThunk " << thunk_name()
                << " unsuccessful.  Will use generic gemm.";
      }
    }

    const StatusOr<se::blas::AlgorithmType>& best_algorithm =
        autotune_it->second;
    if (best_algorithm.ok()) {
      auto algorithm = best_algorithm.ValueOrDie();
      VLOG(2) << "Using algorithm " << algorithm
              << " chosen by autotuning on GemmThunk " << thunk_name();
      return GetGemmWithAlgorithmFn(element_type)(
          lhs_matrix, rhs_matrix, output_matrix, alpha_, beta_,
          computation_type, algorithm, stream,
          /*output_profile_result=*/nullptr);
    }

    // Autotune will fail when CUDA 8 and GPU sm_50 or older are used.
    // Use the older Gemm API in this case.
    return GetGemmFn(element_type)(lhs_matrix, rhs_matrix, output_matrix,
                                   alpha_, beta_, stream);
  };

  auto op_profiler = profiler->MakeScopedInstructionProfiler(
      implements_whole_instruction_ ? hlo_instruction() : nullptr);
  bool launch_ok;
  if (LayoutUtil::Minor(output_shape_.layout(), row_dim) == 0) {
    launch_ok = launch(lhs_descriptor, rhs_descriptor,
                       MatrixDescriptor(output_data, false, output_num_rows,
                                        output_num_cols, batch_size),
                       stream);
  } else {
    launch_ok = launch(rhs_descriptor, lhs_descriptor,
                       MatrixDescriptor(output_data, false, output_num_cols,
                                        output_num_rows, batch_size),
                       stream);
  }

  if (!launch_ok) {
    return InternalError("Unable to launch cuBLAS gemm on stream %p", stream);
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
