/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MATMUL_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MATMUL_UTILS_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/stream_executor/blas.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_blas_lt.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {

StatusOr<std::vector<int64_t>> GetNonContractingDims(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contracting_dims);

// Normalize shape to (batch, rows, columns) logical dimensions.
StatusOr<Shape> GetBatchRowColumnShape(const Shape& shape,
                                       absl::Span<const int64_t> batch_dims,
                                       absl::Span<const int64_t> row_dims,
                                       absl::Span<const int64_t> col_dims);

struct MatrixLayout {
  enum class Order {
    kRowMajor,     // Elements in the same row are contiguous in memory.
    kColumnMajor,  // Elements in the same column are contiguous in memory.
  };

  // Returns the matrix layout for a logical shape (batch, rows, columns).
  static StatusOr<MatrixLayout> For(const Shape& shape);
  // Returns the matrix layout with the given batch, row, col dimensions.
  static StatusOr<MatrixLayout> For(const Shape& shape,
                                    absl::Span<const int64_t> batch_dims,
                                    absl::Span<const int64_t> row_dims,
                                    absl::Span<const int64_t> col_dims);
  // Returns the matrix layout for the output.
  static StatusOr<MatrixLayout> For(const Shape& shape,
                                    size_t lhs_num_batch_dims,
                                    size_t lhs_num_row_dims,
                                    size_t rhs_num_batch_dims,
                                    size_t rhs_num_col_dims);

  void Transpose();

  PrimitiveType dtype;
  // `num_rows` / `num_cols` are for the "logical" matrix shape:
  // i.e. the contracting dim has size `num_cols` for LHS operands and
  // `num_rows` for RHS operands.
  int64_t num_rows;
  int64_t num_cols;
  Order order;
  int64_t leading_dim_stride;
  int64_t batch_size;
  int64_t batch_stride;  // `batch_stride` is set to `0` when `batch_size == 1`.
};

// GPU folding rule for the `TransposeFolding` pass.
StatusOr<bool> CanFoldTransposeOperandIntoDot(const HloInstruction& dot,
                                              int64_t operand_idx);

struct GemmConfig {
  static StatusOr<GemmConfig> For(const HloInstruction* gemm);
  static StatusOr<GemmConfig> For(mlir::lmhlo_gpu::GEMMOp op);

  static StatusOr<GemmConfig> For(
      const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
      absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
      absl::Span<const int64_t> rhs_batch_dims,
      absl::Span<const int64_t> rhs_contracting_dims, const Shape& output_shape,
      double alpha_real, double alpha_imag, double beta,
      std::optional<int64_t> algorithm, int64_t compute_precision);

  MatrixLayout lhs_layout;
  MatrixLayout rhs_layout;
  MatrixLayout output_layout;
  complex128 alpha;
  double beta;
  std::optional<int64_t> algorithm;
  int64_t compute_precision;
};

// Run the given GEMM instruction `gemm` subject to the configuration
// in `gemm_config` and the passed buffers.
//
// If `algorithm` is provided, it overrides the one specified in `config`.
Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
               se::DeviceMemoryBase rhs_buffer,
               se::DeviceMemoryBase output_buffer, se::Stream* stream,
               std::optional<se::blas::AlgorithmType> algorithm = std::nullopt,
               se::blas::ProfileResult* profile_result = nullptr);

#if GOOGLE_CUDA

namespace cublas_lt {

class MatmulPlan {
 public:
  static StatusOr<MatmulPlan> For(mlir::lmhlo_gpu::CublasLtMatmulOp op);
  static StatusOr<MatmulPlan> From(const GemmConfig& config,
                                   se::cuda::BlasLt::Epilogue epilogue);

  Status ExecuteOnStream(se::Stream* stream, se::DeviceMemoryBase a_buffer,
                         se::DeviceMemoryBase b_buffer,
                         se::DeviceMemoryBase c_buffer,
                         se::DeviceMemoryBase d_buffer,
                         se::DeviceMemoryBase bias_buffer,  // may be null
                         const se::cuda::BlasLt::MatmulAlgorithm& algorithm,
                         se::ScratchAllocator& scratch_allocator,
                         se::blas::ProfileResult* profile_result = nullptr);

  StatusOr<std::vector<se::cuda::BlasLt::MatmulAlgorithm>> GetAlgorithms(
      se::Stream* stream) const;

 private:
  MatmulPlan(se::cuda::BlasLt::MatmulPlan plan, complex128 alpha, double beta,
             bool must_swap_operands)
      : plan_(std::move(plan)),
        alpha_(alpha),
        beta_(beta),
        must_swap_operands_(must_swap_operands) {}

  template <typename Input, typename Scale = Input>
  Status DoMatmul(se::Stream* stream, se::DeviceMemoryBase a_buffer,
                  se::DeviceMemoryBase b_buffer, se::DeviceMemoryBase c_buffer,
                  se::DeviceMemoryBase d_buffer,
                  se::DeviceMemoryBase bias_buffer,  // may be null
                  const se::cuda::BlasLt::MatmulAlgorithm& algorithm,
                  se::ScratchAllocator& scratch_allocator,
                  se::blas::ProfileResult* profile_result);

  se::cuda::BlasLt::MatmulPlan plan_;
  complex128 alpha_;
  double beta_;
  bool must_swap_operands_;
};

}  // namespace cublas_lt

#endif  // GOOGLE_CUDA

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MATMUL_UTILS_H_
