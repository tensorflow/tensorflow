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
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_blas_lt.h"
#include "tensorflow/compiler/xla/stream_executor/scratch_allocator.h"
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

  // As above with additional `c_shape` parameter.
  static StatusOr<GemmConfig> For(
      const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
      absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
      absl::Span<const int64_t> rhs_batch_dims,
      absl::Span<const int64_t> rhs_contracting_dims, const Shape& c_shape,
      const Shape& output_shape, double alpha_real, double alpha_imag,
      double beta, std::optional<int64_t> algorithm, int64_t compute_precision);

  MatrixLayout lhs_layout;
  MatrixLayout rhs_layout;
  MatrixLayout c_layout;
  MatrixLayout output_layout;
  complex128 alpha;
  double beta;
  std::optional<int64_t> algorithm;
  int64_t compute_precision;
};

StatusOr<se::blas::ComputationType> GetBlasComputationType(
    PrimitiveType lhs_dtype, PrimitiveType dtype, int64_t compute_precision);

namespace cublas_lt {

// Returns the type for the alpha and beta scalars.
se::blas::DataType GetScaleType(se::blas::DataType c_type,
                                se::blas::ComputationType computation_type);

}  // namespace cublas_lt

// Run the given GEMM instruction `gemm` subject to the configuration
// in `gemm_config` and the passed buffers.
//
// If `algorithm` is provided, it overrides the one specified in `config`.
Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
               se::DeviceMemoryBase rhs_buffer,
               se::DeviceMemoryBase output_buffer, se::Stream* stream,
               std::optional<se::blas::AlgorithmType> algorithm = std::nullopt,
               se::blas::ProfileResult* profile_result = nullptr);

namespace cublas_lt {

StatusOr<bool> EpilogueAddsVectorBias(GemmBackendConfig_Epilogue epilogue);
StatusOr<bool> EpilogueHasAuxiliaryOutput(GemmBackendConfig_Epilogue epilogue);

}  // namespace cublas_lt

StatusOr<se::blas::DataType> AsBlasDataType(PrimitiveType dtype);

#if GOOGLE_CUDA

namespace cublas_lt {

StatusOr<se::cuda::BlasLt::Epilogue> AsBlasLtEpilogue(
    mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue);

class MatmulPlan {
 public:
  template <typename CublasLtMatmulMaybeF8Op,
            typename = std::enable_if<
                std::is_same<CublasLtMatmulMaybeF8Op,
                             mlir::lmhlo_gpu::CublasLtMatmulOp>::value ||
                std::is_same<CublasLtMatmulMaybeF8Op,
                             mlir::lmhlo_gpu::CublasLtMatmulF8Op>::value>>
  static StatusOr<MatmulPlan> For(CublasLtMatmulMaybeF8Op op) {
    mlir::mhlo::DotDimensionNumbersAttr dot_dims = op.getDotDimensionNumbers();

    int64_t compute_precision = 0;  // Default
    if (op.getPrecisionConfig().has_value()) {
      auto precision_config = op.getPrecisionConfig();
      for (auto attr : precision_config.value()) {
        int64_t value = static_cast<int64_t>(
            attr.template cast<mlir::mhlo::PrecisionAttr>().getValue());
        if (value > compute_precision) {
          compute_precision = value;
        }
      }
    }

    TF_ASSIGN_OR_RETURN(
        GemmConfig config,
        GemmConfig::For(
            GetShape(op.getA()), dot_dims.getLhsBatchingDimensions(),
            dot_dims.getLhsContractingDimensions(), GetShape(op.getB()),
            dot_dims.getRhsBatchingDimensions(),
            dot_dims.getRhsContractingDimensions(), GetShape(op.getC()),
            GetShape(op.getD()), op.getAlphaReal().convertToDouble(),
            op.getAlphaImag().convertToDouble(), op.getBeta().convertToDouble(),
            op.getAlgorithm(), compute_precision));

    TF_ASSIGN_OR_RETURN(se::cuda::BlasLt::Epilogue epilogue,
                        AsBlasLtEpilogue(op.getEpilogue()));
    return From(config, epilogue);
  }

  static StatusOr<MatmulPlan> From(const GemmConfig& config,
                                   se::cuda::BlasLt::Epilogue epilogue);

  Status ExecuteOnStream(
      se::Stream* stream, se::DeviceMemoryBase a_buffer,
      se::DeviceMemoryBase b_buffer, se::DeviceMemoryBase c_buffer,
      se::DeviceMemoryBase d_buffer,
      se::DeviceMemoryBase bias_buffer,  // may be null
      se::DeviceMemoryBase aux_buffer,   // may be null
      se::DeviceMemoryBase a_scale_buffer, se::DeviceMemoryBase b_scale_buffer,
      se::DeviceMemoryBase c_scale_buffer, se::DeviceMemoryBase d_scale_buffer,
      se::DeviceMemoryBase d_amax_buffer,
      const se::cuda::BlasLt::MatmulAlgorithm& algorithm,
      se::ScratchAllocator& scratch_allocator,
      se::blas::ProfileResult* profile_result = nullptr) const;

  StatusOr<std::vector<se::cuda::BlasLt::MatmulAlgorithm>> GetAlgorithms(
      se::Stream* stream) const;

 private:
  MatmulPlan(se::cuda::BlasLt::MatmulPlan plan, complex128 alpha, double beta,
             bool must_swap_operands)
      : plan_(std::move(plan)),
        alpha_(alpha),
        beta_(beta),
        must_swap_operands_(must_swap_operands) {}

  template <typename Scale, typename A, typename B = A, typename C = A,
            typename D = A>
  Status DoMatmul(se::Stream* stream, se::DeviceMemoryBase a_buffer,
                  se::DeviceMemoryBase b_buffer, se::DeviceMemoryBase c_buffer,
                  se::DeviceMemoryBase d_buffer,
                  se::DeviceMemoryBase bias_buffer,  // may be null
                  se::DeviceMemoryBase aux_buffer,   // may be null
                  se::DeviceMemoryBase a_scale, se::DeviceMemoryBase b_scale,
                  se::DeviceMemoryBase c_scale, se::DeviceMemoryBase d_scale,
                  se::DeviceMemoryBase d_amax,
                  const se::cuda::BlasLt::MatmulAlgorithm& algorithm,
                  se::ScratchAllocator& scratch_allocator,
                  se::blas::ProfileResult* profile_result) const;

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
