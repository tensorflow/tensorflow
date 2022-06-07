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
#include <vector>

#include "absl/types/span.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/matmul_util.h"
#include "tensorflow/stream_executor/scratch_allocator.h"

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
  static StatusOr<GemmConfig> For(mlir::Operation* op, bool use_cublaslt);

  static StatusOr<GemmConfig> For(
      const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
      absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
      absl::Span<const int64_t> rhs_batch_dims,
      absl::Span<const int64_t> rhs_contracting_dims, const Shape& output_shape,
      double alpha_real, double alpha_imag, double beta,
      std::optional<int64_t> algorithm, bool use_cublaslt);

  MatrixLayout lhs_layout;
  MatrixLayout rhs_layout;
  MatrixLayout output_layout;
  complex128 alpha;
  double beta;
  std::optional<int64_t> algorithm;
  bool use_cublaslt;
};

se::blas::MatrixDescriptor GetMatrixDesc(const MatrixLayout& layout,
                                         se::DeviceMemoryBase data);

void MakeBlasGemmCompatible(int64_t& m, int64_t& n,
                            se::blas::MatrixDescriptor& lhs,
                            se::blas::MatrixDescriptor& rhs,
                            se::blas::MatrixDescriptor& output);

// Run the given GEMM instruction `gemm` subject to the configuration
// in `gemm_config` and the passed buffers.
//
// If `algorithm` is provided, it overrides the one specified in `config`.
Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
               se::DeviceMemoryBase rhs_buffer,
               se::DeviceMemoryBase output_buffer, se::Stream* stream,
               std::optional<se::blas::AlgorithmType> algorithm = std::nullopt,
               se::blas::ProfileResult* profile_result = nullptr);

Status RunBlasLtMatmul(
    const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
    se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase output_buffer,
    se::Stream* stream, se::ScratchAllocator& scratch_allocator,
    const se::blas::IBlasLtMatmulAlgorithm* algorithm = nullptr,
    se::blas::ProfileResult* profile_result = nullptr);

class BlasPlansAutotuneCache {
 public:
  BlasPlansAutotuneCache() = default;

  std::optional<se::blas::AlgorithmConfig> Find(
      const se::BatchMatmulParameters& params) const;
  void Insert(se::BatchMatmulParameters params,
              se::blas::AlgorithmConfig config);

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<se::BatchMatmulParameters, se::blas::AlgorithmConfig>
      blas_plans_algorithms_map_ ABSL_GUARDED_BY(mu_);
};

BlasPlansAutotuneCache& GetBlasPlansAutotuneCache();

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MATMUL_UTILS_H_
