/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MATMUL_UTILS_H_
#define XLA_SERVICE_GPU_MATMUL_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Normalize shape to (batch, rows, columns) logical dimensions.
absl::StatusOr<Shape> GetBatchRowColumnShape(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> row_dims, absl::Span<const int64_t> col_dims);

// GPU folding rule for the `TransposeFolding` pass.
absl::StatusOr<bool> CanFoldTransposeOperandIntoDot(const HloInstruction& dot,
                                                    int64_t operand_idx);

// Returns true if the sum of the sizes of the unbatched operand matrices
// for the dot is smaller than the given threshold.
absl::StatusOr<bool> IsMatrixMultiplicationTooSmallForRewriting(
    const HloInstruction& dot, int64_t threshold);

// Returns true if the backend can lower the dot. Currently the classical
// emitters cannot handle some dots, e.g., i8[] x i8[] -> i32[] dots,
// so we need to always use cuBLAS or Triton for those.
bool IsDotSupportedByClassicalEmitters(const HloInstruction& dot);

// extending plain MatrixLayout struct with creator functions
struct MatrixLayout : public se::gpu::MatrixLayout {
  // Returns the matrix layout for a logical shape (batch, rows, columns).
  static absl::StatusOr<MatrixLayout> For(const Shape& shape);
  // Returns the matrix layout with the given batch, row, col dimensions.
  static absl::StatusOr<MatrixLayout> For(const Shape& shape,
                                          absl::Span<const int64_t> batch_dims,
                                          absl::Span<const int64_t> row_dims,
                                          absl::Span<const int64_t> col_dims);
  // Returns the matrix layout for the output.
  static absl::StatusOr<MatrixLayout> For(const Shape& shape,
                                          size_t lhs_num_batch_dims,
                                          size_t lhs_num_row_dims,
                                          size_t rhs_num_batch_dims,
                                          size_t rhs_num_col_dims);
};

struct GemmConfig : public se::gpu::GemmConfig {
  // For legacy Gemm operations XLA:GPU allocates its own workspace and passes
  // it to all BLAS API calls.
  //
  // Size of the workspace based on NVIDIA recommendation:
  // https://docs.nvidia.com/cuda/cublas/#cublassetworkspace
  static constexpr int64_t kHopperWorkspace = 32 * 1024 * 1024;  // 32 MiB
  static constexpr int64_t kGFX950Workspace = 64 * 1024 * 1024;  // 64 MiB
  static constexpr int64_t kDefaultWorkspace = 4 * 1024 * 1024;  // 4 MiB
  // the number of algorithms to consider for autotuning by default
  static constexpr int64_t kNumAlgorithms = 128;

  static absl::StatusOr<GemmConfig> For(
      const HloInstruction* gemm, const se::GpuComputeCapability& gpu_version);

  // Gets the GemmConfig of the `gemm` instruction with overridden
  // GemmBackendConfig.
  static absl::StatusOr<GemmConfig> For(
      const HloInstruction* gemm, const GemmBackendConfig& config,
      const se::GpuComputeCapability& gpu_version);

  static absl::StatusOr<GemmConfig> For(
      const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
      absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
      absl::Span<const int64_t> rhs_batch_dims,
      absl::Span<const int64_t> rhs_contracting_dims, const Shape& output_shape,
      double alpha_real, double alpha_imag, double beta,
      PrecisionConfig::Algorithm precision_algorithm,
      std::optional<int64_t> algorithm, int64_t compute_precision, bool grad_x,
      bool grad_y, const se::GpuComputeCapability& gpu_version);

  // As above with additional `c_shape` and `bias_shape_ptr` parameter, both
  // which are only necessarily for F8 gemms.
  static absl::StatusOr<GemmConfig> For(
      const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
      absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
      absl::Span<const int64_t> rhs_batch_dims,
      absl::Span<const int64_t> rhs_contracting_dims, const Shape& c_shape,
      const Shape* bias_shape_ptr, const Shape& output_shape, double alpha_real,
      double alpha_imag, double beta,
      PrecisionConfig::Algorithm precision_algorithm,
      std::optional<int64_t> algorithm, int64_t compute_precision, bool grad_x,
      bool grad_y, const se::GpuComputeCapability& gpu_version);

  struct DescriptorsTuple {
    se::gpu::MatrixDescriptor lhs;
    se::gpu::MatrixDescriptor rhs;
    se::gpu::OutputMatrixDescriptor output;
    bool operands_swapped;
  };
  absl::StatusOr<DescriptorsTuple> GetMatrixDescriptors(
      se::DeviceMemoryBase lhs_buf, se::DeviceMemoryBase rhs_buf,
      se::DeviceMemoryBase out_buf) const;
};

// Run the given GEMM instruction `gemm` subject to the configuration
// in `gemm_config` and the passed buffers.
//
// If `algorithm` is provided, it overrides the one specified in `config`.
absl::Status RunGemm(
    const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
    se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase workspace_buffer, bool deterministic_ops,
    se::Stream* stream,
    std::optional<se::blas::AlgorithmType> algorithm = std::nullopt,
    se::blas::ProfileResult* profile_result = nullptr);

namespace gpublas_lt {

absl::StatusOr<bool> EpilogueAddsVectorBias(
    GemmBackendConfig_Epilogue epilogue);
absl::StatusOr<bool> EpilogueHasAuxiliaryOutput(
    GemmBackendConfig_Epilogue epilogue);

absl::StatusOr<se::gpu::BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue);

}  // namespace gpublas_lt

// We should use this in code instead of AutotuneResult::TritonGemmKey.
// This has some advantages, for example it can be used in hashmaps.
struct TritonGemmConfig {
  constexpr TritonGemmConfig() = default;
  constexpr TritonGemmConfig(int block_m, int block_n, int block_k, int split_k,
                             int num_stages, int num_warps, int num_ctas = 1)
      : block_m(block_m),
        block_n(block_n),
        block_k(block_k),
        split_k(split_k),
        num_stages(num_stages),
        num_warps(num_warps),
        num_ctas(num_ctas) {}
  int block_m = 0;
  int block_n = 0;
  int block_k = 0;
  int split_k = 0;
  int num_stages = 0;
  int num_warps = 0;
  // Number of blocks in a block cluster.
  int num_ctas = 0;

  // When adding new members, please update all methods, such as ToTuple,
  // FromProto, ToProto, ToString, etc. Updating ToTuple is not enough.
  // Please also add new members to AutotuneResult::TritonGemmKey in
  // autotuning.proto. Also kVersion has to be incremented in autotuner_util.cc
  // and all the autotuning results stored in tests, repos, etc. will have to
  // be updated.

 private:
  auto ToTuple() const {
    return std::make_tuple(block_m, block_n, block_k, split_k, num_stages,
                           num_warps, num_ctas);
  }

 public:
  // Creates a TritonGemmConfig from the supplied proto, doing a simple sanity
  // check.
  static absl::StatusOr<TritonGemmConfig> FromProto(
      const AutotuneResult::TritonGemmKey& proto);
  AutotuneResult::TritonGemmKey ToProto() const;

  std::string ToString() const;

  bool operator==(const TritonGemmConfig& other) const {
    return ToTuple() == other.ToTuple();
  }

  bool operator<(const TritonGemmConfig& other) const {
    return ToTuple() < other.ToTuple();
  }

  template <typename H>
  friend H AbslHashValue(H h, const TritonGemmConfig& config) {
    return H::combine(std::move(h), config.ToTuple());
  }
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MATMUL_UTILS_H_
