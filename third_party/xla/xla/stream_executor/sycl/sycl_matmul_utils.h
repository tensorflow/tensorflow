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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_MATMUL_UTILS_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_MATMUL_UTILS_H_

#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/sycl/onednn_util.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace sycl {

namespace se = ::stream_executor;

constexpr int64_t kOneDnnGemm = 1;

// Describes a batched matrix in BLAS terms. The rows and columns are the
// stored row-major shape, and the transpose flag says whether the matmul uses
// the matrix transposed. The element type is kept with the descriptor so it
// stays paired with the matrix when operands are swapped.
struct MatrixDescriptor {
  se::DeviceAddressBase data;
  xla::PrimitiveType dtype;
  se::blas::Transpose transpose;
  int64_t num_rows;
  int64_t num_cols;
  int64_t batch_size;
  int64_t batch_stride;
  int64_t leading_dim_stride;

  int64_t reduced_dim() const {
    return transpose == se::blas::Transpose::kTranspose ? num_rows : num_cols;
  }

  template <typename T>
  se::DeviceAddress<T> cast() const {
    return se::DeviceAddress<T>(data);
  }
};

namespace sycl_gemm {

enum class GemmBackendEpilogue {
  DEFAULT,
  RELU,
  GELU,
  BIAS,
  BIAS_RELU,
  BIAS_GELU,
};

absl::StatusOr<GemmBackendEpilogue> EpilogueCast(absl::string_view epilogue);

absl::StatusOr<std::string> EpilogueCast(GemmBackendEpilogue epilogue);

absl::StatusOr<bool> EpilogueAddsVectorBias(GemmBackendEpilogue epilogue);

absl::StatusOr<bool> EpilogueHasAuxiliaryOutput(GemmBackendEpilogue epilogue);

absl::StatusOr<GemmBackendEpilogue> AsSYCLEpilogue(
    xla::gpu::GemmBackendConfig_Epilogue epilogue);
absl::StatusOr<GemmBackendEpilogue> AsSYCLEpilogue(
    se::gpu::BlasLt::Epilogue epilogue);
}  // namespace sycl_gemm

absl::Status RunGemm(const gpu::GemmConfig& config,
                     se::DeviceAddressBase lhs_buffer,
                     se::DeviceAddressBase rhs_buffer,
                     se::DeviceAddressBase add_buffer,
                     se::DeviceAddressBase output_buffer,
                     se::DeviceAddressBase bias_buffer,
                     se::DeviceAddressBase workspace, se::Stream* stream,
                     sycl_gemm::GemmBackendEpilogue epilogue, int64_t algorithm,
                     se::ScratchAllocator* scratch_allocator = nullptr);

// Creates a oneDNN matmul primitive descriptor from a gemm config on the
// default GPU engine. Used at compile time to size the scratchpad. Callers
// must convert the epilogue to its SYCL form first
absl::StatusOr<std::unique_ptr<dnnl::matmul::primitive_desc>>
CreateMatMulPrimDescFromGemmConfig(const xla::gpu::GemmConfig& config,
                                   sycl_gemm::GemmBackendEpilogue epilogue);
absl::StatusOr<dnnl::memory::desc> TransposeLastTwoDims(
    const dnnl::memory::desc& md);
inline absl::Status TransposeLastTwoDimsIf(bool pred,
                                           dnnl::memory::desc& mem_desc) {
  if (pred) {
    ABSL_ASSIGN_OR_RETURN(mem_desc, TransposeLastTwoDims(mem_desc));
  }
  return absl::OkStatus();
}

}  // namespace sycl
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_MATMUL_UTILS_H_
