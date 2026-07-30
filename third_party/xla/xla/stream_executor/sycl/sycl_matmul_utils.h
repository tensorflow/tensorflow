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
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/sycl/onednn_util.h"

namespace stream_executor {
namespace sycl {

namespace se = ::stream_executor;

// This struct contains the metadata of a matrix, e.g., its base address and
// dimensions.
struct MatrixDescriptor {
  se::DeviceMemoryBase data;
  se::blas::Transpose transpose;
  int64_t num_rows;
  int64_t num_cols;
  int64_t batch_stride;
  int64_t leading_dim_stride;

  int64_t reduced_dim() const {
    return transpose == se::blas::Transpose::kTranspose ? num_rows : num_cols;
  }

  template <typename T>
  se::DeviceMemory<T> cast() const {
    return se::DeviceMemory<T>(data);
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
                     se::DeviceMemoryBase lhs_buffer,
                     se::DeviceMemoryBase rhs_buffer,
                     se::DeviceMemoryBase add_buffer,
                     se::DeviceMemoryBase output_buffer,
                     se::DeviceMemoryBase bias_buffer,
                     se::DeviceMemoryBase workspace, se::Stream* stream,
                     sycl_gemm::GemmBackendEpilogue epilogue, int64_t algorithm,
                     se::ScratchAllocator* scratch_allocator = nullptr);

// Creates an OneDNN matmul primitive descriptor from a GemmConfig
// Uses a default GPU engine for OneDNN operations
absl::StatusOr<std::unique_ptr<dnnl::matmul::primitive_desc>>
CreateMatMulPrimDescFromGemmConfig(
    const xla::gpu::GemmConfig& config,
    xla::gpu::GemmBackendConfig_Epilogue epilogue);
absl::StatusOr<dnnl::memory::desc> TransposeLastTwoDims(
    const dnnl::memory::desc& md);
inline absl::Status TransposeLastTwoDimsIf(bool pred,
                                           dnnl::memory::desc& mem_desc) {
  if (pred) {
    ASSIGN_OR_RETURN(mem_desc, TransposeLastTwoDims(mem_desc));
  }
  return absl::OkStatus();
}

}  // namespace sycl
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_MATMUL_UTILS_H_
