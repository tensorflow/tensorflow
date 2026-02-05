/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/autotuner/rocblas.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
RocblasBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }

  if (ShouldUseHipblasLt(instr)) {
    std::vector<std::unique_ptr<BackendConfig>> configs;
    AutotuneResult::GemmKey gemm_key;
    gemm_key.set_algorithm(0);
    configs.push_back(std::make_unique<google::protobuf::Any>());
    configs.back()->PackFrom(gemm_key);
    return configs;
  }

  std::unique_ptr<se::DeviceAddressAllocator> allocator =
      std::make_unique<se::StreamExecutorMemoryAllocator>(stream_executor());
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      allocator->GetStream(stream_executor()->device_ordinal()));

  // We use GemmConfig::For with GemmBackendConfig as a fallback because
  // Matmul_utils.cc relies on backend config to determine gemm contracting
  // dimensions.
  GemmBackendConfig backend_config;
  backend_config =
      instr.backend_config<GpuBackendConfig>()->gemm_backend_config();
  TF_ASSIGN_OR_RETURN(
      GemmConfig gemm_config,
      GemmConfig::For(
          &instr, backend_config,
          target_config().device_description.gpu_compute_capability()));

  auto create_matrix_desc = [](const se::gpu::MatrixLayout& layout)
      -> absl::StatusOr<se::gpu::MatrixDescriptor> {
    TF_ASSIGN_OR_RETURN(se::blas::DataType type,
                        se::gpu::AsBlasDataType(layout.dtype));
    return se::gpu::MatrixDescriptor{
        /*data=*/se::DeviceAddressBase(), layout.leading_dim_stride,
        layout.batch_stride, type,
        // BLAS is column-major by default.
        (layout.order == se::gpu::MatrixLayout::Order::kColumnMajor
             ? se::blas::Transpose::kNoTranspose
             : se::blas::Transpose::kTranspose)};
  };

  TF_ASSIGN_OR_RETURN(se::gpu::MatrixDescriptor lhs_desc,
                      create_matrix_desc(gemm_config.lhs_layout));
  TF_ASSIGN_OR_RETURN(se::gpu::MatrixDescriptor rhs_desc,
                      create_matrix_desc(gemm_config.rhs_layout));
  TF_ASSIGN_OR_RETURN(se::gpu::MatrixDescriptor output_desc_base,
                      create_matrix_desc(gemm_config.output_layout));

  se::gpu::OutputMatrixDescriptor out_desc(std::move(output_desc_base));
  out_desc.batch_size = gemm_config.output_layout.batch_size;
  out_desc.m = gemm_config.output_layout.num_rows;
  out_desc.n = gemm_config.output_layout.num_cols;
  out_desc.k = gemm_config.lhs_layout.num_cols;
  TF_ASSIGN_OR_RETURN(
      out_desc.compute_type,
      se::gpu::GetBlasComputationType(
          gemm_config.precision_algorithm, gemm_config.lhs_layout.dtype,
          gemm_config.output_layout.dtype, gemm_config.compute_precision,
          target_config().device_description.gpu_compute_capability()));

  se::blas::BlasSupport* blas = stream_executor()->AsBlas();
  if (blas == nullptr) {
    return absl::InternalError("Failed to get BLAS support.");
  }
  std::vector<se::blas::AlgorithmType> algorithms;

  blas->GetBlasGemmAlgorithms(stream, lhs_desc, rhs_desc, &out_desc,
                              &gemm_config.alpha, &gemm_config.beta,
                              &algorithms);

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(algorithms.size());
  for (se::blas::AlgorithmType algorithm : algorithms) {
    AutotuneResult::GemmKey gemm_key;
    gemm_key.set_algorithm(algorithm);
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(gemm_key);
    configs.push_back(std::move(any));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> RocblasBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "RocblasBackend does not support this instruction.");
  }
  AutotuneResult::GemmKey gemm_key;
  gemm_key.set_algorithm(se::blas::kDefaultAlgorithm);
  auto any = std::make_unique<google::protobuf::Any>();
  if (ShouldUseHipblasLt(instr)) {
    gemm_key.set_algorithm(0);
  }
  any->PackFrom(gemm_key);
  return any;
}

absl::Status RocblasBackend::ApplyConfig(HloInstruction& instr,
                                         const BackendConfig& config) {
  AutotuneResult::GemmKey gemm_key;
  if (!config.UnpackTo(&gemm_key)) {
    return absl::InvalidArgumentError(
        "Failed to unpack RocblasBackendConfig from Any.");
  }
  if (ShouldUseHipblasLt(instr) && gemm_key.algorithm() == -1) {
    gemm_key.set_algorithm(0);
  }
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  GemmBackendConfig& backend_config = *gpu_config.mutable_gemm_backend_config();
  backend_config.set_selected_algorithm(gemm_key.algorithm());
  backend_config.set_autotune_workspace_size(
      gemm_key.autotune_workspace_size());
  TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  return absl::OkStatus();
}

bool RocblasBackend::IsSupported(const HloInstruction& instr) {
  return IsLegacyCublasMatmul(instr) || ShouldUseHipblasLt(instr);
}

bool RocblasBackend::ShouldUseHipblasLt(const HloInstruction& instr) {
  return fp8_lt_fallback_ && IsCublasLtMatmulF8(instr);
}

}  // namespace gpu
}  // namespace xla
