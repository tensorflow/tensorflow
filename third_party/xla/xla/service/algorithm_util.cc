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

#include "xla/service/algorithm_util.h"

#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace algorithm_util {

namespace {
namespace se = stream_executor;
}  // namespace

absl::StatusOr<se::blas::ComputationType> GetBlasComputationType(
    PrecisionConfig::Algorithm algorithm) {
  // Note: If we will support other algorithm & storage type combinations, such
  // as ALG_DOT_BF16_BF16_F32 with F32 input and output storage types, then
  // we'll have to also depend on the storage types here. For the mentioned
  // example, the computation type would be kBF16AsF32.
  // Only the currently supported algorithms are listed here.
  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
      return se::blas::ComputationType::kF16;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      return se::blas::ComputationType::kBF16AsF32;
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:

    case PrecisionConfig::ALG_DOT_F32_F32_F32:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      return se::blas::ComputationType::kF32;
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
      return se::blas::ComputationType::kTF32AsF32;
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      return se::blas::ComputationType::kF64;
    default:
      return absl::InternalError(
          absl::StrFormat("GetBlasComputationType: unsupported algorithm %s",
                          xla::PrecisionConfig::Algorithm_Name(algorithm)));
  }
}

absl::StatusOr<PrimitiveType> GetDotAccumulatorType(
    PrecisionConfig::Algorithm algorithm) {
  // All dot algorithms should be listed here.
  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
      return F16;
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      return F32;
    case PrecisionConfig::ALG_DOT_BF16_BF16_BF16:
      return BF16;
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      return F64;
    case PrecisionConfig::ALG_UNSET:
    default:
      return absl::InternalError(
          absl::StrFormat("GetDotAccumulatorType: unsupported algorithm %s",
                          xla::PrecisionConfig::Algorithm_Name(algorithm)));
  }
}

bool HasTf32InputType(PrecisionConfig::Algorithm algorithm) {
  return algorithm == PrecisionConfig::ALG_DOT_TF32_TF32_F32 ||
         algorithm == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3;
}

bool HasFastAccum(PrecisionConfig::Algorithm algorithm) {
  return algorithm == PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM;
}

bool IsAmpere(stream_executor::GpuComputeCapability gpu_compute_capability) {
  return std::holds_alternative<se::CudaComputeCapability>(
             gpu_compute_capability) &&
         std::get<se::CudaComputeCapability>(gpu_compute_capability).major ==
             stream_executor::CudaComputeCapability::kAmpere;
}

// It's clear that those libraries could support more, but we only list the ones
// which we explicitly test for now.
bool IsSupportedByCublasOrCublasLt(
    PrecisionConfig::Algorithm algorithm,
    stream_executor::GpuComputeCapability gpu_compute_capability,
    const HloDotInstruction* dot, const int64_t rhs_contracting_index) {
  // 8-bit x 8-bit GEMMs with contracting dim < 4 are not supported by cuBLAS.
  // As this was determined through a failing test, I'm eering on the side of
  // caution and not generalizing this further.
  if (dot) {
    auto lhs_type = dot->operand(0)->shape().element_type();
    auto rhs_type = dot->operand(1)->shape().element_type();
    auto contracting_dim_size =
        dot->operand(1)->shape().dimensions(rhs_contracting_index);
    if (primitive_util::Is8BitIntegralType(lhs_type) &&
        primitive_util::Is8BitIntegralType(rhs_type) &&
        contracting_dim_size < 4) {
      return false;
    }
  }

  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
    case PrecisionConfig::ALG_UNSET:
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
      return true;
    default:
      return false;
  }
}

// Checks if we support the given algorithm using cuDNN.
bool IsSupportedByCudnn(PrecisionConfig::Algorithm algorithm) {
  switch (algorithm) {
    // When the CuDnn backend starts supporting specific algorithms, then
    // those should be listed here.
    case PrecisionConfig::ALG_UNSET:
      return true;
    default:
      return false;
  }
}

bool IsSupportedByElementalIrEmitter(PrecisionConfig::Algorithm algorithm) {
  switch (algorithm) {
    // Probably more can be added.
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
    case PrecisionConfig::ALG_UNSET:
      return true;
    default:
      return false;
  }
}

// Is the given algorithm supported on GPU with the given compute capability and
// input/output storage types.
bool IsSupportedDotAlgorithmOnGpu(
    PrecisionConfig::Algorithm algorithm,
    stream_executor::GpuComputeCapability gpu_compute_capability,
    PrimitiveType input_storage_type, PrimitiveType output_storage_type) {
  // Note: We may want to add some complex types here if people request that.
  const bool is_cuda_ge_ampere =
      std::holds_alternative<se::CudaComputeCapability>(
          gpu_compute_capability) &&
      std::get<se::CudaComputeCapability>(gpu_compute_capability)
          .IsAtLeastAmpere();

  const bool is_cuda_ge_ada =
      std::holds_alternative<se::CudaComputeCapability>(
          gpu_compute_capability) &&
      std::get<se::CudaComputeCapability>(gpu_compute_capability)
          .IsAtLeast(8, 9);

  const bool is_rocm_mi100_and_above =
      std::holds_alternative<se::RocmComputeCapability>(
          gpu_compute_capability) &&
      std::get<se::RocmComputeCapability>(gpu_compute_capability)
          .gfx9_mi100_or_later();

  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
      // Other F8 types are actually not supported by NVIDIA GPUs.
      return (is_cuda_ge_ada || is_rocm_mi100_and_above) &&
             (input_storage_type == F8E5M2 || input_storage_type == F8E4M3FN) &&
             (output_storage_type == F8E5M2 ||
              output_storage_type == F8E4M3FN || output_storage_type == F16 ||
              output_storage_type == BF16 || output_storage_type == F32);
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
      return input_storage_type == F16 &&
             (output_storage_type == F16 || output_storage_type == F32);
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      if (!is_cuda_ge_ampere && !is_rocm_mi100_and_above) return false;
      switch (input_storage_type) {
        case BF16:
          return output_storage_type == BF16 || output_storage_type == F32;
        case F32:
          return output_storage_type == F32;
        default:
          return false;
      }
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      return (is_cuda_ge_ampere || is_rocm_mi100_and_above) &&
             input_storage_type == F32 && output_storage_type == F32;
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
      return (is_cuda_ge_ampere || is_rocm_mi100_and_above) &&
             input_storage_type == F32 && output_storage_type == F32;
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      return input_storage_type == F32 && output_storage_type == F32;
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      return input_storage_type == F64 && output_storage_type == F64;
    default:
      return false;
  }
}

}  // namespace algorithm_util
}  // namespace xla
