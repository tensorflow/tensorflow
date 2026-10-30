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

#include "xla/stream_executor/gpu/blas_lt_matmul_algorithm.h"

#include <variant>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cublas_lt_matmul_algorithm.h"
#include "xla/stream_executor/gpu/blas_lt_matmul_algorithm.pb.h"

namespace stream_executor::gpu {

absl::StatusOr<BlasLtMatmulAlgorithm> BlasLtMatmulAlgorithm::FromProto(
    const BlasLtMatmulAlgorithmProto& proto) {
  switch (proto.algorithm_case()) {
    case BlasLtMatmulAlgorithmProto::kCublaslt: {
      ABSL_ASSIGN_OR_RETURN(
          cuda::CublasLtMatmulAlgorithm cublaslt,
          cuda::CublasLtMatmulAlgorithm::FromProto(proto.cublaslt()));
      return BlasLtMatmulAlgorithm{
          /*library_version=*/proto.library_version(),
          /*algorithm=*/cublaslt,
      };
    }
    case BlasLtMatmulAlgorithmProto::ALGORITHM_NOT_SET:
      return absl::InvalidArgumentError(
          "BlasLtMatmulAlgorithmProto does not have an algorithm set.");
  }
  return absl::InvalidArgumentError(
      "Unrecognized BlasLtMatmulAlgorithmProto algorithm case.");
}

BlasLtMatmulAlgorithmProto BlasLtMatmulAlgorithm::ToProto() const {
  BlasLtMatmulAlgorithmProto proto;
  proto.set_library_version(library_version);
  if (const auto* cublaslt =
          std::get_if<cuda::CublasLtMatmulAlgorithm>(&algorithm)) {
    *proto.mutable_cublaslt() = cublaslt->ToProto();
  }
  return proto;
}

}  // namespace stream_executor::gpu
