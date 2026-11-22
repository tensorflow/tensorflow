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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUBLAS_LT_MATMUL_ALGORITHM_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUBLAS_LT_MATMUL_ALGORITHM_H_

#include <cstdint>
#include <tuple>

#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cublas_lt_matmul_algorithm.pb.h"

namespace stream_executor::cuda {

// CUDA-specific serializable description of a `BlasLtMatmulAlgorithm`, storing
// the algorithm configuration attributes corresponding to a
// `cublasLtMatmulAlgo_t` C struct.
struct CublasLtMatmulAlgorithm {
  int32_t algo_id = 0;
  uint32_t tile_id = 0;
  int32_t split_k = 0;
  uint32_t reduction_scheme = 0;
  uint32_t cta_swizzling = 0;
  uint32_t custom_option = 0;
  uint32_t stages_id = 0;
  uint16_t inner_shape_id = 0;
  uint16_t cluster_shape_id = 0;
  uint64_t numerical_impl_flags = 0;

  bool operator==(const CublasLtMatmulAlgorithm& other) const {
    return std::tie(algo_id, tile_id, split_k, reduction_scheme, cta_swizzling,
                    custom_option, stages_id, inner_shape_id, cluster_shape_id,
                    numerical_impl_flags) ==
           std::tie(other.algo_id, other.tile_id, other.split_k,
                    other.reduction_scheme, other.cta_swizzling,
                    other.custom_option, other.stages_id, other.inner_shape_id,
                    other.cluster_shape_id, other.numerical_impl_flags);
  }
  bool operator!=(const CublasLtMatmulAlgorithm& other) const {
    return !(*this == other);
  }

  static absl::StatusOr<CublasLtMatmulAlgorithm> FromProto(
      const CublasLtMatmulAlgorithmProto& proto);
  CublasLtMatmulAlgorithmProto ToProto() const;
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUBLAS_LT_MATMUL_ALGORITHM_H_
