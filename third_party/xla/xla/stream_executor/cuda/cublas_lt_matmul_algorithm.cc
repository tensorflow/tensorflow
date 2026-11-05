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

#include "xla/stream_executor/cuda/cublas_lt_matmul_algorithm.h"

#include <cstdint>
#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cublas_lt_matmul_algorithm.pb.h"

namespace stream_executor::cuda {

absl::StatusOr<CublasLtMatmulAlgorithm> CublasLtMatmulAlgorithm::FromProto(
    const CublasLtMatmulAlgorithmProto& proto) {
  if (proto.tile_id() < 0 || proto.reduction_scheme() < 0 ||
      proto.cta_swizzling() < 0 || proto.custom_option() < 0 ||
      proto.stages_id() < 0 || proto.inner_shape_id() < 0 ||
      proto.cluster_shape_id() < 0 || proto.numerical_impl_flags() < 0) {
    return absl::InvalidArgumentError(
        "cuBLASLt algorithm attributes must be non-negative.");
  }
  if (proto.inner_shape_id() > std::numeric_limits<uint16_t>::max() ||
      proto.cluster_shape_id() > std::numeric_limits<uint16_t>::max()) {
    return absl::InvalidArgumentError(
        "cuBLASLt inner_shape_id or cluster_shape_id exceeds uint16_t range.");
  }
  CublasLtMatmulAlgorithm result;
  result.algo_id = proto.algo_id();
  result.tile_id = static_cast<uint32_t>(proto.tile_id());
  result.split_k = proto.split_k();
  result.reduction_scheme = static_cast<uint32_t>(proto.reduction_scheme());
  result.cta_swizzling = static_cast<uint32_t>(proto.cta_swizzling());
  result.custom_option = static_cast<uint32_t>(proto.custom_option());
  result.stages_id = static_cast<uint32_t>(proto.stages_id());
  result.inner_shape_id = static_cast<uint16_t>(proto.inner_shape_id());
  result.cluster_shape_id = static_cast<uint16_t>(proto.cluster_shape_id());
  result.numerical_impl_flags =
      static_cast<uint64_t>(proto.numerical_impl_flags());
  return result;
}

CublasLtMatmulAlgorithmProto CublasLtMatmulAlgorithm::ToProto() const {
  CublasLtMatmulAlgorithmProto proto;
  proto.set_algo_id(algo_id);
  proto.set_tile_id(static_cast<int32_t>(tile_id));
  proto.set_split_k(split_k);
  proto.set_reduction_scheme(static_cast<int32_t>(reduction_scheme));
  proto.set_cta_swizzling(static_cast<int32_t>(cta_swizzling));
  proto.set_custom_option(static_cast<int32_t>(custom_option));
  proto.set_stages_id(static_cast<int32_t>(stages_id));
  proto.set_inner_shape_id(inner_shape_id);
  proto.set_cluster_shape_id(cluster_shape_id);
  proto.set_numerical_impl_flags(static_cast<int64_t>(numerical_impl_flags));
  return proto;
}

}  // namespace stream_executor::cuda
