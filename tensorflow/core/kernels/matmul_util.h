/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#ifndef TENSORFLOW_CORE_KERNELS_MATMUL_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_MATMUL_UTIL_H_

#include <optional>
#include <vector>

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

#if GOOGLE_CUDA || TF_HIPBLASLT

#include "absl/container/flat_hash_map.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "tensorflow/core/framework/types.h"
#include "tsl/platform/types.h"

namespace tensorflow {

// Get a workspace limit from the environment variable, which is in MB.
// Return the workspace memory limit in bytes. If no value is set, return the
// default value.
int64_t GetWorkspaceLimit(int64_t default_value_in_bytes);

struct BlasLtMatmulPlanParams {
  std::string ToString() const;
  bool operator==(const BlasLtMatmulPlanParams& other) const;

  se::blas::DataType dtype;
  size_t m;
  size_t n;
  size_t k;
  se::blas::Transpose trans_a;
  se::blas::Transpose trans_b;
  size_t batch_count = 1;
  bool broadcast_a = false;
  bool broadcast_b = false;
  se::gpu::BlasLt::Epilogue epilogue = se::gpu::BlasLt::Epilogue::kDefault;
};

struct PlanAndAlgorithms {
  static StatusOr<const PlanAndAlgorithms*> GetOrCreate(
      se::Stream* stream, const BlasLtMatmulPlanParams& params,
      absl::Mutex** pmu, std::optional<int> max_algorithm_count = std::nullopt);

  Status ExecuteOnStream(
      se::Stream* stream, const se::DeviceMemoryBase& a,
      const se::DeviceMemoryBase& b, se::DeviceMemoryBase& c,
      size_t algorithm_idx, se::ScratchAllocator& scratch_allocator,
      const se::DeviceMemoryBase& bias = se::DeviceMemoryBase{},
      se::blas::ProfileResult* profile_result = nullptr) const;

  se::gpu::BlasLt::MatmulPlanPtr plan;
  std::vector<se::gpu::BlasLt::MatmulAlgorithm> algorithms;
};

namespace internal {

inline auto AsTuple(const BlasLtMatmulPlanParams& p) {
  return std::make_tuple(p.dtype, p.m, p.n, p.k, p.trans_a, p.trans_b,
                         p.batch_count, p.broadcast_a, p.broadcast_b,
                         p.epilogue);
}

}  // namespace internal

template <typename H>
H AbslHashValue(H h, const BlasLtMatmulPlanParams& params) {
  return H::combine(std::move(h), internal::AsTuple(params));
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TF_HIPBLASLT

#endif  // TENSORFLOW_CORE_KERNELS_MATMUL_UTIL_H_
