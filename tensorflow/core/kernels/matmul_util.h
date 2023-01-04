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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_blas_lt.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/core/framework/types.h"

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
  se::cuda::BlasLt::Epilogue epilogue = se::cuda::BlasLt::Epilogue::kDefault;
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

struct PlanAndAlgorithms {
  se::cuda::BlasLt::MatmulPlan plan;
  std::vector<se::cuda::BlasLt::MatmulAlgorithm> algorithms;
};

// Thread-safe map from matmul parameters to their corresponding plan and
// algorithms.
class BlasLtMatmulPlanMap {
 public:
  const PlanAndAlgorithms* Find(const BlasLtMatmulPlanParams& params) const;
  const PlanAndAlgorithms* Insert(const BlasLtMatmulPlanParams& params,
                                  PlanAndAlgorithms value);

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<BlasLtMatmulPlanParams, PlanAndAlgorithms>
      params_plan_map_ ABSL_GUARDED_BY(mu_);
};

StatusOr<se::blas::ComputationType> GetBlasComputationType(
    const DataType& dtype);

StatusOr<const PlanAndAlgorithms*> GetPlanAndAlgorithms(
    se::Stream* stream, const BlasLtMatmulPlanParams& params,
    std::optional<int> max_algorithm_count = std::nullopt);

template <typename T>
Status DoBlasLtMatmul(se::Stream* stream,
                      const se::cuda::BlasLt::MatmulPlan& plan,
                      const se::DeviceMemory<T>& a,
                      const se::DeviceMemory<T>& b, se::DeviceMemory<T>& c,
                      const se::cuda::BlasLt::MatmulAlgorithm& algorithm,
                      se::ScratchAllocator& scratch_allocator,
                      const se::DeviceMemory<T>& bias = {},
                      se::blas::ProfileResult* profile_result = nullptr) {
  se::cuda::BlasLt* blas_lt = se::cuda::GetBlasLt(stream);
  // TF_RET_CHECK(blas_lt != nullptr);

  se::DeviceMemory<T> aux{};  // We don't use the auxilary buffers.

  // The scale type may be f32 if the data type is f16 and bf16.
  if constexpr (std::is_same_v<T, Eigen::half> ||
                std::is_same_v<T, Eigen::bfloat16>) {
    if (plan.op_desc.scale_type() == CUDA_R_32F) {
      return blas_lt->DoMatmul(stream, plan, se::HostOrDeviceScalar<float>(1.0),
                               b, a, se::HostOrDeviceScalar<float>(0.0), c, c,
                               algorithm, scratch_allocator, bias, aux,
                               profile_result);
    }
  }
  return blas_lt->DoMatmul(stream, plan, se::HostOrDeviceScalar<T>(T(1.0)), b,
                           a, se::HostOrDeviceScalar<T>(T(0.0)), c, c,
                           algorithm, scratch_allocator, bias, aux,
                           profile_result);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MATMUL_UTIL_H_
