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
#include "tensorflow/core/framework/types.h"
#include "tensorflow/stream_executor/cuda/cuda_blas_lt.h"

namespace tensorflow {

// Reads the maximum number of algorithms for GEMM autotuning from the
// environment variable TF_MATMUL_AUTOTUNE_MAX_ALGORITHMS. If no value is set,
// return the default value.
int MatmulMaxAutotuneAlgorithmCount();

// Get a workspace limit from the environment variable, which is in MB.
// Return the workspace memory limit in bytes. If no value is set, return the
// default value.
int64_t GetWorkspaceLimit(int64_t default_value_in_bytes);

struct PlanAndAlgorithms {
  se::cuda::BlasLt::MatmulPlan plan;
  std::vector<se::cuda::BlasLt::MatmulAlgorithm> algorithms;
};

// Thread-safe map from matmul parameters to their corresponding plan and
// algorithms.
class BlasLtMatmulPlanMap {
 public:
  const PlanAndAlgorithms* Find(
      const se::cuda::BlasLt::MatmulPlanParams& params) const {
    absl::MutexLock lock(&mu_);
    auto iter = params_plan_map_.find(params);
    if (iter == params_plan_map_.end()) {
      return nullptr;
    }
    return &iter->second;
  }

  const PlanAndAlgorithms* Insert(
      const se::cuda::BlasLt::MatmulPlanParams& params,
      PlanAndAlgorithms value) {
    absl::MutexLock lock(&mu_);
    return &params_plan_map_.emplace(params, std::move(value)).first->second;
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<se::cuda::BlasLt::MatmulPlanParams, PlanAndAlgorithms>
      params_plan_map_ ABSL_GUARDED_BY(mu_);
};

StatusOr<se::blas::ComputationType> GetBlasComputationType(
    const DataType& dtype);

StatusOr<const PlanAndAlgorithms*> GetPlanAndAlgorithms(
    se::Stream* stream, const se::cuda::BlasLt::MatmulPlanParams& params,
    std::optional<int> max_algorithm_count = std::nullopt);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MATMUL_UTIL_H_
