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

#include "tensorflow/core/kernels/matmul_util.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/matmul_autotune.h"
#include "tensorflow/stream_executor/cuda/cuda_blas_lt.h"

namespace tensorflow {

int64_t GetWorkspaceLimit(int64_t default_value_in_bytes) {
  const char* workspace_limit_in_mb_str =
      getenv("TF_CUBLAS_WORKSPACE_LIMIT_IN_MB");
  if (workspace_limit_in_mb_str != nullptr &&
      strcmp(workspace_limit_in_mb_str, "") != 0) {
    int64_t scratch_limit_in_mb = -1;
    if (strings::safe_strto64(workspace_limit_in_mb_str,
                              &scratch_limit_in_mb)) {
      return scratch_limit_in_mb * (1 << 20);
    } else {
      LOG(WARNING) << "Invalid value for TF_CUBLAS_WORKSPACE_LIMIT_IN_MB: "
                   << workspace_limit_in_mb_str;
    }
  }
  return default_value_in_bytes;
}

int MatmulMaxAutotuneAlgorithmCount() {
  int64_t value;
  Status status =
      ReadInt64FromEnvVar("TF_MATMUL_AUTOTUNE_MAX_ALGORITHMS", 10, &value);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
  static constexpr const int kMaxValue = std::numeric_limits<int>::max();
  if (value < 1 || value > kMaxValue) {
    LOG(ERROR) << "Invalid value for TF_MATMUL_AUTOTUNE_MAX_ALGORITHMS: "
               << value << " is not in range [1, " << kMaxValue << "]";
  }
  return value;
}

StatusOr<se::blas::ComputationType> GetBlasComputationType(
    const DataType& dtype) {
  using se::blas::ComputationType;
  static bool use_f32_for_f16_computation = MatmulDoFP32ComputationFP16Input();
  bool allow_tf32 = tensor_float_32_execution_enabled();
  ComputationType f32_type =
      allow_tf32 ? ComputationType::kTF32AsF32 : ComputationType::kF32;
  switch (dtype) {
    case DT_HALF:
    case DT_BFLOAT16:
      return use_f32_for_f16_computation ? f32_type : ComputationType::kF16;
    case DT_FLOAT:
      return f32_type;
    case DT_DOUBLE:
      return ComputationType::kF64;
    case DT_COMPLEX64:
      return f32_type;
    case DT_COMPLEX128:
      return ComputationType::kComplexF64;
    default:
      return errors::Internal("Unsupported dtype for Blas Plans.");
  }
}

StatusOr<const PlanAndAlgorithms*> GetPlanAndAlgorithms(
    se::Stream* stream, const se::cuda::BlasLt::MatmulPlanParams& params,
    std::optional<int> max_algorithm_count) {
  static const int64_t max_scratch_size =
      GetWorkspaceLimit(1LL << 32);  // 4GB by default
  static const int64_t max_autotune_algorithm_count =
      MatmulMaxAutotuneAlgorithmCount();

  if (!max_algorithm_count) max_algorithm_count = max_autotune_algorithm_count;

  static auto& plan_map = *new BlasLtMatmulPlanMap();

  const PlanAndAlgorithms* plan_and_algorithms = plan_map.Find(params);
  if (!plan_and_algorithms) {
    se::cuda::BlasLt* blas_lt = se::cuda::GetBlasLt(stream);
    TF_RET_CHECK(blas_lt != nullptr);

    se::cuda::BlasLt::MatmulPlan plan;
    TF_RETURN_IF_ERROR(plan.init(params));

    TF_ASSIGN_OR_RETURN(
        std::vector<se::cuda::BlasLt::MatmulAlgorithm> algorithms,
        blas_lt->GetMatmulAlgorithms(plan, max_scratch_size,
                                     *max_algorithm_count));

    plan_and_algorithms =
        plan_map.Insert(params, {std::move(plan), std::move(algorithms)});
  }
  return plan_and_algorithms;
}

}  // namespace tensorflow
