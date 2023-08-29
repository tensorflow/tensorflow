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

#include <optional>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_blas_lt.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/matmul_autotune.h"

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

std::string BlasLtMatmulPlanParams::ToString() const {
  return "";  // TODO
}

bool BlasLtMatmulPlanParams::operator==(
    const BlasLtMatmulPlanParams& other) const {
  return internal::AsTuple(*this) == internal::AsTuple(other);
}

const PlanAndAlgorithms* BlasLtMatmulPlanMap::Find(
    const BlasLtMatmulPlanParams& params) const {
  absl::MutexLock lock(&mu_);
  auto it = params_plan_map_.find(params);
  return (it != params_plan_map_.end()) ? &it->second : nullptr;
}

const PlanAndAlgorithms* BlasLtMatmulPlanMap::Insert(
    const BlasLtMatmulPlanParams& params, PlanAndAlgorithms value) {
  absl::MutexLock lock(&mu_);
  return &params_plan_map_.emplace(params, std::move(value)).first->second;
}

namespace {

int MatmulMaxAutotuneAlgorithmCount() {
  int64_t value;
  Status status =
      ReadInt64FromEnvVar("TF_MATMUL_AUTOTUNE_MAX_ALGORITHMS", 10, &value);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
  }
  static constexpr const int kMaxValue = std::numeric_limits<int>::max();
  if (value < 1 || value > kMaxValue) {
    LOG(ERROR) << "Invalid value for TF_MATMUL_AUTOTUNE_MAX_ALGORITHMS: "
               << value << " is not in range [1, " << kMaxValue << "]";
  }
  return value;
}

StatusOr<se::blas::ComputationType> GetBlasComputationType(
    const se::blas::DataType& dtype) {
  using se::blas::ComputationType;
  static bool use_f32_for_f16_computation = MatmulDoFP32ComputationFP16Input();
  switch (dtype) {
    case se::blas::DataType::kHalf:
      return use_f32_for_f16_computation ? ComputationType::kF32
                                         : ComputationType::kF16;
    case se::blas::DataType::kBF16:
      return ComputationType::kF32;
    case se::blas::DataType::kFloat:  // fall-through
    case se::blas::DataType::kComplexFloat:
      return tensor_float_32_execution_enabled() ? ComputationType::kTF32AsF32
                                                 : ComputationType::kF32;
    case se::blas::DataType::kDouble:  // fall-through
    case se::blas::DataType::kComplexDouble:
      return ComputationType::kF64;
    default:
      return errors::Internal("Unsupported dtype for Blas Plans.");
  }
}

se::blas::DataType GetScaleType(se::blas::DataType c_type,
                                se::blas::ComputationType computation_type) {
  return ((computation_type == se::blas::ComputationType::kF32) &&
          (c_type != se::blas::DataType::kComplexFloat))
             ? se::blas::DataType::kFloat
             : c_type;
}

}  // namespace

StatusOr<const PlanAndAlgorithms*> GetPlanAndAlgorithms(
    se::Stream* stream, const BlasLtMatmulPlanParams& params,
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

    TF_ASSIGN_OR_RETURN(se::blas::ComputationType computation_type,
                        GetBlasComputationType(params.dtype));

    se::blas::DataType scale_type =
        GetScaleType(params.dtype, computation_type);

    // cublas_lt's output is column-major. We want row-major so use identity:
    // C^T = (A @ B)^T = B^T @ A^T.
    constexpr auto kColMajor =
        se::cuda::BlasLt::MatrixLayout::Order::kColumnMajor;

    size_t rows_a = params.k;
    size_t cols_a = params.m;
    size_t rows_b = params.n;
    size_t cols_b = params.k;

    if (params.trans_a != se::blas::Transpose::kNoTranspose) {
      std::swap(rows_a, cols_a);
    }

    if (params.trans_b != se::blas::Transpose::kNoTranspose) {
      std::swap(rows_b, cols_b);
    }

    int64_t batch_stride_a =
        params.broadcast_a ? 0 : static_cast<int64_t>(rows_a * cols_a);
    int64_t batch_stride_b =
        params.broadcast_b ? 0 : static_cast<int64_t>(rows_b * cols_b);

    TF_ASSIGN_OR_RETURN(
        auto a_desc,
        se::cuda::BlasLt::MatrixLayout::Create(
            params.dtype, rows_a, cols_a, kColMajor, params.batch_count,
            /*leading_dim_stride=*/std::nullopt, batch_stride_a));
    TF_ASSIGN_OR_RETURN(
        auto b_desc,
        se::cuda::BlasLt::MatrixLayout::Create(
            params.dtype, rows_b, cols_b, kColMajor, params.batch_count,
            /*leading_dim_stride=*/std::nullopt, batch_stride_b));
    TF_ASSIGN_OR_RETURN(auto c_desc, se::cuda::BlasLt::MatrixLayout::Create(
                                         params.dtype, params.n, params.m,
                                         kColMajor, params.batch_count));
    TF_ASSIGN_OR_RETURN(auto d_desc, se::cuda::BlasLt::MatrixLayout::Create(
                                         params.dtype, params.n, params.m,
                                         kColMajor, params.batch_count));

    // `A` and `B` swapped (see above re. column-major output).
    TF_ASSIGN_OR_RETURN(auto op_desc,
                        se::cuda::BlasLt::MatmulDesc::Create(
                            computation_type, scale_type,
                            /*trans_a=*/params.trans_b,
                            /*trans_b=*/params.trans_a, params.epilogue));

    // `A` and `B` swapped (see above re. column-major output).
    se::cuda::BlasLt::MatmulPlan plan{std::move(op_desc), std::move(b_desc),
                                      std::move(a_desc), std::move(c_desc),
                                      std::move(d_desc)};
    TF_ASSIGN_OR_RETURN(
        auto preference,
        se::cuda::BlasLt::MatmulPreference::Create(max_scratch_size));

    TF_ASSIGN_OR_RETURN(
        std::vector<se::cuda::BlasLt::MatmulAlgorithm> algorithms,
        blas_lt->GetMatmulAlgorithms(plan, preference, *max_algorithm_count));

    plan_and_algorithms =
        plan_map.Insert(params, {std::move(plan), std::move(algorithms)});
  }
  return plan_and_algorithms;
}

}  // namespace tensorflow
