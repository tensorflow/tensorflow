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

#include "tensorflow/stream_executor/matmul_util.h"

#include <string>
#include <utility>

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/util/env_var.h"

namespace stream_executor {

int64_t GetWorkspaceLimit(int64_t default_value_in_bytes) {
  const char* workspace_limit_in_mb_str =
      getenv("TF_CUBLAS_WORKSPACE_LIMIT_IN_MB");
  if (workspace_limit_in_mb_str != nullptr &&
      strcmp(workspace_limit_in_mb_str, "") != 0) {
    int64_t scratch_limit_in_mb = -1;
    if (tensorflow::strings::safe_strto64(workspace_limit_in_mb_str,
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
  tensorflow::Status status = tensorflow::ReadInt64FromEnvVar(
      "TF_MATMUL_AUTOTUNE_MAX_ALGORITHMS", 10, &value);
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

static inline port::StatusOr<blas::DataType> GetBlasDataType(
    tensorflow::DataType dtype) {
  switch (dtype) {
    case tensorflow::DT_HALF:
      return blas::ToDataType<Eigen::half>::value;
    case tensorflow::DT_FLOAT:
      return blas::ToDataType<float>::value;
    case tensorflow::DT_DOUBLE:
      return blas::ToDataType<double>::value;
    case tensorflow::DT_COMPLEX64:
      return blas::ToDataType<tensorflow::complex64>::value;
    case tensorflow::DT_COMPLEX128:
      return blas::ToDataType<tensorflow::complex128>::value;
    default:
      return port::InternalError("Unsupported dtype for Blas Plans.");
  }
}

static inline port::StatusOr<blas::ComputationType> GetBlasComputationType(
    const tensorflow::DataType& dtype, bool allow_tf32) {
  using blas::ComputationType;
  static bool use_f32_for_f16_computation =
      tensorflow::MatmulDoFP32ComputationFP16Input();
  ComputationType f32_type =
      allow_tf32 ? ComputationType::kTF32AsF32 : ComputationType::kF32;
  switch (dtype) {
    case tensorflow::DT_HALF:
    case tensorflow::DT_BFLOAT16:
      return use_f32_for_f16_computation ? f32_type : ComputationType::kF16;
    case tensorflow::DT_FLOAT:
      return f32_type;
    case tensorflow::DT_DOUBLE:
      return ComputationType::kF64;
    case tensorflow::DT_COMPLEX64:
      return f32_type;
    case tensorflow::DT_COMPLEX128:
      return ComputationType::kComplexF64;
    default:
      return port::InternalError("Unsupported dtype for Blas Plans.");
  }
}

port::StatusOr<const blas::PlanAndAlgorithms*> GetPlanAndAlgorithms(
    Stream* stream, BatchMatmulParameters matmul_parameters, int64_t batch_size,
    tensorflow::DataType dtype, blas::MatrixDescriptor lhs_matrix,
    blas::MatrixDescriptor rhs_matrix, blas::MatrixDescriptor output_matrix) {
  static const int64_t max_scratch_size =
      GetWorkspaceLimit(1LL << 32);  // 4GB by default
  static const int64_t max_autotune_algorithm_count =
      MatmulMaxAutotuneAlgorithmCount();
  const blas::PlanAndAlgorithms* plan_and_algorithms =
      BatchMatmulPlanMapSingleton::GetInstance()->Find(matmul_parameters);
  if (!plan_and_algorithms) {
    TF_ASSIGN_OR_RETURN(
        blas::BlasLtMatmulPlanParams plan_params,
        CreatePlanParams(batch_size, dtype, matmul_parameters.GetEpilogOp(),
                         lhs_matrix, rhs_matrix, output_matrix));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<blas::IBlasLtMatmulPlan> plan,
                        stream->parent()->CreateBlasLtMatmulPlan(plan_params));
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<blas::IBlasLtMatmulAlgorithm>> algorithms,
        stream->parent()->GetBlasLtMatmulAlgorithms(
            plan.get(), max_scratch_size,
            /* max_algorithm_count */ max_autotune_algorithm_count));

    plan_and_algorithms = BatchMatmulPlanMapSingleton::GetInstance()->Insert(
        matmul_parameters, {std::move(plan), std::move(algorithms)});
  }
  return plan_and_algorithms;
}

port::StatusOr<blas::BlasLtMatmulPlanParams> CreatePlanParams(
    int64_t batch_size, tensorflow::DataType dtype, blas::Epilogue epilog_op,
    blas::MatrixDescriptor lhs_matrix, blas::MatrixDescriptor rhs_matrix,
    blas::MatrixDescriptor output_matrix) {
  blas::BlasLtMatmulPlanParams plan_params;
  int64_t m = output_matrix.num_rows;
  int64_t n = output_matrix.num_cols;
  int64_t k = lhs_matrix.reduced_dim();

  TF_ASSIGN_OR_RETURN(blas::DataType blas_dtype, GetBlasDataType(dtype));
  plan_params.ab_type = blas_dtype;
  plan_params.c_type = blas_dtype;
  bool allow_tf32 = tensorflow::tensor_float_32_execution_enabled();
  TF_ASSIGN_OR_RETURN(blas::ComputationType computation_type,
                      GetBlasComputationType(dtype, allow_tf32));

  plan_params.computation_type = computation_type;

  plan_params.pointer_mode = blas::PointerMode::kHost;
  plan_params.epilogue = blas::Epilogue::kDefault;
  plan_params.epilogue = epilog_op;

  plan_params.transa = lhs_matrix.transpose;
  plan_params.transb = rhs_matrix.transpose;
  plan_params.m = m;
  plan_params.n = n;
  plan_params.k = k;
  plan_params.lda = lhs_matrix.num_rows;
  plan_params.ldb = rhs_matrix.num_rows;
  plan_params.ldc = output_matrix.num_rows;
  plan_params.batch_count = batch_size;

  bool broadcast = batch_size == 1;
  int64_t lhs_stride = broadcast ? 0 : lhs_matrix.stride;
  int64_t rhs_stride = broadcast ? 0 : rhs_matrix.stride;
  plan_params.stride_a = lhs_stride;
  plan_params.stride_b = rhs_stride;
  plan_params.stride_c = output_matrix.stride;

  if (VLOG_IS_ON(4)) {
    bool trans_x = lhs_matrix.transpose == blas::Transpose::kTranspose;
    bool trans_y = rhs_matrix.transpose == blas::Transpose::kTranspose;
    std::string transString[] = {"kNoTranspose", "kTranspose"};
    VLOG(4) << "plan_params.transa: " << transString[trans_x ? 1 : 0]
            << " plan_params.transb: " << transString[trans_y ? 1 : 0]
            << " plan_params.m: " << plan_params.m
            << " plan_params.n: " << plan_params.n
            << " plan_params.k: " << plan_params.k
            << " plan_params.lda: " << plan_params.lda
            << " plan_params.ldb: " << plan_params.ldb
            << " plan_params.ldc: " << plan_params.ldc
            << " plan_params.batch_count: " << plan_params.batch_count
            << " plan_params.stride_a: " << plan_params.stride_a
            << " plan_params.stride_b: " << plan_params.stride_b
            << " plan_params.stride_c: " << plan_params.stride_c;
  }
  return plan_params;
}

}  // namespace stream_executor
