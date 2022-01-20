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

#include "tensorflow/core/framework/op_kernel.h"

namespace stream_executor {

int64_t GetWorkspaceLimit(const string& envvar_in_mb,
                          int64_t default_value_in_bytes) {
  const char* workspace_limit_in_mb_str = getenv(envvar_in_mb.c_str());
  if (workspace_limit_in_mb_str != nullptr &&
      strcmp(workspace_limit_in_mb_str, "") != 0) {
    int64_t scratch_limit_in_mb = -1;
    if (tensorflow::strings::safe_strto64(workspace_limit_in_mb_str,
                                          &scratch_limit_in_mb)) {
      return scratch_limit_in_mb * (1 << 20);
    } else {
      LOG(WARNING) << "Invalid value for env-var " << envvar_in_mb << ": "
                   << workspace_limit_in_mb_str;
    }
  }
  return default_value_in_bytes;
}

port::StatusOr<const blas::PlanAndAlgorithms*> GetPlanAndAlgorithms(
    Stream* stream, BatchMatmulParameters matmul_parameters, int64_t batch_size,
    blas::DataType blas_dtype, tensorflow::DataType dtype,
    blas::MatrixDescriptor lhs_matrix, blas::MatrixDescriptor rhs_matrix,
    blas::MatrixDescriptor output_matrix) {
  static const int64_t max_scratch_size = GetBlasWorkspaceLimit(
      "TF_CUBLAS_WORKSPACE_LIMIT_IN_MB", 1LL << 32);  // 4GB by default
  static const int64_t max_autotune_algorithm_count =
      tensorflow::MatmulMaxAutotuneAlgorithmCount();
  const blas::PlanAndAlgorithms* plan_and_algorithms =
      BatchMatmulPlanMapSingleton::GetInstance()->Find(matmul_parameters);
  if (!plan_and_algorithms) {
    TF_ASSIGN_OR_RETURN(
        blas::BlasLtMatmulPlanParams plan_params,
        CreatePlanParams(batch_size, blas_dtype, dtype, lhs_matrix, rhs_matrix,
                         output_matrix));
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
    int64_t batch_size, blas::DataType blas_dtype, tensorflow::DataType dtype,
    blas::MatrixDescriptor lhs_matrix, blas::MatrixDescriptor rhs_matrix,
    blas::MatrixDescriptor output_matrix) {
  blas::BlasLtMatmulPlanParams plan_params;
  int64_t m = output_matrix.num_rows;
  int64_t n = output_matrix.num_cols;
  int64_t k = lhs_matrix.reduced_dim();

  plan_params.ab_type = blas_dtype;
  plan_params.c_type = blas_dtype;
  bool allow_tf32 = tensorflow::tensor_float_32_execution_enabled();
  blas::ComputationType computation_type;
  TF_CHECK_OK(GetBlasComputationType(dtype, allow_tf32, &computation_type));

  plan_params.computation_type = computation_type;

  plan_params.pointer_mode = blas::PointerMode::kHost;
  plan_params.epilogue = blas::Epilogue::kDefault;

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

GpuScratchAllocator::GpuScratchAllocator(int64_t memory_limit,
                                         tensorflow::OpKernelContext* context)
    : memory_limit_(memory_limit), total_byte_size_(0), context_(context) {}

port::StatusOr<DeviceMemory<uint8>> GpuScratchAllocator::AllocateBytes(
    int64_t byte_size) {
  tensorflow::Tensor temporary_memory;
  if (byte_size < 0) {
    return port::Status{port::error::INVALID_ARGUMENT,
                        "Requested negative byte size!"};
  }
  if (byte_size > memory_limit_) {
    return port::Status{
        port::error::UNAVAILABLE,
        absl::StrCat("Requested memory size (", byte_size,
                     ") exceeds the max memory limit (", memory_limit_, ").")};
  }
  tensorflow::AllocationAttributes allocation_attr;
  allocation_attr.retry_on_failure = false;
  tensorflow::Status allocation_status(context_->allocate_temp(
      tensorflow::DT_UINT8, tensorflow::TensorShape({byte_size}),
      &temporary_memory, tensorflow::AllocatorAttributes(), allocation_attr));
  if (!allocation_status.ok()) {
    return port::Status{
        port::error::UNAVAILABLE,
        absl::StrCat("Failed to allocate the requested memory size (",
                     byte_size, ").")};
  }
  // Hold the reference of the allocated tensors until the end of the
  // allocator.
  // NOTE: We expect tensors to be deallocated when this allocator goes out of
  // scope when allocated_tensors is destructed.
  allocated_tensors_.push_back(temporary_memory);
  total_byte_size_ += byte_size;
  return port::StatusOr<DeviceMemory<uint8>>(
      tensorflow::AsDeviceMemory(temporary_memory.flat<uint8>().data(),
                                 temporary_memory.flat<uint8>().size()));
}

}  // namespace stream_executor