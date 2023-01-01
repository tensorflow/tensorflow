/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_

#if TF_HIPBLASLT
#define cublasStatus_t hipblasStatus_t
#define cudaDataType_t hipblasDatatype_t
#define cublasLtMatrixLayout_t hipblasLtMatrixLayout_t
#define cublasComputeType_t hipblasLtComputeType_t
#define cublasLtPointerMode_t hipblasPointerMode_t
#define cublasLtMatmulDesc_t hipblasLtMatmulDesc_t
#define cublasLtMatmulPreference_t hipblasLtMatmulPreference_t
#define cublasLtMatmulAlgo_t hipblasLtMatmulAlgo_t
#define cublasLtHandle_t hipblasLtHandle_t
#define cublasLtMatrixLayoutAttribute_t hipblasLtMatrixLayoutAttribute_t
#define cublasLtEpilogue_t hipblasLtEpilogue_t
#define cublasLtMatmulDescAttributes_t hipblasLtMatmulDescAttributes_t
#define cublasLtMatmulPreferenceAttributes_t hipblasLtMatmulPreferenceAttributes_t
#define cublasLtMatmulHeuristicResult_t hipblasLtMatmulHeuristicResult_t

#define AsCudaDataType AsHipblasDataType
#define AsCublasLtPointerMode AsHipblasLtPointerMode
#define AsCublasComputeType AsHipblasComputeType
#define AsCublasOperation AsHipblasOperation

#define cublasLtCreate wrap::hipblasLtCreate
#define cublasLtMatrixLayoutCreate wrap::hipblasLtMatrixLayoutCreate
#define cublasLtMatmulDescCreate wrap::hipblasLtMatmulDescCreate
#define cublasLtMatmulPreferenceCreate wrap::hipblasLtMatmulPreferenceCreate
#define cublasLtDestroy wrap::hipblasLtDestroy
#define cublasLtMatrixLayoutDestroy wrap::hipblasLtMatrixLayoutDestroy
#define cublasLtMatmulDescDestroy wrap::hipblasLtMatmulDescDestroy
#define cublasLtMatmulPreferenceDestroy wrap::hipblasLtMatmulPreferenceDestroy
#define cublasLtMatmulPreferenceSetAttribute wrap::hipblasLtMatmulPreferenceSetAttribute
#define cublasLtMatmul wrap::hipblasLtMatmul
#define cublasLtMatmulAlgoGetHeuristic wrap::hipblasLtMatmulAlgoGetHeuristic
#define cublasLtMatrixLayoutSetAttribute wrap::hipblasLtMatrixLayoutSetAttribute
#define cublasLtMatmulDescSetAttribute wrap::hipblasLtMatmulDescSetAttribute

#define CUBLASLT_POINTER_MODE_DEVICE HIPBLAS_POINTER_MODE_DEVICE
#define CUBLASLT_POINTER_MODE_HOST HIPBLAS_POINTER_MODE_HOST
#define CUBLASLT_EPILOGUE_DEFAULT HIPBLASLT_EPILOGUE_DEFAULT
#define CUBLASLT_EPILOGUE_RELU HIPBLASLT_EPILOGUE_RELU
#define CUBLASLT_EPILOGUE_BIAS HIPBLASLT_EPILOGUE_BIAS
#define CUBLASLT_EPILOGUE_RELU_BIAS HIPBLASLT_EPILOGUE_RELU_BIAS
#define CUBLASLT_EPILOGUE_GELU HIPBLASLT_EPILOGUE_GELU
// not supported in 5.5
//#define CUBLASLT_EPILOGUE_GELU_AUX HIPBLASLT_EPILOGUE_GELU_AUX
//#define CUBLASLT_EPILOGUE_GELU_BIAS HIPBLASLT_EPILOGUE_GELU_BIAS
//#define CUBLASLT_EPILOGUE_GELU_AUX_BIAS HIPBLASLT_EPILOGUE_GELU_AUX_BIAS
#define CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT
#define CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
#define CUBLASLT_MATRIX_LAYOUT_TYPE HIPBLASLT_MATRIX_LAYOUT_TYPE
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#define CUBLASLT_MATMUL_DESC_POINTER_MODE HIPBLASLT_MATMUL_DESC_POINTER_MODE
#define CUBLASLT_MATMUL_DESC_TRANSA HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB HIPBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_DESC_EPILOGUE HIPBLASLT_MATMUL_DESC_EPILOGUE
#define CUBLASLT_MATMUL_DESC_BIAS_POINTER HIPBLASLT_MATMUL_DESC_BIAS_POINTER

#define SE_CUBLAS_RETURN_IF_ERROR SE_HIPBLAS_RETURN_IF_ERROR
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS

#include "../cuda/cuda_blas_lt.h"

namespace stream_executor {
namespace rocm {
inline std::string ToHipblasString(hipblasStatus_t status) {
  //return hipblasStatusToString(status);
  return "HipblasLt error " + std::to_string((int)status);
}

inline tsl::Status ToStatus(hipblasStatus_t status, const char* prefix) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return tsl::errors::Internal(
                        absl::StrCat(prefix, ": ", ToHipblasString(status)));
  }
  return tsl::OkStatus();
}

#define SE_HIPBLAS_RETURN_IF_ERROR(expr) \
  TF_RETURN_IF_ERROR(::stream_executor::rocm::ToStatus(expr, #expr))
}
}  // namespace stream_executor

#endif

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_
