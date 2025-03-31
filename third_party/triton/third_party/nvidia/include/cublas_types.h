#ifndef TRITON_CUBLAS_TYPES_H
#define TRITON_CUBLAS_TYPES_H

#include <cstddef>
#include <cstdint>

// Forward declarations of cuBLAS types and functions.

/* CUBLAS status type returns */
typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_MAPPING_ERROR = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
  CUBLAS_STATUS_NOT_SUPPORTED = 15,
  CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

typedef enum {
  CUBLAS_COMPUTE_16F = 64,          /* half - default */
  CUBLAS_COMPUTE_16F_PEDANTIC = 65, /* half - pedantic */
  CUBLAS_COMPUTE_32F = 68,          /* float - default */
  CUBLAS_COMPUTE_32F_PEDANTIC = 69, /* float - pedantic */
  CUBLAS_COMPUTE_32F_FAST_16F =
      74, /* float - fast, allows down-converting inputs to half or TF32 */
  CUBLAS_COMPUTE_32F_FAST_16BF =
      75, /* float - fast, allows down-converting inputs to bfloat16 or TF32 */
  CUBLAS_COMPUTE_32F_FAST_TF32 =
      77, /* float - fast, allows down-converting inputs to TF32 */
  CUBLAS_COMPUTE_64F = 70,          /* double - default */
  CUBLAS_COMPUTE_64F_PEDANTIC = 71, /* double - pedantic */
  CUBLAS_COMPUTE_32I = 72,          /* signed 32-bit int - default */
  CUBLAS_COMPUTE_32I_PEDANTIC = 73, /* signed 32-bit int - pedantic */
} cublasComputeType_t;

typedef enum {
  CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,
  CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1,
  CUBLASLT_MATMUL_DESC_POINTER_MODE = 2,
  CUBLASLT_MATMUL_DESC_TRANSA = 3,
  CUBLASLT_MATMUL_DESC_TRANSB = 4,
  CUBLASLT_MATMUL_DESC_TRANSC = 5,
  CUBLASLT_MATMUL_DESC_FILL_MODE = 6,
  CUBLASLT_MATMUL_DESC_EPILOGUE = 7,
  CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8,
  CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 10,
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11,
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 12,
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 13,
  CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = 14,
  CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = 15,
  CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
  CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
  CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19,
  CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,
  CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = 21,
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 23,
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = 24,
  CUBLASLT_MATMUL_DESC_FAST_ACCUM = 25,
  CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26,
  CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS = 27,
  CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS = 28,
  CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER = 29,
  CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER = 30,
} cublasLtMatmulDescAttributes_t;

typedef enum {
  CUBLAS_OP_N = 0,
  CUBLAS_OP_T = 1,
  CUBLAS_OP_C = 2,
  CUBLAS_OP_HERMITAN = 2, /* synonym if CUBLAS_OP_C */
  CUBLAS_OP_CONJG =
      3 /* conjugate, placeholder - not supported in the current release */
} cublasOperation_t;

typedef enum {
  CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0,
  CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
  CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 3,
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5,
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = 6,
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = 7,
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = 8,
  CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT = 9,
  CUBLASLT_MATMUL_PREF_IMPL_MASK = 12,
} cublasLtMatmulPreferenceAttributes_t;
typedef struct {
  uint64_t data[8];
} cublasLtMatrixLayoutOpaque_t;
typedef cublasLtMatrixLayoutOpaque_t *cublasLtMatrixLayout_t;

typedef struct {
  uint64_t data[8];
} cublasLtMatmulPreferenceOpaque_t;
typedef cublasLtMatmulPreferenceOpaque_t *cublasLtMatmulPreference_t;

typedef struct {
  uint64_t data[8];
} cublasLtMatmulAlgo_t;

typedef struct {
  cublasLtMatmulAlgo_t algo;
  size_t workspaceSize;
  cublasStatus_t state;
  float wavesCount;
  int reserved[4];
} cublasLtMatmulHeuristicResult_t;

typedef enum cudaDataType_t {
  CUDA_R_16F = 2,      /* real as a half */
  CUDA_C_16F = 6,      /* complex as a pair of half numbers */
  CUDA_R_16BF = 14,    /* real as a nv_bfloat16 */
  CUDA_C_16BF = 15,    /* complex as a pair of nv_bfloat16 numbers */
  CUDA_R_32F = 0,      /* real as a float */
  CUDA_C_32F = 4,      /* complex as a pair of float numbers */
  CUDA_R_64F = 1,      /* real as a double */
  CUDA_C_64F = 5,      /* complex as a pair of double numbers */
  CUDA_R_4I = 16,      /* real as a signed 4-bit int */
  CUDA_C_4I = 17,      /* complex as a pair of signed 4-bit int numbers */
  CUDA_R_4U = 18,      /* real as a unsigned 4-bit int */
  CUDA_C_4U = 19,      /* complex as a pair of unsigned 4-bit int numbers */
  CUDA_R_8I = 3,       /* real as a signed 8-bit int */
  CUDA_C_8I = 7,       /* complex as a pair of signed 8-bit int numbers */
  CUDA_R_8U = 8,       /* real as a unsigned 8-bit int */
  CUDA_C_8U = 9,       /* complex as a pair of unsigned 8-bit int numbers */
  CUDA_R_16I = 20,     /* real as a signed 16-bit int */
  CUDA_C_16I = 21,     /* complex as a pair of signed 16-bit int numbers */
  CUDA_R_16U = 22,     /* real as a unsigned 16-bit int */
  CUDA_C_16U = 23,     /* complex as a pair of unsigned 16-bit int numbers */
  CUDA_R_32I = 10,     /* real as a signed 32-bit int */
  CUDA_C_32I = 11,     /* complex as a pair of signed 32-bit int numbers */
  CUDA_R_32U = 12,     /* real as a unsigned 32-bit int */
  CUDA_C_32U = 13,     /* complex as a pair of unsigned 32-bit int numbers */
  CUDA_R_64I = 24,     /* real as a signed 64-bit int */
  CUDA_C_64I = 25,     /* complex as a pair of signed 64-bit int numbers */
  CUDA_R_64U = 26,     /* real as a unsigned 64-bit int */
  CUDA_C_64U = 27,     /* complex as a pair of unsigned 64-bit int numbers */
  CUDA_R_8F_E4M3 = 28, /* real as a nv_fp8_e4m3 */
  CUDA_R_8F_E5M2 = 29, /* real as a nv_fp8_e5m2 */
} cudaDataType;

struct cublasContext;
typedef struct cublasLtContext *cublasLtHandle_t;
struct cublasLtMatmulDescOpaque_t;
typedef cublasLtMatmulDescOpaque_t *cublasLtMatmulDesc_t;
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;

#endif // TRITON_CUBLAS_TYPES_H
