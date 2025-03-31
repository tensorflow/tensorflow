#ifndef TRITON_CUBLAS_INSTANCE_H
#define TRITON_CUBLAS_INSTANCE_H

#include "cublas_types.h"
#include <dlfcn.h>
#include <stdexcept>
#include <string>

class CublasLtInstance {
  // Typedefs for cublas functions
  typedef cublasStatus_t (*cublasLtCreate_t)(cublasLtHandle_t *);
  typedef cublasStatus_t (*cublasLtDestroy_t)(cublasLtHandle_t);
  typedef cublasStatus_t (*cublasLtMatmulDescCreate_t)(cublasLtMatmulDesc_t *,
                                                       cublasComputeType_t,
                                                       cudaDataType_t);
  typedef cublasStatus_t (*cublasLtMatmulDescDestroy_t)(cublasLtMatmulDesc_t);
  typedef cublasStatus_t (*cublasLtMatmulDescSetAttribute_t)(
      cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void *,
      size_t);
  typedef cublasStatus_t (*cublasLtMatrixLayoutCreate_t)(
      cublasLtMatrixLayout_t *, cudaDataType_t, uint64_t, uint64_t, int64_t);
  typedef cublasStatus_t (*cublasLtMatrixLayoutDestroy_t)(
      cublasLtMatrixLayout_t);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceCreate_t)(
      cublasLtMatmulPreference_t *);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceDestroy_t)(
      cublasLtMatmulPreference_t);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceSetAttribute_t)(
      cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t,
      const void *, size_t);
  typedef cublasStatus_t (*cublasLtMatmulAlgoGetHeuristic_t)(
      cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
      cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
      cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t *,
      int *);
  typedef cublasStatus_t (*cublasLtMatmul_t)(
      cublasLtHandle_t, cublasLtMatmulDesc_t, const void *, const void *,
      const cublasLtMatrixLayout_t, const void *, const cublasLtMatrixLayout_t,
      const void *, const void *, const cublasLtMatrixLayout_t, void *,
      const cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t *, void *,
      size_t, cudaStream_t);

  static constexpr const char *name = "libcublas.so";

  cublasLtCreate_t cublasLtCreate;
  cublasLtDestroy_t cublasLtDestroy;
  cublasLtMatmulDescCreate_t cublasLtMatmulDescCreate;
  cublasLtMatmulDescDestroy_t cublasLtMatmulDescDestroy;
  cublasLtMatmulDescSetAttribute_t cublasLtMatmulDescSetAttribute;
  cublasLtMatrixLayoutCreate_t cublasLtMatrixLayoutCreate;
  cublasLtMatrixLayoutDestroy_t cublasLtMatrixLayoutDestroy;
  cublasLtMatmulPreferenceCreate_t cublasLtMatmulPreferenceCreate;
  cublasLtMatmulPreferenceDestroy_t cublasLtMatmulPreferenceDestroy;
  cublasLtMatmulPreferenceSetAttribute_t cublasLtMatmulPreferenceSetAttribute;
  cublasLtMatmulAlgoGetHeuristic_t cublasLtMatmulAlgoGetHeuristic;
  cublasLtMatmul_t cublasLtMatmul;

  void *dylibHandle = nullptr;
  cublasLtHandle_t ltHandle;

  void *workspace = nullptr;
  size_t workspaceSize = 0;

  cublasLtMatmulPreference_t preference = NULL;

  void loadCublasDylib() {
    if (dylibHandle == nullptr) {
      // First reuse the existing handle
      dylibHandle = dlopen(name, RTLD_NOLOAD);
    }
    if (dylibHandle == nullptr) {
      // If not found, try to load it
      dylibHandle = dlopen(name, RTLD_LOCAL | RTLD_LAZY);
    }
    if (dylibHandle == nullptr) {
      throw std::runtime_error("Could not find `" + std::string(name) +
                               "`. Make sure it is in your "
                               "LD_LIBRARY_PATH.");
    }
    dlerror(); // Clear any existing error

    cublasLtCreate = (cublasLtCreate_t)dlsym(dylibHandle, "cublasLtCreate");
    cublasLtDestroy = (cublasLtDestroy_t)dlsym(dylibHandle, "cublasLtDestroy");
    cublasLtMatmulDescCreate = (cublasLtMatmulDescCreate_t)dlsym(
        dylibHandle, "cublasLtMatmulDescCreate");
    cublasLtMatmulDescDestroy = (cublasLtMatmulDescDestroy_t)dlsym(
        dylibHandle, "cublasLtMatmulDescDestroy");
    cublasLtMatmulDescSetAttribute = (cublasLtMatmulDescSetAttribute_t)dlsym(
        dylibHandle, "cublasLtMatmulDescSetAttribute");
    cublasLtMatrixLayoutCreate = (cublasLtMatrixLayoutCreate_t)dlsym(
        dylibHandle, "cublasLtMatrixLayoutCreate");
    cublasLtMatrixLayoutDestroy = (cublasLtMatrixLayoutDestroy_t)dlsym(
        dylibHandle, "cublasLtMatrixLayoutDestroy");
    cublasLtMatmulPreferenceCreate = (cublasLtMatmulPreferenceCreate_t)dlsym(
        dylibHandle, "cublasLtMatmulPreferenceCreate");
    cublasLtMatmulPreferenceDestroy = (cublasLtMatmulPreferenceDestroy_t)dlsym(
        dylibHandle, "cublasLtMatmulPreferenceDestroy");
    cublasLtMatmulPreferenceSetAttribute =
        (cublasLtMatmulPreferenceSetAttribute_t)dlsym(
            dylibHandle, "cublasLtMatmulPreferenceSetAttribute");
    cublasLtMatmulAlgoGetHeuristic = (cublasLtMatmulAlgoGetHeuristic_t)dlsym(
        dylibHandle, "cublasLtMatmulAlgoGetHeuristic");
    cublasLtMatmul = (cublasLtMatmul_t)dlsym(dylibHandle, "cublasLtMatmul");

    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      throw std::runtime_error("Could not load symbol from `" +
                               std::string(name) +
                               "`: " + std::string(dlsym_error));
    }
  }

  void unloadCublasDylib() { dlclose(dylibHandle); }

  void successOrExit(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS Error: " + std::to_string(status) +
                               "\n");
    }
  }

  // Simple wrapper around the cublasLtMatmul function
  void matmul_impl(int m, int n, int k, uint64_t A, uint64_t B, uint64_t D,
                   cudaDataType_t dtype) {
    cublasLtMatmulDesc_t matmulDesc = NULL;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    int8_t fastAccum = 1;

    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                           Ddesc = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    successOrExit(
        cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    if (dtype == CUDA_R_8F_E4M3) {
      successOrExit(cublasLtMatmulDescSetAttribute(
          matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum,
          sizeof(fastAccum)));
    }

    successOrExit(cublasLtMatrixLayoutCreate(&Adesc, dtype, k, m, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Bdesc, dtype, k, n, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, m));
    successOrExit(cublasLtMatrixLayoutCreate(&Ddesc, dtype, m, n, m));

    successOrExit(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
        &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
      throw std::runtime_error(
          "No valid algorithm found by cublasLtMatmulAlgoGetHeuristic");
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    successOrExit(cublasLtMatmul(ltHandle, matmulDesc, &alpha, (void *)A, Adesc,
                                 (void *)B, Bdesc, &beta, nullptr, Cdesc,
                                 (void *)D, Ddesc, &heuristicResult.algo,
                                 (void *)workspace, workspaceSize, 0));
    if (Ddesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(cublasLtMatmulDescDestroy(matmulDesc));
  }

public:
  CublasLtInstance(uint64_t workspace, size_t workspaceSize)
      : workspace((void *)workspace), workspaceSize(workspaceSize) {
    loadCublasDylib();
    cublasLtCreate(&ltHandle);

    successOrExit(cublasLtMatmulPreferenceCreate(&preference));
    successOrExit(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }
  ~CublasLtInstance() {
    if (preference)
      successOrExit(cublasLtMatmulPreferenceDestroy(preference));

    cublasLtDestroy(ltHandle);
    unloadCublasDylib();
  }

  // C = A * B
  // Matrix B needs to be transposed, while matrix A does not. The function
  // *will-not* transpose the matrices, so the caller is responsible for
  // ensuring that the matrices are in the correct format and have the correct
  // dimensions.
  void matmul(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
              cudaDataType_t dtype) {
    // CUDA is column-major, while triton is row-major, therefore we need to
    // reverse the order of the matrices ( A * B = (B^T * A^T)^T ).
    matmul_impl(n, m, k, B, A, C, dtype);
  }
};

#endif // TRITON_CUBLAS_INSTANCE_H
