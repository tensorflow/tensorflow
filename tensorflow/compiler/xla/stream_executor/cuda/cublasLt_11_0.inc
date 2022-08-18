// Auto-generated, do not edit.

extern "C" {

cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t *lightHandle) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(cublasLtHandle_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(lightHandle);
}

cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t lightHandle) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(cublasLtHandle_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(lightHandle);
}

size_t CUBLASWINAPI cublasLtGetVersion(void) {
  using FuncPtr = size_t(CUBLASWINAPI *)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtGetVersion");
  if (!func_ptr) return 0;
  return func_ptr();
}

size_t CUBLASWINAPI cublasLtGetCudartVersion(void) {
  using FuncPtr = size_t(CUBLASWINAPI *)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtGetCudartVersion");
  if (!func_ptr) return 0;
  return func_ptr();
}

cublasStatus_t CUBLASWINAPI cublasLtGetProperty(libraryPropertyType type,
                                                int *value) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(libraryPropertyType, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtGetProperty");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(type, value);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmul(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
    const void *alpha, /* host or device pointer */
    const void *A, cublasLtMatrixLayout_t Adesc, const void *B,
    cublasLtMatrixLayout_t Bdesc, const void *beta, /* host or device pointer */
    const void *C, cublasLtMatrixLayout_t Cdesc, void *D,
    cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo,
    void *workspace, size_t workspaceSizeInBytes, cudaStream_t stream) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasLtHandle_t, cublasLtMatmulDesc_t, const void *, const void *,
      cublasLtMatrixLayout_t, const void *, cublasLtMatrixLayout_t,
      const void *, const void *, cublasLtMatrixLayout_t, void *,
      cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t *, void *, size_t,
      cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmul");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C,
                  Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes,
                  stream);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixTransform(
    cublasLtHandle_t lightHandle, cublasLtMatrixTransformDesc_t transformDesc,
    const void *alpha, /* host or device pointer */
    const void *A, cublasLtMatrixLayout_t Adesc,
    const void *beta, /* host or device pointer */
    const void *B, cublasLtMatrixLayout_t Bdesc, void *C,
    cublasLtMatrixLayout_t Cdesc, cudaStream_t stream) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasLtHandle_t, cublasLtMatrixTransformDesc_t, const void *,
      const void *, cublasLtMatrixLayout_t, const void *, const void *,
      cublasLtMatrixLayout_t, void *, cublasLtMatrixLayout_t, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatrixTransform");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(lightHandle, transformDesc, alpha, A, Adesc, beta, B, Bdesc,
                  C, Cdesc, stream);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutInit_internal(  //
    cublasLtMatrixLayout_t matLayout, size_t size, cudaDataType type,
    uint64_t rows, uint64_t cols, int64_t ld) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtMatrixLayout_t, size_t, cudaDataType, uint64_t, uint64_t,
      int64_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatrixLayoutInit_internal");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(matLayout, size, type, rows, cols, ld);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutCreate(  //
    cublasLtMatrixLayout_t *matLayout, cudaDataType type, uint64_t rows,
    uint64_t cols, int64_t ld) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtMatrixLayout_t *, cudaDataType, uint64_t, uint64_t, int64_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatrixLayoutCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(matLayout, type, rows, cols, ld);
}

cublasStatus_t CUBLASWINAPI
cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(cublasLtMatrixLayout_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatrixLayoutDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(matLayout);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutSetAttribute(  //
    cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr,
    const void *buf, size_t sizeInBytes) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, const void *,
      size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatrixLayoutSetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(matLayout, attr, buf, sizeInBytes);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutGetAttribute(  //
    cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, void *, size_t,
      size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatrixLayoutGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(matLayout, attr, buf, sizeInBytes, sizeWritten);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulDescInit_internal(  //
    cublasLtMatmulDesc_t matmulDesc, size_t size,
    cublasComputeType_t computeType, cudaDataType_t scaleType) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtMatmulDesc_t, size_t, cublasComputeType_t, cudaDataType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmulDescInit_internal");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(matmulDesc, size, computeType, scaleType);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulDescCreate(
    cublasLtMatmulDesc_t *matmulDesc, cublasComputeType_t computeType,
    cudaDataType_t scaleType) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasLtMatmulDesc_t *, cublasComputeType_t, cudaDataType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmulDescCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(matmulDesc, computeType, scaleType);
}

cublasStatus_t CUBLASWINAPI
cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(cublasLtMatmulDesc_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmulDescDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(matmulDesc);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulDescSetAttribute(  //
    cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
    const void *buf, size_t sizeInBytes) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void *,
      size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmulDescSetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(matmulDesc, attr, buf, sizeInBytes);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulDescGetAttribute(  //
    cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, void *, size_t,
      size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmulDescGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(matmulDesc, attr, buf, sizeInBytes, sizeWritten);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescInit_internal(
    cublasLtMatrixTransformDesc_t transformDesc, size_t size,
    cudaDataType scaleType) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(cublasLtMatrixTransformDesc_t,
                                                 size_t, cudaDataType);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatrixTransformDescInit_internal");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(transformDesc, size, scaleType);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescCreate(
    cublasLtMatrixTransformDesc_t *transformDesc, cudaDataType scaleType) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasLtMatrixTransformDesc_t *, cudaDataType);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatrixTransformDescCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(transformDesc, scaleType);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescDestroy(
    cublasLtMatrixTransformDesc_t transformDesc) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(cublasLtMatrixTransformDesc_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatrixTransformDescDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(transformDesc);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescSetAttribute(  //
    cublasLtMatrixTransformDesc_t transformDesc,
    cublasLtMatrixTransformDescAttributes_t attr, const void *buf,
    size_t sizeInBytes) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtMatrixTransformDesc_t, cublasLtMatrixTransformDescAttributes_t,
      const void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatrixTransformDescSetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(transformDesc, attr, buf, sizeInBytes);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescGetAttribute(  //
    cublasLtMatrixTransformDesc_t transformDesc,
    cublasLtMatrixTransformDescAttributes_t attr, void *buf, size_t sizeInBytes,
    size_t *sizeWritten) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtMatrixTransformDesc_t, cublasLtMatrixTransformDescAttributes_t,
      void *, size_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatrixTransformDescGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(transformDesc, attr, buf, sizeInBytes, sizeWritten);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceInit_internal(
    cublasLtMatmulPreference_t pref, size_t size) {
  using FuncPtr =
      cublasStatus_t(CUBLASWINAPI *)(cublasLtMatmulPreference_t, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatmulPreferenceInit_internal");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pref, size);
}

cublasStatus_t CUBLASWINAPI
cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(cublasLtMatmulPreference_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmulPreferenceCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pref);
}

cublasStatus_t CUBLASWINAPI
cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(cublasLtMatmulPreference_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmulPreferenceDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pref);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceSetAttribute(  //
    cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr,
    const void *buf, size_t sizeInBytes) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t,
      const void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatmulPreferenceSetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pref, attr, buf, sizeInBytes);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceGetAttribute(  //
    cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, void *,
      size_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatmulPreferenceGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pref, attr, buf, sizeInBytes, sizeWritten);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference, int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int *returnAlgoCount) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
      cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
      cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t[],
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmulAlgoGetHeuristic");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc,
                  preference, requestedAlgoCount, heuristicResultsArray,
                  returnAlgoCount);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetIds(
    cublasLtHandle_t lightHandle, cublasComputeType_t computeType,
    cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype,
    cudaDataType_t Ctype, cudaDataType_t Dtype, int requestedAlgoCount,
    int algoIdsArray[], int *returnAlgoCount) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasLtHandle_t, cublasComputeType_t, cudaDataType_t, cudaDataType_t,
      cudaDataType_t, cudaDataType_t, cudaDataType_t, int, int[], int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmulAlgoGetIds");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(lightHandle, computeType, scaleType, Atype, Btype, Ctype,
                  Dtype, requestedAlgoCount, algoIdsArray, returnAlgoCount);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoInit(
    cublasLtHandle_t lightHandle, cublasComputeType_t computeType,
    cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype,
    cudaDataType_t Ctype, cudaDataType_t Dtype, int algoId,
    cublasLtMatmulAlgo_t *algo) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasLtHandle_t, cublasComputeType_t, cudaDataType_t, cudaDataType_t,
      cudaDataType_t, cudaDataType_t, cudaDataType_t, int,
      cublasLtMatmulAlgo_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmulAlgoInit");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(lightHandle, computeType, scaleType, Atype, Btype, Ctype,
                  Dtype, algoId, algo);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCheck(  //
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t *algo,  ///< may point to result->algo
    cublasLtMatmulHeuristicResult_t *result) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(  //
      cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
      cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
      const cublasLtMatmulAlgo_t *,  ///< may point to result->algo
      cublasLtMatmulHeuristicResult_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLtMatmulAlgoCheck");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, algo,
                  result);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCapGetAttribute(
    const cublasLtMatmulAlgo_t *algo, cublasLtMatmulAlgoCapAttributes_t attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      const cublasLtMatmulAlgo_t *, cublasLtMatmulAlgoCapAttributes_t, void *,
      size_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatmulAlgoCapGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(algo, attr, buf, sizeInBytes, sizeWritten);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoConfigSetAttribute(
    cublasLtMatmulAlgo_t *algo, cublasLtMatmulAlgoConfigAttributes_t attr,
    const void *buf, size_t sizeInBytes) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      cublasLtMatmulAlgo_t *, cublasLtMatmulAlgoConfigAttributes_t,
      const void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatmulAlgoConfigSetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(algo, attr, buf, sizeInBytes);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoConfigGetAttribute(
    const cublasLtMatmulAlgo_t *algo, cublasLtMatmulAlgoConfigAttributes_t attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten) {
  using FuncPtr = cublasStatus_t(CUBLASWINAPI *)(
      const cublasLtMatmulAlgo_t *, cublasLtMatmulAlgoConfigAttributes_t,
      void *, size_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cublasLtMatmulAlgoConfigGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(algo, attr, buf, sizeInBytes, sizeWritten);
}

}  // extern "C"
