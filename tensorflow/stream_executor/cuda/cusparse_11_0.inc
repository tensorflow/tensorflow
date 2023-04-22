// Auto-generated, do not edit.

#define CUSPARSE_DEPRECATED(new_func)

extern "C" {

cusparseStatus_t CUSPARSEAPI cusparseCreate(cusparseHandle_t *handle) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle);
}

cusparseStatus_t CUSPARSEAPI cusparseDestroy(cusparseHandle_t handle) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle);
}

cusparseStatus_t CUSPARSEAPI cusparseGetVersion(cusparseHandle_t handle,
                                                int *version) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGetVersion");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, version);
}

cusparseStatus_t CUSPARSEAPI cusparseGetProperty(libraryPropertyType type,
                                                 int *value) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(libraryPropertyType, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGetProperty");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(type, value);
}

const char *CUSPARSEAPI cusparseGetErrorName(cusparseStatus_t status) {
  using FuncPtr = const char *(CUSPARSEAPI *)(cusparseStatus_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGetErrorName");
  if (!func_ptr) return "cusparseGetErrorName symbol not found.";
  return func_ptr(status);
}

const char *CUSPARSEAPI cusparseGetErrorString(cusparseStatus_t status) {
  using FuncPtr = const char *(CUSPARSEAPI *)(cusparseStatus_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGetErrorString");
  if (!func_ptr) return "cusparseGetErrorString symbol not found.";
  return func_ptr(status);
}

cusparseStatus_t CUSPARSEAPI cusparseSetStream(cusparseHandle_t handle,
                                               cudaStream_t streamId) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSetStream");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, streamId);
}

cusparseStatus_t CUSPARSEAPI cusparseGetStream(cusparseHandle_t handle,
                                               cudaStream_t *streamId) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, cudaStream_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGetStream");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, streamId);
}

cusparseStatus_t CUSPARSEAPI
cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t *mode) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t,
                                                  cusparsePointerMode_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGetPointerMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode);
}

cusparseStatus_t CUSPARSEAPI
cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, cusparsePointerMode_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSetPointerMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateMatDescr(cusparseMatDescr_t *descrA) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseMatDescr_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateMatDescr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(descrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseMatDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyMatDescr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(descrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseCopyMatDescr(cusparseMatDescr_t dest, const cusparseMatDescr_t src) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseMatDescr_t,
                                                  const cusparseMatDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCopyMatDescr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dest, src);
}

cusparseStatus_t CUSPARSEAPI cusparseSetMatType(cusparseMatDescr_t descrA,
                                                cusparseMatrixType_t type) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseMatDescr_t, cusparseMatrixType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSetMatType");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(descrA, type);
}

cusparseMatrixType_t CUSPARSEAPI
cusparseGetMatType(const cusparseMatDescr_t descrA) {
  using FuncPtr = cusparseMatrixType_t(CUSPARSEAPI *)(const cusparseMatDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGetMatType");
  if (!func_ptr) return cusparseMatrixType_t(-1);
  return func_ptr(descrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseMatDescr_t, cusparseFillMode_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSetMatFillMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(descrA, fillMode);
}

cusparseFillMode_t CUSPARSEAPI
cusparseGetMatFillMode(const cusparseMatDescr_t descrA) {
  using FuncPtr = cusparseFillMode_t(CUSPARSEAPI *)(const cusparseMatDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGetMatFillMode");
  if (!func_ptr) return cusparseFillMode_t(-1);
  return func_ptr(descrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseMatDescr_t, cusparseDiagType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSetMatDiagType");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(descrA, diagType);
}

cusparseDiagType_t CUSPARSEAPI
cusparseGetMatDiagType(const cusparseMatDescr_t descrA) {
  using FuncPtr = cusparseDiagType_t(CUSPARSEAPI *)(const cusparseMatDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGetMatDiagType");
  if (!func_ptr) return cusparseDiagType_t(-1);
  return func_ptr(descrA);
}

cusparseStatus_t CUSPARSEAPI cusparseSetMatIndexBase(cusparseMatDescr_t descrA,
                                                     cusparseIndexBase_t base) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseMatDescr_t, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSetMatIndexBase");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(descrA, base);
}

cusparseIndexBase_t CUSPARSEAPI
cusparseGetMatIndexBase(const cusparseMatDescr_t descrA) {
  using FuncPtr = cusparseIndexBase_t(CUSPARSEAPI *)(const cusparseMatDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGetMatIndexBase");
  if (!func_ptr) return cusparseIndexBase_t(-1);
  return func_ptr(descrA);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseCreateCsrsv2Info(csrsv2Info_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csrsv2Info_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateCsrsv2Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrsv2Info(csrsv2Info_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csrsv2Info_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyCsrsv2Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateCsric02Info(csric02Info_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csric02Info_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateCsric02Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsric02Info(csric02Info_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csric02Info_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyCsric02Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateBsric02Info(bsric02Info_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(bsric02Info_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateBsric02Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseDestroyBsric02Info(bsric02Info_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(bsric02Info_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyBsric02Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateCsrilu02Info(csrilu02Info_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csrilu02Info_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateCsrilu02Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrilu02Info(csrilu02Info_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csrilu02Info_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyCsrilu02Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateBsrilu02Info(bsrilu02Info_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(bsrilu02Info_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateBsrilu02Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrilu02Info(bsrilu02Info_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(bsrilu02Info_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyBsrilu02Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsv2Info(bsrsv2Info_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(bsrsv2Info_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateBsrsv2Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsv2Info(bsrsv2Info_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(bsrsv2Info_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyBsrsv2Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsm2Info(bsrsm2Info_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(bsrsm2Info_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateBsrsm2Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsm2Info(bsrsm2Info_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(bsrsm2Info_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyBsrsm2Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateCsru2csrInfo(csru2csrInfo_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csru2csrInfo_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateCsru2csrInfo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsru2csrInfo(csru2csrInfo_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csru2csrInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyCsru2csrInfo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateColorInfo(cusparseColorInfo_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseColorInfo_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateColorInfo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyColorInfo(cusparseColorInfo_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseColorInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyColorInfo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseSetColorAlgs(cusparseColorInfo_t info,
                                                  cusparseColorAlg_t alg) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseColorInfo_t, cusparseColorAlg_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSetColorAlgs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info, alg);
}

cusparseStatus_t CUSPARSEAPI cusparseGetColorAlgs(cusparseColorInfo_t info,
                                                  cusparseColorAlg_t *alg) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseColorInfo_t,
                                                  cusparseColorAlg_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGetColorAlgs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info, alg);
}

cusparseStatus_t CUSPARSEAPI cusparseCreatePruneInfo(pruneInfo_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(pruneInfo_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreatePruneInfo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseDestroyPruneInfo(pruneInfo_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(pruneInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyPruneInfo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

CUSPARSE_DEPRECATED(cusparseAxpby)
cusparseStatus_t CUSPARSEAPI cusparseSaxpyi(cusparseHandle_t handle, int nnz,
                                            const float *alpha,
                                            const float *xVal, const int *xInd,
                                            float *y,
                                            cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const float *, const float *, const int *, float *,
      cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSaxpyi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, alpha, xVal, xInd, y, idxBase);
}

CUSPARSE_DEPRECATED(cusparseAxpby)
cusparseStatus_t CUSPARSEAPI cusparseDaxpyi(cusparseHandle_t handle, int nnz,
                                            const double *alpha,
                                            const double *xVal, const int *xInd,
                                            double *y,
                                            cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const double *, const double *, const int *,
      double *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDaxpyi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, alpha, xVal, xInd, y, idxBase);
}

CUSPARSE_DEPRECATED(cusparseAxpby)
cusparseStatus_t CUSPARSEAPI cusparseCaxpyi(cusparseHandle_t handle, int nnz,
                                            const cuComplex *alpha,
                                            const cuComplex *xVal,
                                            const int *xInd, cuComplex *y,
                                            cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cuComplex *, const cuComplex *, const int *,
      cuComplex *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCaxpyi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, alpha, xVal, xInd, y, idxBase);
}

CUSPARSE_DEPRECATED(cusparseAxpby)
cusparseStatus_t CUSPARSEAPI cusparseZaxpyi(cusparseHandle_t handle, int nnz,
                                            const cuDoubleComplex *alpha,
                                            const cuDoubleComplex *xVal,
                                            const int *xInd, cuDoubleComplex *y,
                                            cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cuDoubleComplex *, const cuDoubleComplex *,
      const int *, cuDoubleComplex *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZaxpyi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, alpha, xVal, xInd, y, idxBase);
}

CUSPARSE_DEPRECATED(cusparseGather)
cusparseStatus_t CUSPARSEAPI cusparseSgthr(cusparseHandle_t handle, int nnz,
                                           const float *y, float *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const float *, float *, const int *,
      cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgthr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, y, xVal, xInd, idxBase);
}

CUSPARSE_DEPRECATED(cusparseGather)
cusparseStatus_t CUSPARSEAPI cusparseDgthr(cusparseHandle_t handle, int nnz,
                                           const double *y, double *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const double *, double *, const int *,
      cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgthr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, y, xVal, xInd, idxBase);
}

CUSPARSE_DEPRECATED(cusparseGather)
cusparseStatus_t CUSPARSEAPI cusparseCgthr(cusparseHandle_t handle, int nnz,
                                           const cuComplex *y, cuComplex *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cuComplex *, cuComplex *, const int *,
      cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgthr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, y, xVal, xInd, idxBase);
}

CUSPARSE_DEPRECATED(cusparseGather)
cusparseStatus_t CUSPARSEAPI cusparseZgthr(cusparseHandle_t handle, int nnz,
                                           const cuDoubleComplex *y,
                                           cuDoubleComplex *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cuDoubleComplex *, cuDoubleComplex *,
      const int *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgthr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, y, xVal, xInd, idxBase);
}

CUSPARSE_DEPRECATED(cusparseGather)
cusparseStatus_t CUSPARSEAPI cusparseSgthrz(cusparseHandle_t handle, int nnz,
                                            float *y, float *xVal,
                                            const int *xInd,
                                            cusparseIndexBase_t idxBase) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, int, float *, float *,
                                      const int *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgthrz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, y, xVal, xInd, idxBase);
}

CUSPARSE_DEPRECATED(cusparseGather)
cusparseStatus_t CUSPARSEAPI cusparseDgthrz(cusparseHandle_t handle, int nnz,
                                            double *y, double *xVal,
                                            const int *xInd,
                                            cusparseIndexBase_t idxBase) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, int, double *, double *,
                                      const int *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgthrz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, y, xVal, xInd, idxBase);
}

CUSPARSE_DEPRECATED(cusparseGather)
cusparseStatus_t CUSPARSEAPI cusparseCgthrz(cusparseHandle_t handle, int nnz,
                                            cuComplex *y, cuComplex *xVal,
                                            const int *xInd,
                                            cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cuComplex *, cuComplex *, const int *,
      cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgthrz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, y, xVal, xInd, idxBase);
}

CUSPARSE_DEPRECATED(cusparseGather)
cusparseStatus_t CUSPARSEAPI cusparseZgthrz(cusparseHandle_t handle, int nnz,
                                            cuDoubleComplex *y,
                                            cuDoubleComplex *xVal,
                                            const int *xInd,
                                            cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cuDoubleComplex *, cuDoubleComplex *, const int *,
      cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgthrz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, y, xVal, xInd, idxBase);
}

CUSPARSE_DEPRECATED(cusparseScatter)
cusparseStatus_t CUSPARSEAPI cusparseSsctr(cusparseHandle_t handle, int nnz,
                                           const float *xVal, const int *xInd,
                                           float *y,
                                           cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, int,
                                                  const float *, const int *,
                                                  float *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSsctr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, xVal, xInd, y, idxBase);
}

CUSPARSE_DEPRECATED(cusparseScatter)
cusparseStatus_t CUSPARSEAPI cusparseDsctr(cusparseHandle_t handle, int nnz,
                                           const double *xVal, const int *xInd,
                                           double *y,
                                           cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const double *, const int *, double *,
      cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDsctr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, xVal, xInd, y, idxBase);
}

CUSPARSE_DEPRECATED(cusparseScatter)
cusparseStatus_t CUSPARSEAPI cusparseCsctr(cusparseHandle_t handle, int nnz,
                                           const cuComplex *xVal,
                                           const int *xInd, cuComplex *y,
                                           cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cuComplex *, const int *, cuComplex *,
      cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCsctr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, xVal, xInd, y, idxBase);
}

CUSPARSE_DEPRECATED(cusparseScatter)
cusparseStatus_t CUSPARSEAPI cusparseZsctr(cusparseHandle_t handle, int nnz,
                                           const cuDoubleComplex *xVal,
                                           const int *xInd, cuDoubleComplex *y,
                                           cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cuDoubleComplex *, const int *,
      cuDoubleComplex *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZsctr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, xVal, xInd, y, idxBase);
}

CUSPARSE_DEPRECATED(cusparseRot)
cusparseStatus_t CUSPARSEAPI cusparseSroti(cusparseHandle_t handle, int nnz,
                                           float *xVal, const int *xInd,
                                           float *y, const float *c,
                                           const float *s,
                                           cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, float *, const int *, float *, const float *,
      const float *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSroti");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, xVal, xInd, y, c, s, idxBase);
}

CUSPARSE_DEPRECATED(cusparseRot)
cusparseStatus_t CUSPARSEAPI cusparseDroti(cusparseHandle_t handle, int nnz,
                                           double *xVal, const int *xInd,
                                           double *y, const double *c,
                                           const double *s,
                                           cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, double *, const int *, double *, const double *,
      const double *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDroti");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nnz, xVal, xInd, y, c, s, idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m,
               int n, const float *alpha, const float *A, int lda, int nnz,
               const float *xVal, const int *xInd, const float *beta, float *y,
               cusparseIndexBase_t idxBase, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const float *,
      const float *, int, int, const float *, const int *, const float *,
      float *, cusparseIndexBase_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgemvi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y,
                  idxBase, pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA,
                          int m, int n, int nnz, int *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgemvi_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, n, nnz, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m,
               int n, const double *alpha, const double *A, int lda, int nnz,
               const double *xVal, const int *xInd, const double *beta,
               double *y, cusparseIndexBase_t idxBase, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const double *,
      const double *, int, int, const double *, const int *, const double *,
      double *, cusparseIndexBase_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgemvi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y,
                  idxBase, pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA,
                          int m, int n, int nnz, int *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgemvi_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, n, nnz, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseCgemvi(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int n,
    const cuComplex *alpha, const cuComplex *A, int lda, int nnz,
    const cuComplex *xVal, const int *xInd, const cuComplex *beta, cuComplex *y,
    cusparseIndexBase_t idxBase, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cuComplex *,
      const cuComplex *, int, int, const cuComplex *, const int *,
      const cuComplex *, cuComplex *, cusparseIndexBase_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgemvi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y,
                  idxBase, pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA,
                          int m, int n, int nnz, int *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgemvi_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, n, nnz, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseZgemvi(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, int nnz,
    const cuDoubleComplex *xVal, const int *xInd, const cuDoubleComplex *beta,
    cuDoubleComplex *y, cusparseIndexBase_t idxBase, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cuDoubleComplex *,
      const cuDoubleComplex *, int, int, const cuDoubleComplex *, const int *,
      const cuDoubleComplex *, cuDoubleComplex *, cusparseIndexBase_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgemvi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y,
                  idxBase, pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA,
                          int m, int n, int nnz, int *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgemvi_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, n, nnz, pBufferSize);
}

CUSPARSE_DEPRECATED(cusparseSpMV)
cusparseStatus_t CUSPARSEAPI cusparseCsrmvEx_bufferSize(
    cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA,
    int m, int n, int nnz, const void *alpha, cudaDataType alphatype,
    const cusparseMatDescr_t descrA, const void *csrValA,
    cudaDataType csrValAtype, const int *csrRowPtrA, const int *csrColIndA,
    const void *x, cudaDataType xtype, const void *beta, cudaDataType betatype,
    void *y, cudaDataType ytype, cudaDataType executiontype,
    size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseAlgMode_t, cusparseOperation_t, int, int, int,
      const void *, cudaDataType, const cusparseMatDescr_t, const void *,
      cudaDataType, const int *, const int *, const void *, cudaDataType,
      const void *, cudaDataType, void *, cudaDataType, cudaDataType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCsrmvEx_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA,
                  csrValA, csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta,
                  betatype, y, ytype, executiontype, bufferSizeInBytes);
}

CUSPARSE_DEPRECATED(cusparseSpMV)
cusparseStatus_t CUSPARSEAPI cusparseCsrmvEx(
    cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA,
    int m, int n, int nnz, const void *alpha, cudaDataType alphatype,
    const cusparseMatDescr_t descrA, const void *csrValA,
    cudaDataType csrValAtype, const int *csrRowPtrA, const int *csrColIndA,
    const void *x, cudaDataType xtype, const void *beta, cudaDataType betatype,
    void *y, cudaDataType ytype, cudaDataType executiontype, void *buffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseAlgMode_t, cusparseOperation_t, int, int, int,
      const void *, cudaDataType, const cusparseMatDescr_t, const void *,
      cudaDataType, const int *, const int *, const void *, cudaDataType,
      const void *, cudaDataType, void *, cudaDataType, cudaDataType, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCsrmvEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA,
                  csrValA, csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta,
                  betatype, y, ytype, executiontype, buffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrmv(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nb, int nnzb, const float *alpha,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    const float *x, const float *beta, float *y) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
      const float *, const cusparseMatDescr_t, const float *, const int *,
      const int *, int, const float *, const float *, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrmv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim,
                  x, beta, y);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrmv(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nb, int nnzb, const double *alpha,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    const double *x, const double *beta, double *y) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
      const double *, const cusparseMatDescr_t, const double *, const int *,
      const int *, int, const double *, const double *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrmv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim,
                  x, beta, y);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
               cusparseOperation_t transA, int mb, int nb, int nnzb,
               const cuComplex *alpha, const cusparseMatDescr_t descrA,
               const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA, int blockDim, const cuComplex *x,
               const cuComplex *beta, cuComplex *y) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
      const cuComplex *, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, int, const cuComplex *, const cuComplex *,
      cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrmv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim,
                  x, beta, y);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrmv(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nb, int nnzb,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, int blockDim, const cuDoubleComplex *x,
    const cuDoubleComplex *beta, cuDoubleComplex *y) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
      const cuDoubleComplex *, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, int,
      const cuDoubleComplex *, const cuDoubleComplex *, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrmv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim,
                  x, beta, y);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                cusparseOperation_t transA, int sizeOfMask, int mb, int nb,
                int nnzb, const float *alpha, const cusparseMatDescr_t descrA,
                const float *bsrSortedValA, const int *bsrSortedMaskPtrA,
                const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                const int *bsrSortedColIndA, int blockDim, const float *x,
                const float *beta, float *y) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
      int, const float *, const cusparseMatDescr_t, const float *, const int *,
      const int *, const int *, const int *, int, const float *, const float *,
      float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrxmv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA,
                  bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                cusparseOperation_t transA, int sizeOfMask, int mb, int nb,
                int nnzb, const double *alpha, const cusparseMatDescr_t descrA,
                const double *bsrSortedValA, const int *bsrSortedMaskPtrA,
                const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                const int *bsrSortedColIndA, int blockDim, const double *x,
                const double *beta, double *y) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
      int, const double *, const cusparseMatDescr_t, const double *,
      const int *, const int *, const int *, const int *, int, const double *,
      const double *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrxmv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA,
                  bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrxmv(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb,
    const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *bsrSortedValA, const int *bsrSortedMaskPtrA,
    const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
    const int *bsrSortedColIndA, int blockDim, const cuComplex *x,
    const cuComplex *beta, cuComplex *y) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
      int, const cuComplex *, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, const int *, const int *, int,
      const cuComplex *, const cuComplex *, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrxmv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA,
                  bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrxmv(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedValA, const int *bsrSortedMaskPtrA,
    const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
    const int *bsrSortedColIndA, int blockDim, const cuDoubleComplex *x,
    const cuDoubleComplex *beta, cuDoubleComplex *y) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
      int, const cuDoubleComplex *, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, const int *,
      const int *, int, const cuDoubleComplex *, const cuDoubleComplex *,
      cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrxmv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA,
                  bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle,
                                                       csrsv2Info_t info,
                                                       int *position) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, csrsv2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcsrsv2_zeroPivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, position);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      float *, const int *, const int *, csrsv2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrsv2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      double *, const int *, const int *, csrsv2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrsv2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      cuComplex *, const int *, const int *, csrsv2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrsv2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, const int *, csrsv2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrsv2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      float *, const int *, const int *, csrsv2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      double *, const int *, const int *, csrsv2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      cuComplex *, const int *, const int *, csrsv2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, const int *, csrsv2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      const float *, const int *, const int *, csrsv2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrsv2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      const double *, const int *, const int *, csrsv2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrsv2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      const cuComplex *, const int *, const int *, csrsv2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrsv2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, csrsv2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrsv2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_solve(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const float *alpha, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrsv2Info_t info, const float *f, float *x,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const float *,
      const cusparseMatDescr_t, const float *, const int *, const int *,
      csrsv2Info_t, const float *, float *, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrsv2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, alpha, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy,
                  pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_solve(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const double *alpha, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrsv2Info_t info, const double *f, double *x,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const double *,
      const cusparseMatDescr_t, const double *, const int *, const int *,
      csrsv2Info_t, const double *, double *, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrsv2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, alpha, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy,
                  pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_solve(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrsv2Info_t info, const cuComplex *f,
    cuComplex *x, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cuComplex *,
      const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
      csrsv2Info_t, const cuComplex *, cuComplex *, cusparseSolvePolicy_t,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrsv2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, alpha, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy,
                  pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpSV)
cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_solve(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrsv2Info_t info, const cuDoubleComplex *f,
    cuDoubleComplex *x, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, int, int, const cuDoubleComplex *,
      const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
      const int *, csrsv2Info_t, const cuDoubleComplex *, cuDoubleComplex *,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrsv2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transA, m, nnz, alpha, descrA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle,
                                                       bsrsv2Info_t info,
                                                       int *position) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, bsrsv2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXbsrsv2_zeroPivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, position);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, float *, const int *, const int *, int,
      bsrsv2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrsv2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, double *, const int *, const int *, int,
      bsrsv2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrsv2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, cuComplex *, const int *, const int *, int,
      bsrsv2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrsv2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, cuDoubleComplex *, const int *, const int *,
      int, bsrsv2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrsv2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
    bsrsv2Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, float *, const int *, const int *, int,
      bsrsv2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockSize, info,
                  pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
    bsrsv2Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, double *, const int *, const int *, int,
      bsrsv2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockSize, info,
                  pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
    bsrsv2Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, cuComplex *, const int *, const int *, int,
      bsrsv2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockSize, info,
                  pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
    bsrsv2Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, cuDoubleComplex *, const int *, const int *,
      int, bsrsv2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockSize, info,
                  pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, const float *, const int *, const int *, int,
      bsrsv2Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrsv2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, const double *, const int *, const int *, int,
      bsrsv2Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrsv2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
      int, bsrsv2Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrsv2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
      const int *, int, bsrsv2Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrsv2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb, const float *alpha,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, const float *f, float *x, cusparseSolvePolicy_t policy,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const float *, const cusparseMatDescr_t, const float *, const int *,
      const int *, int, bsrsv2Info_t, const float *, float *,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrsv2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x,
                  policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb, const double *alpha,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, const double *f, double *x, cusparseSolvePolicy_t policy,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const double *, const cusparseMatDescr_t, const double *, const int *,
      const int *, int, bsrsv2Info_t, const double *, double *,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrsv2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x,
                  policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb, const cuComplex *alpha,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, const cuComplex *f, cuComplex *x,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cuComplex *, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, int, bsrsv2Info_t, const cuComplex *,
      cuComplex *, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrsv2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x,
                  policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb, const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, const cuDoubleComplex *f, cuDoubleComplex *x,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
      const cuDoubleComplex *, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, int, bsrsv2Info_t,
      const cuDoubleComplex *, cuDoubleComplex *, cusparseSolvePolicy_t,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrsv2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x,
                  policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrmm(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
    int kb, int nnzb, const float *alpha, const cusparseMatDescr_t descrA,
    const float *bsrSortedValA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, const int blockSize, const float *B,
    const int ldb, const float *beta, float *C, int ldc) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, int, const float *,
      const cusparseMatDescr_t, const float *, const int *, const int *,
      const int, const float *, const int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrmm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize,
                  B, ldb, beta, C, ldc);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrmm(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
    int kb, int nnzb, const double *alpha, const cusparseMatDescr_t descrA,
    const double *bsrSortedValA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, const int blockSize, const double *B,
    const int ldb, const double *beta, double *C, int ldc) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, int, const double *,
      const cusparseMatDescr_t, const double *, const int *, const int *,
      const int, const double *, const int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrmm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize,
                  B, ldb, beta, C, ldc);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrmm(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
    int kb, int nnzb, const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, const int blockSize, const cuComplex *B,
    const int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, int, const cuComplex *,
      const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
      const int, const cuComplex *, const int, const cuComplex *, cuComplex *,
      int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrmm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize,
                  B, ldb, beta, C, ldc);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrmm(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
    int kb, int nnzb, const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
    const int blockSize, const cuDoubleComplex *B, const int ldb,
    const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, int, const cuDoubleComplex *,
      const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
      const int *, const int, const cuDoubleComplex *, const int,
      const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrmm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                  bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize,
                  B, ldb, beta, C, ldc);
}

CUSPARSE_DEPRECATED(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI cusparseSgemmi(
    cusparseHandle_t handle, int m, int n, int k, int nnz, const float *alpha,
    const float *A, int lda, const float *cscValB, const int *cscColPtrB,
    const int *cscRowIndB, const float *beta, float *C, int ldc) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, int, const float *, const float *, int,
      const float *, const int *, const int *, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgemmi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB,
                  cscRowIndB, beta, C, ldc);
}

CUSPARSE_DEPRECATED(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI cusparseDgemmi(
    cusparseHandle_t handle, int m, int n, int k, int nnz, const double *alpha,
    const double *A, int lda, const double *cscValB, const int *cscColPtrB,
    const int *cscRowIndB, const double *beta, double *C, int ldc) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, int, const double *, const double *, int,
      const double *, const int *, const int *, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgemmi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB,
                  cscRowIndB, beta, C, ldc);
}

CUSPARSE_DEPRECATED(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI cusparseCgemmi(
    cusparseHandle_t handle, int m, int n, int k, int nnz,
    const cuComplex *alpha, const cuComplex *A, int lda,
    const cuComplex *cscValB, const int *cscColPtrB, const int *cscRowIndB,
    const cuComplex *beta, cuComplex *C, int ldc) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, int, const cuComplex *,
      const cuComplex *, int, const cuComplex *, const int *, const int *,
      const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgemmi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB,
                  cscRowIndB, beta, C, ldc);
}

CUSPARSE_DEPRECATED(cusparseSpMM)
cusparseStatus_t CUSPARSEAPI
cusparseZgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz,
               const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
               const cuDoubleComplex *cscValB, const int *cscColPtrB,
               const int *cscRowIndB, const cuDoubleComplex *beta,
               cuDoubleComplex *C, int ldc) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, int, const cuDoubleComplex *,
      const cuDoubleComplex *, int, const cuDoubleComplex *, const int *,
      const int *, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgemmi");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB,
                  cscRowIndB, beta, C, ldc);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateCsrsm2Info(csrsm2Info_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csrsm2Info_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateCsrsm2Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrsm2Info(csrsm2Info_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csrsm2Info_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyCsrsm2Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrsm2_zeroPivot(cusparseHandle_t handle,
                                                       csrsm2Info_t info,
                                                       int *position) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, csrsm2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcsrsm2_zeroPivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, position);
}

cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B,
    int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
    size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const float *, const cusparseMatDescr_t, const float *, const int *,
      const int *, const float *, int, csrsm2Info_t, cusparseSolvePolicy_t,
      size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrsm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *B,
    int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
    size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const double *, const cusparseMatDescr_t, const double *,
      const int *, const int *, const double *, int, csrsm2Info_t,
      cusparseSolvePolicy_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrsm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cuComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const cuComplex *, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, const cuComplex *, int, csrsm2Info_t,
      cusparseSolvePolicy_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrsm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cuDoubleComplex *B, int ldb,
    csrsm2Info_t info, cusparseSolvePolicy_t policy, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const cuDoubleComplex *, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *,
      const cuDoubleComplex *, int, csrsm2Info_t, cusparseSolvePolicy_t,
      size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrsm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_analysis(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B,
    int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const float *, const cusparseMatDescr_t, const float *, const int *,
      const int *, const float *, int, csrsm2Info_t, cusparseSolvePolicy_t,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrsm2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_analysis(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *B,
    int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const double *, const cusparseMatDescr_t, const double *,
      const int *, const int *, const double *, int, csrsm2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrsm2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_analysis(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cuComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const cuComplex *, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, const cuComplex *, int, csrsm2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrsm2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_analysis(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cuDoubleComplex *B, int ldb,
    csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const cuDoubleComplex *, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *,
      const cuDoubleComplex *, int, csrsm2Info_t, cusparseSolvePolicy_t,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrsm2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_solve(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float *B, int ldb,
    csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const float *, const cusparseMatDescr_t, const float *, const int *,
      const int *, float *, int, csrsm2Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrsm2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_solve(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, double *B,
    int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const double *, const cusparseMatDescr_t, const double *,
      const int *, const int *, double *, int, csrsm2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrsm2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_solve(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, cuComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const cuComplex *, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, cuComplex *, int, csrsm2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrsm2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_solve(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, cuDoubleComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
      int, const cuDoubleComplex *, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, cuDoubleComplex *, int,
      csrsm2Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrsm2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle,
                                                       bsrsm2Info_t info,
                                                       int *position) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, bsrsm2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXbsrsm2_zeroPivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, position);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t, float *,
      const int *, const int *, int, bsrsm2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrsm2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t, double *,
      const int *, const int *, int, bsrsm2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrsm2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t, cuComplex *,
      const int *, const int *, int, bsrsm2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrsm2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, const int *, int, bsrsm2Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrsm2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t, float *,
      const int *, const int *, int, bsrsm2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrsm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transB, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t, double *,
      const int *, const int *, int, bsrsm2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrsm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transB, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t, cuComplex *,
      const int *, const int *, int, bsrsm2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrsm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transB, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, const int *, int, bsrsm2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrsm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transB, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, const float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t,
      const float *, const int *, const int *, int, bsrsm2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrsm2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, const double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t,
      const double *, const int *, const int *, int, bsrsm2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrsm2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, const cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t,
      const cuComplex *, const int *, const int *, int, bsrsm2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrsm2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, int, bsrsm2Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrsm2_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const float *alpha, const cusparseMatDescr_t descrA,
    const float *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
    const float *B, int ldb, float *X, int ldx, cusparseSolvePolicy_t policy,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const float *,
      const cusparseMatDescr_t, const float *, const int *, const int *, int,
      bsrsm2Info_t, const float *, int, float *, int, cusparseSolvePolicy_t,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrsm2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, B, ldb, X, ldx, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const double *alpha, const cusparseMatDescr_t descrA,
    const double *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
    const double *B, int ldb, double *X, int ldx, cusparseSolvePolicy_t policy,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const double *,
      const cusparseMatDescr_t, const double *, const int *, const int *, int,
      bsrsm2Info_t, const double *, int, double *, int, cusparseSolvePolicy_t,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrsm2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, B, ldb, X, ldx, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
    const cuComplex *B, int ldb, cuComplex *X, int ldx,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cuComplex *,
      const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
      int, bsrsm2Info_t, const cuComplex *, int, cuComplex *, int,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrsm2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, B, ldb, X, ldx, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
    const cuDoubleComplex *B, int ldb, cuDoubleComplex *X, int ldx,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
      cusparseOperation_t, int, int, int, const cuDoubleComplex *,
      const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
      const int *, int, bsrsm2Info_t, const cuDoubleComplex *, int,
      cuDoubleComplex *, int, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrsm2_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA,
                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                  info, B, ldb, X, ldx, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_numericBoost(
    cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double *tol,
    float *boost_val) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, csrilu02Info_t, int, double *, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrilu02_numericBoost");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_numericBoost(
    cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double *tol,
    double *boost_val) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, csrilu02Info_t, int, double *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrilu02_numericBoost");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_numericBoost(
    cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double *tol,
    cuComplex *boost_val) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, csrilu02Info_t, int, double *, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrilu02_numericBoost");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_numericBoost(
    cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double *tol,
    cuDoubleComplex *boost_val) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, csrilu02Info_t, int, double *, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrilu02_numericBoost");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02_zeroPivot(
    cusparseHandle_t handle, csrilu02Info_t info, int *position) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, csrilu02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcsrilu02_zeroPivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, position);
}

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, float *,
      const int *, const int *, csrilu02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrilu02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, double *,
      const int *, const int *, csrilu02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrilu02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
      const int *, const int *, csrilu02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrilu02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
      const int *, const int *, csrilu02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrilu02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    float *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd,
    csrilu02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, float *,
      const int *, const int *, csrilu02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrilu02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                  csrSortedColInd, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    double *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, double *,
      const int *, const int *, csrilu02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrilu02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                  csrSortedColInd, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuComplex *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
      const int *, const int *, csrilu02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrilu02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                  csrSortedColInd, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
      const int *, const int *, csrilu02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrilu02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                  csrSortedColInd, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrilu02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrilu02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrilu02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, csrilu02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrilu02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    float *csrSortedValA_valM, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, float *,
      const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrilu02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    double *csrSortedValA_valM, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, double *,
      const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrilu02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuComplex *csrSortedValA_valM, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
      const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrilu02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrSortedValA_valM, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
      const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrilu02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_numericBoost(
    cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double *tol,
    float *boost_val) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, bsrilu02Info_t, int, double *, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrilu02_numericBoost");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_numericBoost(
    cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double *tol,
    double *boost_val) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, bsrilu02Info_t, int, double *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrilu02_numericBoost");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_numericBoost(
    cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double *tol,
    cuComplex *boost_val) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, bsrilu02Info_t, int, double *, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrilu02_numericBoost");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_numericBoost(
    cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double *tol,
    cuDoubleComplex *boost_val) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, bsrilu02Info_t, int, double *, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrilu02_numericBoost");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t CUSPARSEAPI cusparseXbsrilu02_zeroPivot(
    cusparseHandle_t handle, bsrilu02Info_t info, int *position) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, bsrilu02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXbsrilu02_zeroPivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, position);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      float *, const int *, const int *, int, bsrilu02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrilu02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      double *, const int *, const int *, int, bsrilu02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrilu02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuComplex *, const int *, const int *, int, bsrilu02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrilu02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, const int *, int, bsrilu02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrilu02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrilu02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      float *, const int *, const int *, int, bsrilu02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrilu02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockSize, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrilu02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      double *, const int *, const int *, int, bsrilu02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrilu02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockSize, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrilu02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuComplex *, const int *, const int *, int, bsrilu02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrilu02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockSize, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrilu02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, const int *, int, bsrilu02Info_t,
      size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrilu02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockSize, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      float *, const int *, const int *, int, bsrilu02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrilu02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      double *, const int *, const int *, int, bsrilu02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrilu02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuComplex *, const int *, const int *, int, bsrilu02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrilu02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, const int *, int, bsrilu02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrilu02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      float *, const int *, const int *, int, bsrilu02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsrilu02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      double *, const int *, const int *, int, bsrilu02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsrilu02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuComplex *, const int *, const int *, int, bsrilu02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsrilu02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, const int *, int, bsrilu02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsrilu02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsric02_zeroPivot(cusparseHandle_t handle,
                                                        csric02Info_t info,
                                                        int *position) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, csric02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcsric02_zeroPivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, position);
}

cusparseStatus_t CUSPARSEAPI cusparseScsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, float *,
      const int *, const int *, csric02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsric02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, double *,
      const int *, const int *, csric02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsric02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
      const int *, const int *, csric02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsric02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
      const int *, const int *, csric02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsric02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseScsric02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    float *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd,
    csric02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, float *,
      const int *, const int *, csric02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsric02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                  csrSortedColInd, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsric02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    double *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, double *,
      const int *, const int *, csric02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsric02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                  csrSortedColInd, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsric02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuComplex *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
      const int *, const int *, csric02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsric02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                  csrSortedColInd, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsric02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
      const int *, const int *, csric02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsric02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                  csrSortedColInd, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseScsric02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsric02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsric02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsric02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsric02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsric02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsric02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, csric02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsric02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseScsric02(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    float *csrSortedValA_valM, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, float *,
      const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsric02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsric02(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    double *csrSortedValA_valM, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, double *,
      const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsric02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsric02(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuComplex *csrSortedValA_valM, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
      const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsric02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsric02(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrSortedValA_valM, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
      const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsric02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                  csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseXbsric02_zeroPivot(cusparseHandle_t handle,
                                                        bsric02Info_t info,
                                                        int *position) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, bsric02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXbsric02_zeroPivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, position);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsric02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      float *, const int *, const int *, int, bsric02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsric02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsric02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      double *, const int *, const int *, int, bsric02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsric02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsric02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuComplex *, const int *, const int *, int, bsric02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsric02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsric02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, const int *, int, bsric02Info_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsric02_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsric02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsric02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      float *, const int *, const int *, int, bsric02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsric02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockSize, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsric02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsric02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      double *, const int *, const int *, int, bsric02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsric02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockSize, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsric02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsric02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuComplex *, const int *, const int *, int, bsric02Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsric02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockSize, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsric02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsric02Info_t info, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, const int *, int, bsric02Info_t,
      size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsric02_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockSize, info, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const float *, const int *, const int *, int, bsric02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsric02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pInputBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const double *, const int *, const int *, int, bsric02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsric02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pInputBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuComplex *, const int *, const int *, int, bsric02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsric02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pInputBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, int, bsric02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsric02_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pInputBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsric02(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      float *, const int *, const int *, int, bsric02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsric02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsric02(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      double *, const int *, const int *, int, bsric02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsric02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsric02(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuComplex *, const int *, const int *, int, bsric02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsric02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsric02(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, const int *, int, bsric02Info_t,
      cusparseSolvePolicy_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsric02");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, blockDim, info, policy, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *dl, const float *d,
    const float *du, const float *B, int ldb, size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, const float *, const float *,
      const float *, int, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgtsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *dl, const double *d,
    const double *du, const double *B, int ldb, size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, const double *,
      const double *, const double *, int, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgtsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuComplex *dl,
    const cuComplex *d, const cuComplex *du, const cuComplex *B, int ldb,
    size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuComplex *, const cuComplex *,
      const cuComplex *, const cuComplex *, int, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgtsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
    const cuDoubleComplex *d, const cuDoubleComplex *du,
    const cuDoubleComplex *B, int ldb, size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuDoubleComplex *,
      const cuDoubleComplex *, const cuDoubleComplex *, const cuDoubleComplex *,
      int, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgtsv2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2(cusparseHandle_t handle, int m,
                                            int n, const float *dl,
                                            const float *d, const float *du,
                                            float *B, int ldb, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, const float *, const float *,
      float *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgtsv2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2(cusparseHandle_t handle, int m,
                                            int n, const double *dl,
                                            const double *d, const double *du,
                                            double *B, int ldb, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, const double *,
      const double *, double *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgtsv2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2(cusparseHandle_t handle, int m,
                                            int n, const cuComplex *dl,
                                            const cuComplex *d,
                                            const cuComplex *du, cuComplex *B,
                                            int ldb, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuComplex *, const cuComplex *,
      const cuComplex *, cuComplex *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgtsv2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2(cusparseHandle_t handle, int m,
                                            int n, const cuDoubleComplex *dl,
                                            const cuDoubleComplex *d,
                                            const cuDoubleComplex *du,
                                            cuDoubleComplex *B, int ldb,
                                            void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuDoubleComplex *,
      const cuDoubleComplex *, const cuDoubleComplex *, cuDoubleComplex *, int,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgtsv2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *dl, const float *d,
    const float *du, const float *B, int ldb, size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, const float *, const float *,
      const float *, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSgtsv2_nopivot_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *dl, const double *d,
    const double *du, const double *B, int ldb, size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, const double *,
      const double *, const double *, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDgtsv2_nopivot_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuComplex *dl,
    const cuComplex *d, const cuComplex *du, const cuComplex *B, int ldb,
    size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuComplex *, const cuComplex *,
      const cuComplex *, const cuComplex *, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseCgtsv2_nopivot_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
    const cuDoubleComplex *d, const cuDoubleComplex *du,
    const cuDoubleComplex *B, int ldb, size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuDoubleComplex *,
      const cuDoubleComplex *, const cuDoubleComplex *, const cuDoubleComplex *,
      int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseZgtsv2_nopivot_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot(
    cusparseHandle_t handle, int m, int n, const float *dl, const float *d,
    const float *du, float *B, int ldb, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, const float *, const float *,
      float *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgtsv2_nopivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot(
    cusparseHandle_t handle, int m, int n, const double *dl, const double *d,
    const double *du, double *B, int ldb, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, const double *,
      const double *, double *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgtsv2_nopivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot(
    cusparseHandle_t handle, int m, int n, const cuComplex *dl,
    const cuComplex *d, const cuComplex *du, cuComplex *B, int ldb,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuComplex *, const cuComplex *,
      const cuComplex *, cuComplex *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgtsv2_nopivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
    const cuDoubleComplex *d, const cuDoubleComplex *du, cuDoubleComplex *B,
    int ldb, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuDoubleComplex *,
      const cuDoubleComplex *, const cuDoubleComplex *, cuDoubleComplex *, int,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgtsv2_nopivot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle, int m, const float *dl, const float *d,
    const float *du, const float *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const float *, const float *, const float *,
      const float *, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSgtsv2StridedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, dl, d, du, x, batchCount, batchStride,
                  bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle, int m, const double *dl, const double *d,
    const double *du, const double *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const double *, const double *, const double *,
      const double *, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDgtsv2StridedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, dl, d, du, x, batchCount, batchStride,
                  bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle, int m, const cuComplex *dl, const cuComplex *d,
    const cuComplex *du, const cuComplex *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cuComplex *, const cuComplex *,
      const cuComplex *, const cuComplex *, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseCgtsv2StridedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, dl, d, du, x, batchCount, batchStride,
                  bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle, int m, const cuDoubleComplex *dl,
    const cuDoubleComplex *d, const cuDoubleComplex *du,
    const cuDoubleComplex *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cuDoubleComplex *, const cuDoubleComplex *,
      const cuDoubleComplex *, const cuDoubleComplex *, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseZgtsv2StridedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, dl, d, du, x, batchCount, batchStride,
                  bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch(
    cusparseHandle_t handle, int m, const float *dl, const float *d,
    const float *du, float *x, int batchCount, int batchStride, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const float *, const float *, const float *,
      float *, int, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgtsv2StridedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2StridedBatch(cusparseHandle_t handle, int m, const double *dl,
                           const double *d, const double *du, double *x,
                           int batchCount, int batchStride, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const double *, const double *, const double *,
      double *, int, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgtsv2StridedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch(
    cusparseHandle_t handle, int m, const cuComplex *dl, const cuComplex *d,
    const cuComplex *du, cuComplex *x, int batchCount, int batchStride,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cuComplex *, const cuComplex *,
      const cuComplex *, cuComplex *, int, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgtsv2StridedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch(
    cusparseHandle_t handle, int m, const cuDoubleComplex *dl,
    const cuDoubleComplex *d, const cuDoubleComplex *du, cuDoubleComplex *x,
    int batchCount, int batchStride, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cuDoubleComplex *, const cuDoubleComplex *,
      const cuDoubleComplex *, cuDoubleComplex *, int, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgtsv2StridedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const float *dl, const float *d,
    const float *du, const float *x, int batchCount,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, const float *, const float *,
      const float *, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSgtsvInterleavedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, dl, d, du, x, batchCount,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const double *dl, const double *d,
    const double *du, const double *x, int batchCount,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, const double *,
      const double *, const double *, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDgtsvInterleavedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, dl, d, du, x, batchCount,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuComplex *dl,
    const cuComplex *d, const cuComplex *du, const cuComplex *x, int batchCount,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuComplex *, const cuComplex *,
      const cuComplex *, const cuComplex *, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseCgtsvInterleavedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, dl, d, du, x, batchCount,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuDoubleComplex *dl,
    const cuDoubleComplex *d, const cuDoubleComplex *du,
    const cuDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuDoubleComplex *,
      const cuDoubleComplex *, const cuDoubleComplex *, const cuDoubleComplex *,
      int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseZgtsvInterleavedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, dl, d, du, x, batchCount,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSgtsvInterleavedBatch(
    cusparseHandle_t handle, int algo, int m, float *dl, float *d, float *du,
    float *x, int batchCount, void *pBuffer) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, int, int, float *,
                                      float *, float *, float *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgtsvInterleavedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDgtsvInterleavedBatch(
    cusparseHandle_t handle, int algo, int m, double *dl, double *d, double *du,
    double *x, int batchCount, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, int, int,
                                                  double *, double *, double *,
                                                  double *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgtsvInterleavedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCgtsvInterleavedBatch(
    cusparseHandle_t handle, int algo, int m, cuComplex *dl, cuComplex *d,
    cuComplex *du, cuComplex *x, int batchCount, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, cuComplex *, cuComplex *, cuComplex *,
      cuComplex *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgtsvInterleavedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZgtsvInterleavedBatch(
    cusparseHandle_t handle, int algo, int m, cuDoubleComplex *dl,
    cuDoubleComplex *d, cuDoubleComplex *du, cuDoubleComplex *x, int batchCount,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, cuDoubleComplex *, cuDoubleComplex *,
      cuDoubleComplex *, cuDoubleComplex *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgtsvInterleavedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const float *ds, const float *dl,
    const float *d, const float *du, const float *dw, const float *x,
    int batchCount, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, const float *, const float *,
      const float *, const float *, const float *, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSgpsvInterleavedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, ds, dl, d, du, dw, x, batchCount,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const double *ds,
    const double *dl, const double *d, const double *du, const double *dw,
    const double *x, int batchCount, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, const double *,
      const double *, const double *, const double *, const double *, int,
      size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDgpsvInterleavedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, ds, dl, d, du, dw, x, batchCount,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuComplex *ds,
    const cuComplex *dl, const cuComplex *d, const cuComplex *du,
    const cuComplex *dw, const cuComplex *x, int batchCount,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuComplex *, const cuComplex *,
      const cuComplex *, const cuComplex *, const cuComplex *,
      const cuComplex *, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseCgpsvInterleavedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, ds, dl, d, du, dw, x, batchCount,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuDoubleComplex *ds,
    const cuDoubleComplex *dl, const cuDoubleComplex *d,
    const cuDoubleComplex *du, const cuDoubleComplex *dw,
    const cuDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuDoubleComplex *,
      const cuDoubleComplex *, const cuDoubleComplex *, const cuDoubleComplex *,
      const cuDoubleComplex *, const cuDoubleComplex *, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseZgpsvInterleavedBatch_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, ds, dl, d, du, dw, x, batchCount,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSgpsvInterleavedBatch(
    cusparseHandle_t handle, int algo, int m, float *ds, float *dl, float *d,
    float *du, float *dw, float *x, int batchCount, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, float *, float *, float *, float *, float *,
      float *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgpsvInterleavedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDgpsvInterleavedBatch(
    cusparseHandle_t handle, int algo, int m, double *ds, double *dl, double *d,
    double *du, double *dw, double *x, int batchCount, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, double *, double *, double *, double *,
      double *, double *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgpsvInterleavedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCgpsvInterleavedBatch(
    cusparseHandle_t handle, int algo, int m, cuComplex *ds, cuComplex *dl,
    cuComplex *d, cuComplex *du, cuComplex *dw, cuComplex *x, int batchCount,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, cuComplex *, cuComplex *, cuComplex *,
      cuComplex *, cuComplex *, cuComplex *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgpsvInterleavedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZgpsvInterleavedBatch(
    cusparseHandle_t handle, int algo, int m, cuDoubleComplex *ds,
    cuDoubleComplex *dl, cuDoubleComplex *d, cuDoubleComplex *du,
    cuDoubleComplex *dw, cuDoubleComplex *x, int batchCount, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, cuDoubleComplex *, cuDoubleComplex *,
      cuDoubleComplex *, cuDoubleComplex *, cuDoubleComplex *,
      cuDoubleComplex *, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgpsvInterleavedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpGEMM)
cusparseStatus_t CUSPARSEAPI cusparseCreateCsrgemm2Info(csrgemm2Info_t *info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csrgemm2Info_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateCsrgemm2Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

CUSPARSE_DEPRECATED(cusparseSpGEMM)
cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrgemm2Info(csrgemm2Info_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(csrgemm2Info_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyCsrgemm2Info");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

CUSPARSE_DEPRECATED(cusparseSpGEMM)
cusparseStatus_t CUSPARSEAPI cusparseScsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const float *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const float *beta,
    const cusparseMatDescr_t descrD, int nnzD, const int *csrSortedRowPtrD,
    const int *csrSortedColIndD, csrgemm2Info_t info,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const float *, const cusparseMatDescr_t,
      int, const int *, const int *, const cusparseMatDescr_t, int, const int *,
      const int *, const float *, const cusparseMatDescr_t, int, const int *,
      const int *, csrgemm2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrgemm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA,
                  csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
                  csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD,
                  csrSortedColIndD, info, pBufferSizeInBytes);
}

CUSPARSE_DEPRECATED(cusparseSpGEMM)
cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const double *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const double *beta, const cusparseMatDescr_t descrD, int nnzD,
    const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    csrgemm2Info_t info, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const double *, const cusparseMatDescr_t,
      int, const int *, const int *, const cusparseMatDescr_t, int, const int *,
      const int *, const double *, const cusparseMatDescr_t, int, const int *,
      const int *, csrgemm2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrgemm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA,
                  csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
                  csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD,
                  csrSortedColIndD, info, pBufferSizeInBytes);
}

CUSPARSE_DEPRECATED(cusparseSpGEMM)
cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const cuComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cuComplex *beta, const cusparseMatDescr_t descrD, int nnzD,
    const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    csrgemm2Info_t info, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cuComplex *,
      const cusparseMatDescr_t, int, const int *, const int *,
      const cusparseMatDescr_t, int, const int *, const int *,
      const cuComplex *, const cusparseMatDescr_t, int, const int *,
      const int *, csrgemm2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrgemm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA,
                  csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
                  csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD,
                  csrSortedColIndD, info, pBufferSizeInBytes);
}

CUSPARSE_DEPRECATED(cusparseSpGEMM)
cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cuDoubleComplex *beta, const cusparseMatDescr_t descrD, int nnzD,
    const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    csrgemm2Info_t info, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cuDoubleComplex *,
      const cusparseMatDescr_t, int, const int *, const int *,
      const cusparseMatDescr_t, int, const int *, const int *,
      const cuDoubleComplex *, const cusparseMatDescr_t, int, const int *,
      const int *, csrgemm2Info_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrgemm2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA,
                  csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
                  csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD,
                  csrSortedColIndD, info, pBufferSizeInBytes);
}

CUSPARSE_DEPRECATED(cusparseSpGEMM)
cusparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Nnz(
    cusparseHandle_t handle, int m, int n, int k,
    const cusparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cusparseMatDescr_t descrD, int nnzD, const int *csrSortedRowPtrD,
    const int *csrSortedColIndD, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, const csrgemm2Info_t info,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, int,
      const int *, const int *, const cusparseMatDescr_t, int, const int *,
      const int *, const cusparseMatDescr_t, int, const int *, const int *,
      const cusparseMatDescr_t, int *, int *, const csrgemm2Info_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcsrgemm2Nnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, descrA, nnzA, csrSortedRowPtrA,
                  csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
                  csrSortedColIndB, descrD, nnzD, csrSortedRowPtrD,
                  csrSortedColIndD, descrC, csrSortedRowPtrC,
                  nnzTotalDevHostPtr, info, pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpGEMM)
cusparseStatus_t CUSPARSEAPI cusparseScsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const float *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const float *beta,
    const cusparseMatDescr_t descrD, int nnzD, const float *csrSortedValD,
    const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    const cusparseMatDescr_t descrC, float *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC,
    const csrgemm2Info_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const float *, const cusparseMatDescr_t,
      int, const float *, const int *, const int *, const cusparseMatDescr_t,
      int, const float *, const int *, const int *, const float *,
      const cusparseMatDescr_t, int, const float *, const int *, const int *,
      const cusparseMatDescr_t, float *, const int *, int *,
      const csrgemm2Info_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrgemm2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta,
                  descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
                  csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC,
                  csrSortedColIndC, info, pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpGEMM)
cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const double *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const double *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const double *beta, const cusparseMatDescr_t descrD, int nnzD,
    const double *csrSortedValD, const int *csrSortedRowPtrD,
    const int *csrSortedColIndD, const cusparseMatDescr_t descrC,
    double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    const csrgemm2Info_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const double *, const cusparseMatDescr_t,
      int, const double *, const int *, const int *, const cusparseMatDescr_t,
      int, const double *, const int *, const int *, const double *,
      const cusparseMatDescr_t, int, const double *, const int *, const int *,
      const cusparseMatDescr_t, double *, const int *, int *,
      const csrgemm2Info_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrgemm2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta,
                  descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
                  csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC,
                  csrSortedColIndC, info, pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpGEMM)
cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const cuComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const cuComplex *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cuComplex *beta, const cusparseMatDescr_t descrD, int nnzD,
    const cuComplex *csrSortedValD, const int *csrSortedRowPtrD,
    const int *csrSortedColIndD, const cusparseMatDescr_t descrC,
    cuComplex *csrSortedValC, const int *csrSortedRowPtrC,
    int *csrSortedColIndC, const csrgemm2Info_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cuComplex *,
      const cusparseMatDescr_t, int, const cuComplex *, const int *,
      const int *, const cusparseMatDescr_t, int, const cuComplex *,
      const int *, const int *, const cuComplex *, const cusparseMatDescr_t,
      int, const cuComplex *, const int *, const int *,
      const cusparseMatDescr_t, cuComplex *, const int *, int *,
      const csrgemm2Info_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrgemm2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta,
                  descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
                  csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC,
                  csrSortedColIndC, info, pBuffer);
}

CUSPARSE_DEPRECATED(cusparseSpGEMM)
cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const cuDoubleComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cuDoubleComplex *beta,
    const cusparseMatDescr_t descrD, int nnzD,
    const cuDoubleComplex *csrSortedValD, const int *csrSortedRowPtrD,
    const int *csrSortedColIndD, const cusparseMatDescr_t descrC,
    cuDoubleComplex *csrSortedValC, const int *csrSortedRowPtrC,
    int *csrSortedColIndC, const csrgemm2Info_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cuDoubleComplex *,
      const cusparseMatDescr_t, int, const cuDoubleComplex *, const int *,
      const int *, const cusparseMatDescr_t, int, const cuDoubleComplex *,
      const int *, const int *, const cuDoubleComplex *,
      const cusparseMatDescr_t, int, const cuDoubleComplex *, const int *,
      const int *, const cusparseMatDescr_t, cuDoubleComplex *, const int *,
      int *, const csrgemm2Info_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrgemm2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta,
                  descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
                  csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC,
                  csrSortedColIndC, info, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseScsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cusparseMatDescr_t descrC, const float *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, const cusparseMatDescr_t, int,
      const float *, const int *, const int *, const float *,
      const cusparseMatDescr_t, int, const float *, const int *, const int *,
      const cusparseMatDescr_t, const float *, const int *, const int *,
      size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrgeam2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const double *beta, const cusparseMatDescr_t descrB, int nnzB,
    const double *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, const cusparseMatDescr_t, int,
      const double *, const int *, const int *, const double *,
      const cusparseMatDescr_t, int, const double *, const int *, const int *,
      const cusparseMatDescr_t, const double *, const int *, const int *,
      size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrgeam2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cuComplex *beta, const cusparseMatDescr_t descrB, int nnzB,
    const cuComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    const cuComplex *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuComplex *, const cusparseMatDescr_t,
      int, const cuComplex *, const int *, const int *, const cuComplex *,
      const cusparseMatDescr_t, int, const cuComplex *, const int *,
      const int *, const cusparseMatDescr_t, const cuComplex *, const int *,
      const int *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrgeam2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cuDoubleComplex *beta,
    const cusparseMatDescr_t descrB, int nnzB,
    const cuDoubleComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    const cuDoubleComplex *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuDoubleComplex *,
      const cusparseMatDescr_t, int, const cuDoubleComplex *, const int *,
      const int *, const cuDoubleComplex *, const cusparseMatDescr_t, int,
      const cuDoubleComplex *, const int *, const int *,
      const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
      const int *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrgeam2_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrgeam2Nnz(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    int nnzA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *workspace) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, int, const int *,
      const int *, const cusparseMatDescr_t, int, const int *, const int *,
      const cusparseMatDescr_t, int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcsrgeam2Nnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, nnzA, csrSortedRowPtrA,
                  csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
                  csrSortedColIndB, descrC, csrSortedRowPtrC,
                  nnzTotalDevHostPtr, workspace);
}

cusparseStatus_t CUSPARSEAPI cusparseScsrgeam2(
    cusparseHandle_t handle, int m, int n, const float *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cusparseMatDescr_t descrC, float *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, const cusparseMatDescr_t, int,
      const float *, const int *, const int *, const float *,
      const cusparseMatDescr_t, int, const float *, const int *, const int *,
      const cusparseMatDescr_t, float *, int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrgeam2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam2(
    cusparseHandle_t handle, int m, int n, const double *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const double *beta, const cusparseMatDescr_t descrB, int nnzB,
    const double *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    double *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, const cusparseMatDescr_t, int,
      const double *, const int *, const int *, const double *,
      const cusparseMatDescr_t, int, const double *, const int *, const int *,
      const cusparseMatDescr_t, double *, int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrgeam2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam2(
    cusparseHandle_t handle, int m, int n, const cuComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cuComplex *beta, const cusparseMatDescr_t descrB, int nnzB,
    const cuComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    cuComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuComplex *, const cusparseMatDescr_t,
      int, const cuComplex *, const int *, const int *, const cuComplex *,
      const cusparseMatDescr_t, int, const cuComplex *, const int *,
      const int *, const cusparseMatDescr_t, cuComplex *, int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrgeam2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam2(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cuDoubleComplex *beta,
    const cusparseMatDescr_t descrB, int nnzB,
    const cuDoubleComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC,
    int *csrSortedColIndC, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cuDoubleComplex *,
      const cusparseMatDescr_t, int, const cuDoubleComplex *, const int *,
      const int *, const cuDoubleComplex *, const cusparseMatDescr_t, int,
      const cuDoubleComplex *, const int *, const int *,
      const cusparseMatDescr_t, cuDoubleComplex *, int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrgeam2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                  csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseScsrcolor(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const float *fractionToColor, int *ncolors,
    int *coloring, int *reordering, const cusparseColorInfo_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, const float *, int *, int *, int *,
      const cusparseColorInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsrcolor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, fractionToColor, ncolors, coloring,
                  reordering, info);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsrcolor(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const double *fractionToColor, int *ncolors,
    int *coloring, int *reordering, const cusparseColorInfo_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, const double *, int *, int *, int *,
      const cusparseColorInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsrcolor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, fractionToColor, ncolors, coloring,
                  reordering, info);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsrcolor(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const float *fractionToColor, int *ncolors,
    int *coloring, int *reordering, const cusparseColorInfo_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, const float *, int *, int *, int *,
      const cusparseColorInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsrcolor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, fractionToColor, ncolors, coloring,
                  reordering, info);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsrcolor(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const double *fractionToColor, int *ncolors,
    int *coloring, int *reordering, const cusparseColorInfo_t info) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, const double *, int *,
      int *, int *, const cusparseColorInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsrcolor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, fractionToColor, ncolors, coloring,
                  reordering, info);
}

cusparseStatus_t CUSPARSEAPI
cusparseSnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
             const cusparseMatDescr_t descrA, const float *A, int lda,
             int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const float *, int, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSnnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol,
                  nnzTotalDevHostPtr);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
             const cusparseMatDescr_t descrA, const double *A, int lda,
             int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const double *, int, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDnnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol,
                  nnzTotalDevHostPtr);
}

cusparseStatus_t CUSPARSEAPI
cusparseCnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
             const cusparseMatDescr_t descrA, const cuComplex *A, int lda,
             int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuComplex *, int, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCnnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol,
                  nnzTotalDevHostPtr);
}

cusparseStatus_t CUSPARSEAPI
cusparseZnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
             const cusparseMatDescr_t descrA, const cuDoubleComplex *A, int lda,
             int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, int, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZnnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol,
                  nnzTotalDevHostPtr);
}

cusparseStatus_t CUSPARSEAPI cusparseSnnz_compress(
    cusparseHandle_t handle, int m, const cusparseMatDescr_t descr,
    const float *csrSortedValA, const int *csrSortedRowPtrA, int *nnzPerRow,
    int *nnzC, float tol) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cusparseMatDescr_t, const float *,
      const int *, int *, int *, float);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSnnz_compress");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow,
                  nnzC, tol);
}

cusparseStatus_t CUSPARSEAPI cusparseDnnz_compress(
    cusparseHandle_t handle, int m, const cusparseMatDescr_t descr,
    const double *csrSortedValA, const int *csrSortedRowPtrA, int *nnzPerRow,
    int *nnzC, double tol) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cusparseMatDescr_t, const double *,
      const int *, int *, int *, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDnnz_compress");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow,
                  nnzC, tol);
}

cusparseStatus_t CUSPARSEAPI cusparseCnnz_compress(
    cusparseHandle_t handle, int m, const cusparseMatDescr_t descr,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, int *nnzPerRow,
    int *nnzC, cuComplex tol) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cusparseMatDescr_t, const cuComplex *,
      const int *, int *, int *, cuComplex);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCnnz_compress");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow,
                  nnzC, tol);
}

cusparseStatus_t CUSPARSEAPI cusparseZnnz_compress(
    cusparseHandle_t handle, int m, const cusparseMatDescr_t descr,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    int *nnzPerRow, int *nnzC, cuDoubleComplex tol) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, const cusparseMatDescr_t, const cuDoubleComplex *,
      const int *, int *, int *, cuDoubleComplex);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZnnz_compress");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow,
                  nnzC, tol);
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    float *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC,
    float tol) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, int, const int *, float *, int *, int *, float);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsr2csr_compress");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, csrSortedValA, csrSortedColIndA,
                  csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC,
                  csrSortedColIndC, csrSortedRowPtrC, tol);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    double *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC,
    double tol) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, int, const int *, double *, int *, int *,
      double);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsr2csr_compress");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, csrSortedValA, csrSortedColIndA,
                  csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC,
                  csrSortedColIndC, csrSortedRowPtrC, tol);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    cuComplex *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC,
    cuComplex tol) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, int, const int *, cuComplex *, int *, int *,
      cuComplex);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsr2csr_compress");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, csrSortedValA, csrSortedColIndA,
                  csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC,
                  csrSortedColIndC, csrSortedRowPtrC, tol);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    cuDoubleComplex *csrSortedValC, int *csrSortedColIndC,
    int *csrSortedRowPtrC, cuDoubleComplex tol) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, int, const int *,
      cuDoubleComplex *, int *, int *, cuDoubleComplex);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsr2csr_compress");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, csrSortedValA, csrSortedColIndA,
                  csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC,
                  csrSortedColIndC, csrSortedRowPtrC, tol);
}

CUSPARSE_DEPRECATED(cusparseDenseToSparse)
cusparseStatus_t CUSPARSEAPI cusparseSdense2csr(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const float *A, int lda, const int *nnzPerRow, float *csrSortedValA,
    int *csrSortedRowPtrA, int *csrSortedColIndA) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *, int,
      const int *, float *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSdense2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA);
}

CUSPARSE_DEPRECATED(cusparseDenseToSparse)
cusparseStatus_t CUSPARSEAPI cusparseDdense2csr(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const double *A, int lda, const int *nnzPerRow, double *csrSortedValA,
    int *csrSortedRowPtrA, int *csrSortedColIndA) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *, int,
      const int *, double *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDdense2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA);
}

CUSPARSE_DEPRECATED(cusparseDenseToSparse)
cusparseStatus_t CUSPARSEAPI cusparseCdense2csr(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuComplex *A, int lda, const int *nnzPerRow, cuComplex *csrSortedValA,
    int *csrSortedRowPtrA, int *csrSortedColIndA) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
      int, const int *, cuComplex *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCdense2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA);
}

CUSPARSE_DEPRECATED(cusparseDenseToSparse)
cusparseStatus_t CUSPARSEAPI cusparseZdense2csr(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *A, int lda, const int *nnzPerRow,
    cuDoubleComplex *csrSortedValA, int *csrSortedRowPtrA,
    int *csrSortedColIndA) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, int, const int *, cuDoubleComplex *, int *,
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZdense2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA,
                  csrSortedRowPtrA, csrSortedColIndA);
}

CUSPARSE_DEPRECATED(cusparseSparseToDense)
cusparseStatus_t CUSPARSEAPI cusparseScsr2dense(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, float *A, int lda) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsr2dense");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, A, lda);
}

CUSPARSE_DEPRECATED(cusparseSparseToDense)
cusparseStatus_t CUSPARSEAPI cusparseDcsr2dense(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, double *A, int lda) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsr2dense");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, A, lda);
}

CUSPARSE_DEPRECATED(cusparseSparseToDense)
cusparseStatus_t CUSPARSEAPI cusparseCcsr2dense(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, cuComplex *A, int lda) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsr2dense");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, A, lda);
}

CUSPARSE_DEPRECATED(cusparseSparseToDense)
cusparseStatus_t CUSPARSEAPI cusparseZcsr2dense(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, cuDoubleComplex *A, int lda) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, cuDoubleComplex *,
      int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsr2dense");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, A, lda);
}

CUSPARSE_DEPRECATED(cusparseDenseToSparse)
cusparseStatus_t CUSPARSEAPI cusparseSdense2csc(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const float *A, int lda, const int *nnzPerCol, float *cscSortedValA,
    int *cscSortedRowIndA, int *cscSortedColPtrA) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *, int,
      const int *, float *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSdense2csc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA,
                  cscSortedRowIndA, cscSortedColPtrA);
}

CUSPARSE_DEPRECATED(cusparseDenseToSparse)
cusparseStatus_t CUSPARSEAPI cusparseDdense2csc(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const double *A, int lda, const int *nnzPerCol, double *cscSortedValA,
    int *cscSortedRowIndA, int *cscSortedColPtrA) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *, int,
      const int *, double *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDdense2csc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA,
                  cscSortedRowIndA, cscSortedColPtrA);
}

CUSPARSE_DEPRECATED(cusparseDenseToSparse)
cusparseStatus_t CUSPARSEAPI cusparseCdense2csc(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuComplex *A, int lda, const int *nnzPerCol, cuComplex *cscSortedValA,
    int *cscSortedRowIndA, int *cscSortedColPtrA) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
      int, const int *, cuComplex *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCdense2csc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA,
                  cscSortedRowIndA, cscSortedColPtrA);
}

CUSPARSE_DEPRECATED(cusparseDenseToSparse)
cusparseStatus_t CUSPARSEAPI cusparseZdense2csc(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *A, int lda, const int *nnzPerCol,
    cuDoubleComplex *cscSortedValA, int *cscSortedRowIndA,
    int *cscSortedColPtrA) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, int, const int *, cuDoubleComplex *, int *,
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZdense2csc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA,
                  cscSortedRowIndA, cscSortedColPtrA);
}

CUSPARSE_DEPRECATED(cusparseSparseToDense)
cusparseStatus_t CUSPARSEAPI cusparseScsc2dense(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const float *cscSortedValA, const int *cscSortedRowIndA,
    const int *cscSortedColPtrA, float *A, int lda) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsc2dense");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                  cscSortedColPtrA, A, lda);
}

CUSPARSE_DEPRECATED(cusparseSparseToDense)
cusparseStatus_t CUSPARSEAPI cusparseDcsc2dense(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const double *cscSortedValA, const int *cscSortedRowIndA,
    const int *cscSortedColPtrA, double *A, int lda) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsc2dense");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                  cscSortedColPtrA, A, lda);
}

CUSPARSE_DEPRECATED(cusparseSparseToDense)
cusparseStatus_t CUSPARSEAPI cusparseCcsc2dense(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuComplex *cscSortedValA, const int *cscSortedRowIndA,
    const int *cscSortedColPtrA, cuComplex *A, int lda) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
      const int *, const int *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsc2dense");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                  cscSortedColPtrA, A, lda);
}

CUSPARSE_DEPRECATED(cusparseSparseToDense)
cusparseStatus_t CUSPARSEAPI cusparseZcsc2dense(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *cscSortedValA, const int *cscSortedRowIndA,
    const int *cscSortedColPtrA, cuDoubleComplex *A, int lda) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, cuDoubleComplex *,
      int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsc2dense");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                  cscSortedColPtrA, A, lda);
}

cusparseStatus_t CUSPARSEAPI cusparseXcoo2csr(cusparseHandle_t handle,
                                              const int *cooRowInd, int nnz,
                                              int m, int *csrSortedRowPtr,
                                              cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, const int *, int, int, int *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcoo2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, cooRowInd, nnz, m, csrSortedRowPtr, idxBase);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsr2coo(cusparseHandle_t handle,
                                              const int *csrSortedRowPtr,
                                              int nnz, int m, int *cooRowInd,
                                              cusparseIndexBase_t idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, const int *, int, int, int *, cusparseIndexBase_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcsr2coo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, csrSortedRowPtr, nnz, m, cooRowInd, idxBase);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsr2bsrNnz(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC,
    int *bsrSortedRowPtrC, int *nnzTotalDevHostPtr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const int *, const int *, int, const cusparseMatDescr_t, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcsr2bsrNnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedRowPtrA,
                  csrSortedColIndA, blockDim, descrC, bsrSortedRowPtrC,
                  nnzTotalDevHostPtr);
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2bsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, float *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const float *, const int *, const int *, int, const cusparseMatDescr_t,
      float *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsr2bsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, blockDim, descrC, bsrSortedValC,
                  bsrSortedRowPtrC, bsrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2bsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, double *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const double *, const int *, const int *, int, const cusparseMatDescr_t,
      double *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsr2bsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, blockDim, descrC, bsrSortedValC,
                  bsrSortedRowPtrC, bsrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2bsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, cuComplex *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuComplex *, const int *, const int *, int,
      const cusparseMatDescr_t, cuComplex *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsr2bsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, blockDim, descrC, bsrSortedValC,
                  bsrSortedRowPtrC, bsrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2bsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, int,
      const cusparseMatDescr_t, cuDoubleComplex *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsr2bsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, blockDim, descrC, bsrSortedValC,
                  bsrSortedRowPtrC, bsrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseSbsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, float *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const float *, const int *, const int *, int, const cusparseMatDescr_t,
      float *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSbsr2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                  bsrSortedColIndA, blockDim, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseDbsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, double *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const double *, const int *, const int *, int, const cusparseMatDescr_t,
      double *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDbsr2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                  bsrSortedColIndA, blockDim, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseCbsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, cuComplex *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuComplex *, const int *, const int *, int,
      const cusparseMatDescr_t, cuComplex *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCbsr2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                  bsrSortedColIndA, blockDim, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseZbsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, cuDoubleComplex *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, int,
      const cusparseMatDescr_t, cuDoubleComplex *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZbsr2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                  bsrSortedColIndA, blockDim, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc_bufferSize(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const float *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const float *, const int *, const int *,
      int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgebsr2gebsc_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc_bufferSize(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const double *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const double *, const int *, const int *,
      int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgebsr2gebsc_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc_bufferSize(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cuComplex *, const int *,
      const int *, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgebsr2gebsc_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc_bufferSize(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cuDoubleComplex *, const int *,
      const int *, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgebsr2gebsc_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc_bufferSizeExt(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const float *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const float *, const int *, const int *,
      int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSgebsr2gebsc_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc_bufferSizeExt(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const double *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const double *, const int *, const int *,
      int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDgebsr2gebsc_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc_bufferSizeExt(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cuComplex *, const int *,
      const int *, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseCgebsr2gebsc_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc_bufferSizeExt(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cuDoubleComplex *, const int *,
      const int *, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseZgebsr2gebsc_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const float *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, float *bscVal,
    int *bscRowInd, int *bscColPtr, cusparseAction_t copyValues,
    cusparseIndexBase_t idxBase, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const float *, const int *, const int *,
      int, int, float *, int *, int *, cusparseAction_t, cusparseIndexBase_t,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgebsr2gebsc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd,
                  bscColPtr, copyValues, idxBase, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const double *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    double *bscVal, int *bscRowInd, int *bscColPtr, cusparseAction_t copyValues,
    cusparseIndexBase_t idxBase, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const double *, const int *, const int *,
      int, int, double *, int *, int *, cusparseAction_t, cusparseIndexBase_t,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgebsr2gebsc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd,
                  bscColPtr, copyValues, idxBase, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    cuComplex *bscVal, int *bscRowInd, int *bscColPtr,
    cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cuComplex *, const int *,
      const int *, int, int, cuComplex *, int *, int *, cusparseAction_t,
      cusparseIndexBase_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgebsr2gebsc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd,
                  bscColPtr, copyValues, idxBase, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    cuDoubleComplex *bscVal, int *bscRowInd, int *bscColPtr,
    cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cuDoubleComplex *, const int *,
      const int *, int, int, cuDoubleComplex *, int *, int *, cusparseAction_t,
      cusparseIndexBase_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgebsr2gebsc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                  bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd,
                  bscColPtr, copyValues, idxBase, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseXgebsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, int rowBlockDim, int colBlockDim,
    const cusparseMatDescr_t descrC, int *csrSortedRowPtrC,
    int *csrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const int *, const int *, int, int, const cusparseMatDescr_t, int *,
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXgebsr2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, descrA, bsrSortedRowPtrA,
                  bsrSortedColIndA, rowBlockDim, colBlockDim, descrC,
                  csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const cusparseMatDescr_t descrC, float *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const float *, const int *, const int *, int, int,
      const cusparseMatDescr_t, float *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgebsr2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                  bsrSortedColIndA, rowBlockDim, colBlockDim, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const cusparseMatDescr_t descrC, double *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const double *, const int *, const int *, int, int,
      const cusparseMatDescr_t, double *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgebsr2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                  bsrSortedColIndA, rowBlockDim, colBlockDim, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const cusparseMatDescr_t descrC, cuComplex *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuComplex *, const int *, const int *, int, int,
      const cusparseMatDescr_t, cuComplex *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgebsr2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                  bsrSortedColIndA, rowBlockDim, colBlockDim, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const cusparseMatDescr_t descrC,
    cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC,
    int *csrSortedColIndC) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, int, int,
      const cusparseMatDescr_t, cuDoubleComplex *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgebsr2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                  bsrSortedColIndA, rowBlockDim, colBlockDim, descrC,
                  csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const float *, const int *, const int *, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsr2gebsr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, rowBlockDim, colBlockDim,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const double *, const int *, const int *, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsr2gebsr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, rowBlockDim, colBlockDim,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuComplex *, const int *, const int *, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsr2gebsr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, rowBlockDim, colBlockDim,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsr2gebsr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, rowBlockDim, colBlockDim,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const float *, const int *, const int *, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseScsr2gebsr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const double *, const int *, const int *, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDcsr2gebsr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuComplex *, const int *, const int *, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseCcsr2gebsr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseZcsr2gebsr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsr2gebsrNnz(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrC,
    int *bsrSortedRowPtrC, int rowBlockDim, int colBlockDim,
    int *nnzTotalDevHostPtr, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const int *, const int *, const cusparseMatDescr_t, int *, int, int,
      int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcsr2gebsrNnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedRowPtrA,
                  csrSortedColIndA, descrC, bsrSortedRowPtrC, rowBlockDim,
                  colBlockDim, nnzTotalDevHostPtr, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrC, float *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
    int colBlockDim, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const float *, const int *, const int *, const cusparseMatDescr_t,
      float *, int *, int *, int, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsr2gebsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                  bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrC, double *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
    int colBlockDim, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const double *, const int *, const int *, const cusparseMatDescr_t,
      double *, int *, int *, int, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsr2gebsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                  bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrC, cuComplex *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
    int colBlockDim, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuComplex *, const int *, const int *, const cusparseMatDescr_t,
      cuComplex *, int *, int *, int, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsr2gebsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                  bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
    int colBlockDim, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
      const cuDoubleComplex *, const int *, const int *,
      const cusparseMatDescr_t, cuDoubleComplex *, int *, int *, int, int,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsr2gebsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                  bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const float *, const int *, const int *, int,
      int, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgebsr2gebsr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const double *, const int *, const int *, int,
      int, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgebsr2gebsr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
      int, int, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgebsr2gebsr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
      const int *, int, int, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgebsr2gebsr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const float *, const int *, const int *, int,
      int, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSgebsr2gebsr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const double *, const int *, const int *, int,
      int, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDgebsr2gebsr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
      int, int, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseCgebsr2gebsr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
      const int *, int, int, int, int, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseZgebsr2gebsr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseXgebsr2gebsrNnz(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
    const cusparseMatDescr_t descrC, int *bsrSortedRowPtrC, int rowBlockDimC,
    int colBlockDimC, int *nnzTotalDevHostPtr, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const int *, const int *, int, int,
      const cusparseMatDescr_t, int *, int, int, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXgebsr2gebsrNnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedRowPtrA,
                  bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC,
                  bsrSortedRowPtrC, rowBlockDimC, colBlockDimC,
                  nnzTotalDevHostPtr, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC, float *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
    int colBlockDimC, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const float *, const int *, const int *, int,
      int, const cusparseMatDescr_t, float *, int *, int *, int, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSgebsr2gebsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                  bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC, double *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
    int colBlockDimC, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const double *, const int *, const int *, int,
      int, const cusparseMatDescr_t, double *, int *, int *, int, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDgebsr2gebsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                  bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC, cuComplex *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
    int colBlockDimC, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
      int, int, const cusparseMatDescr_t, cuComplex *, int *, int *, int, int,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCgebsr2gebsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                  bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC,
    cuDoubleComplex *bsrSortedValC, int *bsrSortedRowPtrC,
    int *bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDirection_t, int, int, int,
      const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
      const int *, int, int, const cusparseMatDescr_t, cuDoubleComplex *, int *,
      int *, int, int, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZgebsr2gebsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                  bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                  colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                  bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateIdentityPermutation(cusparseHandle_t handle, int n, int *p) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseHandle_t, int, int *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseCreateIdentityPermutation");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, p);
}

cusparseStatus_t CUSPARSEAPI cusparseXcoosort_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnz, const int *cooRowsA,
    const int *cooColsA, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const int *, const int *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcoosort_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, cooRowsA, cooColsA, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseXcoosortByRow(cusparseHandle_t handle,
                                                   int m, int n, int nnz,
                                                   int *cooRowsA, int *cooColsA,
                                                   int *P, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, int *, int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcoosortByRow");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseXcoosortByColumn(cusparseHandle_t handle,
                                                      int m, int n, int nnz,
                                                      int *cooRowsA,
                                                      int *cooColsA, int *P,
                                                      void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, int *, int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcoosortByColumn");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrsort_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnz, const int *csrRowPtrA,
    const int *csrColIndA, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const int *, const int *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcsrsort_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, csrRowPtrA, csrColIndA,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseXcsrsort(cusparseHandle_t handle, int m,
                                              int n, int nnz,
                                              const cusparseMatDescr_t descrA,
                                              const int *csrRowPtrA,
                                              int *csrColIndA, int *P,
                                              void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const int *,
      int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcsrsort");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, descrA, csrRowPtrA, csrColIndA, P,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseXcscsort_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnz, const int *cscColPtrA,
    const int *cscRowIndA, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const int *, const int *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcscsort_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, cscColPtrA, cscRowIndA,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseXcscsort(cusparseHandle_t handle, int m,
                                              int n, int nnz,
                                              const cusparseMatDescr_t descrA,
                                              const int *cscColPtrA,
                                              int *cscRowIndA, int *P,
                                              void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const int *,
      int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseXcscsort");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, descrA, cscColPtrA, cscRowIndA, P,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseScsru2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnz, float *csrVal,
    const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, float *, const int *, int *,
      csru2csrInfo_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsru2csr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsru2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnz, double *csrVal,
    const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, double *, const int *, int *,
      csru2csrInfo_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsru2csr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsru2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnz, cuComplex *csrVal,
    const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, cuComplex *, const int *, int *,
      csru2csrInfo_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsru2csr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsru2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnz, cuDoubleComplex *csrVal,
    const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, cuDoubleComplex *, const int *, int *,
      csru2csrInfo_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsru2csr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info,
                  pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseScsru2csr(
    cusparseHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, float *csrVal, const int *csrRowPtr,
    int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, float *,
      const int *, int *, csru2csrInfo_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsru2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsru2csr(
    cusparseHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, double *csrVal, const int *csrRowPtr,
    int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, double *,
      const int *, int *, csru2csrInfo_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsru2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsru2csr(
    cusparseHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, cuComplex *csrVal, const int *csrRowPtr,
    int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, cuComplex *,
      const int *, int *, csru2csrInfo_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsru2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsru2csr(
    cusparseHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, cuDoubleComplex *csrVal,
    const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, int *, csru2csrInfo_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsru2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseScsr2csru(
    cusparseHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, float *csrVal, const int *csrRowPtr,
    int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, float *,
      const int *, int *, csru2csrInfo_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScsr2csru");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDcsr2csru(
    cusparseHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, double *csrVal, const int *csrRowPtr,
    int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, double *,
      const int *, int *, csru2csrInfo_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDcsr2csru");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCcsr2csru(
    cusparseHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, cuComplex *csrVal, const int *csrRowPtr,
    int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, cuComplex *,
      const int *, int *, csru2csrInfo_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCcsr2csru");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseZcsr2csru(
    cusparseHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, cuDoubleComplex *csrVal,
    const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t,
      cuDoubleComplex *, const int *, int *, csru2csrInfo_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseZcsr2csru");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                  pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *A, int lda,
    const float *threshold, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, int, const float *,
      const cusparseMatDescr_t, const float *, const int *, const int *,
      size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSpruneDense2csr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, threshold, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *A, int lda,
    const double *threshold, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, int, const double *,
      const cusparseMatDescr_t, const double *, const int *, const int *,
      size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDpruneDense2csr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, threshold, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrNnz(
    cusparseHandle_t handle, int m, int n, const float *A, int lda,
    const float *threshold, const cusparseMatDescr_t descrC, int *csrRowPtrC,
    int *nnzTotalDevHostPtr, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, int, const float *,
      const cusparseMatDescr_t, int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpruneDense2csrNnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, threshold, descrC, csrRowPtrC,
                  nnzTotalDevHostPtr, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrNnz(
    cusparseHandle_t handle, int m, int n, const double *A, int lda,
    const double *threshold, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, int, const double *,
      const cusparseMatDescr_t, int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDpruneDense2csrNnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, threshold, descrC, csrSortedRowPtrC,
                  nnzTotalDevHostPtr, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csr(
    cusparseHandle_t handle, int m, int n, const float *A, int lda,
    const float *threshold, const cusparseMatDescr_t descrC,
    float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, int, const float *,
      const cusparseMatDescr_t, float *, const int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpruneDense2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, threshold, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csr(
    cusparseHandle_t handle, int m, int n, const double *A, int lda,
    const double *threshold, const cusparseMatDescr_t descrC,
    double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, int, const double *,
      const cusparseMatDescr_t, double *, const int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDpruneDense2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, threshold, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const float *threshold, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, const float *, const cusparseMatDescr_t,
      const float *, const int *, const int *, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSpruneCsr2csr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, threshold, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const double *threshold, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, const double *, const cusparseMatDescr_t,
      const double *, const int *, const int *, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDpruneCsr2csr_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, threshold, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrNnz(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const float *threshold, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, const float *, const cusparseMatDescr_t, int *,
      int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpruneCsr2csrNnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, threshold, descrC, csrSortedRowPtrC,
                  nnzTotalDevHostPtr, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrNnz(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const double *threshold, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, const double *, const cusparseMatDescr_t, int *,
      int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDpruneCsr2csrNnz");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, threshold, descrC, csrSortedRowPtrC,
                  nnzTotalDevHostPtr, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csr(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const float *threshold, const cusparseMatDescr_t descrC,
    float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, const float *, const cusparseMatDescr_t,
      float *, const int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpruneCsr2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, threshold, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csr(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const double *threshold, const cusparseMatDescr_t descrC,
    double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, const double *, const cusparseMatDescr_t,
      double *, const int *, int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDpruneCsr2csr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, threshold, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *A, int lda,
    float percentage, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, int, float,
      const cusparseMatDescr_t, const float *, const int *, const int *,
      pruneInfo_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSpruneDense2csrByPercentage_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, percentage, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *A, int lda,
    float percentage, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, int, float,
      const cusparseMatDescr_t, const double *, const int *, const int *,
      pruneInfo_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDpruneDense2csrByPercentage_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, percentage, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, const float *A, int lda,
    float percentage, const cusparseMatDescr_t descrC, int *csrRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, int, float,
      const cusparseMatDescr_t, int *, int *, pruneInfo_t, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSpruneDense2csrNnzByPercentage");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, percentage, descrC, csrRowPtrC,
                  nnzTotalDevHostPtr, info, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, const double *A, int lda,
    float percentage, const cusparseMatDescr_t descrC, int *csrRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, int, float,
      const cusparseMatDescr_t, int *, int *, pruneInfo_t, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDpruneDense2csrNnzByPercentage");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, percentage, descrC, csrRowPtrC,
                  nnzTotalDevHostPtr, info, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrByPercentage(
    cusparseHandle_t handle, int m, int n, const float *A, int lda,
    float percentage, const cusparseMatDescr_t descrC, float *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const float *, int, float,
      const cusparseMatDescr_t, float *, const int *, int *, pruneInfo_t,
      void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSpruneDense2csrByPercentage");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, percentage, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrByPercentage(
    cusparseHandle_t handle, int m, int n, const double *A, int lda,
    float percentage, const cusparseMatDescr_t descrC, double *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, const double *, int, float,
      const cusparseMatDescr_t, double *, const int *, int *, pruneInfo_t,
      void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDpruneDense2csrByPercentage");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, percentage, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, const float *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, float, const cusparseMatDescr_t, const float *,
      const int *, const int *, pruneInfo_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSpruneCsr2csrByPercentage_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, percentage, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, const double *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, float, const cusparseMatDescr_t, const double *,
      const int *, const int *, pruneInfo_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDpruneCsr2csrByPercentage_bufferSizeExt");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, percentage, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, float, const cusparseMatDescr_t, int *, int *,
      pruneInfo_t, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSpruneCsr2csrNnzByPercentage");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, percentage, descrC, csrSortedRowPtrC,
                  nnzTotalDevHostPtr, info, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, float, const cusparseMatDescr_t, int *, int *,
      pruneInfo_t, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDpruneCsr2csrNnzByPercentage");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, percentage, descrC, csrSortedRowPtrC,
                  nnzTotalDevHostPtr, info, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, float *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
      const int *, const int *, float, const cusparseMatDescr_t, float *,
      const int *, int *, pruneInfo_t, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSpruneCsr2csrByPercentage");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, percentage, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, double *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
      const int *, const int *, float, const cusparseMatDescr_t, double *,
      const int *, int *, pruneInfo_t, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDpruneCsr2csrByPercentage");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                  csrSortedColIndA, percentage, descrC, csrSortedValC,
                  csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCsr2cscEx2(
    cusparseHandle_t handle, int m, int n, int nnz, const void *csrVal,
    const int *csrRowPtr, const int *csrColInd, void *cscVal, int *cscColPtr,
    int *cscRowInd, cudaDataType valType, cusparseAction_t copyValues,
    cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, void *buffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const void *, const int *, const int *,
      void *, int *, int *, cudaDataType, cusparseAction_t, cusparseIndexBase_t,
      cusparseCsr2CscAlg_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCsr2cscEx2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal,
                  cscColPtr, cscRowInd, valType, copyValues, idxBase, alg,
                  buffer);
}

cusparseStatus_t CUSPARSEAPI cusparseCsr2cscEx2_bufferSize(
    cusparseHandle_t handle, int m, int n, int nnz, const void *csrVal,
    const int *csrRowPtr, const int *csrColInd, void *cscVal, int *cscColPtr,
    int *cscRowInd, cudaDataType valType, cusparseAction_t copyValues,
    cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, size_t *bufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, int, int, int, const void *, const int *, const int *,
      void *, int *, int *, cudaDataType, cusparseAction_t, cusparseIndexBase_t,
      cusparseCsr2CscAlg_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCsr2cscEx2_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal,
                  cscColPtr, cscRowInd, valType, copyValues, idxBase, alg,
                  bufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateSpVec(cusparseSpVecDescr_t *spVecDescr, int64_t size, int64_t nnz,
                    void *indices, void *values, cusparseIndexType_t idxType,
                    cusparseIndexBase_t idxBase, cudaDataType valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpVecDescr_t *, int64_t, int64_t, void *, void *,
      cusparseIndexType_t, cusparseIndexBase_t, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateSpVec");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spVecDescr, size, nnz, indices, values, idxType, idxBase,
                  valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpVecDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroySpVec");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spVecDescr);
}

cusparseStatus_t CUSPARSEAPI cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr,
                                              int64_t *size, int64_t *nnz,
                                              void **indices, void **values,
                                              cusparseIndexType_t *idxType,
                                              cusparseIndexBase_t *idxBase,
                                              cudaDataType *valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpVecDescr_t, int64_t *, int64_t *, void **, void **,
      cusparseIndexType_t *, cusparseIndexBase_t *, cudaDataType *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpVecGet");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spVecDescr, size, nnz, indices, values, idxType, idxBase,
                  valueType);
}

cusparseStatus_t CUSPARSEAPI cusparseSpVecGetIndexBase(
    cusparseSpVecDescr_t spVecDescr, cusparseIndexBase_t *idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpVecDescr_t,
                                                  cusparseIndexBase_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpVecGetIndexBase");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spVecDescr, idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr, void **values) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseSpVecDescr_t, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpVecGetValues");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spVecDescr, values);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr, void *values) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpVecDescr_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpVecSetValues");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spVecDescr, values);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateDnVec(cusparseDnVecDescr_t *dnVecDescr, int64_t size,
                    void *values, cudaDataType valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseDnVecDescr_t *, int64_t, void *, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateDnVec");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnVecDescr, size, values, valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseDnVecDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyDnVec");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnVecDescr);
}

cusparseStatus_t CUSPARSEAPI cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr,
                                              int64_t *size, void **values,
                                              cudaDataType *valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseDnVecDescr_t, int64_t *, void **, cudaDataType *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDnVecGet");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnVecDescr, size, values, valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr, void **values) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseDnVecDescr_t, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDnVecGetValues");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnVecDescr, values);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr, void *values) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseDnVecDescr_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDnVecSetValues");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnVecDescr, values);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroySpMat");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr);
}

cusparseStatus_t CUSPARSEAPI cusparseSpMatGetFormat(
    cusparseSpMatDescr_t spMatDescr, cusparseFormat_t *format) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t, cusparseFormat_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMatGetFormat");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, format);
}

cusparseStatus_t CUSPARSEAPI cusparseSpMatGetIndexBase(
    cusparseSpMatDescr_t spMatDescr, cusparseIndexBase_t *idxBase) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t,
                                                  cusparseIndexBase_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMatGetIndexBase");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr, void **values) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMatGetValues");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, values);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr, void *values) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMatSetValues");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, values);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetSize(cusparseSpMatDescr_t spMatDescr, int64_t *rows,
                     int64_t *cols, int64_t *nnz) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t, int64_t *, int64_t *, int64_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMatGetSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, rows, cols, nnz);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMatSetStridedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, batchCount);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetStridedBatch(cusparseSpMatDescr_t spMatDescr, int *batchCount) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMatGetStridedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, batchCount);
}

cusparseStatus_t CUSPARSEAPI cusparseCooSetStridedBatch(
    cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t, int, int64_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCooSetStridedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, batchCount, batchStride);
}

cusparseStatus_t CUSPARSEAPI cusparseCsrSetStridedBatch(
    cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride,
    int64_t columnsValuesBatchStride) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t, int,
                                                  int64_t, int64_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCsrSetStridedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, batchCount, offsetsBatchStride,
                  columnsValuesBatchStride);
}

#if CUDA_VERSION >= 11030

cusparseStatus_t CUSPARSEAPI cusparseSpMatGetAttribute(
    cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute,
    void *data, size_t dataSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t, cusparseSpMatAttribute_t, void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMatGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, attribute, data, dataSize);
}

cusparseStatus_t CUSPARSEAPI cusparseSpMatSetAttribute(
    cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute,
    void *data, size_t dataSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t, cusparseSpMatAttribute_t, void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMatSetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, attribute, data, dataSize);
}

#endif  // CUDA_VERSION >= 11030

cusparseStatus_t CUSPARSEAPI cusparseCreateCsr(
    cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *csrRowOffsets, void *csrColInd, void *csrValues,
    cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType,
    cusparseIndexBase_t idxBase, cudaDataType valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t *, int64_t, int64_t, int64_t, void *, void *, void *,
      cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t,
      cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateCsr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd,
                  csrValues, csrRowOffsetsType, csrColIndType, idxBase,
                  valueType);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateCsc(
    cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *cscColOffsets, void *cscRowInd, void *cscValues,
    cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType,
    cusparseIndexBase_t idxBase, cudaDataType valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t *, int64_t, int64_t, int64_t, void *, void *, void *,
      cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t,
      cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateCsc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd,
                  cscValues, cscColOffsetsType, cscRowIndType, idxBase,
                  valueType);
}

cusparseStatus_t CUSPARSEAPI cusparseCsrGet(
    cusparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz,
    void **csrRowOffsets, void **csrColInd, void **csrValues,
    cusparseIndexType_t *csrRowOffsetsType, cusparseIndexType_t *csrColIndType,
    cusparseIndexBase_t *idxBase, cudaDataType *valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t, int64_t *, int64_t *, int64_t *, void **, void **,
      void **, cusparseIndexType_t *, cusparseIndexType_t *,
      cusparseIndexBase_t *, cudaDataType *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCsrGet");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd,
                  csrValues, csrRowOffsetsType, csrColIndType, idxBase,
                  valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void *csrRowOffsets,
                       void *csrColInd, void *csrValues) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t, void *,
                                                  void *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCsrSetPointers");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, csrRowOffsets, csrColInd, csrValues);
}

cusparseStatus_t CUSPARSEAPI
cusparseCscSetPointers(cusparseSpMatDescr_t spMatDescr, void *cscColOffsets,
                       void *cscRowInd, void *cscValues) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t, void *,
                                                  void *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCscSetPointers");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, cscColOffsets, cscRowInd, cscValues);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateCoo(cusparseSpMatDescr_t *spMatDescr,
                                               int64_t rows, int64_t cols,
                                               int64_t nnz, void *cooRowInd,
                                               void *cooColInd, void *cooValues,
                                               cusparseIndexType_t cooIdxType,
                                               cusparseIndexBase_t idxBase,
                                               cudaDataType valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t *, int64_t, int64_t, int64_t, void *, void *, void *,
      cusparseIndexType_t, cusparseIndexBase_t, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateCoo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues,
                  cooIdxType, idxBase, valueType);
}

CUSPARSE_DEPRECATED(cusparseCreateCoo)
cusparseStatus_t CUSPARSEAPI cusparseCreateCooAoS(
    cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *cooInd, void *cooValues, cusparseIndexType_t cooIdxType,
    cusparseIndexBase_t idxBase, cudaDataType valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t *, int64_t, int64_t, int64_t, void *, void *,
      cusparseIndexType_t, cusparseIndexBase_t, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateCooAoS");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, rows, cols, nnz, cooInd, cooValues, cooIdxType,
                  idxBase, valueType);
}

cusparseStatus_t CUSPARSEAPI cusparseCooGet(
    cusparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz,
    void **cooRowInd,  // COO row indices
    void **cooColInd,  // COO column indices
    void **cooValues,  // COO values
    cusparseIndexType_t *idxType, cusparseIndexBase_t *idxBase,
    cudaDataType *valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t, int64_t *, int64_t *, int64_t *, void **, void **,
      void **, cusparseIndexType_t *, cusparseIndexBase_t *, cudaDataType *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCooGet");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues,
                  idxType, idxBase, valueType);
}

CUSPARSE_DEPRECATED(cusparseCooGet)
cusparseStatus_t CUSPARSEAPI cusparseCooAoSGet(cusparseSpMatDescr_t spMatDescr,
                                               int64_t *rows, int64_t *cols,
                                               int64_t *nnz,
                                               void **cooInd,     // COO indices
                                               void **cooValues,  // COO values
                                               cusparseIndexType_t *idxType,
                                               cusparseIndexBase_t *idxBase,
                                               cudaDataType *valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t, int64_t *, int64_t *, int64_t *, void **, void **,
      cusparseIndexType_t *, cusparseIndexBase_t *, cudaDataType *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCooAoSGet");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, rows, cols, nnz, cooInd, cooValues, idxType,
                  idxBase, valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseCooSetPointers(cusparseSpMatDescr_t spMatDescr, void *cooRows,
                       void *cooColumns, void *cooValues) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpMatDescr_t, void *,
                                                  void *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCooSetPointers");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, cooRows, cooColumns, cooValues);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateBlockedEll(
    cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
    int64_t ellBlockSize, int64_t ellCols, void *ellColInd, void *ellValue,
    cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase,
    cudaDataType valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t *, int64_t, int64_t, int64_t, int64_t, void *,
      void *, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateBlockedEll");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd,
                  ellValue, ellIdxType, idxBase, valueType);
}

cusparseStatus_t CUSPARSEAPI cusparseBlockedEllGet(
    cusparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
    int64_t *ellBlockSize, int64_t *ellCols, void **ellColInd, void **ellValue,
    cusparseIndexType_t *ellIdxType, cusparseIndexBase_t *idxBase,
    cudaDataType *valueType) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseSpMatDescr_t, int64_t *, int64_t *, int64_t *, int64_t *, void **,
      void **, cusparseIndexType_t *, cusparseIndexBase_t *, cudaDataType *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseBlockedEllGet");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd,
                  ellValue, ellIdxType, idxBase, valueType);
}

cusparseStatus_t CUSPARSEAPI cusparseCreateDnMat(
    cusparseDnMatDescr_t *dnMatDescr, int64_t rows, int64_t cols, int64_t ld,
    void *values, cudaDataType valueType, cusparseOrder_t order) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseDnMatDescr_t *, int64_t, int64_t, int64_t, void *, cudaDataType,
      cusparseOrder_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseCreateDnMat");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnMatDescr, rows, cols, ld, values, valueType, order);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseDnMatDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDestroyDnMat");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnMatDescr);
}

cusparseStatus_t CUSPARSEAPI cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr,
                                              int64_t *rows, int64_t *cols,
                                              int64_t *ld, void **values,
                                              cudaDataType *type,
                                              cusparseOrder_t *order) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseDnMatDescr_t, int64_t *, int64_t *, int64_t *, void **,
      cudaDataType *, cusparseOrder_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDnMatGet");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnMatDescr, rows, cols, ld, values, type, order);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr, void **values) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseDnMatDescr_t, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDnMatGetValues");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnMatDescr, values);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr, void *values) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseDnMatDescr_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDnMatSetValues");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnMatDescr, values);
}

cusparseStatus_t CUSPARSEAPI cusparseDnMatSetStridedBatch(
    cusparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseDnMatDescr_t, int, int64_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDnMatSetStridedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnMatDescr, batchCount, batchStride);
}

cusparseStatus_t CUSPARSEAPI cusparseDnMatGetStridedBatch(
    cusparseDnMatDescr_t dnMatDescr, int *batchCount, int64_t *batchStride) {
  using FuncPtr =
      cusparseStatus_t(CUSPARSEAPI *)(cusparseDnMatDescr_t, int *, int64_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDnMatGetStridedBatch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dnMatDescr, batchCount, batchStride);
}

cusparseStatus_t CUSPARSEAPI cusparseAxpby(cusparseHandle_t handle,
                                           const void *alpha,
                                           cusparseSpVecDescr_t vecX,
                                           const void *beta,
                                           cusparseDnVecDescr_t vecY) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, const void *, cusparseSpVecDescr_t, const void *,
      cusparseDnVecDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseAxpby");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha, vecX, beta, vecY);
}

cusparseStatus_t CUSPARSEAPI cusparseGather(cusparseHandle_t handle,
                                            cusparseDnVecDescr_t vecY,
                                            cusparseSpVecDescr_t vecX) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDnVecDescr_t, cusparseSpVecDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseGather");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, vecY, vecX);
}

cusparseStatus_t CUSPARSEAPI cusparseScatter(cusparseHandle_t handle,
                                             cusparseSpVecDescr_t vecX,
                                             cusparseDnVecDescr_t vecY) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseScatter");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, vecX, vecY);
}

cusparseStatus_t CUSPARSEAPI cusparseRot(cusparseHandle_t handle,
                                         const void *c_coeff,
                                         const void *s_coeff,
                                         cusparseSpVecDescr_t vecX,
                                         cusparseDnVecDescr_t vecY) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, const void *, const void *, cusparseSpVecDescr_t,
      cusparseDnVecDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseRot");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, c_coeff, s_coeff, vecX, vecY);
}

cusparseStatus_t CUSPARSEAPI cusparseSpVV_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opX, cusparseSpVecDescr_t vecX,
    cusparseDnVecDescr_t vecY, const void *result, cudaDataType computeType,
    size_t *bufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t,
      cusparseDnVecDescr_t, const void *, cudaDataType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpVV_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opX, vecX, vecY, result, computeType, bufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX,
             cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY, void *result,
             cudaDataType computeType, void *externalBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t,
      cusparseDnVecDescr_t, void *, cudaDataType, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpVV");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opX, vecX, vecY, result, computeType, externalBuffer);
}

#if CUSPARSE_VERSION >= 11300

cusparseStatus_t CUSPARSEAPI cusparseSparseToDense_bufferSize(
    cusparseHandle_t handle, cusparseSpMatDescr_t matA,
    cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg,
    size_t *bufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseSpMatDescr_t, cusparseDnMatDescr_t,
      cusparseSparseToDenseAlg_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseSparseToDense_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, matA, matB, alg, bufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseSparseToDense(
    cusparseHandle_t handle, cusparseSpMatDescr_t matA,
    cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, void *buffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseSpMatDescr_t, cusparseDnMatDescr_t,
      cusparseSparseToDenseAlg_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSparseToDense");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, matA, matB, alg, buffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_bufferSize(
    cusparseHandle_t handle, cusparseDnMatDescr_t matA,
    cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg,
    size_t *bufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDnMatDescr_t, cusparseSpMatDescr_t,
      cusparseDenseToSparseAlg_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseDenseToSparse_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, matA, matB, alg, bufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_analysis(
    cusparseHandle_t handle, cusparseDnMatDescr_t matA,
    cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void *buffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDnMatDescr_t, cusparseSpMatDescr_t,
      cusparseDenseToSparseAlg_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDenseToSparse_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, matA, matB, alg, buffer);
}

cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_convert(
    cusparseHandle_t handle, cusparseDnMatDescr_t matA,
    cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void *buffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseDnMatDescr_t, cusparseSpMatDescr_t,
      cusparseDenseToSparseAlg_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseDenseToSparse_convert");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, matA, matB, alg, buffer);
}

#endif  // CUSPARSE_VERSION >= 11300

cusparseStatus_t CUSPARSEAPI cusparseSpMV(
    cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
    cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, const void *beta,
    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg,
    void *externalBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, const void *, cusparseSpMatDescr_t,
      cusparseDnVecDescr_t, const void *, cusparseDnVecDescr_t, cudaDataType,
      cusparseSpMVAlg_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMV");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg,
                  externalBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSpMV_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
    cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, const void *beta,
    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg,
    size_t *bufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, const void *, cusparseSpMatDescr_t,
      cusparseDnVecDescr_t, const void *, cusparseDnVecDescr_t, cudaDataType,
      cusparseSpMVAlg_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMV_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg,
                  bufferSize);
}

#if CUDA_VERSION >= 11030

cusparseStatus_t CUSPARSEAPI
cusparseSpSV_createDescr(cusparseSpSVDescr_t *descr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpSVDescr_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpSV_createDescr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(descr);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpSVDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpSV_destroyDescr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(descr);
}

cusparseStatus_t CUSPARSEAPI cusparseSpSV_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
    cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX,
    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
    cusparseSpSVDescr_t spsvDescr, size_t *bufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, const void *, cusparseSpMatDescr_t,
      cusparseDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType,
      cusparseSpSVAlg_t, cusparseSpSVDescr_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpSV_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, alpha, matA, vecX, vecY, computeType, alg,
                  spsvDescr, bufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseSpSV_analysis(
    cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
    cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX,
    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
    cusparseSpSVDescr_t spsvDescr, void *externalBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, const void *, cusparseSpMatDescr_t,
      cusparseDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType,
      cusparseSpSVAlg_t, cusparseSpSVDescr_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpSV_analysis");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, alpha, matA, vecX, vecY, computeType, alg,
                  spsvDescr, externalBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSpSV_solve(
    cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
    cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX,
    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
    cusparseSpSVDescr_t spsvDescr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, const void *, cusparseSpMatDescr_t,
      cusparseDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType,
      cusparseSpSVAlg_t, cusparseSpSVDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpSV_solve");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, alpha, matA, vecX, vecY, computeType, alg,
                  spsvDescr);
}

#endif  // CUDA_VERSION >= 11030

cusparseStatus_t CUSPARSEAPI cusparseSpMM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
    const void *beta, cusparseDnMatDescr_t matC, cudaDataType computeType,
    cusparseSpMMAlg_t alg, size_t *bufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
      cusparseSpMatDescr_t, cusparseDnMatDescr_t, const void *,
      cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMM_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                  alg, bufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseSpMM_preprocess(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
    const void *beta, cusparseDnMatDescr_t matC, cudaDataType computeType,
    cusparseSpMMAlg_t alg, void *externalBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
      cusparseSpMatDescr_t, cusparseDnMatDescr_t, const void *,
      cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMM_preprocess");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                  alg, externalBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSpMM(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
    const void *beta, cusparseDnMatDescr_t matC, cudaDataType computeType,
    cusparseSpMMAlg_t alg, void *externalBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
      cusparseSpMatDescr_t, cusparseDnMatDescr_t, const void *,
      cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpMM");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                  alg, externalBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t *descr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpGEMMDescr_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpGEMM_createDescr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(descr);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(cusparseSpGEMMDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpGEMM_destroyDescr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(descr);
}

cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_workEstimation(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB,
    const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
    cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize1, void *externalBuffer1) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
      cusparseSpMatDescr_t, cusparseSpMatDescr_t, const void *,
      cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t,
      cusparseSpGEMMDescr_t, size_t *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpGEMM_workEstimation");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                  alg, spgemmDescr, bufferSize1, externalBuffer1);
}

cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_compute(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB,
    const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
    cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize2, void *externalBuffer2) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
      cusparseSpMatDescr_t, cusparseSpMatDescr_t, const void *,
      cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t,
      cusparseSpGEMMDescr_t, size_t *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpGEMM_compute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                  alg, spgemmDescr, bufferSize2, externalBuffer2);
}

cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_copy(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB,
    const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
    cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
      cusparseSpMatDescr_t, cusparseSpMatDescr_t, const void *,
      cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t,
      cusparseSpGEMMDescr_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSpGEMM_copy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                  alg, spgemmDescr);
}

CUSPARSE_DEPRECATED(cusparseSDDMM)
cusparseStatus_t CUSPARSEAPI cusparseConstrainedGeMM(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB,
    const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
    void *externalBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
      cusparseDnMatDescr_t, cusparseDnMatDescr_t, const void *,
      cusparseSpMatDescr_t, cudaDataType, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseConstrainedGeMM");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                  externalBuffer);
}

CUSPARSE_DEPRECATED(cusparseSDDMM)
cusparseStatus_t CUSPARSEAPI cusparseConstrainedGeMM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB,
    const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
    size_t *bufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
      cusparseDnMatDescr_t, cusparseDnMatDescr_t, const void *,
      cusparseSpMatDescr_t, cudaDataType, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusparseConstrainedGeMM_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                  bufferSize);
}

#if CUSPARSE_VERSION >= 11400

cusparseStatus_t CUSPARSEAPI cusparseSDDMM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB,
    const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
    cusparseSDDMMAlg_t alg, size_t *bufferSize) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
      cusparseDnMatDescr_t, cusparseDnMatDescr_t, const void *,
      cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSDDMM_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                  alg, bufferSize);
}

cusparseStatus_t CUSPARSEAPI cusparseSDDMM_preprocess(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB,
    const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
    cusparseSDDMMAlg_t alg, void *externalBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
      cusparseDnMatDescr_t, cusparseDnMatDescr_t, const void *,
      cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSDDMM_preprocess");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                  alg, externalBuffer);
}

cusparseStatus_t CUSPARSEAPI cusparseSDDMM(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB,
    const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
    cusparseSDDMMAlg_t alg, void *externalBuffer) {
  using FuncPtr = cusparseStatus_t(CUSPARSEAPI *)(
      cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
      cusparseDnMatDescr_t, cusparseDnMatDescr_t, const void *,
      cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusparseSDDMM");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                  alg, externalBuffer);
}

#endif  // CUSPARSE_VERSION >= 11400

}  // extern "C"
