// Auto-generated, do not edit.

extern "C" {
cublasStatus_t CUBLASWINAPI cublasCreate_v2 (cublasHandle_t *handle) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCreate_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle);
}

cublasStatus_t CUBLASWINAPI cublasDestroy_v2 (cublasHandle_t handle) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDestroy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle);
}

cublasStatus_t CUBLASWINAPI cublasGetVersion_v2(cublasHandle_t handle, int *version) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetVersion_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, version);
}

cublasStatus_t CUBLASWINAPI cublasGetProperty(libraryPropertyType type, int *value) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(libraryPropertyType, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetProperty");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(type, value);
}

size_t CUBLASWINAPI cublasGetCudartVersion(void) {
  using FuncPtr = size_t (CUBLASWINAPI *)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetCudartVersion");
  if (!func_ptr) return 0;
  return func_ptr();
}

cublasStatus_t CUBLASWINAPI cublasSetStream_v2 (cublasHandle_t handle, cudaStream_t streamId) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSetStream_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, streamId);
}

cublasStatus_t CUBLASWINAPI cublasGetStream_v2 (cublasHandle_t handle, cudaStream_t *streamId) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cudaStream_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetStream_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, streamId);
}

cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2 (cublasHandle_t handle, cublasPointerMode_t *mode) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasPointerMode_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetPointerMode_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode);
}

cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2 (cublasHandle_t handle, cublasPointerMode_t mode) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasPointerMode_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSetPointerMode_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode);
}

cublasStatus_t  CUBLASWINAPI cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasAtomicsMode_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetAtomicsMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode);
}

cublasStatus_t  CUBLASWINAPI cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasAtomicsMode_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSetAtomicsMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode);
}

cublasStatus_t  CUBLASWINAPI cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasMath_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetMathMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode);
}

cublasStatus_t  CUBLASWINAPI cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasMath_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSetMathMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode);
}

cublasStatus_t CUBLASWINAPI cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char* logFileName) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(int, int, int, const char *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasLoggerConfigure");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(logIsOn, logToStdOut, logToStdErr, logFileName);
}

cublasStatus_t CUBLASWINAPI cublasSetLoggerCallback(cublasLogCallback userCallback) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasLogCallback);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSetLoggerCallback");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(userCallback);
}

cublasStatus_t CUBLASWINAPI cublasGetLoggerCallback(cublasLogCallback* userCallback) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasLogCallback *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetLoggerCallback");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(userCallback);
}

cublasStatus_t CUBLASWINAPI cublasSetVector (int n, int elemSize, const void *x, 
                                             int incx, void *devicePtr, int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(int, int, const void *, int, void *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSetVector");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(n, elemSize, x, incx, devicePtr, incy);
}

cublasStatus_t CUBLASWINAPI cublasGetVector (int n, int elemSize, const void *x, 
                                             int incx, void *y, int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(int, int, const void *, int, void *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetVector");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(n, elemSize, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasSetMatrix (int rows, int cols, int elemSize, 
                                             const void *A, int lda, void *B, 
                                             int ldb) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(int, int, int, const void *, int, void *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSetMatrix");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rows, cols, elemSize, A, lda, B, ldb);
}

cublasStatus_t CUBLASWINAPI cublasGetMatrix (int rows, int cols, int elemSize, 
                                             const void *A, int lda, void *B,
                                             int ldb) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(int, int, int, const void *, int, void *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetMatrix");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rows, cols, elemSize, A, lda, B, ldb);
}

cublasStatus_t CUBLASWINAPI cublasSetVectorAsync (int n, int elemSize, 
                                                  const void *hostPtr, int incx, 
                                                  void *devicePtr, int incy,
                                                  cudaStream_t stream) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(int, int, const void *, int, void *, int, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSetVectorAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(n, elemSize, hostPtr, incx, devicePtr, incy, stream);
}

cublasStatus_t CUBLASWINAPI cublasGetVectorAsync (int n, int elemSize,
                                                  const void *devicePtr, int incx,
                                                  void *hostPtr, int incy,
                                                  cudaStream_t stream) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(int, int, const void *, int, void *, int, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetVectorAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(n, elemSize, devicePtr, incx, hostPtr, incy, stream);
}

cublasStatus_t CUBLASWINAPI cublasSetMatrixAsync (int rows, int cols, int elemSize,
                                                  const void *A, int lda, void *B,
                                                  int ldb, cudaStream_t stream) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(int, int, int, const void *, int, void *, int, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSetMatrixAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rows, cols, elemSize, A, lda, B, ldb, stream);
}

cublasStatus_t CUBLASWINAPI cublasGetMatrixAsync (int rows, int cols, int elemSize,
                                                  const void *A, int lda, void *B,
                                                  int ldb, cudaStream_t stream) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(int, int, int, const void *, int, void *, int, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetMatrixAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rows, cols, elemSize, A, lda, B, ldb, stream);
}

void CUBLASWINAPI cublasXerbla (const char *srName, int info) {
  using FuncPtr = void (CUBLASWINAPI *)(const char *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasXerbla");
  if (!func_ptr) LogFatalSymbolNotFound("cublasXerbla");
  return func_ptr(srName, info);
}

cublasStatus_t CUBLASWINAPI cublasNrm2Ex(cublasHandle_t handle, 
                                                     int n, 
                                                     const void *x, 
                                                     cudaDataType xType,
                                                     int incx, 
                                                     void *result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const void *, cudaDataType, int, void *, cudaDataType, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasNrm2Ex");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, xType, incx, result, resultType, executionType);
}

cublasStatus_t CUBLASWINAPI cublasSnrm2_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const float *x, 
                                                     int incx, 
                                                     float *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *, int, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSnrm2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasDnrm2_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const double *x, 
                                                     int incx, 
                                                     double *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *, int, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDnrm2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasScnrm2_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      float *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *, int, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasScnrm2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasDznrm2_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      double *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *, int, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDznrm2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasDotEx (cublasHandle_t handle,
                                                     int n, 
                                                     const void *x,
                                                     cudaDataType xType, 
                                                     int incx, 
                                                     const void *y, 
                                                     cudaDataType yType,
                                                     int incy,
                                                     void *result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const void *, cudaDataType, int, const void *, cudaDataType, int, void *, cudaDataType, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDotEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
}

cublasStatus_t CUBLASWINAPI cublasDotcEx (cublasHandle_t handle,
                                                     int n, 
                                                     const void *x,
                                                     cudaDataType xType, 
                                                     int incx, 
                                                     const void *y, 
                                                     cudaDataType yType,
                                                     int incy,
                                                     void *result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const void *, cudaDataType, int, const void *, cudaDataType, int, void *, cudaDataType, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDotcEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
}

cublasStatus_t CUBLASWINAPI cublasSdot_v2 (cublasHandle_t handle,
                                                     int n, 
                                                     const float *x, 
                                                     int incx, 
                                                     const float *y, 
                                                     int incy,
                                                     float *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *, int, const float *, int, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSdot_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, result);
}

cublasStatus_t CUBLASWINAPI cublasDdot_v2 (cublasHandle_t handle,
                                                     int n, 
                                                     const double *x, 
                                                     int incx, 
                                                     const double *y,
                                                     int incy,
                                                     double *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *, int, const double *, int, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDdot_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, result);
}

cublasStatus_t CUBLASWINAPI cublasCdotu_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      const cuComplex *y, 
                                                      int incy,
                                                      cuComplex *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *, int, const cuComplex *, int, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCdotu_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, result);
}

cublasStatus_t CUBLASWINAPI cublasCdotc_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      const cuComplex *y, 
                                                      int incy,
                                                      cuComplex *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *, int, const cuComplex *, int, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCdotc_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, result);
}

cublasStatus_t CUBLASWINAPI cublasZdotu_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      const cuDoubleComplex *y, 
                                                      int incy,
                                                      cuDoubleComplex *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZdotu_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, result);
}

cublasStatus_t CUBLASWINAPI cublasZdotc_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx,
                                                      const cuDoubleComplex *y, 
                                                      int incy,
                                                      cuDoubleComplex *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZdotc_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, result);
}

cublasStatus_t CUBLASWINAPI cublasScalEx(cublasHandle_t handle, 
                                                     int n, 
                                                     const void *alpha,  /* host or device pointer */
                                                     cudaDataType alphaType,
                                                     void *x, 
                                                     cudaDataType xType,
                                                     int incx,
                                                     cudaDataType executionType) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const void *, cudaDataType, void *, cudaDataType, int, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasScalEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, alphaType, x, xType, incx, executionType);
}

cublasStatus_t CUBLASWINAPI cublasSscal_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const float *alpha,  /* host or device pointer */
                                                     float *x, 
                                                     int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSscal_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasDscal_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const double *alpha,  /* host or device pointer */
                                                     double *x, 
                                                     int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDscal_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasCscal_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     cuComplex *x, 
                                                     int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCscal_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasCsscal_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */
                                                      cuComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsscal_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasZscal_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     cuDoubleComplex *x, 
                                                     int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZscal_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasZdscal_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const double *alpha, /* host or device pointer */
                                                      cuDoubleComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZdscal_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasAxpyEx (cublasHandle_t handle,
                                                      int n,
                                                      const void *alpha, /* host or device pointer */
                                                      cudaDataType alphaType,
                                                      const void *x,
                                                      cudaDataType xType,
                                                      int incx,
                                                      void *y,
                                                      cudaDataType yType,
                                                      int incy,
                                                      cudaDataType executiontype) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const void *, cudaDataType, const void *, cudaDataType, int, void *, cudaDataType, int, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasAxpyEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype);
}

cublasStatus_t CUBLASWINAPI cublasSaxpy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */
                                                      const float *x, 
                                                      int incx, 
                                                      float *y, 
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSaxpy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasDaxpy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const double *alpha, /* host or device pointer */
                                                      const double *x, 
                                                      int incx, 
                                                      double *y, 
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDaxpy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasCaxpy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      cuComplex *y, 
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCaxpy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasZaxpy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      cuDoubleComplex *y, 
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZaxpy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasCopyEx (cublasHandle_t handle,
                                                      int n, 
                                                      const void *x,
                                                      cudaDataType xType,
                                                      int incx, 
                                                      void *y,
                                                      cudaDataType yType,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const void *, cudaDataType, int, void *, cudaDataType, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCopyEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, xType, incx, y, yType, incy);
}

cublasStatus_t CUBLASWINAPI cublasScopy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const float *x, 
                                                      int incx, 
                                                      float *y, 
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasScopy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasDcopy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const double *x, 
                                                      int incx, 
                                                      double *y, 
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDcopy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasCcopy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      cuComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCcopy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasZcopy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      cuDoubleComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZcopy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasSswap_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      float *x, 
                                                      int incx, 
                                                      float *y, 
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSswap_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasDswap_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      double *x, 
                                                      int incx, 
                                                      double *y, 
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDswap_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasCswap_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      cuComplex *x, 
                                                      int incx, 
                                                      cuComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCswap_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasZswap_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      cuDoubleComplex *x, 
                                                      int incx, 
                                                      cuDoubleComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZswap_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasSwapEx (cublasHandle_t handle,
                                                      int n,
                                                      void *x,
                                                      cudaDataType xType,
                                                      int incx,
                                                      void *y,
                                                      cudaDataType yType,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, void *, cudaDataType, int, void *, cudaDataType, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSwapEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, xType, incx, y, yType, incy);
}

cublasStatus_t CUBLASWINAPI cublasIsamax_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const float *x, 
                                                      int incx, 
                                                      int *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIsamax_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasIdamax_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const double *x, 
                                                      int incx, 
                                                      int *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIdamax_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasIcamax_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      int *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIcamax_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasIzamax_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      int *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIzamax_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasIamaxEx(cublasHandle_t handle,
                                                      int n,
                                                      const void *x, cudaDataType xType,
                                                      int incx,
                                                      int *result  /* host or device pointer */
                                                    ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const void *, cudaDataType, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIamaxEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, xType, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasIsamin_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const float *x, 
                                                      int incx, 
                                                      int *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIsamin_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasIdamin_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const double *x, 
                                                      int incx, 
                                                      int *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIdamin_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasIcamin_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      int *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIcamin_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasIzamin_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      int *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIzamin_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasIaminEx(cublasHandle_t handle,
                                                      int n,
                                                      const void *x, cudaDataType xType,
                                                      int incx,
                                                      int *result /* host or device pointer */
                                                    ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const void *, cudaDataType, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIaminEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, xType, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasAsumEx(cublasHandle_t handle,
                                                     int n, 
                                                     const void *x,
                                                     cudaDataType xType,
                                                     int incx, 
                                                     void *result,
                                                     cudaDataType resultType, /* host or device pointer */
                                                     cudaDataType executiontype
                                                  ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const void *, cudaDataType, int, void *, cudaDataType, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasAsumEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, xType, incx, result, resultType, executiontype);
}

cublasStatus_t CUBLASWINAPI cublasSasum_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const float *x, 
                                                     int incx, 
                                                     float *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *, int, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSasum_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasDasum_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const double *x, 
                                                     int incx, 
                                                     double *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *, int, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDasum_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasScasum_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      float *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *, int, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasScasum_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasDzasum_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      double *result) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *, int, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDzasum_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, result);
}

cublasStatus_t CUBLASWINAPI cublasSrot_v2 (cublasHandle_t handle, 
                                                     int n, 
                                                     float *x, 
                                                     int incx, 
                                                     float *y, 
                                                     int incy, 
                                                     const float *c,  /* host or device pointer */
                                                     const float *s) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, float *, int, float *, int, const float *, const float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSrot_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, c, s);
}

cublasStatus_t CUBLASWINAPI cublasDrot_v2 (cublasHandle_t handle, 
                                                     int n, 
                                                     double *x, 
                                                     int incx, 
                                                     double *y, 
                                                     int incy, 
                                                     const double *c,  /* host or device pointer */
                                                     const double *s) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, double *, int, double *, int, const double *, const double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDrot_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, c, s);
}

cublasStatus_t CUBLASWINAPI cublasCrot_v2 (cublasHandle_t handle, 
                                                     int n, 
                                                     cuComplex *x, 
                                                     int incx, 
                                                     cuComplex *y, 
                                                     int incy, 
                                                     const float *c,      /* host or device pointer */
                                                     const cuComplex *s) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, cuComplex *, int, cuComplex *, int, const float *, const cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCrot_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, c, s);
}

cublasStatus_t CUBLASWINAPI cublasCsrot_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     cuComplex *x, 
                                                     int incx, 
                                                     cuComplex *y, 
                                                     int incy, 
                                                     const float *c,  /* host or device pointer */
                                                     const float *s) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, cuComplex *, int, cuComplex *, int, const float *, const float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsrot_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, c, s);
}

cublasStatus_t CUBLASWINAPI cublasZrot_v2 (cublasHandle_t handle, 
                                                     int n, 
                                                     cuDoubleComplex *x, 
                                                     int incx, 
                                                     cuDoubleComplex *y, 
                                                     int incy, 
                                                     const double *c,            /* host or device pointer */
                                                     const cuDoubleComplex *s) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, cuDoubleComplex *, int, cuDoubleComplex *, int, const double *, const cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZrot_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, c, s);
}

cublasStatus_t CUBLASWINAPI cublasZdrot_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     cuDoubleComplex *x, 
                                                     int incx, 
                                                     cuDoubleComplex *y, 
                                                     int incy, 
                                                     const double *c,  /* host or device pointer */
                                                     const double *s) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, cuDoubleComplex *, int, cuDoubleComplex *, int, const double *, const double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZdrot_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, c, s);
}

cublasStatus_t CUBLASWINAPI cublasRotEx (cublasHandle_t handle,
                                                     int n,
                                                     void *x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     void *y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     const void *c,  /* host or device pointer */
                                                     const void *s,
                                                     cudaDataType csType,
                                                     cudaDataType executiontype) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, void *, cudaDataType, int, void *, cudaDataType, int, const void *, const void *, cudaDataType, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasRotEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype);
}

cublasStatus_t CUBLASWINAPI cublasSrotg_v2(cublasHandle_t handle, 
                                                     float *a,   /* host or device pointer */
                                                     float *b,   /* host or device pointer */
                                                     float *c,   /* host or device pointer */
                                                     float *s) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, float *, float *, float *, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSrotg_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, a, b, c, s);
}

cublasStatus_t CUBLASWINAPI cublasDrotg_v2(cublasHandle_t handle, 
                                                     double *a,  /* host or device pointer */
                                                     double *b,  /* host or device pointer */
                                                     double *c,  /* host or device pointer */
                                                     double *s) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, double *, double *, double *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDrotg_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, a, b, c, s);
}

cublasStatus_t CUBLASWINAPI cublasCrotg_v2(cublasHandle_t handle, 
                                                     cuComplex *a,  /* host or device pointer */
                                                     cuComplex *b,  /* host or device pointer */
                                                     float *c,      /* host or device pointer */
                                                     cuComplex *s) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cuComplex *, cuComplex *, float *, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCrotg_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, a, b, c, s);
}

cublasStatus_t CUBLASWINAPI cublasZrotg_v2(cublasHandle_t handle, 
                                                     cuDoubleComplex *a,  /* host or device pointer */
                                                     cuDoubleComplex *b,  /* host or device pointer */
                                                     double *c,           /* host or device pointer */
                                                     cuDoubleComplex *s) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cuDoubleComplex *, cuDoubleComplex *, double *, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZrotg_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, a, b, c, s);
}

cublasStatus_t CUBLASWINAPI cublasRotgEx(cublasHandle_t handle,
                                                     void *a,   /* host or device pointer */
                                                     void *b,   /* host or device pointer */
                                                     cudaDataType abType,
                                                     void *c,   /* host or device pointer */
                                                     void *s,   /* host or device pointer */
                                                     cudaDataType csType,
                                                     cudaDataType executiontype) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, void *, void *, cudaDataType, void *, void *, cudaDataType, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasRotgEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, a, b, abType, c, s, csType, executiontype);
}

cublasStatus_t CUBLASWINAPI cublasSrotm_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     float *x, 
                                                     int incx, 
                                                     float *y, 
                                                     int incy, 
                                                     const float* param) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, float *, int, float *, int, const float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSrotm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, param);
}

cublasStatus_t CUBLASWINAPI cublasDrotm_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     double *x, 
                                                     int incx, 
                                                     double *y, 
                                                     int incy, 
                                                     const double* param) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, double *, int, double *, int, const double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDrotm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, incx, y, incy, param);
}

cublasStatus_t CUBLASWINAPI cublasRotmEx(cublasHandle_t handle,
                                                     int n,
                                                     void *x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     void *y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     const void* param, /* host or device pointer */
                                                     cudaDataType paramType,
                                                     cudaDataType executiontype) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, void *, cudaDataType, int, void *, cudaDataType, int, const void *, cudaDataType, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasRotmEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype);
}

cublasStatus_t CUBLASWINAPI cublasSrotmg_v2(cublasHandle_t handle, 
                                                      float *d1,        /* host or device pointer */
                                                      float *d2,        /* host or device pointer */
                                                      float *x1,        /* host or device pointer */
                                                      const float *y1,  /* host or device pointer */
                                                      float *param) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, float *, float *, float *, const float *, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSrotmg_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, d1, d2, x1, y1, param);
}

cublasStatus_t CUBLASWINAPI cublasDrotmg_v2(cublasHandle_t handle, 
                                                      double *d1,        /* host or device pointer */  
                                                      double *d2,        /* host or device pointer */  
                                                      double *x1,        /* host or device pointer */  
                                                      const double *y1,  /* host or device pointer */  
                                                      double *param) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, double *, double *, double *, const double *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDrotmg_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, d1, d2, x1, y1, param);
}

cublasStatus_t CUBLASWINAPI cublasRotmgEx(cublasHandle_t handle,
                                                      void *d1,        /* host or device pointer */
                                                      cudaDataType d1Type,
                                                      void *d2,        /* host or device pointer */
                                                      cudaDataType d2Type,
                                                      void *x1,        /* host or device pointer */
                                                      cudaDataType x1Type,
                                                      const void *y1,  /* host or device pointer */
                                                      cudaDataType y1Type,
                                                      void *param,     /* host or device pointer */
                                                      cudaDataType paramType,
                                                      cudaDataType executiontype
                                                      ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, void *, cudaDataType, void *, cudaDataType, void *, cudaDataType, const void *, cudaDataType, void *, cudaDataType, cudaDataType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasRotmgEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype);
}

cublasStatus_t CUBLASWINAPI cublasSgemv_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t trans, 
                                                      int m, 
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */
                                                      const float *A, 
                                                      int lda, 
                                                      const float *x, 
                                                      int incx, 
                                                      const float *beta,  /* host or device pointer */
                                                      float *y, 
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgemv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasDgemv_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */ 
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */
                                                      double *y, 
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const double *, const double *, int, const double *, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgemv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasCgemv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */ 
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x, 
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */ 
                                                      cuComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgemv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasZgemv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda, 
                                                      const cuDoubleComplex *x, 
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgemv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasSgbmv_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A, 
                                                      int lda, 
                                                      const float *x,
                                                      int incx,
                                                      const float *beta, /* host or device pointer */  
                                                      float *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasDgbmv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const double *alpha, /* host or device pointer */ 
                                                      const double *A,
                                                      int lda, 
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */ 
                                                      double *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, int, const double *, const double *, int, const double *, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasCgbmv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const cuComplex *alpha, /* host or device pointer */ 
                                                      const cuComplex *A,
                                                      int lda, 
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */ 
                                                      cuComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasZgbmv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const cuDoubleComplex *alpha, /* host or device pointer */ 
                                                      const cuDoubleComplex *A,
                                                      int lda, 
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */ 
                                                      cuDoubleComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasStrmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStrmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasDtrmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtrmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasCtrmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuComplex *A, 
                                                      int lda, 
                                                      cuComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtrmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasZtrmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuDoubleComplex *A, 
                                                      int lda, 
                                                      cuDoubleComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtrmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasStbmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasDtbmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasCtbmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const cuComplex *A, 
                                                      int lda, 
                                                      cuComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasZtbmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const cuDoubleComplex *A, 
                                                      int lda, 
                                                      cuDoubleComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasStpmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const float *AP, 
                                                      float *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStpmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, AP, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasDtpmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const double *AP, 
                                                      double *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtpmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, AP, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasCtpmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuComplex *AP, 
                                                      cuComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtpmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, AP, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasZtpmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuDoubleComplex *AP, 
                                                      cuDoubleComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtpmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, AP, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasStrsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStrsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasDtrsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtrsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasCtrsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuComplex *A, 
                                                      int lda, 
                                                      cuComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtrsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasZtrsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuDoubleComplex *A, 
                                                      int lda, 
                                                      cuDoubleComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtrsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasStpsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const float *AP, 
                                                      float *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStpsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, AP, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasDtpsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const double *AP, 
                                                      double *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtpsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, AP, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasCtpsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuComplex *AP, 
                                                      cuComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtpsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, AP, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasZtpsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuDoubleComplex *AP, 
                                                      cuDoubleComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtpsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, AP, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasStbsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStbsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasDtbsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtbsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasCtbsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const cuComplex *A, 
                                                      int lda, 
                                                      cuComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtbsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasZtbsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const cuDoubleComplex *A, 
                                                      int lda, 
                                                      cuDoubleComplex *x, 
                                                      int incx) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtbsv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

cublasStatus_t CUBLASWINAPI cublasSsymv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const float *alpha, /* host or device pointer */ 
                                                      const float *A,
                                                      int lda,
                                                      const float *x,
                                                      int incx,
                                                      const float *beta, /* host or device pointer */ 
                                                      float *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *, int, const float *, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsymv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasDsymv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const double *alpha, /* host or device pointer */ 
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */ 
                                                      double *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *, int, const double *, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsymv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasCsymv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */ 
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */ 
                                                      cuComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsymv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasZsymv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */ 
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta,   /* host or device pointer */ 
                                                      cuDoubleComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZsymv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasChemv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */ 
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */ 
                                                      cuComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChemv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasZhemv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */ 
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta,   /* host or device pointer */ 
                                                      cuDoubleComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhemv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasSsbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const float *alpha,   /* host or device pointer */ 
                                                      const float *A,
                                                      int lda,
                                                      const float *x, 
                                                      int incx,
                                                      const float *beta,  /* host or device pointer */ 
                                                      float *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasDsbmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const double *alpha,   /* host or device pointer */ 
                                                      const double *A,
                                                      int lda,
                                                      const double *x, 
                                                      int incx,
                                                      const double *beta,   /* host or device pointer */ 
                                                      double *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, int, const double *, const double *, int, const double *, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasChbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */ 
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x, 
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */ 
                                                      cuComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasZhbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x, 
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */ 
                                                      cuDoubleComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhbmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasSspmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo,
                                                      int n, 
                                                      const float *alpha,  /* host or device pointer */                                           
                                                      const float *AP,
                                                      const float *x,
                                                      int incx,
                                                      const float *beta,   /* host or device pointer */  
                                                      float *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *, const float *, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSspmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasDspmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *AP,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta,  /* host or device pointer */  
                                                      double *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *, const double *, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDspmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasChpmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *AP,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *y,
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChpmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasZhpmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *AP,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *y, 
                                                      int incy) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhpmv_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

cublasStatus_t CUBLASWINAPI cublasSger_v2 (cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const float *x,
                                                     int incx,
                                                     const float *y,
                                                     int incy,
                                                     float *A,
                                                     int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, const float *, const float *, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSger_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasDger_v2 (cublasHandle_t handle, 
                                                     int m,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */   
                                                     const double *x,
                                                     int incx,
                                                     const double *y,
                                                     int incy,
                                                     double *A,
                                                     int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, const double *, const double *, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDger_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasCgeru_v2 (cublasHandle_t handle, 
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *A,
                                                      int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgeru_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasCgerc_v2 (cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *A,
                                                      int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgerc_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasZgeru_v2 (cublasHandle_t handle, 
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgeru_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasZgerc_v2 (cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgerc_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasSsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const float *x,
                                                     int incx,
                                                     float *A, 
                                                     int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsyr_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasDsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const double *x,
                                                     int incx,
                                                     double *A, 
                                                     int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsyr_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasCsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex *alpha, /* host or device pointer */  
                                                     const cuComplex *x,
                                                     int incx,
                                                     cuComplex *A, 
                                                     int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsyr_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasZsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */  
                                                     const cuDoubleComplex *x,
                                                     int incx,
                                                     cuDoubleComplex *A, 
                                                     int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZsyr_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasCher_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const cuComplex *x,
                                                     int incx,
                                                     cuComplex *A, 
                                                     int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const float *, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCher_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasZher_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const cuDoubleComplex *x,
                                                     int incx,
                                                     cuDoubleComplex *A, 
                                                     int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const double *, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZher_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasSspr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const float *x,
                                                     int incx,
                                                     float *AP) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *, int, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSspr_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, AP);
}

cublasStatus_t CUBLASWINAPI cublasDspr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const double *x,
                                                     int incx,
                                                     double *AP) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *, int, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDspr_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, AP);
}

cublasStatus_t CUBLASWINAPI cublasChpr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const cuComplex *x,
                                                     int incx,
                                                     cuComplex *AP) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const float *, const cuComplex *, int, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChpr_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, AP);
}

cublasStatus_t CUBLASWINAPI cublasZhpr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const cuDoubleComplex *x,
                                                     int incx,
                                                     cuDoubleComplex *AP) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const double *, const cuDoubleComplex *, int, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhpr_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, AP);
}

cublasStatus_t CUBLASWINAPI cublasSsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *x,
                                                      int incx,
                                                      const float *y,
                                                      int incy,
                                                      float *A,
                                                      int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsyr2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasDsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n, 
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *x,
                                                      int incx,
                                                      const double *y,
                                                      int incy,
                                                      double *A,
                                                      int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsyr2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasCsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, int n, 
                                                      const cuComplex *alpha,  /* host or device pointer */  
                                                      const cuComplex *x,
                                                      int incx, 
                                                      const cuComplex *y,
                                                      int incy, 
                                                      cuComplex *A, 
                                                      int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsyr2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasZsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n, 
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */  
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZsyr2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasCher2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, int n, 
                                                      const cuComplex *alpha,  /* host or device pointer */  
                                                      const cuComplex *x,
                                                      int incx, 
                                                      const cuComplex *y,
                                                      int incy, 
                                                      cuComplex *A, 
                                                      int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCher2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasZher2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n, 
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */  
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZher2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasSspr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const float *alpha,  /* host or device pointer */  
                                                      const float *x,
                                                      int incx,
                                                      const float *y,
                                                      int incy,
                                                      float *AP) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *, int, const float *, int, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSspr2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

cublasStatus_t CUBLASWINAPI cublasDspr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha,  /* host or device pointer */  
                                                      const double *x,
                                                      int incx, 
                                                      const double *y,
                                                      int incy,
                                                      double *AP) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *, int, const double *, int, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDspr2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

cublasStatus_t CUBLASWINAPI cublasChpr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *AP) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChpr2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

cublasStatus_t CUBLASWINAPI cublasZhpr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *AP) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhpr2_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

cublasStatus_t CUBLASWINAPI cublasSgemm_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A, 
                                                      int lda,
                                                      const float *B,
                                                      int ldb, 
                                                      const float *beta, /* host or device pointer */  
                                                      float *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgemm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasDgemm_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A, 
                                                      int lda,
                                                      const double *B,
                                                      int ldb, 
                                                      const double *beta, /* host or device pointer */  
                                                      double *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double *, const double *, int, const double *, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgemm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCgemm_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *A, 
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb, 
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgemm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCgemm3m  (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *A, 
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb, 
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgemm3m");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCgemm3mEx (cublasHandle_t handle, 
                                                     cublasOperation_t transa, cublasOperation_t transb,  
                                                     int m, int n, int k, 
                                                     const cuComplex *alpha, 
                                                     const void *A, 
                                                     cudaDataType Atype, 
                                                     int lda, 
                                                     const void *B, 
                                                     cudaDataType Btype, 
                                                     int ldb,
                                                     const cuComplex *beta, 
                                                     void *C, 
                                                     cudaDataType Ctype, 
                                                     int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *, const void *, cudaDataType, int, const void *, cudaDataType, int, const cuComplex *, void *, cudaDataType, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgemm3mEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZgemm_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A, 
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb, 
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgemm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZgemm3m  (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A, 
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb, 
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgemm3m");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasSgemmEx  (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const float *alpha, /* host or device pointer */  
                                                      const void *A, 
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const void *B,
                                                      cudaDataType Btype,
                                                      int ldb, 
                                                      const float *beta, /* host or device pointer */  
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const void *, cudaDataType, int, const void *, cudaDataType, int, const float *, void *, cudaDataType, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgemmEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
}

cublasStatus_t CUBLASWINAPI cublasGemmEx  (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const void *alpha, /* host or device pointer */  
                                                      const void *A, 
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const void *B,
                                                      cudaDataType Btype,
                                                      int ldb, 
                                                      const void *beta, /* host or device pointer */  
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc,
                                                      cublasComputeType_t computeType,
                                                      cublasGemmAlgo_t algo) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void *, cudaDataType, int, const void *, cudaDataType, int, const void *, void *, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGemmEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
}

cublasStatus_t CUBLASWINAPI cublasCgemmEx (cublasHandle_t handle, 
                                                     cublasOperation_t transa, cublasOperation_t transb,  
                                                     int m, int n, int k, 
                                                     const cuComplex *alpha, 
                                                     const void *A, 
                                                     cudaDataType Atype, 
                                                     int lda, 
                                                     const void *B, 
                                                     cudaDataType Btype, 
                                                     int ldb,
                                                     const cuComplex *beta, 
                                                     void *C, 
                                                     cudaDataType Ctype, 
                                                     int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *, const void *, cudaDataType, int, const void *, cudaDataType, int, const cuComplex *, void *, cudaDataType, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgemmEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
}

cublasStatus_t CUBLASWINAPI cublasUint8gemmBias (cublasHandle_t handle, 
                                                           cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc,  
                                                           int m, int n, int k, 
                                                           const unsigned char *A, int A_bias, int lda, 
                                                           const unsigned char *B, int B_bias, int ldb,
                                                                 unsigned char *C, int C_bias, int ldc,
                                                           int C_mult, int C_shift) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, cublasOperation_t, int, int, int, const unsigned char *, int, int, const unsigned char *, int, int, unsigned char *, int, int, int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasUint8gemmBias");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift);
}

cublasStatus_t CUBLASWINAPI cublasSsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda,
                                                      const float *beta, /* host or device pointer */  
                                                      float *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *, const float *, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsyrk_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasDsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const double *alpha,  /* host or device pointer */  
                                                      const double *A,
                                                      int lda,
                                                      const double *beta,  /* host or device pointer */  
                                                      double *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double *, const double *, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsyrk_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsyrk_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *C, 
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZsyrk_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCsyrkEx ( cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const void *A, 
                                                      cudaDataType Atype, 
                                                      int lda,
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      void *C, 
                                                      cudaDataType Ctype, 
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *, const void *, cudaDataType, int, const cuComplex *, void *, cudaDataType, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsyrkEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCsyrk3mEx(cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      int n, 
                                                      int k,
                                                      const cuComplex *alpha, 
                                                      const void *A, 
                                                      cudaDataType Atype, 
                                                      int lda,
                                                      const cuComplex *beta, 
                                                      void *C, 
                                                      cudaDataType Ctype, 
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *, const void *, cudaDataType, int, const cuComplex *, void *, cudaDataType, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsyrk3mEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCherk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha,  /* host or device pointer */  
                                                      const cuComplex *A,
                                                      int lda,
                                                      const float *beta,   /* host or device pointer */  
                                                      cuComplex *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *, const cuComplex *, int, const float *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCherk_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZherk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const double *alpha,  /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const double *beta,  /* host or device pointer */  
                                                      cuDoubleComplex *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double *, const cuDoubleComplex *, int, const double *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZherk_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCherkEx  (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha,  /* host or device pointer */  
                                                      const void *A, 
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const float *beta,   /* host or device pointer */  
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *, const void *, cudaDataType, int, const float *, void *, cudaDataType, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCherkEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCherk3mEx (cublasHandle_t handle,
                                                       cublasFillMode_t uplo, 
                                                       cublasOperation_t trans, 
                                                       int n, 
                                                       int k,
                                                       const float *alpha, 
                                                       const void *A, cudaDataType Atype, 
                                                       int lda,
                                                       const float *beta, 
                                                       void *C, 
                                                       cudaDataType Ctype, 
                                                       int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *, const void *, cudaDataType, int, const float *, void *, cudaDataType, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCherk3mEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
}

cublasStatus_t CUBLASWINAPI cublasSsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const float *alpha, /* host or device pointer */  
                                                       const float *A,
                                                       int lda,
                                                       const float *B,
                                                       int ldb,
                                                       const float *beta, /* host or device pointer */  
                                                       float *C,
                                                       int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsyr2k_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasDsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const double *alpha, /* host or device pointer */  
                                                       const double *A,
                                                       int lda,
                                                       const double *B,
                                                       int ldb,
                                                       const double *beta, /* host or device pointer */  
                                                       double *C,
                                                       int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double *, const double *, int, const double *, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsyr2k_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const cuComplex *alpha, /* host or device pointer */  
                                                       const cuComplex *A,
                                                       int lda,
                                                       const cuComplex *B,
                                                       int ldb,
                                                       const cuComplex *beta, /* host or device pointer */  
                                                       cuComplex *C,
                                                       int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsyr2k_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const cuDoubleComplex *alpha,  /* host or device pointer */  
                                                       const cuDoubleComplex *A,
                                                       int lda,
                                                       const cuDoubleComplex *B,
                                                       int ldb,
                                                       const cuDoubleComplex *beta,  /* host or device pointer */  
                                                       cuDoubleComplex *C,
                                                       int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZsyr2k_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCher2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const cuComplex *alpha, /* host or device pointer */  
                                                       const cuComplex *A,
                                                       int lda,
                                                       const cuComplex *B,
                                                       int ldb,
                                                       const float *beta,   /* host or device pointer */  
                                                       cuComplex *C,
                                                       int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const float *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCher2k_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZher2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans, 
                                                       int n,
                                                       int k,
                                                       const cuDoubleComplex *alpha, /* host or device pointer */  
                                                       const cuDoubleComplex *A, 
                                                       int lda,
                                                       const cuDoubleComplex *B,
                                                       int ldb,
                                                       const double *beta, /* host or device pointer */  
                                                       cuDoubleComplex *C,
                                                       int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const double *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZher2k_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasSsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const float *alpha, /* host or device pointer */ 
                                                    const float *A,
                                                    int lda,
                                                    const float *B,
                                                    int ldb,
                                                    const float *beta, /* host or device pointer */ 
                                                    float *C,
                                                    int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsyrkx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasDsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const double *alpha, /* host or device pointer */ 
                                                    const double *A,
                                                    int lda,
                                                    const double *B,
                                                    int ldb,
                                                    const double *beta, /* host or device pointer */ 
                                                    double *C,
                                                    int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double *, const double *, int, const double *, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsyrkx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuComplex *alpha, /* host or device pointer */ 
                                                    const cuComplex *A,
                                                    int lda,
                                                    const cuComplex *B,
                                                    int ldb,
                                                    const cuComplex *beta, /* host or device pointer */ 
                                                    cuComplex *C, 
                                                    int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsyrkx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo, 
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuDoubleComplex *alpha, /* host or device pointer */ 
                                                    const cuDoubleComplex *A,
                                                    int lda,
                                                    const cuDoubleComplex *B,
                                                    int ldb,
                                                    const cuDoubleComplex *beta, /* host or device pointer */ 
                                                    cuDoubleComplex *C, 
                                                    int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZsyrkx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCherkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuComplex *alpha, /* host or device pointer */ 
                                                    const cuComplex *A,
                                                    int lda,
                                                    const cuComplex *B,
                                                    int ldb,
                                                    const float *beta, /* host or device pointer */ 
                                                    cuComplex *C,
                                                    int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const float *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCherkx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZherkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuDoubleComplex *alpha, /* host or device pointer */ 
                                                    const cuDoubleComplex *A,
                                                    int lda,
                                                    const cuDoubleComplex *B,
                                                    int ldb,
                                                    const double *beta, /* host or device pointer */ 
                                                    cuDoubleComplex *C,
                                                    int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const double *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZherkx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasSsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda,
                                                      const float *B,
                                                      int ldb,
                                                      const float *beta, /* host or device pointer */  
                                                      float *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsymm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasDsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m, 
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      const double *beta, /* host or device pointer */  
                                                      double *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const double *, const double *, int, const double *, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsymm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsymm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZsymm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasChemm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *C, 
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChemm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZhemm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhemm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasStrsm_v2 (cublasHandle_t handle, 
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda,
                                                      float *B,
                                                      int ldb) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float *, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStrsm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

cublasStatus_t CUBLASWINAPI cublasDtrsm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A, 
                                                      int lda, 
                                                      double *B,
                                                      int ldb) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double *, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtrsm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

cublasStatus_t CUBLASWINAPI cublasCtrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex *alpha, /* host or device pointer */  
                                                     const cuComplex *A,
                                                     int lda,
                                                     cuComplex *B,
                                                     int ldb) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex *, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtrsm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

cublasStatus_t CUBLASWINAPI cublasZtrsm_v2(cublasHandle_t handle, 
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */  
                                                     const cuDoubleComplex *A,                                        
                                                     int lda,
                                                     cuDoubleComplex *B,
                                                     int ldb) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtrsm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

cublasStatus_t CUBLASWINAPI cublasStrmm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda, 
                                                      const float *B,
                                                      int ldb,
                                                      float *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float *, const float *, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStrmm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasDtrmm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      double *C,
                                                      int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double *, const double *, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtrmm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCtrmm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex *alpha, /* host or device pointer */  
                                                     const cuComplex *A,
                                                     int lda,
                                                     const cuComplex *B,
                                                     int ldb,
                                                     cuComplex *C,
                                                     int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtrmm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, 
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */  
                                                     const cuDoubleComplex *A,
                                                     int lda,
                                                     const cuDoubleComplex *B,
                                                     int ldb,
                                                     cuDoubleComplex *C,
                                                     int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtrmm_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasSgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const float *alpha,  /* host or device pointer */  
                                                          const float *const Aarray[], 
                                                          int lda,
                                                          const float *const Barray[],
                                                          int ldb, 
                                                          const float *beta,   /* host or device pointer */  
                                                          float *const Carray[],
                                                          int ldc,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const float *const [], int, const float *const [], int, const float *, float *const [], int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgemmBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasDgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const double *alpha,  /* host or device pointer */ 
                                                          const double *const Aarray[], 
                                                          int lda,
                                                          const double *const Barray[],
                                                          int ldb, 
                                                          const double *beta,  /* host or device pointer */ 
                                                          double *const Carray[],
                                                          int ldc,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double *, const double *const [], int, const double *const [], int, const double *, double *const [], int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgemmBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasCgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuComplex *alpha, /* host or device pointer */ 
                                                          const cuComplex *const Aarray[], 
                                                          int lda,
                                                          const cuComplex *const Barray[],
                                                          int ldb, 
                                                          const cuComplex *beta, /* host or device pointer */ 
                                                          cuComplex *const Carray[],
                                                          int ldc,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *, const cuComplex *const [], int, const cuComplex *const [], int, const cuComplex *, cuComplex *const [], int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgemmBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasCgemm3mBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuComplex *alpha, /* host or device pointer */ 
                                                          const cuComplex *const Aarray[], 
                                                          int lda,
                                                          const cuComplex *const Barray[],
                                                          int ldb, 
                                                          const cuComplex *beta, /* host or device pointer */ 
                                                          cuComplex *const Carray[],
                                                          int ldc,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *, const cuComplex *const [], int, const cuComplex *const [], int, const cuComplex *, cuComplex *const [], int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgemm3mBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasZgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuDoubleComplex *alpha, /* host or device pointer */ 
                                                          const cuDoubleComplex *const Aarray[], 
                                                          int lda,
                                                          const cuDoubleComplex *const Barray[],
                                                          int ldb, 
                                                          const cuDoubleComplex *beta, /* host or device pointer */ 
                                                          cuDoubleComplex *const Carray[],
                                                          int ldc,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex *, const cuDoubleComplex *const [], int, const cuDoubleComplex *const [], int, const cuDoubleComplex *, cuDoubleComplex *const [], int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgemmBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx  (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const void *alpha, /* host or device pointer */  
                                                      const void *const Aarray[], 
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const void *const Barray[],
                                                      cudaDataType Btype,
                                                      int ldb, 
                                                      const void *beta, /* host or device pointer */  
                                                      void *const Carray[],
                                                      cudaDataType Ctype,
                                                      int ldc,
                                                      int batchCount,
                                                      cublasComputeType_t computeType,
                                                      cublasGemmAlgo_t algo) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void *const [], cudaDataType, int, const void *const [], cudaDataType, int, const void *, void *const [], cudaDataType, int, int, cublasComputeType_t, cublasGemmAlgo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGemmBatchedEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo);
}

cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const void *alpha,  /* host or device pointer */
                                                                 const void *A,
                                                                 cudaDataType Atype,
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const void *B,
                                                                 cudaDataType Btype,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const void *beta,   /* host or device pointer */
                                                                 void *C,
                                                                 cudaDataType Ctype,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount,
                                                                 cublasComputeType_t computeType,
                                                                 cublasGemmAlgo_t algo) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void *, cudaDataType, int, long long, const void *, cudaDataType, int, long long, const void *, void *, cudaDataType, int, long long, int, cublasComputeType_t, cublasGemmAlgo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGemmStridedBatchedEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
}

cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const float *alpha,  /* host or device pointer */
                                                                 const float *A,
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const float *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const float *beta,   /* host or device pointer */
                                                                 float *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const float *, int, long long, const float *, int, long long, const float *, float *, int, long long, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgemmStridedBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasDgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const double *alpha,  /* host or device pointer */
                                                                 const double *A, 
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const double *B,
                                                                 int ldb, 
                                                                 long long int strideB,
                                                                 const double *beta,   /* host or device pointer */
                                                                 double *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double *, const double *, int, long long, const double *, int, long long, const double *, double *, int, long long, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgemmStridedBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasCgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const cuComplex *alpha,  /* host or device pointer */
                                                                 const cuComplex *A, 
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const cuComplex *B,
                                                                 int ldb, 
                                                                 long long int strideB,
                                                                 const cuComplex *beta,   /* host or device pointer */
                                                                 cuComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *, const cuComplex *, int, long long, const cuComplex *, int, long long, const cuComplex *, cuComplex *, int, long long, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgemmStridedBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasCgemm3mStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const cuComplex *alpha,  /* host or device pointer */
                                                                 const cuComplex *A, 
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const cuComplex *B,
                                                                 int ldb, 
                                                                 long long int strideB,
                                                                 const cuComplex *beta,   /* host or device pointer */
                                                                 cuComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *, const cuComplex *, int, long long, const cuComplex *, int, long long, const cuComplex *, cuComplex *, int, long long, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgemm3mStridedBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasZgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb, 
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const cuDoubleComplex *alpha,  /* host or device pointer */
                                                                 const cuDoubleComplex *A, 
                                                                 int lda,
                                                                 long long int strideA,   /* purposely signed */
                                                                 const cuDoubleComplex *B,
                                                                 int ldb, 
                                                                 long long int strideB,
                                                                 const cuDoubleComplex *beta,   /* host or device poi */
                                                                 cuDoubleComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, long long, const cuDoubleComplex *, int, long long, const cuDoubleComplex *, cuDoubleComplex *, int, long long, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgemmStridedBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasSgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa, 
                                                  cublasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const float *alpha, /* host or device pointer */ 
                                                  const float *A, 
                                                  int lda,
                                                  const float *beta , /* host or device pointer */ 
                                                  const float *B, 
                                                  int ldb,
                                                  float *C, 
                                                  int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const float *, const float *, int, const float *, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgeam");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasDgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa, 
                                                  cublasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const double *alpha, /* host or device pointer */ 
                                                  const double *A, 
                                                  int lda,
                                                  const double *beta, /* host or device pointer */ 
                                                  const double *B, 
                                                  int ldb,
                                                  double *C, 
                                                  int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const double *, const double *, int, const double *, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgeam");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa, 
                                                  cublasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const cuComplex *alpha, /* host or device pointer */ 
                                                  const cuComplex *A, 
                                                  int lda,
                                                  const cuComplex *beta, /* host or device pointer */  
                                                  const cuComplex *B, 
                                                  int ldb,
                                                  cuComplex *C, 
                                                  int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const cuComplex *, const cuComplex *, int, const cuComplex *, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgeam");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa, 
                                                  cublasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const cuDoubleComplex *alpha, /* host or device pointer */ 
                                                  const cuDoubleComplex *A, 
                                                  int lda,
                                                  const cuDoubleComplex *beta, /* host or device pointer */  
                                                  const cuDoubleComplex *B, 
                                                  int ldb,
                                                  cuDoubleComplex *C, 
                                                  int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgeam");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasSgetrfBatched(cublasHandle_t handle,
                                                  int n, 
                                                  float *const A[],                /*Device pointer*/
                                                  int lda, 
                                                  int *P,                          /*Device Pointer*/
                                                  int *info,                       /*Device Pointer*/
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, float *const [], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgetrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasDgetrfBatched(cublasHandle_t handle,
                                                  int n, 
                                                  double *const A[],               /*Device pointer*/
                                                  int lda, 
                                                  int *P,                          /*Device Pointer*/
                                                  int *info,                       /*Device Pointer*/
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, double *const [], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgetrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasCgetrfBatched(cublasHandle_t handle,
                                                  int n, 
                                                  cuComplex *const A[],           /*Device pointer*/
                                                  int lda, 
                                                  int *P,                         /*Device Pointer*/
                                                  int *info,                      /*Device Pointer*/
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, cuComplex *const [], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgetrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasZgetrfBatched(cublasHandle_t handle,
                                                  int n, 
                                                  cuDoubleComplex *const A[],     /*Device pointer*/
                                                  int lda, 
                                                  int *P,                         /*Device Pointer*/
                                                  int *info,                      /*Device Pointer*/
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, cuDoubleComplex *const [], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgetrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasSgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const float *const A[],         /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  float *const C[],               /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *const [], int, const int *, float *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgetriBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, C, ldc, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasDgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const double *const A[],        /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  double *const C[],              /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *const [], int, const int *, double *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgetriBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, C, ldc, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasCgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const cuComplex *const A[],     /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  cuComplex *const C[],           /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *const [], int, const int *, cuComplex *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgetriBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, C, ldc, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasZgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const cuDoubleComplex *const A[], /*Device pointer*/
                                                  int lda,
                                                  const int *P,                     /*Device pointer*/
                                                  cuDoubleComplex *const C[],       /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *const [], int, const int *, cuDoubleComplex *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgetriBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, C, ldc, info, batchSize);
}

cublasStatus_t  CUBLASWINAPI cublasSgetrsBatched( cublasHandle_t handle, 
                                                            cublasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const float *const Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            float *const Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const float *const [], int, const int *, float *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgetrsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasDgetrsBatched( cublasHandle_t handle, 
                                                           cublasOperation_t trans, 
                                                           int n, 
                                                           int nrhs, 
                                                           const double *const Aarray[], 
                                                           int lda, 
                                                           const int *devIpiv, 
                                                           double *const Barray[], 
                                                           int ldb, 
                                                           int *info,
                                                           int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const double *const [], int, const int *, double *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgetrsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

cublasStatus_t  CUBLASWINAPI cublasCgetrsBatched( cublasHandle_t handle, 
                                                            cublasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const cuComplex *const Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            cuComplex *const Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex *const [], int, const int *, cuComplex *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgetrsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

cublasStatus_t  CUBLASWINAPI cublasZgetrsBatched( cublasHandle_t handle, 
                                                            cublasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const cuDoubleComplex *const Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            cuDoubleComplex *const Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex *const [], int, const int *, cuDoubleComplex *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgetrsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasStrsmBatched( cublasHandle_t    handle, 
                                                          cublasSideMode_t  side, 
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans, 
                                                          cublasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const float *alpha,           /*Host or Device Pointer*/
                                                          const float *const A[], 
                                                          int lda,
                                                          float *const B[], 
                                                          int ldb,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float *, const float *const [], int, float *const [], int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStrsmBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasDtrsmBatched( cublasHandle_t    handle, 
                                                          cublasSideMode_t  side, 
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans, 
                                                          cublasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const double *alpha,          /*Host or Device Pointer*/
                                                          const double *const A[], 
                                                          int lda,
                                                          double *const B[], 
                                                          int ldb,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double *, const double *const [], int, double *const [], int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtrsmBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasCtrsmBatched( cublasHandle_t    handle, 
                                                          cublasSideMode_t  side, 
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans, 
                                                          cublasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const cuComplex *alpha,       /*Host or Device Pointer*/
                                                          const cuComplex *const A[], 
                                                          int lda,
                                                          cuComplex *const B[], 
                                                          int ldb,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex *, const cuComplex *const [], int, cuComplex *const [], int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtrsmBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasZtrsmBatched( cublasHandle_t    handle, 
                                                          cublasSideMode_t  side, 
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans, 
                                                          cublasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const cuDoubleComplex *alpha, /*Host or Device Pointer*/
                                                          const cuDoubleComplex *const A[], 
                                                          int lda,
                                                          cuDoubleComplex *const B[], 
                                                          int ldb,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *const [], int, cuDoubleComplex *const [], int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtrsmBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasSmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const float *const A[],      /*Device pointer*/
                                                          int lda, 
                                                          float *const Ainv[],         /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *const [], int, float *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSmatinvBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasDmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const double *const A[],     /*Device pointer*/
                                                          int lda, 
                                                          double *const Ainv[],        /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *const [], int, double *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDmatinvBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasCmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const cuComplex *const A[],  /*Device pointer*/
                                                          int lda, 
                                                          cuComplex *const Ainv[],     /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *const [], int, cuComplex *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCmatinvBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasZmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const cuDoubleComplex *const A[], /*Device pointer*/
                                                          int lda, 
                                                          cuDoubleComplex *const Ainv[],    /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                        /*Device Pointer*/
                                                          int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *const [], int, cuDoubleComplex *const [], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZmatinvBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasSgeqrfBatched( cublasHandle_t handle, 
                                                           int m, 
                                                           int n,
                                                           float *const Aarray[],      /*Device pointer*/
                                                           int lda,
                                                           float *const TauArray[],    /*Device pointer*/                                                           
                                                           int *info,
                                                           int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, float *const [], int, float *const [], int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgeqrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasDgeqrfBatched( cublasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            double *const Aarray[],     /*Device pointer*/
                                                            int lda, 
                                                            double *const TauArray[],   /*Device pointer*/                                                            
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, double *const [], int, double *const [], int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgeqrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasCgeqrfBatched( cublasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            cuComplex *const Aarray[],          /*Device pointer*/
                                                            int lda, 
                                                            cuComplex *const TauArray[],        /*Device pointer*/                                                            
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, cuComplex *const [], int, cuComplex *const [], int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgeqrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasZgeqrfBatched( cublasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            cuDoubleComplex *const Aarray[],    /*Device pointer*/
                                                            int lda,
                                                            cuDoubleComplex *const TauArray[],  /*Device pointer*/
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, cuDoubleComplex *const [], int, cuDoubleComplex *const [], int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgeqrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasSgelsBatched( cublasHandle_t handle, 
                                                           cublasOperation_t trans, 
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           float *const Aarray[],      /*Device pointer*/
                                                           int lda, 
                                                           float *const Carray[],      /*Device pointer*/
                                                           int ldc,
                                                           int *info, 
                                                           int *devInfoArray,          /*Device pointer*/
                                                           int batchSize ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, float *const [], int, float *const [], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgelsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasDgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           double *const Aarray[],     /*Device pointer*/
                                                           int lda, 
                                                           double *const Carray[],     /*Device pointer*/
                                                           int ldc,
                                                           int *info, 
                                                           int *devInfoArray,          /*Device pointer*/
                                                           int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, double *const [], int, double *const [], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgelsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasCgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           cuComplex *const Aarray[],  /*Device pointer*/
                                                           int lda,
                                                           cuComplex *const Carray[],  /*Device pointer*/
                                                           int ldc,
                                                           int *info,
                                                           int *devInfoArray,
                                                           int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, cuComplex *const [], int, cuComplex *const [], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgelsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasZgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           cuDoubleComplex *const Aarray[],  /*Device pointer*/
                                                           int lda,
                                                           cuDoubleComplex *const Carray[],  /*Device pointer*/
                                                           int ldc,
                                                           int *info,
                                                           int *devInfoArray,
                                                           int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, cuDoubleComplex *const [], int, cuDoubleComplex *const [], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgelsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasSdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const float *A, 
                                                  int lda,
                                                  const float *x, 
                                                  int incx,
                                                  float *C, 
                                                  int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, int, int, const float *, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSdgmm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasDdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const double *A, 
                                                  int lda,
                                                  const double *x, 
                                                  int incx,
                                                  double *C, 
                                                  int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, int, int, const double *, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDdgmm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasCdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const cuComplex *A, 
                                                  int lda,
                                                  const cuComplex *x, 
                                                  int incx,
                                                  cuComplex *C, 
                                                  int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, int, int, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCdgmm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasZdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const cuDoubleComplex *A, 
                                                  int lda,
                                                  const cuDoubleComplex *x, 
                                                  int incx,
                                                  cuDoubleComplex *C, 
                                                  int ldc) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, int, int, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZdgmm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

cublasStatus_t CUBLASWINAPI cublasStpttr ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const float *AP,
                                                     float *A,  
                                                     int lda ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStpttr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, AP, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasDtpttr ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const double *AP,
                                                     double *A,  
                                                     int lda ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtpttr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, AP, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasCtpttr ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const cuComplex *AP,
                                                     cuComplex *A,  
                                                     int lda ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtpttr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, AP, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasZtpttr ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const cuDoubleComplex *AP,
                                                     cuDoubleComplex *A,  
                                                     int lda ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtpttr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, AP, A, lda);
}

cublasStatus_t CUBLASWINAPI cublasStrttp ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const float *A,
                                                     int lda,
                                                     float *AP ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const float *, int, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStrttp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, AP);
}

cublasStatus_t CUBLASWINAPI cublasDtrttp ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const double *A,
                                                     int lda,
                                                     double *AP ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const double *, int, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtrttp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, AP);
}

cublasStatus_t CUBLASWINAPI cublasCtrttp ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const cuComplex *A,
                                                     int lda,
                                                     cuComplex *AP ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, int, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtrttp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, AP);
}

cublasStatus_t CUBLASWINAPI cublasZtrttp ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const cuDoubleComplex *A,
                                                     int lda,
                                                     cuDoubleComplex *AP ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, int, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtrttp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, AP);
}

}  // extern "C"
