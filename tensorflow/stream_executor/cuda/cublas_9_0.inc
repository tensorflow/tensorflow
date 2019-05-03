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
                                                      cudaDataType computeType,
                                                      cublasGemmAlgo_t algo) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void *, cudaDataType, int, const void *, cudaDataType, int, const void *, void *, cudaDataType, int, cudaDataType, cublasGemmAlgo_t);
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
                                                          const float *Aarray[], 
                                                          int lda,
                                                          const float *Barray[],
                                                          int ldb, 
                                                          const float *beta,   /* host or device pointer */  
                                                          float *Carray[],
                                                          int ldc,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const float *[], int, const float *[], int, const float *, float *[], int, int);
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
                                                          const double *Aarray[], 
                                                          int lda,
                                                          const double *Barray[],
                                                          int ldb, 
                                                          const double *beta,  /* host or device pointer */ 
                                                          double *Carray[],
                                                          int ldc,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double *, const double *[], int, const double *[], int, const double *, double *[], int, int);
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
                                                          const cuComplex *Aarray[], 
                                                          int lda,
                                                          const cuComplex *Barray[],
                                                          int ldb, 
                                                          const cuComplex *beta, /* host or device pointer */ 
                                                          cuComplex *Carray[],
                                                          int ldc,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *, const cuComplex *[], int, const cuComplex *[], int, const cuComplex *, cuComplex *[], int, int);
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
                                                          const cuComplex *Aarray[], 
                                                          int lda,
                                                          const cuComplex *Barray[],
                                                          int ldb, 
                                                          const cuComplex *beta, /* host or device pointer */ 
                                                          cuComplex *Carray[],
                                                          int ldc,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *, const cuComplex *[], int, const cuComplex *[], int, const cuComplex *, cuComplex *[], int, int);
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
                                                          const cuDoubleComplex *Aarray[], 
                                                          int lda,
                                                          const cuDoubleComplex *Barray[],
                                                          int ldb, 
                                                          const cuDoubleComplex *beta, /* host or device pointer */ 
                                                          cuDoubleComplex *Carray[],
                                                          int ldc,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex *, const cuDoubleComplex *[], int, const cuDoubleComplex *[], int, const cuDoubleComplex *, cuDoubleComplex *[], int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgemmBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
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
                                                  float *A[],                      /*Device pointer*/
                                                  int lda, 
                                                  int *P,                          /*Device Pointer*/
                                                  int *info,                       /*Device Pointer*/
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, float *[], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgetrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasDgetrfBatched(cublasHandle_t handle,
                                                  int n, 
                                                  double *A[],                     /*Device pointer*/
                                                  int lda, 
                                                  int *P,                          /*Device Pointer*/
                                                  int *info,                       /*Device Pointer*/
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, double *[], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgetrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasCgetrfBatched(cublasHandle_t handle,
                                                  int n, 
                                                  cuComplex *A[],                 /*Device pointer*/
                                                  int lda, 
                                                  int *P,                         /*Device Pointer*/
                                                  int *info,                      /*Device Pointer*/
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, cuComplex *[], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgetrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasZgetrfBatched(cublasHandle_t handle,
                                                  int n, 
                                                  cuDoubleComplex *A[],           /*Device pointer*/
                                                  int lda, 
                                                  int *P,                         /*Device Pointer*/
                                                  int *info,                      /*Device Pointer*/
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, cuDoubleComplex *[], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgetrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasSgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const float *A[],               /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  float *C[],                     /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *[], int, const int *, float *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgetriBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, C, ldc, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasDgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const double *A[],              /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  double *C[],                    /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *[], int, const int *, double *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgetriBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, C, ldc, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasCgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const cuComplex *A[],            /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  cuComplex *C[],                 /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *[], int, const int *, cuComplex *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgetriBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, C, ldc, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasZgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const cuDoubleComplex *A[],     /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  cuDoubleComplex *C[],           /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *[], int, const int *, cuDoubleComplex *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgetriBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, P, C, ldc, info, batchSize);
}

cublasStatus_t  CUBLASWINAPI cublasSgetrsBatched( cublasHandle_t handle, 
                                                            cublasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const float *Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            float *Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const float *[], int, const int *, float *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgetrsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasDgetrsBatched( cublasHandle_t handle, 
                                                           cublasOperation_t trans, 
                                                           int n, 
                                                           int nrhs, 
                                                           const double *Aarray[], 
                                                           int lda, 
                                                           const int *devIpiv, 
                                                           double *Barray[], 
                                                           int ldb, 
                                                           int *info,
                                                           int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const double *[], int, const int *, double *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgetrsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

cublasStatus_t  CUBLASWINAPI cublasCgetrsBatched( cublasHandle_t handle, 
                                                            cublasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const cuComplex *Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            cuComplex *Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex *[], int, const int *, cuComplex *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgetrsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

cublasStatus_t  CUBLASWINAPI cublasZgetrsBatched( cublasHandle_t handle, 
                                                            cublasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const cuDoubleComplex *Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            cuDoubleComplex *Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex *[], int, const int *, cuDoubleComplex *[], int, int *, int);
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
                                                          const float *A[], 
                                                          int lda,
                                                          float *B[], 
                                                          int ldb,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float *, const float *[], int, float *[], int, int);
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
                                                          const double *A[], 
                                                          int lda,
                                                          double *B[], 
                                                          int ldb,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double *, const double *[], int, double *[], int, int);
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
                                                          const cuComplex *A[], 
                                                          int lda,
                                                          cuComplex *B[], 
                                                          int ldb,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex *, const cuComplex *[], int, cuComplex *[], int, int);
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
                                                          const cuDoubleComplex *A[], 
                                                          int lda,
                                                          cuDoubleComplex *B[], 
                                                          int ldb,
                                                          int batchCount) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *[], int, cuDoubleComplex *[], int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtrsmBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

cublasStatus_t CUBLASWINAPI cublasSmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const float *A[],                  /*Device pointer*/
                                                          int lda, 
                                                          float *Ainv[],               /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const float *[], int, float *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSmatinvBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasDmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const double *A[],                 /*Device pointer*/
                                                          int lda, 
                                                          double *Ainv[],              /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const double *[], int, double *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDmatinvBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasCmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const cuComplex *A[],              /*Device pointer*/
                                                          int lda, 
                                                          cuComplex *Ainv[],           /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuComplex *[], int, cuComplex *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCmatinvBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasZmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const cuDoubleComplex *A[],        /*Device pointer*/
                                                          int lda, 
                                                          cuDoubleComplex *Ainv[],     /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, const cuDoubleComplex *[], int, cuDoubleComplex *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZmatinvBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

cublasStatus_t CUBLASWINAPI cublasSgeqrfBatched( cublasHandle_t handle, 
                                                           int m, 
                                                           int n,
                                                           float *Aarray[],           /*Device pointer*/
                                                           int lda, 
                                                           float *TauArray[],        /* Device pointer*/                                                           
                                                           int *info,
                                                           int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, float *[], int, float *[], int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgeqrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasDgeqrfBatched( cublasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            double *Aarray[],           /*Device pointer*/
                                                            int lda, 
                                                            double *TauArray[],        /* Device pointer*/                                                            
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, double *[], int, double *[], int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgeqrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasCgeqrfBatched( cublasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            cuComplex *Aarray[],           /*Device pointer*/
                                                            int lda, 
                                                            cuComplex *TauArray[],        /* Device pointer*/                                                            
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, cuComplex *[], int, cuComplex *[], int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgeqrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasZgeqrfBatched( cublasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            cuDoubleComplex *Aarray[],           /*Device pointer*/
                                                            int lda, 
                                                            cuDoubleComplex *TauArray[],        /* Device pointer*/                                                          
                                                            int *info,
                                                            int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, int, int, cuDoubleComplex *[], int, cuDoubleComplex *[], int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgeqrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasSgelsBatched( cublasHandle_t handle, 
                                                           cublasOperation_t trans, 
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           float *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           float *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray, /* Device pointer*/
                                                           int batchSize ) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, float *[], int, float *[], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgelsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasDgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t trans,  
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           double *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           double *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray, /* Device pointer*/
                                                           int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, double *[], int, double *[], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgelsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasCgelsBatched( cublasHandle_t handle, 
                                                           cublasOperation_t trans, 
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           cuComplex *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           cuComplex *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray,
                                                           int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, cuComplex *[], int, cuComplex *[], int, int *, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgelsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

cublasStatus_t CUBLASWINAPI  cublasZgelsBatched( cublasHandle_t handle, 
                                                           cublasOperation_t trans, 
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           cuDoubleComplex *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           cuDoubleComplex *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray,
                                                           int batchSize) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cublasHandle_t, cublasOperation_t, int, int, int, cuDoubleComplex *[], int, cuDoubleComplex *[], int, int *, int *, int);
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

cublasStatus CUBLASWINAPI cublasInit (void) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasInit");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr();
}

cublasStatus CUBLASWINAPI cublasShutdown (void) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasShutdown");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr();
}

cublasStatus CUBLASWINAPI cublasGetError (void) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetError");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr();
}

cublasStatus CUBLASWINAPI cublasGetVersion(int *version) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasGetVersion");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(version);
}

cublasStatus CUBLASWINAPI cublasAlloc (int n, int elemSize, void **devicePtr) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(int, int, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasAlloc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(n, elemSize, devicePtr);
}

cublasStatus CUBLASWINAPI cublasFree (void *devicePtr) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasFree");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(devicePtr);
}

cublasStatus CUBLASWINAPI cublasSetKernelStream (cudaStream_t stream) {
  using FuncPtr = cublasStatus_t (CUBLASWINAPI *)(cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSetKernelStream");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(stream);
}

float CUBLASWINAPI cublasSnrm2 (int n, const float *x, int incx) {
  using FuncPtr = float (CUBLASWINAPI *)(int, const float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSnrm2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSnrm2");
  return func_ptr(n, x, incx);
}

double CUBLASWINAPI cublasDnrm2 (int n, const double *x, int incx) {
  using FuncPtr = double (CUBLASWINAPI *)(int, const double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDnrm2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDnrm2");
  return func_ptr(n, x, incx);
}

float CUBLASWINAPI cublasScnrm2 (int n, const cuComplex *x, int incx) {
  using FuncPtr = float (CUBLASWINAPI *)(int, const cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasScnrm2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasScnrm2");
  return func_ptr(n, x, incx);
}

double CUBLASWINAPI cublasDznrm2 (int n, const cuDoubleComplex *x, int incx) {
  using FuncPtr = double (CUBLASWINAPI *)(int, const cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDznrm2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDznrm2");
  return func_ptr(n, x, incx);
}

float CUBLASWINAPI cublasSdot (int n, const float *x, int incx, const float *y, 
                               int incy) {
  using FuncPtr = float (CUBLASWINAPI *)(int, const float *, int, const float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSdot");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSdot");
  return func_ptr(n, x, incx, y, incy);
}

double CUBLASWINAPI cublasDdot (int n, const double *x, int incx, const double *y, 
                               int incy) {
  using FuncPtr = double (CUBLASWINAPI *)(int, const double *, int, const double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDdot");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDdot");
  return func_ptr(n, x, incx, y, incy);
}

cuComplex CUBLASWINAPI cublasCdotu (int n, const cuComplex *x, int incx, const cuComplex *y, 
                               int incy) {
  using FuncPtr = cuComplex (CUBLASWINAPI *)(int, const cuComplex *, int, const cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCdotu");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCdotu");
  return func_ptr(n, x, incx, y, incy);
}

cuComplex CUBLASWINAPI cublasCdotc (int n, const cuComplex *x, int incx, const cuComplex *y, 
                               int incy) {
  using FuncPtr = cuComplex (CUBLASWINAPI *)(int, const cuComplex *, int, const cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCdotc");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCdotc");
  return func_ptr(n, x, incx, y, incy);
}

cuDoubleComplex CUBLASWINAPI cublasZdotu (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, 
                               int incy) {
  using FuncPtr = cuDoubleComplex (CUBLASWINAPI *)(int, const cuDoubleComplex *, int, const cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZdotu");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZdotu");
  return func_ptr(n, x, incx, y, incy);
}

cuDoubleComplex CUBLASWINAPI cublasZdotc (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, 
                               int incy) {
  using FuncPtr = cuDoubleComplex (CUBLASWINAPI *)(int, const cuDoubleComplex *, int, const cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZdotc");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZdotc");
  return func_ptr(n, x, incx, y, incy);
}

void CUBLASWINAPI cublasSscal (int n, float alpha, float *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(int, float, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSscal");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSscal");
  return func_ptr(n, alpha, x, incx);
}

void CUBLASWINAPI cublasDscal (int n, double alpha, double *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(int, double, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDscal");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDscal");
  return func_ptr(n, alpha, x, incx);
}

void CUBLASWINAPI cublasCscal (int n, cuComplex alpha, cuComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(int, cuComplex, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCscal");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCscal");
  return func_ptr(n, alpha, x, incx);
}

void CUBLASWINAPI cublasZscal (int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(int, cuDoubleComplex, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZscal");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZscal");
  return func_ptr(n, alpha, x, incx);
}

void CUBLASWINAPI cublasCsscal (int n, float alpha, cuComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(int, float, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsscal");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCsscal");
  return func_ptr(n, alpha, x, incx);
}

void CUBLASWINAPI cublasZdscal (int n, double alpha, cuDoubleComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(int, double, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZdscal");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZdscal");
  return func_ptr(n, alpha, x, incx);
}

void CUBLASWINAPI cublasSaxpy (int n, float alpha, const float *x, int incx, 
                               float *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, float, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSaxpy");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSaxpy");
  return func_ptr(n, alpha, x, incx, y, incy);
}

void CUBLASWINAPI cublasDaxpy (int n, double alpha, const double *x, 
                               int incx, double *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, double, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDaxpy");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDaxpy");
  return func_ptr(n, alpha, x, incx, y, incy);
}

void CUBLASWINAPI cublasCaxpy (int n, cuComplex alpha, const cuComplex *x, 
                               int incx, cuComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, cuComplex, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCaxpy");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCaxpy");
  return func_ptr(n, alpha, x, incx, y, incy);
}

void CUBLASWINAPI cublasZaxpy (int n, cuDoubleComplex alpha, const cuDoubleComplex *x, 
                               int incx, cuDoubleComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, cuDoubleComplex, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZaxpy");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZaxpy");
  return func_ptr(n, alpha, x, incx, y, incy);
}

void CUBLASWINAPI cublasScopy (int n, const float *x, int incx, float *y, 
                               int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasScopy");
  if (!func_ptr) LogFatalSymbolNotFound("cublasScopy");
  return func_ptr(n, x, incx, y, incy);
}

void CUBLASWINAPI cublasDcopy (int n, const double *x, int incx, double *y, 
                               int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDcopy");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDcopy");
  return func_ptr(n, x, incx, y, incy);
}

void CUBLASWINAPI cublasCcopy (int n, const cuComplex *x, int incx, cuComplex *y,
                               int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCcopy");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCcopy");
  return func_ptr(n, x, incx, y, incy);
}

void CUBLASWINAPI cublasZcopy (int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y,
                               int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZcopy");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZcopy");
  return func_ptr(n, x, incx, y, incy);
}

void CUBLASWINAPI cublasSswap (int n, float *x, int incx, float *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSswap");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSswap");
  return func_ptr(n, x, incx, y, incy);
}

void CUBLASWINAPI cublasDswap (int n, double *x, int incx, double *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDswap");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDswap");
  return func_ptr(n, x, incx, y, incy);
}

void CUBLASWINAPI cublasCswap (int n, cuComplex *x, int incx, cuComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCswap");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCswap");
  return func_ptr(n, x, incx, y, incy);
}

void CUBLASWINAPI cublasZswap (int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(int, cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZswap");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZswap");
  return func_ptr(n, x, incx, y, incy);
}

int CUBLASWINAPI cublasIsamax (int n, const float *x, int incx) {
  using FuncPtr = int (CUBLASWINAPI *)(int, const float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIsamax");
  if (!func_ptr) LogFatalSymbolNotFound("cublasIsamax");
  return func_ptr(n, x, incx);
}

int CUBLASWINAPI cublasIdamax (int n, const double *x, int incx) {
  using FuncPtr = int (CUBLASWINAPI *)(int, const double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIdamax");
  if (!func_ptr) LogFatalSymbolNotFound("cublasIdamax");
  return func_ptr(n, x, incx);
}

int CUBLASWINAPI cublasIcamax (int n, const cuComplex *x, int incx) {
  using FuncPtr = int (CUBLASWINAPI *)(int, const cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIcamax");
  if (!func_ptr) LogFatalSymbolNotFound("cublasIcamax");
  return func_ptr(n, x, incx);
}

int CUBLASWINAPI cublasIzamax (int n, const cuDoubleComplex *x, int incx) {
  using FuncPtr = int (CUBLASWINAPI *)(int, const cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIzamax");
  if (!func_ptr) LogFatalSymbolNotFound("cublasIzamax");
  return func_ptr(n, x, incx);
}

int CUBLASWINAPI cublasIsamin (int n, const float *x, int incx) {
  using FuncPtr = int (CUBLASWINAPI *)(int, const float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIsamin");
  if (!func_ptr) LogFatalSymbolNotFound("cublasIsamin");
  return func_ptr(n, x, incx);
}

int CUBLASWINAPI cublasIdamin (int n, const double *x, int incx) {
  using FuncPtr = int (CUBLASWINAPI *)(int, const double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIdamin");
  if (!func_ptr) LogFatalSymbolNotFound("cublasIdamin");
  return func_ptr(n, x, incx);
}

int CUBLASWINAPI cublasIcamin (int n, const cuComplex *x, int incx) {
  using FuncPtr = int (CUBLASWINAPI *)(int, const cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIcamin");
  if (!func_ptr) LogFatalSymbolNotFound("cublasIcamin");
  return func_ptr(n, x, incx);
}

int CUBLASWINAPI cublasIzamin (int n, const cuDoubleComplex *x, int incx) {
  using FuncPtr = int (CUBLASWINAPI *)(int, const cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasIzamin");
  if (!func_ptr) LogFatalSymbolNotFound("cublasIzamin");
  return func_ptr(n, x, incx);
}

float CUBLASWINAPI cublasSasum (int n, const float *x, int incx) {
  using FuncPtr = float (CUBLASWINAPI *)(int, const float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSasum");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSasum");
  return func_ptr(n, x, incx);
}

double CUBLASWINAPI cublasDasum (int n, const double *x, int incx) {
  using FuncPtr = double (CUBLASWINAPI *)(int, const double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDasum");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDasum");
  return func_ptr(n, x, incx);
}

float CUBLASWINAPI cublasScasum (int n, const cuComplex *x, int incx) {
  using FuncPtr = float (CUBLASWINAPI *)(int, const cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasScasum");
  if (!func_ptr) LogFatalSymbolNotFound("cublasScasum");
  return func_ptr(n, x, incx);
}

double CUBLASWINAPI cublasDzasum (int n, const cuDoubleComplex *x, int incx) {
  using FuncPtr = double (CUBLASWINAPI *)(int, const cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDzasum");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDzasum");
  return func_ptr(n, x, incx);
}

void CUBLASWINAPI cublasSrot (int n, float *x, int incx, float *y, int incy, 
                              float sc, float ss) {
  using FuncPtr = void (CUBLASWINAPI *)(int, float *, int, float *, int, float, float);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSrot");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSrot");
  return func_ptr(n, x, incx, y, incy, sc, ss);
}

void CUBLASWINAPI cublasDrot (int n, double *x, int incx, double *y, int incy, 
                              double sc, double ss) {
  using FuncPtr = void (CUBLASWINAPI *)(int, double *, int, double *, int, double, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDrot");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDrot");
  return func_ptr(n, x, incx, y, incy, sc, ss);
}

void CUBLASWINAPI cublasCrot (int n, cuComplex *x, int incx, cuComplex *y, 
                              int incy, float c, cuComplex s) {
  using FuncPtr = void (CUBLASWINAPI *)(int, cuComplex *, int, cuComplex *, int, float, cuComplex);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCrot");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCrot");
  return func_ptr(n, x, incx, y, incy, c, s);
}

void CUBLASWINAPI cublasZrot (int n, cuDoubleComplex *x, int incx, 
                              cuDoubleComplex *y, int incy, double sc, 
                              cuDoubleComplex cs) {
  using FuncPtr = void (CUBLASWINAPI *)(int, cuDoubleComplex *, int, cuDoubleComplex *, int, double, cuDoubleComplex);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZrot");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZrot");
  return func_ptr(n, x, incx, y, incy, sc, cs);
}

void CUBLASWINAPI cublasCsrot (int n, cuComplex *x, int incx, cuComplex *y,
                               int incy, float c, float s) {
  using FuncPtr = void (CUBLASWINAPI *)(int, cuComplex *, int, cuComplex *, int, float, float);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsrot");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCsrot");
  return func_ptr(n, x, incx, y, incy, c, s);
}

void CUBLASWINAPI cublasZdrot (int n, cuDoubleComplex *x, int incx, 
                               cuDoubleComplex *y, int incy, double c, double s) {
  using FuncPtr = void (CUBLASWINAPI *)(int, cuDoubleComplex *, int, cuDoubleComplex *, int, double, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZdrot");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZdrot");
  return func_ptr(n, x, incx, y, incy, c, s);
}

void CUBLASWINAPI cublasSrotg (float *sa, float *sb, float *sc, float *ss) {
  using FuncPtr = void (CUBLASWINAPI *)(float *, float *, float *, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSrotg");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSrotg");
  return func_ptr(sa, sb, sc, ss);
}

void CUBLASWINAPI cublasDrotg (double *sa, double *sb, double *sc, double *ss) {
  using FuncPtr = void (CUBLASWINAPI *)(double *, double *, double *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDrotg");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDrotg");
  return func_ptr(sa, sb, sc, ss);
}

void CUBLASWINAPI cublasCrotg (cuComplex *ca, cuComplex cb, float *sc,
                               cuComplex *cs) {
  using FuncPtr = void (CUBLASWINAPI *)(cuComplex *, cuComplex, float *, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCrotg");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCrotg");
  return func_ptr(ca, cb, sc, cs);
}

void CUBLASWINAPI cublasZrotg (cuDoubleComplex *ca, cuDoubleComplex cb, double *sc,
                               cuDoubleComplex *cs) {
  using FuncPtr = void (CUBLASWINAPI *)(cuDoubleComplex *, cuDoubleComplex, double *, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZrotg");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZrotg");
  return func_ptr(ca, cb, sc, cs);
}

void CUBLASWINAPI cublasSrotm(int n, float *x, int incx, float *y, int incy, 
                              const float* sparam) {
  using FuncPtr = void (CUBLASWINAPI *)(int, float *, int, float *, int, const float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSrotm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSrotm");
  return func_ptr(n, x, incx, y, incy, sparam);
}

void CUBLASWINAPI cublasDrotm(int n, double *x, int incx, double *y, int incy, 
                              const double* sparam) {
  using FuncPtr = void (CUBLASWINAPI *)(int, double *, int, double *, int, const double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDrotm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDrotm");
  return func_ptr(n, x, incx, y, incy, sparam);
}

void CUBLASWINAPI cublasSrotmg (float *sd1, float *sd2, float *sx1, 
                                const float *sy1, float* sparam) {
  using FuncPtr = void (CUBLASWINAPI *)(float *, float *, float *, const float *, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSrotmg");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSrotmg");
  return func_ptr(sd1, sd2, sx1, sy1, sparam);
}

void CUBLASWINAPI cublasDrotmg (double *sd1, double *sd2, double *sx1, 
                                const double *sy1, double* sparam) {
  using FuncPtr = void (CUBLASWINAPI *)(double *, double *, double *, const double *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDrotmg");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDrotmg");
  return func_ptr(sd1, sd2, sx1, sy1, sparam);
}

void CUBLASWINAPI cublasSgemv (char trans, int m, int n, float alpha,
                               const float *A, int lda, const float *x, int incx,
                               float beta, float *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, float, const float *, int, const float *, int, float, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgemv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSgemv");
  return func_ptr(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasDgemv (char trans, int m, int n, double alpha,
                               const double *A, int lda, const double *x, int incx,
                               double beta, double *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, double, const double *, int, const double *, int, double, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgemv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDgemv");
  return func_ptr(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasCgemv (char trans, int m, int n, cuComplex alpha,
                               const cuComplex *A, int lda, const cuComplex *x, int incx,
                               cuComplex beta, cuComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgemv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCgemv");
  return func_ptr(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasZgemv (char trans, int m, int n, cuDoubleComplex alpha,
                               const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
                               cuDoubleComplex beta, cuDoubleComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgemv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZgemv");
  return func_ptr(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasSgbmv (char trans, int m, int n, int kl, int ku, 
                               float alpha, const float *A, int lda, 
                               const float *x, int incx, float beta, float *y, 
                               int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, int, int, float, const float *, int, const float *, int, float, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSgbmv");
  return func_ptr(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasDgbmv (char trans, int m, int n, int kl, int ku, 
                               double alpha, const double *A, int lda, 
                               const double *x, int incx, double beta, double *y, 
                               int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDgbmv");
  return func_ptr(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasCgbmv (char trans, int m, int n, int kl, int ku, 
                               cuComplex alpha, const cuComplex *A, int lda, 
                               const cuComplex *x, int incx, cuComplex beta, cuComplex *y, 
                               int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCgbmv");
  return func_ptr(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasZgbmv (char trans, int m, int n, int kl, int ku, 
                               cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, 
                               const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, 
                               int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZgbmv");
  return func_ptr(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasStrmv (char uplo, char trans, char diag, int n, 
                               const float *A, int lda, float *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStrmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasStrmv");
  return func_ptr(uplo, trans, diag, n, A, lda, x, incx);
}

void CUBLASWINAPI cublasDtrmv (char uplo, char trans, char diag, int n, 
                               const double *A, int lda, double *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtrmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDtrmv");
  return func_ptr(uplo, trans, diag, n, A, lda, x, incx);
}

void CUBLASWINAPI cublasCtrmv (char uplo, char trans, char diag, int n, 
                               const cuComplex *A, int lda, cuComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtrmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCtrmv");
  return func_ptr(uplo, trans, diag, n, A, lda, x, incx);
}

void CUBLASWINAPI cublasZtrmv (char uplo, char trans, char diag, int n, 
                               const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtrmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZtrmv");
  return func_ptr(uplo, trans, diag, n, A, lda, x, incx);
}

void CUBLASWINAPI cublasStbmv (char uplo, char trans, char diag, int n, int k, 
                               const float *A, int lda, float *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasStbmv");
  return func_ptr(uplo, trans, diag, n, k, A, lda, x, incx);
}

void CUBLASWINAPI cublasDtbmv (char uplo, char trans, char diag, int n, int k, 
                               const double *A, int lda, double *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDtbmv");
  return func_ptr(uplo, trans, diag, n, k, A, lda, x, incx);
}

void CUBLASWINAPI cublasCtbmv (char uplo, char trans, char diag, int n, int k, 
                               const cuComplex *A, int lda, cuComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCtbmv");
  return func_ptr(uplo, trans, diag, n, k, A, lda, x, incx);
}

void CUBLASWINAPI cublasZtbmv (char uplo, char trans, char diag, int n, int k, 
                               const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZtbmv");
  return func_ptr(uplo, trans, diag, n, k, A, lda, x, incx);
}

void CUBLASWINAPI cublasStpmv(char uplo, char trans, char diag, int n, const float *AP, float *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStpmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasStpmv");
  return func_ptr(uplo, trans, diag, n, AP, x, incx);
}

void CUBLASWINAPI cublasDtpmv(char uplo, char trans, char diag, int n, const double *AP, double *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtpmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDtpmv");
  return func_ptr(uplo, trans, diag, n, AP, x, incx);
}

void CUBLASWINAPI cublasCtpmv(char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtpmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCtpmv");
  return func_ptr(uplo, trans, diag, n, AP, x, incx);
}

void CUBLASWINAPI cublasZtpmv(char uplo, char trans, char diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtpmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZtpmv");
  return func_ptr(uplo, trans, diag, n, AP, x, incx);
}

void CUBLASWINAPI cublasStrsv(char uplo, char trans, char diag, int n, const float *A, int lda, float *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStrsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasStrsv");
  return func_ptr(uplo, trans, diag, n, A, lda, x, incx);
}

void CUBLASWINAPI cublasDtrsv(char uplo, char trans, char diag, int n, const double *A, int lda, double *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtrsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDtrsv");
  return func_ptr(uplo, trans, diag, n, A, lda, x, incx);
}

void CUBLASWINAPI cublasCtrsv(char uplo, char trans, char diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtrsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCtrsv");
  return func_ptr(uplo, trans, diag, n, A, lda, x, incx);
}

void CUBLASWINAPI cublasZtrsv(char uplo, char trans, char diag, int n, const cuDoubleComplex *A, int lda, 
                              cuDoubleComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtrsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZtrsv");
  return func_ptr(uplo, trans, diag, n, A, lda, x, incx);
}

void CUBLASWINAPI cublasStpsv(char uplo, char trans, char diag, int n, const float *AP, 
                              float *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const float *, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStpsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasStpsv");
  return func_ptr(uplo, trans, diag, n, AP, x, incx);
}

void CUBLASWINAPI cublasDtpsv(char uplo, char trans, char diag, int n, const double *AP, double *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const double *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtpsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDtpsv");
  return func_ptr(uplo, trans, diag, n, AP, x, incx);
}

void CUBLASWINAPI cublasCtpsv(char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const cuComplex *, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtpsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCtpsv");
  return func_ptr(uplo, trans, diag, n, AP, x, incx);
}

void CUBLASWINAPI cublasZtpsv(char uplo, char trans, char diag, int n, const cuDoubleComplex *AP, 
                              cuDoubleComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, const cuDoubleComplex *, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtpsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZtpsv");
  return func_ptr(uplo, trans, diag, n, AP, x, incx);
}

void CUBLASWINAPI cublasStbsv(char uplo, char trans, 
                              char diag, int n, int k, const float *A, 
                              int lda, float *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStbsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasStbsv");
  return func_ptr(uplo, trans, diag, n, k, A, lda, x, incx);
}

void CUBLASWINAPI cublasDtbsv(char uplo, char trans, 
                              char diag, int n, int k, const double *A, 
                              int lda, double *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtbsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDtbsv");
  return func_ptr(uplo, trans, diag, n, k, A, lda, x, incx);
}

void CUBLASWINAPI cublasCtbsv(char uplo, char trans, 
                              char diag, int n, int k, const cuComplex *A, 
                              int lda, cuComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtbsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCtbsv");
  return func_ptr(uplo, trans, diag, n, k, A, lda, x, incx);
}

void CUBLASWINAPI cublasZtbsv(char uplo, char trans, 
                              char diag, int n, int k, const cuDoubleComplex *A, 
                              int lda, cuDoubleComplex *x, int incx) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, int, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtbsv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZtbsv");
  return func_ptr(uplo, trans, diag, n, k, A, lda, x, incx);
}

void CUBLASWINAPI cublasSsymv (char uplo, int n, float alpha, const float *A,
                               int lda, const float *x, int incx, float beta, 
                               float *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, float, const float *, int, const float *, int, float, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsymv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSsymv");
  return func_ptr(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasDsymv (char uplo, int n, double alpha, const double *A,
                               int lda, const double *x, int incx, double beta, 
                               double *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, double, const double *, int, const double *, int, double, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsymv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDsymv");
  return func_ptr(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasChemv (char uplo, int n, cuComplex alpha, const cuComplex *A,
                               int lda, const cuComplex *x, int incx, cuComplex beta, 
                               cuComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChemv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasChemv");
  return func_ptr(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasZhemv (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *A,
                               int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta, 
                               cuDoubleComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhemv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZhemv");
  return func_ptr(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasSsbmv (char uplo, int n, int k, float alpha, 
                               const float *A, int lda, const float *x, int incx, 
                               float beta, float *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, float, const float *, int, const float *, int, float, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSsbmv");
  return func_ptr(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasDsbmv (char uplo, int n, int k, double alpha, 
                               const double *A, int lda, const double *x, int incx, 
                               double beta, double *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, double, const double *, int, const double *, int, double, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDsbmv");
  return func_ptr(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasChbmv (char uplo, int n, int k, cuComplex alpha, 
                               const cuComplex *A, int lda, const cuComplex *x, int incx, 
                               cuComplex beta, cuComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasChbmv");
  return func_ptr(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasZhbmv (char uplo, int n, int k, cuDoubleComplex alpha, 
                               const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, 
                               cuDoubleComplex beta, cuDoubleComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhbmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZhbmv");
  return func_ptr(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasSspmv(char uplo, int n, float alpha,
                              const float *AP, const float *x,
                              int incx, float beta, float *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, float, const float *, const float *, int, float, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSspmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSspmv");
  return func_ptr(uplo, n, alpha, AP, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasDspmv(char uplo, int n, double alpha,
                              const double *AP, const double *x,
                              int incx, double beta, double *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, double, const double *, const double *, int, double, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDspmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDspmv");
  return func_ptr(uplo, n, alpha, AP, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasChpmv(char uplo, int n, cuComplex alpha,
                              const cuComplex *AP, const cuComplex *x,
                              int incx, cuComplex beta, cuComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, cuComplex, const cuComplex *, const cuComplex *, int, cuComplex, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChpmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasChpmv");
  return func_ptr(uplo, n, alpha, AP, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasZhpmv(char uplo, int n, cuDoubleComplex alpha,
                              const cuDoubleComplex *AP, const cuDoubleComplex *x,
                              int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, cuDoubleComplex, const cuDoubleComplex *, const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhpmv");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZhpmv");
  return func_ptr(uplo, n, alpha, AP, x, incx, beta, y, incy);
}

void CUBLASWINAPI cublasSger (int m, int n, float alpha, const float *x, int incx,
                              const float *y, int incy, float *A, int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(int, int, float, const float *, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSger");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSger");
  return func_ptr(m, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLASWINAPI cublasDger (int m, int n, double alpha, const double *x, int incx,
                              const double *y, int incy, double *A, int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(int, int, double, const double *, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDger");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDger");
  return func_ptr(m, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLASWINAPI cublasCgeru (int m, int n, cuComplex alpha, const cuComplex *x,
                               int incx, const cuComplex *y, int incy,
                               cuComplex *A, int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgeru");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCgeru");
  return func_ptr(m, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLASWINAPI cublasCgerc (int m, int n, cuComplex alpha, const cuComplex *x,
                               int incx, const cuComplex *y, int incy,
                               cuComplex *A, int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgerc");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCgerc");
  return func_ptr(m, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLASWINAPI cublasZgeru (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x,
                               int incx, const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *A, int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgeru");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZgeru");
  return func_ptr(m, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLASWINAPI cublasZgerc (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x,
                               int incx, const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *A, int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgerc");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZgerc");
  return func_ptr(m, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLASWINAPI cublasSsyr (char uplo, int n, float alpha, const float *x,
                              int incx, float *A, int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, float, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsyr");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSsyr");
  return func_ptr(uplo, n, alpha, x, incx, A, lda);
}

void CUBLASWINAPI cublasDsyr (char uplo, int n, double alpha, const double *x,
                              int incx, double *A, int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, double, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsyr");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDsyr");
  return func_ptr(uplo, n, alpha, x, incx, A, lda);
}

void CUBLASWINAPI cublasCher (char uplo, int n, float alpha, 
                              const cuComplex *x, int incx, cuComplex *A, int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, float, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCher");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCher");
  return func_ptr(uplo, n, alpha, x, incx, A, lda);
}

void CUBLASWINAPI cublasZher (char uplo, int n, double alpha, 
                              const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, double, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZher");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZher");
  return func_ptr(uplo, n, alpha, x, incx, A, lda);
}

void CUBLASWINAPI cublasSspr (char uplo, int n, float alpha, const float *x,
                              int incx, float *AP) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, float, const float *, int, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSspr");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSspr");
  return func_ptr(uplo, n, alpha, x, incx, AP);
}

void CUBLASWINAPI cublasDspr (char uplo, int n, double alpha, const double *x,
                              int incx, double *AP) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, double, const double *, int, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDspr");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDspr");
  return func_ptr(uplo, n, alpha, x, incx, AP);
}

void CUBLASWINAPI cublasChpr (char uplo, int n, float alpha, const cuComplex *x,
                              int incx, cuComplex *AP) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, float, const cuComplex *, int, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChpr");
  if (!func_ptr) LogFatalSymbolNotFound("cublasChpr");
  return func_ptr(uplo, n, alpha, x, incx, AP);
}

void CUBLASWINAPI cublasZhpr (char uplo, int n, double alpha, const cuDoubleComplex *x,
                              int incx, cuDoubleComplex *AP) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, double, const cuDoubleComplex *, int, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhpr");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZhpr");
  return func_ptr(uplo, n, alpha, x, incx, AP);
}

void CUBLASWINAPI cublasSsyr2 (char uplo, int n, float alpha, const float *x, 
                               int incx, const float *y, int incy, float *A, 
                               int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, float, const float *, int, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsyr2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSsyr2");
  return func_ptr(uplo, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLASWINAPI cublasDsyr2 (char uplo, int n, double alpha, const double *x, 
                               int incx, const double *y, int incy, double *A, 
                               int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, double, const double *, int, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsyr2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDsyr2");
  return func_ptr(uplo, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLASWINAPI cublasCher2 (char uplo, int n, cuComplex alpha, const cuComplex *x, 
                               int incx, const cuComplex *y, int incy, cuComplex *A, 
                               int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCher2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCher2");
  return func_ptr(uplo, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLASWINAPI cublasZher2 (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, 
                               int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, 
                               int lda) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZher2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZher2");
  return func_ptr(uplo, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLASWINAPI cublasSspr2 (char uplo, int n, float alpha, const float *x, 
                               int incx, const float *y, int incy, float *AP) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, float, const float *, int, const float *, int, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSspr2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSspr2");
  return func_ptr(uplo, n, alpha, x, incx, y, incy, AP);
}

void CUBLASWINAPI cublasDspr2 (char uplo, int n, double alpha,
                               const double *x, int incx, const double *y,
                               int incy, double *AP) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, double, const double *, int, const double *, int, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDspr2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDspr2");
  return func_ptr(uplo, n, alpha, x, incx, y, incy, AP);
}

void CUBLASWINAPI cublasChpr2 (char uplo, int n, cuComplex alpha,
                               const cuComplex *x, int incx, const cuComplex *y,
                               int incy, cuComplex *AP) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChpr2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasChpr2");
  return func_ptr(uplo, n, alpha, x, incx, y, incy, AP);
}

void CUBLASWINAPI cublasZhpr2 (char uplo, int n, cuDoubleComplex alpha,
                               const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,
                               int incy, cuDoubleComplex *AP) {
  using FuncPtr = void (CUBLASWINAPI *)(char, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhpr2");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZhpr2");
  return func_ptr(uplo, n, alpha, x, incx, y, incy, AP);
}

void CUBLASWINAPI cublasSgemm (char transa, char transb, int m, int n, int k, 
                               float alpha, const float *A, int lda, 
                               const float *B, int ldb, float beta, float *C, 
                               int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, int, float, const float *, int, const float *, int, float, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSgemm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSgemm");
  return func_ptr(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasDgemm (char transa, char transb, int m, int n, int k,
                               double alpha, const double *A, int lda, 
                               const double *B, int ldb, double beta, double *C, 
                               int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, int, double, const double *, int, const double *, int, double, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDgemm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDgemm");
  return func_ptr(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasCgemm (char transa, char transb, int m, int n, int k, 
                               cuComplex alpha, const cuComplex *A, int lda,
                               const cuComplex *B, int ldb, cuComplex beta,
                               cuComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCgemm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCgemm");
  return func_ptr(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasZgemm (char transa, char transb, int m, int n,
                               int k, cuDoubleComplex alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *B, int ldb,
                               cuDoubleComplex beta, cuDoubleComplex *C,
                               int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZgemm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZgemm");
  return func_ptr(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasSsyrk (char uplo, char trans, int n, int k, float alpha, 
                               const float *A, int lda, float beta, float *C, 
                               int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, float, const float *, int, float, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsyrk");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSsyrk");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

void CUBLASWINAPI cublasDsyrk (char uplo, char trans, int n, int k,
                               double alpha, const double *A, int lda,
                               double beta, double *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, double, const double *, int, double, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsyrk");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDsyrk");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

void CUBLASWINAPI cublasCsyrk (char uplo, char trans, int n, int k,
                               cuComplex alpha, const cuComplex *A, int lda,
                               cuComplex beta, cuComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, cuComplex, const cuComplex *, int, cuComplex, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsyrk");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCsyrk");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

void CUBLASWINAPI cublasZsyrk (char uplo, char trans, int n, int k,
                               cuDoubleComplex alpha,
                               const cuDoubleComplex *A, int lda,
                               cuDoubleComplex beta,
                               cuDoubleComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZsyrk");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZsyrk");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

void CUBLASWINAPI cublasCherk (char uplo, char trans, int n, int k,
                               float alpha, const cuComplex *A, int lda,
                               float beta, cuComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, float, const cuComplex *, int, float, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCherk");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCherk");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

void CUBLASWINAPI cublasZherk (char uplo, char trans, int n, int k,
                               double alpha,
                               const cuDoubleComplex *A, int lda,
                               double beta,
                               cuDoubleComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, double, const cuDoubleComplex *, int, double, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZherk");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZherk");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

void CUBLASWINAPI cublasSsyr2k (char uplo, char trans, int n, int k, float alpha, 
                                const float *A, int lda, const float *B, int ldb, 
                                float beta, float *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, float, const float *, int, const float *, int, float, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsyr2k");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSsyr2k");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasDsyr2k (char uplo, char trans, int n, int k,
                                double alpha, const double *A, int lda,
                                const double *B, int ldb, double beta,
                                double *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, double, const double *, int, const double *, int, double, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsyr2k");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDsyr2k");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasCsyr2k (char uplo, char trans, int n, int k,
                                cuComplex alpha, const cuComplex *A, int lda,
                                const cuComplex *B, int ldb, cuComplex beta,
                                cuComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsyr2k");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCsyr2k");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasZsyr2k (char uplo, char trans, int n, int k,
                                cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                const cuDoubleComplex *B, int ldb, cuDoubleComplex beta,
                                cuDoubleComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZsyr2k");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZsyr2k");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasCher2k (char uplo, char trans, int n, int k,
                                cuComplex alpha, const cuComplex *A, int lda,
                                const cuComplex *B, int ldb, float beta,
                                cuComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, float, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCher2k");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCher2k");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasZher2k (char uplo, char trans, int n, int k,
                                cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                const cuDoubleComplex *B, int ldb, double beta,
                                cuDoubleComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, double, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZher2k");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZher2k");
  return func_ptr(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasSsymm (char side, char uplo, int m, int n, float alpha, 
                               const float *A, int lda, const float *B, int ldb,
                               float beta, float *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, float, const float *, int, const float *, int, float, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasSsymm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasSsymm");
  return func_ptr(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasDsymm (char side, char uplo, int m, int n, double alpha, 
                               const double *A, int lda, const double *B, int ldb,
                               double beta, double *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, double, const double *, int, const double *, int, double, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDsymm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDsymm");
  return func_ptr(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasCsymm (char side, char uplo, int m, int n, cuComplex alpha, 
                               const cuComplex *A, int lda, const cuComplex *B, int ldb,
                               cuComplex beta, cuComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCsymm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCsymm");
  return func_ptr(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasZsymm (char side, char uplo, int m, int n, cuDoubleComplex alpha, 
                               const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
                               cuDoubleComplex beta, cuDoubleComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZsymm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZsymm");
  return func_ptr(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasChemm (char side, char uplo, int m, int n,
                               cuComplex alpha, const cuComplex *A, int lda,
                               const cuComplex *B, int ldb, cuComplex beta,
                               cuComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasChemm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasChemm");
  return func_ptr(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasZhemm (char side, char uplo, int m, int n,
                               cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *B, int ldb, cuDoubleComplex beta,
                               cuDoubleComplex *C, int ldc) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZhemm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZhemm");
  return func_ptr(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CUBLASWINAPI cublasStrsm (char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A, int lda,
                               float *B, int ldb) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, char, int, int, float, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStrsm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasStrsm");
  return func_ptr(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

void CUBLASWINAPI cublasDtrsm (char side, char uplo, char transa,
                               char diag, int m, int n, double alpha,
                               const double *A, int lda, double *B,
                               int ldb) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, char, int, int, double, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtrsm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDtrsm");
  return func_ptr(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

void CUBLASWINAPI cublasCtrsm (char side, char uplo, char transa, char diag,
                               int m, int n, cuComplex alpha, const cuComplex *A,
                               int lda, cuComplex *B, int ldb) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, char, int, int, cuComplex, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtrsm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCtrsm");
  return func_ptr(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

void CUBLASWINAPI cublasZtrsm (char side, char uplo, char transa,
                               char diag, int m, int n, cuDoubleComplex alpha,
                               const cuDoubleComplex *A, int lda,
                               cuDoubleComplex *B, int ldb) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtrsm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZtrsm");
  return func_ptr(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

void CUBLASWINAPI cublasStrmm (char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A, int lda,
                               float *B, int ldb) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, char, int, int, float, const float *, int, float *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasStrmm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasStrmm");
  return func_ptr(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

void CUBLASWINAPI cublasDtrmm (char side, char uplo, char transa,
                               char diag, int m, int n, double alpha,
                               const double *A, int lda, double *B,
                               int ldb) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, char, int, int, double, const double *, int, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasDtrmm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasDtrmm");
  return func_ptr(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

void CUBLASWINAPI cublasCtrmm (char side, char uplo, char transa, char diag,
                               int m, int n, cuComplex alpha, const cuComplex *A,
                               int lda, cuComplex *B, int ldb) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, char, int, int, cuComplex, const cuComplex *, int, cuComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasCtrmm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasCtrmm");
  return func_ptr(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

void CUBLASWINAPI cublasZtrmm (char side, char uplo, char transa,
                               char diag, int m, int n, cuDoubleComplex alpha,
                               const cuDoubleComplex *A, int lda, cuDoubleComplex *B,
                               int ldb) {
  using FuncPtr = void (CUBLASWINAPI *)(char, char, char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, cuDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cublasZtrmm");
  if (!func_ptr) LogFatalSymbolNotFound("cublasZtrmm");
  return func_ptr(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

}  // extern "C"
