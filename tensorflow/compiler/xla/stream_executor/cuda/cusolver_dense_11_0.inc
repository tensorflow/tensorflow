// Auto-generated, do not edit.

extern "C" {

cusolverStatus_t CUSOLVERAPI cusolverGetProperty(libraryPropertyType type,
                                                 int *value) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(libraryPropertyType, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverGetProperty");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(type, value);
}

cusolverStatus_t CUSOLVERAPI cusolverGetVersion(int *version) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverGetVersion");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(version);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCreate(cusolverDnHandle_t *handle) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDestroy(cusolverDnHandle_t handle) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSetStream(cusolverDnHandle_t handle,
                                                 cudaStream_t streamId) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSetStream");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, streamId);
}

cusolverStatus_t CUSOLVERAPI cusolverDnGetStream(cusolverDnHandle_t handle,
                                                 cudaStream_t *streamId) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cudaStream_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnGetStream");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, streamId);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSParamsCreate(cusolverDnIRSParams_t *params_ptr) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSParamsCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params_ptr);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSParamsDestroy(cusolverDnIRSParams_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSParamsDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetRefinementSolver(
    cusolverDnIRSParams_t params, cusolverIRSRefinement_t refinement_solver) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t,
                                                  cusolverIRSRefinement_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnIRSParamsSetRefinementSolver");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params, refinement_solver);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetSolverMainPrecision(
    cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t,
                                                  cusolverPrecType_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnIRSParamsSetSolverMainPrecision");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params, solver_main_precision);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetSolverLowestPrecision(
    cusolverDnIRSParams_t params, cusolverPrecType_t solver_lowest_precision) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t,
                                                  cusolverPrecType_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnIRSParamsSetSolverLowestPrecision");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params, solver_lowest_precision);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetSolverPrecisions(
    cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision,
    cusolverPrecType_t solver_lowest_precision) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnIRSParams_t, cusolverPrecType_t, cusolverPrecType_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnIRSParamsSetSolverPrecisions");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params, solver_main_precision, solver_lowest_precision);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSParamsSetTol(cusolverDnIRSParams_t params, double val) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSParamsSetTol");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params, val);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSParamsSetTolInner(cusolverDnIRSParams_t params, double val) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSParamsSetTolInner");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params, val);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetMaxIters(
    cusolverDnIRSParams_t params, cusolver_int_t maxiters) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t, cusolver_int_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSParamsSetMaxIters");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params, maxiters);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetMaxItersInner(
    cusolverDnIRSParams_t params, cusolver_int_t maxiters_inner) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t, cusolver_int_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnIRSParamsSetMaxItersInner");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params, maxiters_inner);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsGetMaxIters(
    cusolverDnIRSParams_t params, cusolver_int_t *maxiters) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSParamsGetMaxIters");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params, maxiters);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSParamsEnableFallback(cusolverDnIRSParams_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnIRSParamsEnableFallback");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSParamsDisableFallback(cusolverDnIRSParams_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSParams_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnIRSParamsDisableFallback");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSInfosDestroy(cusolverDnIRSInfos_t infos) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSInfos_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSInfosDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(infos);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSInfosCreate(cusolverDnIRSInfos_t *infos_ptr) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSInfos_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSInfosCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(infos_ptr);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSInfosGetNiters(
    cusolverDnIRSInfos_t infos, cusolver_int_t *niters) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSInfos_t, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSInfosGetNiters");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(infos, niters);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSInfosGetOuterNiters(
    cusolverDnIRSInfos_t infos, cusolver_int_t *outer_niters) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSInfos_t, cusolver_int_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnIRSInfosGetOuterNiters");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(infos, outer_niters);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSInfosRequestResidual(cusolverDnIRSInfos_t infos) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSInfos_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnIRSInfosRequestResidual");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(infos);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSInfosGetResidualHistory(
    cusolverDnIRSInfos_t infos, void **residual_history) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSInfos_t, void **);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnIRSInfosGetResidualHistory");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(infos, residual_history);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSInfosGetMaxIters(
    cusolverDnIRSInfos_t infos, cusolver_int_t *maxiters) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnIRSInfos_t, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSInfosGetMaxIters");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(infos, maxiters);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZZgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
      cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZZgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZCgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
      cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZCgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZKgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
      cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZKgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZEgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
      cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZEgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZYgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
      cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZYgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCCgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
      cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t,
      cuComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCCgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCEgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
      cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t,
      cuComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCEgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCKgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
      cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t,
      cuComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCKgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCYgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
      cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t,
      cuComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCYgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDDgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *,
      cusolver_int_t, cusolver_int_t *, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDDgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDSgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *,
      cusolver_int_t, cusolver_int_t *, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDSgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDHgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *,
      cusolver_int_t, cusolver_int_t *, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDHgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDBgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *,
      cusolver_int_t, cusolver_int_t *, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDBgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDXgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *,
      cusolver_int_t, cusolver_int_t *, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDXgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSSgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *,
      cusolver_int_t, cusolver_int_t *, float *, cusolver_int_t, float *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSSgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSHgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *,
      cusolver_int_t, cusolver_int_t *, float *, cusolver_int_t, float *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSHgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSBgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *,
      cusolver_int_t, cusolver_int_t *, float *, cusolver_int_t, float *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSBgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSXgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *,
      cusolver_int_t, cusolver_int_t *, float *, cusolver_int_t, float *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSXgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZZgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
      cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZZgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZCgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
      cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZCgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZKgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
      cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZKgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZEgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
      cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZEgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZYgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
      cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZYgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCCgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
      cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t,
      cuComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCCgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCKgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
      cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t,
      cuComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCKgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCEgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
      cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t,
      cuComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCEgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCYgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
      cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t,
      cuComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCYgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDDgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *,
      cusolver_int_t, cusolver_int_t *, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDDgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDSgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *,
      cusolver_int_t, cusolver_int_t *, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDSgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDHgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *,
      cusolver_int_t, cusolver_int_t *, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDHgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDBgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *,
      cusolver_int_t, cusolver_int_t *, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDBgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDXgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *,
      cusolver_int_t, cusolver_int_t *, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDXgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSSgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *,
      cusolver_int_t, cusolver_int_t *, float *, cusolver_int_t, float *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSSgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSHgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *,
      cusolver_int_t, cusolver_int_t *, float *, cusolver_int_t, float *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSHgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSBgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *,
      cusolver_int_t, cusolver_int_t *, float *, cusolver_int_t, float *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSBgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSXgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *,
      cusolver_int_t, cusolver_int_t *, float *, cusolver_int_t, float *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSXgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                  dWorkspace, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZZgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
                 cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
                 cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
                 cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZZgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZCgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
                 cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
                 cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
                 cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZCgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZKgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
                 cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
                 cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
                 cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZKgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZEgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
                 cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
                 cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
                 cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZEgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZYgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
                 cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
                 cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
                 cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZYgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCCgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCCgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCKgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCKgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCEgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCEgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCYgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCYgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDDgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      double *, cusolver_int_t, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDDgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDSgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      double *, cusolver_int_t, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDSgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDHgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      double *, cusolver_int_t, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDHgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDBgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      double *, cusolver_int_t, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDBgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDXgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      double *, cusolver_int_t, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDXgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSSgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, float *dA, cusolver_int_t ldda, float *dB,
    cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      float *, cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t,
      void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSSgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSHgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, float *dA, cusolver_int_t ldda, float *dB,
    cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      float *, cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t,
      void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSHgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSBgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, float *dA, cusolver_int_t ldda, float *dB,
    cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      float *, cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t,
      void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSBgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSXgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, float *dA, cusolver_int_t ldda, float *dB,
    cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      float *, cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t,
      void *, size_t, cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSXgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes, iter, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZZgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZZgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZCgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZCgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZKgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZKgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZEgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZEgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZYgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
      cuDoubleComplex *, cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZYgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCCgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCCgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCKgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCKgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCEgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCEgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCYgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCYgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDDgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      double *, cusolver_int_t, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDDgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDSgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      double *, cusolver_int_t, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDSgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDHgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      double *, cusolver_int_t, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDHgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDBgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      double *, cusolver_int_t, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDBgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDXgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      double *, cusolver_int_t, double *, cusolver_int_t, double *,
      cusolver_int_t, void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDXgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSSgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, float *dA, cusolver_int_t ldda, float *dB,
    cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      float *, cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t,
      void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSSgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSHgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, float *dA, cusolver_int_t ldda, float *dB,
    cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      float *, cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t,
      void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSHgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSBgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, float *dA, cusolver_int_t ldda, float *dB,
    cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      float *, cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t,
      void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSBgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSXgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, float *dA, cusolver_int_t ldda, float *dB,
    cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
      float *, cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t,
      void *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSXgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                  lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgesv(
    cusolverDnHandle_t handle, cusolverDnIRSParams_t gesv_irs_params,
    cusolverDnIRSInfos_t gesv_irs_infos, cusolver_int_t n, cusolver_int_t nrhs,
    void *dA, cusolver_int_t ldda, void *dB, cusolver_int_t lddb, void *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    cusolver_int_t *niters, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t,
      cusolver_int_t, cusolver_int_t, void *, cusolver_int_t, void *,
      cusolver_int_t, void *, cusolver_int_t, void *, size_t, cusolver_int_t *,
      cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSXgesv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA, ldda,
                  dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgesv_bufferSize(
    cusolverDnHandle_t handle, cusolverDnIRSParams_t params, cusolver_int_t n,
    cusolver_int_t nrhs, size_t *lwork_bytes) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cusolverDnIRSParams_t,
                                      cusolver_int_t, cusolver_int_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSXgesv_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, n, nrhs, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgels(
    cusolverDnHandle_t handle, cusolverDnIRSParams_t gels_irs_params,
    cusolverDnIRSInfos_t gels_irs_infos, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, void *dA, cusolver_int_t ldda, void *dB,
    cusolver_int_t lddb, void *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *niters, cusolver_int_t *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t,
      cusolver_int_t, cusolver_int_t, cusolver_int_t, void *, cusolver_int_t,
      void *, cusolver_int_t, void *, cusolver_int_t, void *, size_t,
      cusolver_int_t *, cusolver_int_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSXgels");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, gels_irs_params, gels_irs_infos, m, n, nrhs, dA, ldda,
                  dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgels_bufferSize(
    cusolverDnHandle_t handle, cusolverDnIRSParams_t params, cusolver_int_t m,
    cusolver_int_t n, cusolver_int_t nrhs, size_t *lwork_bytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t, cusolver_int_t,
      cusolver_int_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnIRSXgels_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, m, n, nrhs, lwork_bytes);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, float *A, int lda, int *Lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSpotrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, Lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, double *A, int lda, int *Lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDpotrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, Lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnCpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, cuComplex *A, int lda, int *Lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCpotrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, Lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, cuDoubleComplex *A, int lda, int *Lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZpotrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, Lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              float *A, int lda,
                                              float *Workspace, int Lwork,
                                              int *devInfo) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      float *, int, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSpotrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              double *A, int lda,
                                              double *Workspace, int Lwork,
                                              int *devInfo) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      double *, int, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDpotrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              cuComplex *A, int lda,
                                              cuComplex *Workspace, int Lwork,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, cuComplex *,
      int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCpotrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              cuDoubleComplex *A, int lda,
                                              cuDoubleComplex *Workspace,
                                              int Lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
      cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZpotrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSpotrs(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              int nrhs, const float *A, int lda,
                                              float *B, int ldb, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const float *, int,
      float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSpotrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrs(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              int nrhs, const double *A,
                                              int lda, double *B, int ldb,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const double *, int,
      double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDpotrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCpotrs(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              int nrhs, const cuComplex *A,
                                              int lda, cuComplex *B, int ldb,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const cuComplex *, int,
      cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCpotrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrs(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              int nrhs,
                                              const cuDoubleComplex *A, int lda,
                                              cuDoubleComplex *B, int ldb,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const cuDoubleComplex *,
      int, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZpotrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSpotrfBatched(cusolverDnHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n, float *Aarray[],
                                                     int lda, int *infoArray,
                                                     int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, float *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSpotrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrfBatched(cusolverDnHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n, double *Aarray[],
                                                     int lda, int *infoArray,
                                                     int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, double *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDpotrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCpotrfBatched(cusolverDnHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n, cuComplex *Aarray[],
                                                     int lda, int *infoArray,
                                                     int batchSize) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      cuComplex *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCpotrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrfBatched(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
    cuDoubleComplex *Aarray[], int lda, int *infoArray, int batchSize) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      cuDoubleComplex *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZpotrfBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSpotrsBatched(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
    int nrhs, /* only support rhs = 1*/
    float *A[], int lda, float *B[], int ldb, int *d_info, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, float *[], int, float *[],
      int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSpotrsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrsBatched(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
    int nrhs, /* only support rhs = 1*/
    double *A[], int lda, double *B[], int ldb, int *d_info, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, double *[], int,
      double *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDpotrsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnCpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                        int nrhs, /* only support rhs = 1*/
                        cuComplex *A[], int lda, cuComplex *B[], int ldb,
                        int *d_info, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, cuComplex *[], int,
      cuComplex *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCpotrsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                        int nrhs, /* only support rhs = 1*/
                        cuDoubleComplex *A[], int lda, cuDoubleComplex *B[],
                        int ldb, int *d_info, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, cuDoubleComplex *[], int,
      cuDoubleComplex *[], int, int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZpotrsBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnSpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, float *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSpotri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnDpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, double *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDpotri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnCpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, cuComplex *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCpotri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, cuDoubleComplex *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZpotri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSpotri(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              float *A, int lda, float *work,
                                              int lwork, int *devInfo) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      float *, int, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSpotri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDpotri(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              double *A, int lda, double *work,
                                              int lwork, int *devInfo) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      double *, int, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDpotri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCpotri(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              cuComplex *A, int lda,
                                              cuComplex *work, int lwork,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, cuComplex *,
      int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCpotri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZpotri(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              cuDoubleComplex *A, int lda,
                                              cuDoubleComplex *work, int lwork,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
      cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZpotri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnStrtri_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag,
    int n, float *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int, float *, int,
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnStrtri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, diag, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDtrtri_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag,
    int n, double *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int, double *,
      int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDtrtri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, diag, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCtrtri_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag,
    int n, cuComplex *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int, cuComplex *,
      int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCtrtri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, diag, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZtrtri_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag,
    int n, cuDoubleComplex *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int,
      cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZtrtri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, diag, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnStrtri(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo,
                                              cublasDiagType_t diag, int n,
                                              float *A, int lda, float *work,
                                              int lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int, float *, int,
      float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnStrtri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, diag, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDtrtri(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo,
                                              cublasDiagType_t diag, int n,
                                              double *A, int lda, double *work,
                                              int lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int, double *,
      int, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDtrtri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, diag, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCtrtri(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag,
    int n, cuComplex *A, int lda, cuComplex *work, int lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int, cuComplex *,
      int, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCtrtri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, diag, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZtrtri(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo,
                                              cublasDiagType_t diag, int n,
                                              cuDoubleComplex *A, int lda,
                                              cuDoubleComplex *work, int lwork,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int,
      cuDoubleComplex *, int, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZtrtri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, diag, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnSlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, float *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSlauum_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnDlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, double *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDlauum_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnClauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, cuComplex *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnClauum_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                            int n, cuDoubleComplex *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZlauum_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSlauum(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              float *A, int lda, float *work,
                                              int lwork, int *devInfo) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      float *, int, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSlauum");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDlauum(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              double *A, int lda, double *work,
                                              int lwork, int *devInfo) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      double *, int, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDlauum");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnClauum(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              cuComplex *A, int lda,
                                              cuComplex *work, int lwork,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, cuComplex *,
      int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnClauum");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZlauum(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              cuDoubleComplex *A, int lda,
                                              cuDoubleComplex *work, int lwork,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
      cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZlauum");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, work, lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf_bufferSize(
    cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int,
                                                  float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgetrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf_bufferSize(
    cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int,
                                                  double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgetrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnCgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                            cuComplex *A, int lda, int *Lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int,
                                                  cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgetrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                            cuDoubleComplex *A, int lda, int *Lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgetrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf(cusolverDnHandle_t handle, int m,
                                              int n, float *A, int lda,
                                              float *Workspace, int *devIpiv,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, float *, int, float *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgetrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf(cusolverDnHandle_t handle, int m,
                                              int n, double *A, int lda,
                                              double *Workspace, int *devIpiv,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, double *, int, double *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgetrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgetrf(cusolverDnHandle_t handle, int m,
                                              int n, cuComplex *A, int lda,
                                              cuComplex *Workspace,
                                              int *devIpiv, int *devInfo) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int, cuComplex *,
                                      int, cuComplex *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgetrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgetrf(cusolverDnHandle_t handle, int m,
                                              int n, cuDoubleComplex *A,
                                              int lda,
                                              cuDoubleComplex *Workspace,
                                              int *devIpiv, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, cuDoubleComplex *, int, cuDoubleComplex *,
      int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgetrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSlaswp(cusolverDnHandle_t handle, int n,
                                              float *A, int lda, int k1, int k2,
                                              const int *devIpiv, int incx) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, float *, int, int, int, const int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSlaswp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, k1, k2, devIpiv, incx);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDlaswp(cusolverDnHandle_t handle, int n,
                                              double *A, int lda, int k1,
                                              int k2, const int *devIpiv,
                                              int incx) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, double *, int, int, int, const int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDlaswp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, k1, k2, devIpiv, incx);
}

cusolverStatus_t CUSOLVERAPI cusolverDnClaswp(cusolverDnHandle_t handle, int n,
                                              cuComplex *A, int lda, int k1,
                                              int k2, const int *devIpiv,
                                              int incx) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, cuComplex *, int, int, int, const int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnClaswp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, k1, k2, devIpiv, incx);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZlaswp(cusolverDnHandle_t handle, int n,
                                              cuDoubleComplex *A, int lda,
                                              int k1, int k2,
                                              const int *devIpiv, int incx) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int,
                                                  cuDoubleComplex *, int, int,
                                                  int, const int *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZlaswp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, k1, k2, devIpiv, incx);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgetrs(cusolverDnHandle_t handle,
                                              cublasOperation_t trans, int n,
                                              int nrhs, const float *A, int lda,
                                              const int *devIpiv, float *B,
                                              int ldb, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasOperation_t, int, int, const float *, int,
      const int *, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgetrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgetrs(cusolverDnHandle_t handle,
                                              cublasOperation_t trans, int n,
                                              int nrhs, const double *A,
                                              int lda, const int *devIpiv,
                                              double *B, int ldb,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasOperation_t, int, int, const double *, int,
      const int *, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgetrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgetrs(cusolverDnHandle_t handle,
                                              cublasOperation_t trans, int n,
                                              int nrhs, const cuComplex *A,
                                              int lda, const int *devIpiv,
                                              cuComplex *B, int ldb,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasOperation_t, int, int, const cuComplex *, int,
      const int *, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgetrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgetrs(
    cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
    const cuDoubleComplex *A, int lda, const int *devIpiv, cuDoubleComplex *B,
    int ldb, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasOperation_t, int, int, const cuDoubleComplex *,
      int, const int *, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgetrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf_bufferSize(
    cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int,
                                                  float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgeqrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf_bufferSize(
    cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int,
                                                  double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgeqrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnCgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                            cuComplex *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int,
                                                  cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgeqrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                            cuDoubleComplex *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgeqrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf(cusolverDnHandle_t handle, int m,
                                              int n, float *A, int lda,
                                              float *TAU, float *Workspace,
                                              int Lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, float *, int, float *, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgeqrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf(cusolverDnHandle_t handle, int m,
                                              int n, double *A, int lda,
                                              double *TAU, double *Workspace,
                                              int Lwork, int *devInfo) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int, double *,
                                      int, double *, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgeqrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgeqrf(cusolverDnHandle_t handle, int m,
                                              int n, cuComplex *A, int lda,
                                              cuComplex *TAU,
                                              cuComplex *Workspace, int Lwork,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int,
                                                  cuComplex *, int, cuComplex *,
                                                  cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgeqrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgeqrf(cusolverDnHandle_t handle, int m,
                                              int n, cuDoubleComplex *A,
                                              int lda, cuDoubleComplex *TAU,
                                              cuDoubleComplex *Workspace,
                                              int Lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, cuDoubleComplex *, int, cuDoubleComplex *,
      cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgeqrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int k, const float *A, int lda,
    const float *tau, int *lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int, int,
                                      const float *, int, const float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSorgqr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int k, const double *A, int lda,
    const double *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int,
                                                  int, const double *, int,
                                                  const double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDorgqr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCungqr_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int k, const cuComplex *A, int lda,
    const cuComplex *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int,
                                                  int, const cuComplex *, int,
                                                  const cuComplex *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCungqr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZungqr_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int k, const cuDoubleComplex *A,
    int lda, const cuDoubleComplex *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, int, const cuDoubleComplex *, int,
      const cuDoubleComplex *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZungqr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr(cusolverDnHandle_t handle, int m,
                                              int n, int k, float *A, int lda,
                                              const float *tau, float *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, int, float *, int, const float *, float *,
      int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSorgqr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr(cusolverDnHandle_t handle, int m,
                                              int n, int k, double *A, int lda,
                                              const double *tau, double *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, int, double *, int, const double *,
      double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDorgqr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCungqr(cusolverDnHandle_t handle, int m,
                                              int n, int k, cuComplex *A,
                                              int lda, const cuComplex *tau,
                                              cuComplex *work, int lwork,
                                              int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, int, cuComplex *, int, const cuComplex *,
      cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCungqr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZungqr(
    cusolverDnHandle_t handle, int m, int n, int k, cuDoubleComplex *A, int lda,
    const cuDoubleComplex *tau, cuDoubleComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, int, cuDoubleComplex *, int,
      const cuDoubleComplex *, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZungqr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, k, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSormqr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k, const float *A, int lda, const float *tau,
    const float *C, int ldc, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
      const float *, int, const float *, const float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSormqr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDormqr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k, const double *A, int lda, const double *tau,
    const double *C, int ldc, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
      const double *, int, const double *, const double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDormqr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k, const cuComplex *A, int lda, const cuComplex *tau,
    const cuComplex *C, int ldc, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
      const cuComplex *, int, const cuComplex *, const cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCunmqr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *tau, const cuDoubleComplex *C, int ldc, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
      const cuDoubleComplex *, int, const cuDoubleComplex *,
      const cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZunmqr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSormqr(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k, const float *A, int lda, const float *tau, float *C,
    int ldc, float *work, int lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
      const float *, int, const float *, float *, int, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSormqr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                  lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDormqr(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k, const double *A, int lda, const double *tau, double *C,
    int ldc, double *work, int lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
      const double *, int, const double *, double *, int, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDormqr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                  lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k, const cuComplex *A, int lda, const cuComplex *tau,
    cuComplex *C, int ldc, cuComplex *work, int lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
      const cuComplex *, int, const cuComplex *, cuComplex *, int, cuComplex *,
      int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCunmqr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                  lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *tau, cuDoubleComplex *C, int ldc,
    cuDoubleComplex *work, int lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
      const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *,
      int, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZunmqr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work,
                  lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf_bufferSize(
    cusolverDnHandle_t handle, int n, float *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int,
                                                  float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsytrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf_bufferSize(
    cusolverDnHandle_t handle, int n, double *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int,
                                                  double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsytrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf_bufferSize(
    cusolverDnHandle_t handle, int n, cuComplex *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int,
                                                  cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCsytrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf_bufferSize(
    cusolverDnHandle_t handle, int n, cuDoubleComplex *A, int lda, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZsytrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, n, A, lda, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              float *A, int lda, int *ipiv,
                                              float *work, int lwork,
                                              int *info) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      float *, int, int *, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsytrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              double *A, int lda, int *ipiv,
                                              double *work, int lwork,
                                              int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, double *, int, int *, double *,
      int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsytrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              cuComplex *A, int lda, int *ipiv,
                                              cuComplex *work, int lwork,
                                              int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, int *,
      cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCsytrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              cuDoubleComplex *A, int lda,
                                              int *ipiv, cuDoubleComplex *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int, int *,
      cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZsytrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsytrs_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
    const float *A, int lda, const int *ipiv, float *B, int ldb, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const float *, int,
      const int *, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsytrs_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrs_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
    const double *A, int lda, const int *ipiv, double *B, int ldb, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const double *, int,
      const int *, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsytrs_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCsytrs_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
    const cuComplex *A, int lda, const int *ipiv, cuComplex *B, int ldb,
    int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const cuComplex *, int,
      const int *, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCsytrs_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZsytrs_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
    const cuDoubleComplex *A, int lda, const int *ipiv, cuDoubleComplex *B,
    int ldb, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const cuDoubleComplex *,
      int, const int *, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZsytrs_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsytrs(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              int nrhs, const float *A, int lda,
                                              const int *ipiv, float *B,
                                              int ldb, float *work, int lwork,
                                              int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const float *, int,
      const int *, float *, int, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsytrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork,
                  info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrs(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              int nrhs, const double *A,
                                              int lda, const int *ipiv,
                                              double *B, int ldb, double *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const double *, int,
      const int *, double *, int, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsytrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork,
                  info);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnCsytrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                 int nrhs, const cuComplex *A, int lda, const int *ipiv,
                 cuComplex *B, int ldb, cuComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const cuComplex *, int,
      const int *, cuComplex *, int, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCsytrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork,
                  info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZsytrs(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
    const cuDoubleComplex *A, int lda, const int *ipiv, cuDoubleComplex *B,
    int ldb, cuDoubleComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, int, const cuDoubleComplex *,
      int, const int *, cuDoubleComplex *, int, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZsytrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork,
                  info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsytri_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda,
    const int *ipiv, int *lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      float *, int, const int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsytri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsytri_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda,
    const int *ipiv, int *lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      double *, int, const int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsytri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCsytri_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A,
    int lda, const int *ipiv, int *lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      cuComplex *, int, const int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCsytri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZsytri_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A,
    int lda, const int *ipiv, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
      const int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZsytri_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsytri(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              float *A, int lda,
                                              const int *ipiv, float *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, float *, int, const int *,
      float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsytri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsytri(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              double *A, int lda,
                                              const int *ipiv, double *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, double *, int, const int *,
      double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsytri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCsytri(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              cuComplex *A, int lda,
                                              const int *ipiv, cuComplex *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, const int *,
      cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCsytri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZsytri(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A,
    int lda, const int *ipiv, cuDoubleComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
      const int *, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZsytri");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int *Lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgebrd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int *Lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgebrd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int *Lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgebrd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int *Lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgebrd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, Lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd(cusolverDnHandle_t handle, int m,
                                              int n, float *A, int lda,
                                              float *D, float *E, float *TAUQ,
                                              float *TAUP, float *Work,
                                              int Lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, float *, int, float *, float *, float *,
      float *, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgebrd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd(cusolverDnHandle_t handle, int m,
                                              int n, double *A, int lda,
                                              double *D, double *E,
                                              double *TAUQ, double *TAUP,
                                              double *Work, int Lwork,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, double *, int, double *, double *, double *,
      double *, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgebrd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd(cusolverDnHandle_t handle, int m,
                                              int n, cuComplex *A, int lda,
                                              float *D, float *E,
                                              cuComplex *TAUQ, cuComplex *TAUP,
                                              cuComplex *Work, int Lwork,
                                              int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, cuComplex *, int, float *, float *,
      cuComplex *, cuComplex *, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgebrd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd(
    cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda,
    double *D, double *E, cuDoubleComplex *TAUQ, cuDoubleComplex *TAUP,
    cuDoubleComplex *Work, int Lwork, int *devInfo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, int, int, cuDoubleComplex *, int, double *, double *,
      cuDoubleComplex *, cuDoubleComplex *, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgebrd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k,
    const float *A, int lda, const float *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, int, int, int, const float *, int,
      const float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSorgbr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k,
    const double *A, int lda, const double *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, int, int, int, const double *, int,
      const double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDorgbr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCungbr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k,
    const cuComplex *A, int lda, const cuComplex *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, int, int, int, const cuComplex *,
      int, const cuComplex *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCungbr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZungbr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k,
    const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, int, int, int,
      const cuDoubleComplex *, int, const cuDoubleComplex *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZungbr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr(cusolverDnHandle_t handle,
                                              cublasSideMode_t side, int m,
                                              int n, int k, float *A, int lda,
                                              const float *tau, float *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, int, int, int, float *, int,
      const float *, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSorgbr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr(cusolverDnHandle_t handle,
                                              cublasSideMode_t side, int m,
                                              int n, int k, double *A, int lda,
                                              const double *tau, double *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, int, int, int, double *, int,
      const double *, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDorgbr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCungbr(cusolverDnHandle_t handle,
                                              cublasSideMode_t side, int m,
                                              int n, int k, cuComplex *A,
                                              int lda, const cuComplex *tau,
                                              cuComplex *work, int lwork,
                                              int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, int, int, int, cuComplex *, int,
      const cuComplex *, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCungbr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n,
                 int k, cuDoubleComplex *A, int lda, const cuDoubleComplex *tau,
                 cuDoubleComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, int, int, int, cuDoubleComplex *,
      int, const cuDoubleComplex *, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZungbr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float *A,
    int lda, const float *d, const float *e, const float *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, const float *, int,
      const float *, const float *, const float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsytrd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, d, e, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double *A,
    int lda, const double *d, const double *e, const double *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, const double *, int,
      const double *, const double *, const double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsytrd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, d, e, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnChetrd_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *A,
    int lda, const float *d, const float *e, const cuComplex *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, const cuComplex *, int,
      const float *, const float *, const cuComplex *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnChetrd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, d, e, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
    const cuDoubleComplex *A, int lda, const double *d, const double *e,
    const cuDoubleComplex *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, int,
      const double *, const double *, const cuDoubleComplex *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZhetrd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, d, e, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              float *A, int lda, float *d,
                                              float *e, float *tau, float *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, float *, int, float *, float *,
      float *, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsytrd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, d, e, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda,
    double *d, double *e, double *tau, double *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, double *, int, double *,
      double *, double *, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsytrd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, d, e, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnChetrd(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              cuComplex *A, int lda, float *d,
                                              float *e, cuComplex *tau,
                                              cuComplex *work, int lwork,
                                              int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, float *,
      float *, cuComplex *, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnChetrd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, d, e, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A,
    int lda, double *d, double *e, cuDoubleComplex *tau, cuDoubleComplex *work,
    int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
      double *, double *, cuDoubleComplex *, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZhetrd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, d, e, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float *A,
    int lda, const float *tau, int *lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, cublasFillMode_t, int,
                                      const float *, int, const float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSorgtr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double *A,
    int lda, const double *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, const double *, int,
      const double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDorgtr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCungtr_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *A,
    int lda, const cuComplex *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, const cuComplex *, int,
      const cuComplex *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCungtr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZungtr_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
    const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, int,
      const cuDoubleComplex *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZungtr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, tau, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              float *A, int lda,
                                              const float *tau, float *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, float *, int, const float *,
      float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSorgtr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              double *A, int lda,
                                              const double *tau, double *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, double *, int, const double *,
      double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDorgtr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCungtr(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A,
    int lda, const cuComplex *tau, cuComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int,
      const cuComplex *, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCungtr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZungtr(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              cuDoubleComplex *A, int lda,
                                              const cuDoubleComplex *tau,
                                              cuDoubleComplex *work, int lwork,
                                              int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
      const cuDoubleComplex *, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZungtr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, uplo, n, A, lda, tau, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSormtr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, int m, int n, const float *A, int lda,
    const float *tau, const float *C, int ldc, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
      int, int, const float *, int, const float *, const float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSormtr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDormtr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, int m, int n, const double *A, int lda,
    const double *tau, const double *C, int ldc, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
      int, int, const double *, int, const double *, const double *, int,
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDormtr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCunmtr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, int m, int n, const cuComplex *A, int lda,
    const cuComplex *tau, const cuComplex *C, int ldc, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
      int, int, const cuComplex *, int, const cuComplex *, const cuComplex *,
      int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCunmtr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, int m, int n, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *tau, const cuDoubleComplex *C, int ldc, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
      int, int, const cuDoubleComplex *, int, const cuDoubleComplex *,
      const cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZunmtr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSormtr(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, int m, int n, float *A, int lda, float *tau,
    float *C, int ldc, float *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
      int, int, float *, int, float *, float *, int, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSormtr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                  lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDormtr(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, int m, int n, double *A, int lda, double *tau,
    double *C, int ldc, double *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
      int, int, double *, int, double *, double *, int, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDormtr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                  lwork, info);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnCunmtr(cusolverDnHandle_t handle, cublasSideMode_t side,
                 cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
                 cuComplex *A, int lda, cuComplex *tau, cuComplex *C, int ldc,
                 cuComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
      int, int, cuComplex *, int, cuComplex *, cuComplex *, int, cuComplex *,
      int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCunmtr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                  lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, int m, int n, cuDoubleComplex *A, int lda,
    cuDoubleComplex *tau, cuDoubleComplex *C, int ldc, cuDoubleComplex *work,
    int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
      int, int, cuDoubleComplex *, int, cuDoubleComplex *, cuDoubleComplex *,
      int, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZunmtr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work,
                  lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int *lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgesvd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int *lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgesvd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvd_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int *lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgesvd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvd_bufferSize(
    cusolverDnHandle_t handle, int m, int n, int *lwork) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, int, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgesvd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, m, n, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd(
    cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m,
    int n, float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt,
    float *work, int lwork, float *rwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, signed char, signed char, int, int, float *, int,
      float *, float *, int, float *, int, float *, int, float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgesvd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                  lwork, rwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd(
    cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m,
    int n, double *A, int lda, double *S, double *U, int ldu, double *VT,
    int ldvt, double *work, int lwork, double *rwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, signed char, signed char, int, int, double *, int,
      double *, double *, int, double *, int, double *, int, double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgesvd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                  lwork, rwork, info);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnCgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt,
                 int m, int n, cuComplex *A, int lda, float *S, cuComplex *U,
                 int ldu, cuComplex *VT, int ldvt, cuComplex *work, int lwork,
                 float *rwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, signed char, signed char, int, int, cuComplex *, int,
      float *, cuComplex *, int, cuComplex *, int, cuComplex *, int, float *,
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgesvd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                  lwork, rwork, info);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt,
                 int m, int n, cuDoubleComplex *A, int lda, double *S,
                 cuDoubleComplex *U, int ldu, cuDoubleComplex *VT, int ldvt,
                 cuDoubleComplex *work, int lwork, double *rwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, signed char, signed char, int, int, cuDoubleComplex *,
      int, double *, cuDoubleComplex *, int, cuDoubleComplex *, int,
      cuDoubleComplex *, int, double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgesvd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work,
                  lwork, rwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const float *A, int lda, const float *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const float *, int, const float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsyevd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const double *A, int lda, const double *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const double *, int, const double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsyevd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCheevd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const cuComplex *A, int lda, const float *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const cuComplex *, int, const float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCheevd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZheevd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const cuDoubleComplex *A, int lda, const double *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const cuDoubleComplex *, int, const double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZheevd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, float *A, int lda, float *W, float *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float *,
      int, float *, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsyevd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, double *A, int lda, double *W, double *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double *,
      int, double *, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsyevd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCheevd(cusolverDnHandle_t handle,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo, int n,
                                              cuComplex *A, int lda, float *W,
                                              cuComplex *work, int lwork,
                                              int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex *,
      int, float *, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCheevd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZheevd(cusolverDnHandle_t handle,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo, int n,
                                              cuDoubleComplex *A, int lda,
                                              double *W, cuDoubleComplex *work,
                                              int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      cuDoubleComplex *, int, double *, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZheevd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const float *A, int lda, float vl, float vu,
    int il, int iu, int *meig, const float *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
      cublasFillMode_t, int, const float *, int, float, float, int, int, int *,
      const float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsyevdx_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                  lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const double *A, int lda, double vl,
    double vu, int il, int iu, int *meig, const double *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
      cublasFillMode_t, int, const double *, int, double, double, int, int,
      int *, const double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsyevdx_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                  lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCheevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const cuComplex *A, int lda, float vl,
    float vu, int il, int iu, int *meig, const float *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
      cublasFillMode_t, int, const cuComplex *, int, float, float, int, int,
      int *, const float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCheevdx_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                  lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, double vl,
    double vu, int il, int iu, int *meig, const double *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
      cublasFillMode_t, int, const cuDoubleComplex *, int, double, double, int,
      int, int *, const double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZheevdx_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                  lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, float *A, int lda, float vl, float vu, int il,
    int iu, int *meig, float *W, float *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
      cublasFillMode_t, int, float *, int, float, float, int, int, int *,
      float *, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsyevdx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                  work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, double *A, int lda, double vl, double vu,
    int il, int iu, int *meig, double *W, double *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
      cublasFillMode_t, int, double *, int, double, double, int, int, int *,
      double *, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsyevdx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                  work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnCheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                  cusolverEigRange_t range, cublasFillMode_t uplo, int n,
                  cuComplex *A, int lda, float vl, float vu, int il, int iu,
                  int *meig, float *W, cuComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
      cublasFillMode_t, int, cuComplex *, int, float, float, int, int, int *,
      float *, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCheevdx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                  work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, double vl,
    double vu, int il, int iu, int *meig, double *W, cuDoubleComplex *work,
    int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t,
      cublasFillMode_t, int, cuDoubleComplex *, int, double, double, int, int,
      int *, double *, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZheevdx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W,
                  work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float *A,
    int lda, const float *B, int ldb, float vl, float vu, int il, int iu,
    int *meig, const float *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int, const float *, int,
      const float *, int, float, float, int, int, int *, const float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsygvdx_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                  il, iu, meig, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double *A,
    int lda, const double *B, int ldb, double vl, double vu, int il, int iu,
    int *meig, const double *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int, const double *, int,
      const double *, int, double, double, int, int, int *, const double *,
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsygvdx_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                  il, iu, meig, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex *A,
    int lda, const cuComplex *B, int ldb, float vl, float vu, int il, int iu,
    int *meig, const float *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int, const cuComplex *, int,
      const cuComplex *, int, float, float, int, int, int *, const float *,
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnChegvdx_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                  il, iu, meig, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n,
    const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
    double vl, double vu, int il, int iu, int *meig, const double *W,
    int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int, const cuDoubleComplex *, int,
      const cuDoubleComplex *, int, double, double, int, int, int *,
      const double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZhegvdx_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                  il, iu, meig, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, float *A, int lda,
    float *B, int ldb, float vl, float vu, int il, int iu, int *meig, float *W,
    float *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int, float *, int, float *, int,
      float, float, int, int, int *, float *, float *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsygvdx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                  il, iu, meig, W, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, double *A, int lda,
    double *B, int ldb, double vl, double vu, int il, int iu, int *meig,
    double *W, double *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int, double *, int, double *, int,
      double, double, int, int, int *, double *, double *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsygvdx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                  il, iu, meig, W, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuComplex *A,
    int lda, cuComplex *B, int ldb, float vl, float vu, int il, int iu,
    int *meig, float *W, cuComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int, cuComplex *, int, cuComplex *,
      int, float, float, int, int, int *, float *, cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnChegvdx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                  il, iu, meig, W, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex *A,
    int lda, cuDoubleComplex *B, int ldb, double vl, double vu, int il, int iu,
    int *meig, double *W, cuDoubleComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int, cuDoubleComplex *, int,
      cuDoubleComplex *, int, double, double, int, int, int *, double *,
      cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZhegvdx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                  il, iu, meig, W, work, lwork, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const float *A, int lda, const float *B,
    int ldb, const float *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, const float *, int, const float *, int,
      const float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsygvd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const double *A, int lda, const double *B,
    int ldb, const double *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, const double *, int, const double *, int,
      const double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsygvd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnChegvd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const cuComplex *A, int lda,
    const cuComplex *B, int ldb, const float *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, const cuComplex *, int, const cuComplex *, int,
      const float *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnChegvd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb, const double *W, int *lwork) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, const cuDoubleComplex *, int,
      const cuDoubleComplex *, int, const double *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZhegvd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, float *A, int lda, float *B, int ldb,
    float *W, float *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, float *, int, float *, int, float *, float *, int,
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsygvd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                  info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, double *A, int lda, double *B, int ldb,
    double *W, double *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, double *, int, double *, int, double *, double *,
      int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsygvd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                  info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnChegvd(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, cuComplex *A, int lda, cuComplex *B, int ldb,
    float *W, cuComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, cuComplex *, int, cuComplex *, int, float *,
      cuComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnChegvd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                  info);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnZhegvd(cusolverDnHandle_t handle, cusolverEigType_t itype,
                 cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                 cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb,
                 double *W, cuDoubleComplex *work, int lwork, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, cuDoubleComplex *, int, cuDoubleComplex *, int,
      double *, cuDoubleComplex *, int, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZhegvd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                  info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCreateSyevjInfo(syevjInfo_t *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(syevjInfo_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCreateSyevjInfo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDestroySyevjInfo(syevjInfo_t info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDestroySyevjInfo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetTolerance(syevjInfo_t info,
                                                          double tolerance) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(syevjInfo_t, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXsyevjSetTolerance");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info, tolerance);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info,
                                                          int max_sweeps) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(syevjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXsyevjSetMaxSweeps");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info, max_sweeps);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetSortEig(syevjInfo_t info,
                                                        int sort_eig) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(syevjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXsyevjSetSortEig");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info, sort_eig);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetResidual(
    cusolverDnHandle_t handle, syevjInfo_t info, double *residual) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t,
                                                  syevjInfo_t, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXsyevjGetResidual");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, residual);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetSweeps(
    cusolverDnHandle_t handle, syevjInfo_t info, int *executed_sweeps) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, syevjInfo_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXsyevjGetSweeps");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, executed_sweeps);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const float *A, int lda, const float *W, int *lwork,
    syevjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const float *, int, const float *, int *, syevjInfo_t, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnSsyevjBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const double *A, int lda, const double *W, int *lwork,
    syevjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const double *, int, const double *, int *, syevjInfo_t, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnDsyevjBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCheevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const cuComplex *A, int lda, const float *W, int *lwork,
    syevjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const cuComplex *, int, const float *, int *, syevjInfo_t, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnCheevjBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZheevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const cuDoubleComplex *A, int lda, const double *W, int *lwork,
    syevjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const cuDoubleComplex *, int, const double *, int *, syevjInfo_t, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnZheevjBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevjBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, float *A, int lda, float *W, float *work, int lwork, int *info,
    syevjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float *,
      int, float *, float *, int, int *, syevjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsyevjBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params,
                  batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevjBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, double *A, int lda, double *W, double *work, int lwork, int *info,
    syevjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double *,
      int, double *, double *, int, int *, syevjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsyevjBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params,
                  batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCheevjBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, cuComplex *A, int lda, float *W, cuComplex *work, int lwork,
    int *info, syevjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex *,
      int, float *, cuComplex *, int, int *, syevjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCheevjBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params,
                  batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZheevjBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, cuDoubleComplex *A, int lda, double *W, cuDoubleComplex *work,
    int lwork, int *info, syevjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      cuDoubleComplex *, int, double *, cuDoubleComplex *, int, int *,
      syevjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZheevjBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params,
                  batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const float *A, int lda, const float *W, int *lwork,
    syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const float *, int, const float *, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsyevj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const double *A, int lda, const double *W, int *lwork,
    syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const double *, int, const double *, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsyevj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCheevj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const cuComplex *A, int lda, const float *W, int *lwork,
    syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const cuComplex *, int, const float *, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCheevj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZheevj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const cuDoubleComplex *A, int lda, const double *W, int *lwork,
    syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      const cuDoubleComplex *, int, const double *, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZheevj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, lwork, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj(cusolverDnHandle_t handle,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo, int n,
                                              float *A, int lda, float *W,
                                              float *work, int lwork, int *info,
                                              syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float *,
      int, float *, float *, int, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsyevj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj(cusolverDnHandle_t handle,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo, int n,
                                              double *A, int lda, double *W,
                                              double *work, int lwork,
                                              int *info, syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double *,
      int, double *, double *, int, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsyevj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCheevj(cusolverDnHandle_t handle,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo, int n,
                                              cuComplex *A, int lda, float *W,
                                              cuComplex *work, int lwork,
                                              int *info, syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex *,
      int, float *, cuComplex *, int, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCheevj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZheevj(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, cuDoubleComplex *A, int lda, double *W, cuDoubleComplex *work,
    int lwork, int *info, syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
      cuDoubleComplex *, int, double *, cuDoubleComplex *, int, int *,
      syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZheevj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const float *A, int lda, const float *B,
    int ldb, const float *W, int *lwork, syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, const float *, int, const float *, int,
      const float *, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsygvj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork,
                  params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const double *A, int lda, const double *B,
    int ldb, const double *W, int *lwork, syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, const double *, int, const double *, int,
      const double *, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsygvj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork,
                  params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnChegvj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const cuComplex *A, int lda,
    const cuComplex *B, int ldb, const float *W, int *lwork,
    syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, const cuComplex *, int, const cuComplex *, int,
      const float *, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnChegvj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork,
                  params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb, const double *W, int *lwork,
    syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, const cuDoubleComplex *, int,
      const cuDoubleComplex *, int, const double *, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZhegvj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork,
                  params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, float *A, int lda, float *B, int ldb,
    float *W, float *work, int lwork, int *info, syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, float *, int, float *, int, float *, float *, int,
      int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSsygvj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                  info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, double *A, int lda, double *B, int ldb,
    double *W, double *work, int lwork, int *info, syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, double *, int, double *, int, double *, double *,
      int, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDsygvj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                  info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnChegvj(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, cuComplex *A, int lda, cuComplex *B, int ldb,
    float *W, cuComplex *work, int lwork, int *info, syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, cuComplex *, int, cuComplex *, int, float *,
      cuComplex *, int, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnChegvj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                  info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda,
    cuDoubleComplex *B, int ldb, double *W, cuDoubleComplex *work, int lwork,
    int *info, syevjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t,
      cublasFillMode_t, int, cuDoubleComplex *, int, cuDoubleComplex *, int,
      double *, cuDoubleComplex *, int, int *, syevjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZhegvj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                  info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCreateGesvdjInfo(gesvdjInfo_t *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(gesvdjInfo_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCreateGesvdjInfo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(gesvdjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDestroyGesvdjInfo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info,
                                                           double tolerance) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(gesvdjInfo_t, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgesvdjSetTolerance");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info, tolerance);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info,
                                                           int max_sweeps) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(gesvdjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgesvdjSetMaxSweeps");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info, max_sweeps);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info,
                                                         int sort_svd) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(gesvdjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgesvdjSetSortEig");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(info, sort_svd);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetResidual(
    cusolverDnHandle_t handle, gesvdjInfo_t info, double *residual) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t,
                                                  gesvdjInfo_t, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgesvdjGetResidual");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, residual);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetSweeps(
    cusolverDnHandle_t handle, gesvdjInfo_t info, int *executed_sweeps) {
  using FuncPtr =
      cusolverStatus_t(CUSOLVERAPI *)(cusolverDnHandle_t, gesvdjInfo_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgesvdjGetSweeps");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, info, executed_sweeps);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    const float *A, int lda, const float *S, const float *U, int ldu,
    const float *V, int ldv, int *lwork, gesvdjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, const float *, int,
      const float *, const float *, int, const float *, int, int *,
      gesvdjInfo_t, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnSgesvdjBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params,
                  batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    const double *A, int lda, const double *S, const double *U, int ldu,
    const double *V, int ldv, int *lwork, gesvdjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, const double *, int,
      const double *, const double *, int, const double *, int, int *,
      gesvdjInfo_t, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnDgesvdjBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params,
                  batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    const cuComplex *A, int lda, const float *S, const cuComplex *U, int ldu,
    const cuComplex *V, int ldv, int *lwork, gesvdjInfo_t params,
    int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, const cuComplex *, int,
      const float *, const cuComplex *, int, const cuComplex *, int, int *,
      gesvdjInfo_t, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnCgesvdjBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params,
                  batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    const cuDoubleComplex *A, int lda, const double *S,
    const cuDoubleComplex *U, int ldu, const cuDoubleComplex *V, int ldv,
    int *lwork, gesvdjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, const cuDoubleComplex *,
      int, const double *, const cuDoubleComplex *, int,
      const cuDoubleComplex *, int, int *, gesvdjInfo_t, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnZgesvdjBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params,
                  batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, float *A,
    int lda, float *S, float *U, int ldu, float *V, int ldv, float *work,
    int lwork, int *info, gesvdjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, float *, int, float *,
      float *, int, float *, int, float *, int, int *, gesvdjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgesvdjBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork,
                  info, params, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, double *A,
    int lda, double *S, double *U, int ldu, double *V, int ldv, double *work,
    int lwork, int *info, gesvdjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, double *, int, double *,
      double *, int, double *, int, double *, int, int *, gesvdjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgesvdjBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork,
                  info, params, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    cuComplex *A, int lda, float *S, cuComplex *U, int ldu, cuComplex *V,
    int ldv, cuComplex *work, int lwork, int *info, gesvdjInfo_t params,
    int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, cuComplex *, int,
      float *, cuComplex *, int, cuComplex *, int, cuComplex *, int, int *,
      gesvdjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgesvdjBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork,
                  info, params, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    cuDoubleComplex *A, int lda, double *S, cuDoubleComplex *U, int ldu,
    cuDoubleComplex *V, int ldv, cuDoubleComplex *work, int lwork, int *info,
    gesvdjInfo_t params, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, cuDoubleComplex *, int,
      double *, cuDoubleComplex *, int, cuDoubleComplex *, int,
      cuDoubleComplex *, int, int *, gesvdjInfo_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgesvdjBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork,
                  info, params, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
    const float *A, int lda, const float *S, const float *U, int ldu,
    const float *V, int ldv, int *lwork, gesvdjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float *, int,
      const float *, const float *, int, const float *, int, int *,
      gesvdjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgesvdj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork,
                  params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
    const double *A, int lda, const double *S, const double *U, int ldu,
    const double *V, int ldv, int *lwork, gesvdjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double *, int,
      const double *, const double *, int, const double *, int, int *,
      gesvdjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgesvdj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork,
                  params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
    const cuComplex *A, int lda, const float *S, const cuComplex *U, int ldu,
    const cuComplex *V, int ldv, int *lwork, gesvdjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex *,
      int, const float *, const cuComplex *, int, const cuComplex *, int, int *,
      gesvdjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgesvdj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork,
                  params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
    const cuDoubleComplex *A, int lda, const double *S,
    const cuDoubleComplex *U, int ldu, const cuDoubleComplex *V, int ldv,
    int *lwork, gesvdjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int,
      const cuDoubleComplex *, int, const double *, const cuDoubleComplex *,
      int, const cuDoubleComplex *, int, int *, gesvdjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgesvdj_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork,
                  params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
    float *A, int lda, float *S, float *U, int ldu, float *V, int ldv,
    float *work, int lwork, int *info, gesvdjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, float *, int,
      float *, float *, int, float *, int, float *, int, int *, gesvdjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgesvdj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                  lwork, info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
    double *A, int lda, double *S, double *U, int ldu, double *V, int ldv,
    double *work, int lwork, int *info, gesvdjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, double *, int,
      double *, double *, int, double *, int, double *, int, int *,
      gesvdjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgesvdj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                  lwork, info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
    cuComplex *A, int lda, float *S, cuComplex *U, int ldu, cuComplex *V,
    int ldv, cuComplex *work, int lwork, int *info, gesvdjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, cuComplex *, int,
      float *, cuComplex *, int, cuComplex *, int, cuComplex *, int, int *,
      gesvdjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgesvdj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                  lwork, info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
    cuDoubleComplex *A, int lda, double *S, cuDoubleComplex *U, int ldu,
    cuDoubleComplex *V, int ldv, cuDoubleComplex *work, int lwork, int *info,
    gesvdjInfo_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, cuDoubleComplex *,
      int, double *, cuDoubleComplex *, int, cuDoubleComplex *, int,
      cuDoubleComplex *, int, int *, gesvdjInfo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgesvdj");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work,
                  lwork, info, params);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const float *d_A, int lda, long long int strideA, const float *d_S,
    long long int strideS, const float *d_U, int ldu, long long int strideU,
    const float *d_V, int ldv, long long int strideV, int *lwork,
    int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float *, int,
      long long, const float *, long long, const float *, int, long long,
      const float *, int, long long, int *, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnSgesvdaStridedBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS,
                  d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const double *d_A, int lda, long long int strideA, const double *d_S,
    long long int strideS, const double *d_U, int ldu, long long int strideU,
    const double *d_V, int ldv, long long int strideV, int *lwork,
    int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double *, int,
      long long, const double *, long long, const double *, int, long long,
      const double *, int, long long, int *, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnDgesvdaStridedBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS,
                  d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const cuComplex *d_A, int lda, long long int strideA, const float *d_S,
    long long int strideS, const cuComplex *d_U, int ldu, long long int strideU,
    const cuComplex *d_V, int ldv, long long int strideV, int *lwork,
    int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex *,
      int, long long, const float *, long long, const cuComplex *, int,
      long long, const cuComplex *, int, long long, int *, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnCgesvdaStridedBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS,
                  d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const cuDoubleComplex *d_A, int lda, long long int strideA,
    const double *d_S, long long int strideS, const cuDoubleComplex *d_U,
    int ldu, long long int strideU, const cuDoubleComplex *d_V, int ldv,
    long long int strideV, int *lwork, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int,
      const cuDoubleComplex *, int, long long, const double *, long long,
      const cuDoubleComplex *, int, long long, const cuDoubleComplex *, int,
      long long, int *, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cusolverDnZgesvdaStridedBatched_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS,
                  d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const float *d_A, int lda, long long int strideA, float *d_S,
    long long int strideS, float *d_U, int ldu, long long int strideU,
    float *d_V, int ldv, long long int strideV, float *d_work, int lwork,
    int *d_info, double *h_R_nrmF, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float *, int,
      long long, float *, long long, float *, int, long long, float *, int,
      long long, float *, int, int *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSgesvdaStridedBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS,
                  d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info,
                  h_R_nrmF, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const double *d_A, int lda, long long int strideA, double *d_S,
    long long int strideS, double *d_U, int ldu, long long int strideU,
    double *d_V, int ldv, long long int strideV, double *d_work, int lwork,
    int *d_info, double *h_R_nrmF, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double *, int,
      long long, double *, long long, double *, int, long long, double *, int,
      long long, double *, int, int *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDgesvdaStridedBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS,
                  d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info,
                  h_R_nrmF, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const cuComplex *d_A, int lda, long long int strideA, float *d_S,
    long long int strideS, cuComplex *d_U, int ldu, long long int strideU,
    cuComplex *d_V, int ldv, long long int strideV, cuComplex *d_work,
    int lwork, int *d_info, double *h_R_nrmF, int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex *,
      int, long long, float *, long long, cuComplex *, int, long long,
      cuComplex *, int, long long, cuComplex *, int, int *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCgesvdaStridedBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS,
                  d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info,
                  h_R_nrmF, batchSize);
}

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const cuDoubleComplex *d_A, int lda, long long int strideA, double *d_S,
    long long int strideS, cuDoubleComplex *d_U, int ldu, long long int strideU,
    cuDoubleComplex *d_V, int ldv, long long int strideV,
    cuDoubleComplex *d_work, int lwork, int *d_info, double *h_R_nrmF,
    int batchSize) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverEigMode_t, int, int, int,
      const cuDoubleComplex *, int, long long, double *, long long,
      cuDoubleComplex *, int, long long, cuDoubleComplex *, int, long long,
      cuDoubleComplex *, int, int *, double *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnZgesvdaStridedBatched");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS,
                  d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info,
                  h_R_nrmF, batchSize);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnCreateParams(cusolverDnParams_t *params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnParams_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnCreateParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnDestroyParams(cusolverDnParams_t params) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(cusolverDnParams_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnDestroyParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnSetAdvOptions(cusolverDnParams_t params,
                        cusolverDnFunction_t function, cusolverAlgMode_t algo) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnParams_t, cusolverDnFunction_t, cusolverAlgMode_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSetAdvOptions");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(params, function, algo);
}

cusolverStatus_t CUSOLVERAPI cusolverDnPotrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo,
    int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType computeType, size_t *workspaceInBytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t,
      cudaDataType, const void *, int64_t, cudaDataType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnPotrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, uplo, n, dataTypeA, A, lda, computeType,
                  workspaceInBytes);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnPotrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA,
                void *A, int64_t lda, cudaDataType computeType, void *pBuffer,
                size_t workspaceInBytes, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t,
      cudaDataType, void *, int64_t, cudaDataType, void *, size_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnPotrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, uplo, n, dataTypeA, A, lda, computeType,
                  pBuffer, workspaceInBytes, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnPotrs(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo,
    int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType dataTypeB, void *B, int64_t ldb, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t,
      int64_t, cudaDataType, const void *, int64_t, cudaDataType, void *,
      int64_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnPotrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB,
                  B, ldb, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnGeqrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType dataTypeTau, const void *tau, cudaDataType computeType,
    size_t *workspaceInBytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
      const void *, int64_t, cudaDataType, const void *, cudaDataType,
      size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnGeqrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau,
                  computeType, workspaceInBytes);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnGeqrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m,
                int64_t n, cudaDataType dataTypeA, void *A, int64_t lda,
                cudaDataType dataTypeTau, void *tau, cudaDataType computeType,
                void *pBuffer, size_t workspaceInBytes, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
      void *, int64_t, cudaDataType, void *, cudaDataType, void *, size_t,
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnGeqrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau,
                  computeType, pBuffer, workspaceInBytes, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnGetrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType computeType, size_t *workspaceInBytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
      const void *, int64_t, cudaDataType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnGetrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, m, n, dataTypeA, A, lda, computeType,
                  workspaceInBytes);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnGetrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m,
                int64_t n, cudaDataType dataTypeA, void *A, int64_t lda,
                int64_t *ipiv, cudaDataType computeType, void *pBuffer,
                size_t workspaceInBytes, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
      void *, int64_t, int64_t *, cudaDataType, void *, size_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnGetrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType,
                  pBuffer, workspaceInBytes, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnGetrs(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cublasOperation_t trans, int64_t n, int64_t nrhs, cudaDataType dataTypeA,
    const void *A, int64_t lda, const int64_t *ipiv, cudaDataType dataTypeB,
    void *B, int64_t ldb, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cublasOperation_t, int64_t,
      int64_t, cudaDataType, const void *, int64_t, const int64_t *,
      cudaDataType, void *, int64_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnGetrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv,
                  dataTypeB, B, ldb, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeW,
    const void *W, cudaDataType computeType, size_t *workspaceInBytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
      cublasFillMode_t, int64_t, cudaDataType, const void *, int64_t,
      cudaDataType, const void *, cudaDataType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSyevd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW,
                  W, computeType, workspaceInBytes);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnSyevd(cusolverDnHandle_t handle, cusolverDnParams_t params,
                cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                cudaDataType dataTypeA, void *A, int64_t lda,
                cudaDataType dataTypeW, void *W, cudaDataType computeType,
                void *pBuffer, size_t workspaceInBytes, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
      cublasFillMode_t, int64_t, cudaDataType, void *, int64_t, cudaDataType,
      void *, cudaDataType, void *, size_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSyevd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW,
                  W, computeType, pBuffer, workspaceInBytes, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo,
    int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, void *vl,
    void *vu, int64_t il, int64_t iu, int64_t *h_meig, cudaDataType dataTypeW,
    const void *W, cudaDataType computeType, size_t *workspaceInBytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int64_t, cudaDataType, const void *,
      int64_t, void *, void *, int64_t, int64_t, int64_t *, cudaDataType,
      const void *, cudaDataType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSyevdx_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl,
                  vu, il, iu, h_meig, dataTypeW, W, computeType,
                  workspaceInBytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnSyevdx(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo,
    int64_t n, cudaDataType dataTypeA, void *A, int64_t lda, void *vl, void *vu,
    int64_t il, int64_t iu, int64_t *meig64, cudaDataType dataTypeW, void *W,
    cudaDataType computeType, void *pBuffer, size_t workspaceInBytes,
    int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int64_t, cudaDataType, void *,
      int64_t, void *, void *, int64_t, int64_t, int64_t *, cudaDataType,
      void *, cudaDataType, void *, size_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnSyevdx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl,
                  vu, il, iu, meig64, dataTypeW, W, computeType, pBuffer,
                  workspaceInBytes, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnGesvd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu,
    signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA,
    const void *A, int64_t lda, cudaDataType dataTypeS, const void *S,
    cudaDataType dataTypeU, const void *U, int64_t ldu, cudaDataType dataTypeVT,
    const void *VT, int64_t ldvt, cudaDataType computeType,
    size_t *workspaceInBytes) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t,
      int64_t, cudaDataType, const void *, int64_t, cudaDataType, const void *,
      cudaDataType, const void *, int64_t, cudaDataType, const void *, int64_t,
      cudaDataType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnGesvd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda,
                  dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt,
                  computeType, workspaceInBytes);
}

cusolverStatus_t CUSOLVERAPI cusolverDnGesvd(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu,
    signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA, void *A,
    int64_t lda, cudaDataType dataTypeS, void *S, cudaDataType dataTypeU,
    void *U, int64_t ldu, cudaDataType dataTypeVT, void *VT, int64_t ldvt,
    cudaDataType computeType, void *pBuffer, size_t workspaceInBytes,
    int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t,
      int64_t, cudaDataType, void *, int64_t, cudaDataType, void *,
      cudaDataType, void *, int64_t, cudaDataType, void *, int64_t,
      cudaDataType, void *, size_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnGesvd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda,
                  dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt,
                  computeType, pBuffer, workspaceInBytes, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXpotrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo,
    int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t,
      cudaDataType, const void *, int64_t, cudaDataType, size_t *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXpotrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, uplo, n, dataTypeA, A, lda, computeType,
                  workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnXpotrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA,
                 void *A, int64_t lda, cudaDataType computeType,
                 void *bufferOnDevice, size_t workspaceInBytesOnDevice,
                 void *bufferOnHost, size_t workspaceInBytesOnHost, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t,
      cudaDataType, void *, int64_t, cudaDataType, void *, size_t, void *,
      size_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXpotrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, uplo, n, dataTypeA, A, lda, computeType,
                  bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                  workspaceInBytesOnHost, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXpotrs(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo,
    int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType dataTypeB, void *B, int64_t ldb, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t,
      int64_t, cudaDataType, const void *, int64_t, cudaDataType, void *,
      int64_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXpotrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB,
                  B, ldb, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgeqrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType dataTypeTau, const void *tau, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
      const void *, int64_t, cudaDataType, const void *, cudaDataType, size_t *,
      size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgeqrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau,
                  computeType, workspaceInBytesOnDevice,
                  workspaceInBytesOnHost);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgeqrf(
    cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n,
    cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeTau,
    void *tau, cudaDataType computeType, void *bufferOnDevice,
    size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
      void *, int64_t, cudaDataType, void *, cudaDataType, void *, size_t,
      void *, size_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgeqrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau,
                  computeType, bufferOnDevice, workspaceInBytesOnDevice,
                  bufferOnHost, workspaceInBytesOnHost, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgetrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
      const void *, int64_t, cudaDataType, size_t *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgetrf_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, m, n, dataTypeA, A, lda, computeType,
                  workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnXgetrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 int64_t m, int64_t n, cudaDataType dataTypeA, void *A,
                 int64_t lda, int64_t *ipiv, cudaDataType computeType,
                 void *bufferOnDevice, size_t workspaceInBytesOnDevice,
                 void *bufferOnHost, size_t workspaceInBytesOnHost, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
      void *, int64_t, int64_t *, cudaDataType, void *, size_t, void *, size_t,
      int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgetrf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType,
                  bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                  workspaceInBytesOnHost, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgetrs(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cublasOperation_t trans, int64_t n, int64_t nrhs, cudaDataType dataTypeA,
    const void *A, int64_t lda, const int64_t *ipiv, cudaDataType dataTypeB,
    void *B, int64_t ldb, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cublasOperation_t, int64_t,
      int64_t, cudaDataType, const void *, int64_t, const int64_t *,
      cudaDataType, void *, int64_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgetrs");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv,
                  dataTypeB, B, ldb, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeW,
    const void *W, cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
      cublasFillMode_t, int64_t, cudaDataType, const void *, int64_t,
      cudaDataType, const void *, cudaDataType, size_t *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXsyevd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW,
                  W, computeType, workspaceInBytesOnDevice,
                  workspaceInBytesOnHost);
}

cusolverStatus_t CUSOLVERAPI
cusolverDnXsyevd(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                 cudaDataType dataTypeA, void *A, int64_t lda,
                 cudaDataType dataTypeW, void *W, cudaDataType computeType,
                 void *bufferOnDevice, size_t workspaceInBytesOnDevice,
                 void *bufferOnHost, size_t workspaceInBytesOnHost, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
      cublasFillMode_t, int64_t, cudaDataType, void *, int64_t, cudaDataType,
      void *, cudaDataType, void *, size_t, void *, size_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXsyevd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW,
                  W, computeType, bufferOnDevice, workspaceInBytesOnDevice,
                  bufferOnHost, workspaceInBytesOnHost, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo,
    int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, void *vl,
    void *vu, int64_t il, int64_t iu, int64_t *h_meig, cudaDataType dataTypeW,
    const void *W, cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int64_t, cudaDataType, const void *,
      int64_t, void *, void *, int64_t, int64_t, int64_t *, cudaDataType,
      const void *, cudaDataType, size_t *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXsyevdx_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl,
                  vu, il, iu, h_meig, dataTypeW, W, computeType,
                  workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevdx(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo,
    int64_t n, cudaDataType dataTypeA, void *A, int64_t lda, void *vl, void *vu,
    int64_t il, int64_t iu, int64_t *meig64, cudaDataType dataTypeW, void *W,
    cudaDataType computeType, void *bufferOnDevice,
    size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
      cusolverEigRange_t, cublasFillMode_t, int64_t, cudaDataType, void *,
      int64_t, void *, void *, int64_t, int64_t, int64_t *, cudaDataType,
      void *, cudaDataType, void *, size_t, void *, size_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXsyevdx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl,
                  vu, il, iu, meig64, dataTypeW, W, computeType, bufferOnDevice,
                  workspaceInBytesOnDevice, bufferOnHost,
                  workspaceInBytesOnHost, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu,
    signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA,
    const void *A, int64_t lda, cudaDataType dataTypeS, const void *S,
    cudaDataType dataTypeU, const void *U, int64_t ldu, cudaDataType dataTypeVT,
    const void *VT, int64_t ldvt, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t,
      int64_t, cudaDataType, const void *, int64_t, cudaDataType, const void *,
      cudaDataType, const void *, int64_t, cudaDataType, const void *, int64_t,
      cudaDataType, size_t *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgesvd_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda,
                  dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt,
                  computeType, workspaceInBytesOnDevice,
                  workspaceInBytesOnHost);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvd(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu,
    signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA, void *A,
    int64_t lda, cudaDataType dataTypeS, void *S, cudaDataType dataTypeU,
    void *U, int64_t ldu, cudaDataType dataTypeVT, void *VT, int64_t ldvt,
    cudaDataType computeType, void *bufferOnDevice,
    size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, int *info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t,
      int64_t, cudaDataType, void *, int64_t, cudaDataType, void *,
      cudaDataType, void *, int64_t, cudaDataType, void *, int64_t,
      cudaDataType, void *, size_t, void *, size_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgesvd");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda,
                  dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt,
                  computeType, bufferOnDevice, workspaceInBytesOnDevice,
                  bufferOnHost, workspaceInBytesOnHost, info);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdp_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, int econ, int64_t m, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeS,
    const void *S, cudaDataType dataTypeU, const void *U, int64_t ldu,
    cudaDataType dataTypeV, const void *V, int64_t ldv,
    cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, int, int64_t,
      int64_t, cudaDataType, const void *, int64_t, cudaDataType, const void *,
      cudaDataType, const void *, int64_t, cudaDataType, const void *, int64_t,
      cudaDataType, size_t *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgesvdp_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobz, econ, m, n, dataTypeA, A, lda,
                  dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv,
                  computeType, workspaceInBytesOnDevice,
                  workspaceInBytesOnHost);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdp(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, int econ, int64_t m, int64_t n,
    cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeS,
    void *S, cudaDataType dataTypeU, void *U, int64_t ldu,
    cudaDataType dataTypeV, void *V, int64_t ldv, cudaDataType computeType,
    void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, int *d_info, double *h_err_sigma) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, int, int64_t,
      int64_t, cudaDataType, void *, int64_t, cudaDataType, void *,
      cudaDataType, void *, int64_t, cudaDataType, void *, int64_t,
      cudaDataType, void *, size_t, void *, size_t, int *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgesvdp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobz, econ, m, n, dataTypeA, A, lda,
                  dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv,
                  computeType, bufferOnDevice, workspaceInBytesOnDevice,
                  bufferOnHost, workspaceInBytesOnHost, d_info, h_err_sigma);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdr_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu,
    signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p,
    int64_t niters, cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType dataTypeSrand, const void *Srand, cudaDataType dataTypeUrand,
    const void *Urand, int64_t ldUrand, cudaDataType dataTypeVrand,
    const void *Vrand, int64_t ldVrand, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t,
      int64_t, int64_t, int64_t, int64_t, cudaDataType, const void *, int64_t,
      cudaDataType, const void *, cudaDataType, const void *, int64_t,
      cudaDataType, const void *, int64_t, cudaDataType, size_t *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgesvdr_bufferSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A,
                  lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand,
                  dataTypeVrand, Vrand, ldVrand, computeType,
                  workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdr(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu,
    signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p,
    int64_t niters, cudaDataType dataTypeA, void *A, int64_t lda,
    cudaDataType dataTypeSrand, void *Srand, cudaDataType dataTypeUrand,
    void *Urand, int64_t ldUrand, cudaDataType dataTypeVrand, void *Vrand,
    int64_t ldVrand, cudaDataType computeType, void *bufferOnDevice,
    size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, int *d_info) {
  using FuncPtr = cusolverStatus_t(CUSOLVERAPI *)(
      cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t,
      int64_t, int64_t, int64_t, int64_t, cudaDataType, void *, int64_t,
      cudaDataType, void *, cudaDataType, void *, int64_t, cudaDataType, void *,
      int64_t, cudaDataType, void *, size_t, void *, size_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cusolverDnXgesvdr");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A,
                  lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand,
                  dataTypeVrand, Vrand, ldVrand, computeType, bufferOnDevice,
                  workspaceInBytesOnDevice, bufferOnHost,
                  workspaceInBytesOnHost, d_info);
}

}  // extern "C"
