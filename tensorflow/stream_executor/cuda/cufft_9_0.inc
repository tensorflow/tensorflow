// Auto-generated, do not edit.

extern "C" {

cufftResult CUFFTAPI cufftPlan1d(cufftHandle *plan, int nx, cufftType type,
                                 int batch) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle *, int, cufftType, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftPlan1d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, nx, type, batch);
}

cufftResult CUFFTAPI cufftPlan2d(cufftHandle *plan, int nx, int ny,
                                 cufftType type) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle *, int, int, cufftType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftPlan2d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, nx, ny, type);
}

cufftResult CUFFTAPI cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz,
                                 cufftType type) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle *, int, int, int, cufftType);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftPlan3d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, nx, ny, nz, type);
}

cufftResult CUFFTAPI cufftPlanMany(cufftHandle *plan, int rank, int *n,
                                   int *inembed, int istride, int idist,
                                   int *onembed, int ostride, int odist,
                                   cufftType type, int batch) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle *, int, int *, int *, int,
                                          int, int *, int, int, cufftType, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftPlanMany");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, rank, n, inembed, istride, idist, onembed, ostride,
                  odist, type, batch);
}

cufftResult CUFFTAPI cufftMakePlan1d(cufftHandle plan, int nx, cufftType type,
                                     int batch, size_t *workSize) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle, int, cufftType, int, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftMakePlan1d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, nx, type, batch, workSize);
}

cufftResult CUFFTAPI cufftMakePlan2d(cufftHandle plan, int nx, int ny,
                                     cufftType type, size_t *workSize) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle, int, int, cufftType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftMakePlan2d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, nx, ny, type, workSize);
}

cufftResult CUFFTAPI cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz,
                                     cufftType type, size_t *workSize) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle, int, int, int, cufftType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftMakePlan3d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, nx, ny, nz, type, workSize);
}

cufftResult CUFFTAPI cufftMakePlanMany(cufftHandle plan, int rank, int *n,
                                       int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist,
                                       cufftType type, int batch,
                                       size_t *workSize) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle, int, int *, int *, int, int, int *,
                              int, int, cufftType, int, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftMakePlanMany");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, rank, n, inembed, istride, idist, onembed, ostride,
                  odist, type, batch, workSize);
}

cufftResult CUFFTAPI cufftMakePlanMany64(
    cufftHandle plan, int rank, long long int *n, long long int *inembed,
    long long int istride, long long int idist, long long int *onembed,
    long long int ostride, long long int odist, cufftType type,
    long long int batch, size_t *workSize) {
  using FuncPtr = cufftResult(CUFFTAPI *)(
      cufftHandle, int, long long *, long long *, long long, long long,
      long long *, long long, long long, cufftType, long long, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftMakePlanMany64");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, rank, n, inembed, istride, idist, onembed, ostride,
                  odist, type, batch, workSize);
}

cufftResult CUFFTAPI cufftGetSizeMany64(
    cufftHandle plan, int rank, long long int *n, long long int *inembed,
    long long int istride, long long int idist, long long int *onembed,
    long long int ostride, long long int odist, cufftType type,
    long long int batch, size_t *workSize) {
  using FuncPtr = cufftResult(CUFFTAPI *)(
      cufftHandle, int, long long *, long long *, long long, long long,
      long long *, long long, long long, cufftType, long long, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftGetSizeMany64");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, rank, n, inembed, istride, idist, onembed, ostride,
                  odist, type, batch, workSize);
}

cufftResult CUFFTAPI cufftEstimate1d(int nx, cufftType type, int batch,
                                     size_t *workSize) {
  using FuncPtr = cufftResult(CUFFTAPI *)(int, cufftType, int, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftEstimate1d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(nx, type, batch, workSize);
}

cufftResult CUFFTAPI cufftEstimate2d(int nx, int ny, cufftType type,
                                     size_t *workSize) {
  using FuncPtr = cufftResult(CUFFTAPI *)(int, int, cufftType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftEstimate2d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(nx, ny, type, workSize);
}

cufftResult CUFFTAPI cufftEstimate3d(int nx, int ny, int nz, cufftType type,
                                     size_t *workSize) {
  using FuncPtr = cufftResult(CUFFTAPI *)(int, int, int, cufftType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftEstimate3d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(nx, ny, nz, type, workSize);
}

cufftResult CUFFTAPI cufftEstimateMany(int rank, int *n, int *inembed,
                                       int istride, int idist, int *onembed,
                                       int ostride, int odist, cufftType type,
                                       int batch, size_t *workSize) {
  using FuncPtr = cufftResult(CUFFTAPI *)(int, int *, int *, int, int, int *,
                                          int, int, cufftType, int, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftEstimateMany");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rank, n, inembed, istride, idist, onembed, ostride, odist,
                  type, batch, workSize);
}

cufftResult CUFFTAPI cufftCreate(cufftHandle *handle) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle);
}

cufftResult CUFFTAPI cufftGetSize1d(cufftHandle handle, int nx, cufftType type,
                                    int batch, size_t *workSize) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle, int, cufftType, int, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftGetSize1d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nx, type, batch, workSize);
}

cufftResult CUFFTAPI cufftGetSize2d(cufftHandle handle, int nx, int ny,
                                    cufftType type, size_t *workSize) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle, int, int, cufftType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftGetSize2d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nx, ny, type, workSize);
}

cufftResult CUFFTAPI cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz,
                                    cufftType type, size_t *workSize) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle, int, int, int, cufftType, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftGetSize3d");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, nx, ny, nz, type, workSize);
}

cufftResult CUFFTAPI cufftGetSizeMany(cufftHandle handle, int rank, int *n,
                                      int *inembed, int istride, int idist,
                                      int *onembed, int ostride, int odist,
                                      cufftType type, int batch,
                                      size_t *workArea) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle, int, int *, int *, int, int, int *,
                              int, int, cufftType, int, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftGetSizeMany");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rank, n, inembed, istride, idist, onembed, ostride,
                  odist, type, batch, workArea);
}

cufftResult CUFFTAPI cufftGetSize(cufftHandle handle, size_t *workSize) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftGetSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, workSize);
}

cufftResult CUFFTAPI cufftSetWorkArea(cufftHandle plan, void *workArea) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftSetWorkArea");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, workArea);
}

cufftResult CUFFTAPI cufftSetAutoAllocation(cufftHandle plan,
                                            int autoAllocate) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftSetAutoAllocation");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, autoAllocate);
}

cufftResult CUFFTAPI cufftExecC2C(cufftHandle plan, cufftComplex *idata,
                                  cufftComplex *odata, int direction) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle, cufftComplex *, cufftComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftExecC2C");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, idata, odata, direction);
}

cufftResult CUFFTAPI cufftExecR2C(cufftHandle plan, cufftReal *idata,
                                  cufftComplex *odata) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle, cufftReal *, cufftComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftExecR2C");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, idata, odata);
}

cufftResult CUFFTAPI cufftExecC2R(cufftHandle plan, cufftComplex *idata,
                                  cufftReal *odata) {
  using FuncPtr =
      cufftResult(CUFFTAPI *)(cufftHandle, cufftComplex *, cufftReal *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftExecC2R");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, idata, odata);
}

cufftResult CUFFTAPI cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata,
                                  cufftDoubleComplex *odata, int direction) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle, cufftDoubleComplex *,
                                          cufftDoubleComplex *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftExecZ2Z");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, idata, odata, direction);
}

cufftResult CUFFTAPI cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata,
                                  cufftDoubleComplex *odata) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle, cufftDoubleReal *,
                                          cufftDoubleComplex *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftExecD2Z");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, idata, odata);
}

cufftResult CUFFTAPI cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata,
                                  cufftDoubleReal *odata) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle, cufftDoubleComplex *,
                                          cufftDoubleReal *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftExecZ2D");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, idata, odata);
}

cufftResult CUFFTAPI cufftSetStream(cufftHandle plan, cudaStream_t stream) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftSetStream");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, stream);
}

cufftResult CUFFTAPI cufftSetCompatibilityMode(cufftHandle plan,
                                               cufftCompatibility mode) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle, cufftCompatibility);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftSetCompatibilityMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, mode);
}

cufftResult CUFFTAPI cufftDestroy(cufftHandle plan) {
  using FuncPtr = cufftResult(CUFFTAPI *)(cufftHandle);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan);
}

cufftResult CUFFTAPI cufftGetVersion(int *version) {
  using FuncPtr = cufftResult(CUFFTAPI *)(int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftGetVersion");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(version);
}

cufftResult CUFFTAPI cufftGetProperty(libraryPropertyType type, int *value) {
  using FuncPtr = cufftResult(CUFFTAPI *)(libraryPropertyType, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cufftGetProperty");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(type, value);
}

}  // extern "C"
