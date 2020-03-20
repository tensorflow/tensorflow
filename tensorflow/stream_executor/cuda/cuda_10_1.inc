// Auto-generated, do not edit.

extern "C" {

CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
  using FuncPtr = CUresult(CUDAAPI *)(CUresult, const char **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGetErrorString");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(error, pStr);
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
  using FuncPtr = CUresult(CUDAAPI *)(CUresult, const char **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGetErrorName");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(error, pStr);
}

CUresult CUDAAPI cuInit(unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuInit");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(Flags);
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
  using FuncPtr = CUresult(CUDAAPI *)(int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDriverGetVersion");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(driverVersion);
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdevice *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceGet");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device, ordinal);
}

CUresult CUDAAPI cuDeviceGetCount(int *count) {
  using FuncPtr = CUresult(CUDAAPI *)(int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceGetCount");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(count);
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
  using FuncPtr = CUresult(CUDAAPI *)(char *, int, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceGetName");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(name, len, dev);
}

CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
  using FuncPtr = CUresult(CUDAAPI *)(CUuuid *, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceGetUuid");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(uuid, dev);
}

CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  using FuncPtr = CUresult(CUDAAPI *)(size_t *, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceTotalMem_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(bytes, dev);
}

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                                      CUdevice dev) {
  using FuncPtr = CUresult(CUDAAPI *)(int *, CUdevice_attribute, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pi, attrib, dev);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuDeviceGetProperties(CUdevprop *prop,
                                                         CUdevice dev) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdevprop *, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceGetProperties");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(prop, dev);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuDeviceComputeCapability(int *major,
                                                             int *minor,
                                                             CUdevice dev) {
  using FuncPtr = CUresult(CUDAAPI *)(int *, int *, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceComputeCapability");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(major, minor, dev);
}

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext *, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDevicePrimaryCtxRetain");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pctx, dev);
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDevicePrimaryCtxRelease");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dev);
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdevice, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDevicePrimaryCtxSetFlags");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dev, flags);
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                            int *active) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdevice, unsigned int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDevicePrimaryCtxGetState");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dev, flags, active);
}

CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDevicePrimaryCtxReset");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dev);
}

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags,
                             CUdevice dev) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext *, unsigned int, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxCreate_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pctx, flags, dev);
}

CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxDestroy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctx);
}

CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxPushCurrent_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctx);
}

CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxPopCurrent_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pctx);
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxSetCurrent");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctx);
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxGetCurrent");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pctx);
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdevice *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxGetDevice");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device);
}

CUresult CUDAAPI cuCtxGetFlags(unsigned int *flags) {
  using FuncPtr = CUresult(CUDAAPI *)(unsigned int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxGetFlags");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(flags);
}

CUresult CUDAAPI cuCtxSynchronize(void) {
  using FuncPtr = CUresult(CUDAAPI *)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxSynchronize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr();
}

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
  using FuncPtr = CUresult(CUDAAPI *)(CUlimit, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxSetLimit");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(limit, value);
}

CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
  using FuncPtr = CUresult(CUDAAPI *)(size_t *, CUlimit);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxGetLimit");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pvalue, limit);
}

CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunc_cache *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxGetCacheConfig");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pconfig);
}

CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunc_cache);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxSetCacheConfig");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(config);
}

CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
  using FuncPtr = CUresult(CUDAAPI *)(CUsharedconfig *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxGetSharedMemConfig");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pConfig);
}

CUresult CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig config) {
  using FuncPtr = CUresult(CUDAAPI *)(CUsharedconfig);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxSetSharedMemConfig");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(config);
}

CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext, unsigned int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxGetApiVersion");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctx, version);
}

CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
  using FuncPtr = CUresult(CUDAAPI *)(int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxGetStreamPriorityRange");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(leastPriority, greatestPriority);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuCtxAttach(CUcontext *pctx,
                                               unsigned int flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext *, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxAttach");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pctx, flags);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuCtxDetach(CUcontext ctx) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxDetach");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctx);
}

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
  using FuncPtr = CUresult(CUDAAPI *)(CUmodule *, const char *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuModuleLoad");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(module, fname);
}

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
  using FuncPtr = CUresult(CUDAAPI *)(CUmodule *, const void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuModuleLoadData");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(module, image);
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                                    unsigned int numOptions,
                                    CUjit_option *options,
                                    void **optionValues) {
  using FuncPtr = CUresult(CUDAAPI *)(CUmodule *, const void *, unsigned int,
                                      CUjit_option *, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuModuleLoadDataEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(module, image, numOptions, options, optionValues);
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
  using FuncPtr = CUresult(CUDAAPI *)(CUmodule *, const void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuModuleLoadFatBinary");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(module, fatCubin);
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
  using FuncPtr = CUresult(CUDAAPI *)(CUmodule);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuModuleUnload");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hmod);
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                     const char *name) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction *, CUmodule, const char *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuModuleGetFunction");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hfunc, hmod, name);
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                                   CUmodule hmod, const char *name) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr *, size_t *, CUmodule, const char *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuModuleGetGlobal_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dptr, bytes, hmod, name);
}

CUresult CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod,
                                   const char *name) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref *, CUmodule, const char *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuModuleGetTexRef");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pTexRef, hmod, name);
}

CUresult CUDAAPI cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod,
                                    const char *name) {
  using FuncPtr = CUresult(CUDAAPI *)(CUsurfref *, CUmodule, const char *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuModuleGetSurfRef");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pSurfRef, hmod, name);
}

CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                              void **optionValues, CUlinkState *stateOut) {
  using FuncPtr =
      CUresult(CUDAAPI *)(unsigned int, CUjit_option *, void **, CUlinkState *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuLinkCreate_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(numOptions, options, optionValues, stateOut);
}

CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void *data, size_t size, const char *name,
                               unsigned int numOptions, CUjit_option *options,
                               void **optionValues) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUlinkState, CUjitInputType, void *, size_t,
                          const char *, unsigned int, CUjit_option *, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuLinkAddData_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(state, type, data, size, name, numOptions, options,
                  optionValues);
}

CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues) {
  using FuncPtr = CUresult(CUDAAPI *)(CUlinkState, CUjitInputType, const char *,
                                      unsigned int, CUjit_option *, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuLinkAddFile_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(state, type, path, numOptions, options, optionValues);
}

CUresult CUDAAPI cuLinkComplete(CUlinkState state, void **cubinOut,
                                size_t *sizeOut) {
  using FuncPtr = CUresult(CUDAAPI *)(CUlinkState, void **, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuLinkComplete");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(state, cubinOut, sizeOut);
}

CUresult CUDAAPI cuLinkDestroy(CUlinkState state) {
  using FuncPtr = CUresult(CUDAAPI *)(CUlinkState);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuLinkDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(state);
}

CUresult CUDAAPI cuMemGetInfo(size_t *free, size_t *total) {
  using FuncPtr = CUresult(CUDAAPI *)(size_t *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemGetInfo_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(free, total);
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemAlloc_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dptr, bytesize);
}

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                                 size_t WidthInBytes, size_t Height,
                                 unsigned int ElementSizeBytes) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr *, size_t *, size_t, size_t,
                                      unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemAllocPitch_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemFree_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dptr);
}

CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                                      CUdeviceptr dptr) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr *, size_t *, CUdeviceptr);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemGetAddressRange_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pbase, psize, dptr);
}

CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize) {
  using FuncPtr = CUresult(CUDAAPI *)(void **, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemAllocHost_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pp, bytesize);
}

CUresult CUDAAPI cuMemFreeHost(void *p) {
  using FuncPtr = CUresult(CUDAAPI *)(void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemFreeHost");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(p);
}

CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize,
                                unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(void **, size_t, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemHostAlloc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pp, bytesize, Flags);
}

CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                           unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr *, void *, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemHostGetDevicePointer_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pdptr, p, Flags);
}

CUresult CUDAAPI cuMemHostGetFlags(unsigned int *pFlags, void *p) {
  using FuncPtr = CUresult(CUDAAPI *)(unsigned int *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemHostGetFlags");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pFlags, p);
}

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                                   unsigned int flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr *, size_t, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemAllocManaged");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dptr, bytesize, flags);
}

CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdevice *, const char *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceGetByPCIBusId");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dev, pciBusId);
}

CUresult CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
  using FuncPtr = CUresult(CUDAAPI *)(char *, int, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceGetPCIBusId");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pciBusId, len, dev);
}

CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
  using FuncPtr = CUresult(CUDAAPI *)(CUipcEventHandle *, CUevent);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuIpcGetEventHandle");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pHandle, event);
}

CUresult CUDAAPI cuIpcOpenEventHandle(CUevent *phEvent,
                                      CUipcEventHandle handle) {
  using FuncPtr = CUresult(CUDAAPI *)(CUevent *, CUipcEventHandle);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuIpcOpenEventHandle");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phEvent, handle);
}

CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
  using FuncPtr = CUresult(CUDAAPI *)(CUipcMemHandle *, CUdeviceptr);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuIpcGetMemHandle");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pHandle, dptr);
}

CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle,
                                    unsigned int Flags) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr *, CUipcMemHandle, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuIpcOpenMemHandle");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pdptr, handle, Flags);
}

CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuIpcCloseMemHandle");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dptr);
}

CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize,
                                   unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(void *, size_t, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemHostRegister_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(p, bytesize, Flags);
}

CUresult CUDAAPI cuMemHostUnregister(void *p) {
  using FuncPtr = CUresult(CUDAAPI *)(void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemHostUnregister");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(p);
}

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, CUdeviceptr, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dst, src, ByteCount);
}

CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, CUcontext, CUdeviceptr,
                                      CUcontext, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyPeer");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, const void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyHtoD_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, srcHost, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  using FuncPtr = CUresult(CUDAAPI *)(void *, CUdeviceptr, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyDtoH_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstHost, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, CUdeviceptr, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyDtoD_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoA(CUarray dstArray, size_t dstOffset,
                              CUdeviceptr srcDevice, size_t ByteCount) {
  using FuncPtr = CUresult(CUDAAPI *)(CUarray, size_t, CUdeviceptr, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyDtoA_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstArray, dstOffset, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray,
                              size_t srcOffset, size_t ByteCount) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, CUarray, size_t, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyAtoD_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, srcArray, srcOffset, ByteCount);
}

CUresult CUDAAPI cuMemcpyHtoA(CUarray dstArray, size_t dstOffset,
                              const void *srcHost, size_t ByteCount) {
  using FuncPtr = CUresult(CUDAAPI *)(CUarray, size_t, const void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyHtoA_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstArray, dstOffset, srcHost, ByteCount);
}

CUresult CUDAAPI cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  using FuncPtr = CUresult(CUDAAPI *)(void *, CUarray, size_t, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyAtoH_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstHost, srcArray, srcOffset, ByteCount);
}

CUresult CUDAAPI cuMemcpyAtoA(CUarray dstArray, size_t dstOffset,
                              CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  using FuncPtr = CUresult(CUDAAPI *)(CUarray, size_t, CUarray, size_t, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyAtoA_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}

CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
  using FuncPtr = CUresult(CUDAAPI *)(const CUDA_MEMCPY2D *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpy2D_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pCopy);
}

CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
  using FuncPtr = CUresult(CUDAAPI *)(const CUDA_MEMCPY2D *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpy2DUnaligned_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pCopy);
}

CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
  using FuncPtr = CUresult(CUDAAPI *)(const CUDA_MEMCPY3D *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpy3D_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pCopy);
}

CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  using FuncPtr = CUresult(CUDAAPI *)(const CUDA_MEMCPY3D_PEER *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpy3DPeer");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pCopy);
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dst, src, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, CUcontext, CUdeviceptr,
                                      CUcontext, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyPeerAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, dstContext, srcDevice, srcContext, ByteCount,
                  hStream);
}

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                                   size_t ByteCount, CUstream hStream) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr, const void *, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyHtoDAsync_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, srcHost, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(void *, CUdeviceptr, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyDtoHAsync_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstHost, srcDevice, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyDtoDAsync_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset,
                                   const void *srcHost, size_t ByteCount,
                                   CUstream hStream) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUarray, size_t, const void *, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyHtoAAsync_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstArray, dstOffset, srcHost, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray,
                                   size_t srcOffset, size_t ByteCount,
                                   CUstream hStream) {
  using FuncPtr =
      CUresult(CUDAAPI *)(void *, CUarray, size_t, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpyAtoHAsync_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstHost, srcArray, srcOffset, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(const CUDA_MEMCPY2D *, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpy2DAsync_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pCopy, hStream);
}

CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(const CUDA_MEMCPY3D *, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpy3DAsync_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pCopy, hStream);
}

CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                                     CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(const CUDA_MEMCPY3D_PEER *, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemcpy3DPeerAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pCopy, hStream);
}

CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, unsigned char, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD8_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, uc, N);
}

CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us,
                             size_t N) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, unsigned short, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD16_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, us, N);
}

CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, unsigned int, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD32_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, ui, N);
}

CUresult CUDAAPI cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
                              unsigned char uc, size_t Width, size_t Height) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr, size_t, unsigned char, size_t, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD2D8_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, dstPitch, uc, Width, Height);
}

CUresult CUDAAPI cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned short us, size_t Width, size_t Height) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr, size_t, unsigned short, size_t, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD2D16_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, dstPitch, us, Width, Height);
}

CUresult CUDAAPI cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned int ui, size_t Width, size_t Height) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr, size_t, unsigned int, size_t, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD2D32_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, dstPitch, ui, Width, Height);
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr, unsigned char, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD8Async");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, uc, N, hStream);
}

CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr, unsigned short, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD16Async");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, us, N, hStream);
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr, unsigned int, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD32Async");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, ui, N, hStream);
}

CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                                   unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, size_t, unsigned char,
                                      size_t, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD2D8Async");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, dstPitch, uc, Width, Height, hStream);
}

CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, size_t, unsigned short,
                                      size_t, size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD2D16Async");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, dstPitch, us, Width, Height, hStream);
}

CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, size_t, unsigned int, size_t,
                                      size_t, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemsetD2D32Async");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dstDevice, dstPitch, ui, Width, Height, hStream);
}

CUresult CUDAAPI cuArrayCreate(CUarray *pHandle,
                               const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  using FuncPtr = CUresult(CUDAAPI *)(CUarray *, const CUDA_ARRAY_DESCRIPTOR *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuArrayCreate_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pHandle, pAllocateArray);
}

CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                                      CUarray hArray) {
  using FuncPtr = CUresult(CUDAAPI *)(CUDA_ARRAY_DESCRIPTOR *, CUarray);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuArrayGetDescriptor_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pArrayDescriptor, hArray);
}

CUresult CUDAAPI cuArrayDestroy(CUarray hArray) {
  using FuncPtr = CUresult(CUDAAPI *)(CUarray);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuArrayDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hArray);
}

CUresult CUDAAPI cuArray3DCreate(
    CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuArray3DCreate_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pHandle, pAllocateArray);
}

CUresult CUDAAPI cuArray3DGetDescriptor(
    CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
  using FuncPtr = CUresult(CUDAAPI *)(CUDA_ARRAY3D_DESCRIPTOR *, CUarray);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuArray3DGetDescriptor_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pArrayDescriptor, hArray);
}

CUresult CUDAAPI
cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels) {
  using FuncPtr = CUresult(CUDAAPI *)(
      CUmipmappedArray *, const CUDA_ARRAY3D_DESCRIPTOR *, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMipmappedArrayCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pHandle, pMipmappedArrayDesc, numMipmapLevels);
}

CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                                          CUmipmappedArray hMipmappedArray,
                                          unsigned int level) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUarray *, CUmipmappedArray, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMipmappedArrayGetLevel");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pLevelArray, hMipmappedArray, level);
}

CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
  using FuncPtr = CUresult(CUDAAPI *)(CUmipmappedArray);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMipmappedArrayDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hMipmappedArray);
}

CUresult CUDAAPI cuPointerGetAttribute(void *data,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  using FuncPtr = CUresult(CUDAAPI *)(void *, CUpointer_attribute, CUdeviceptr);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuPointerGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(data, attribute, ptr);
}

CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr, size_t, CUdevice, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemPrefetchAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(devPtr, count, dstDevice, hStream);
}

CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count,
                             CUmem_advise advice, CUdevice device) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr, size_t, CUmem_advise, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemAdvise");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(devPtr, count, advice, device);
}

CUresult CUDAAPI cuMemRangeGetAttribute(void *data, size_t dataSize,
                                        CUmem_range_attribute attribute,
                                        CUdeviceptr devPtr, size_t count) {
  using FuncPtr = CUresult(CUDAAPI *)(void *, size_t, CUmem_range_attribute,
                                      CUdeviceptr, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemRangeGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(data, dataSize, attribute, devPtr, count);
}

CUresult CUDAAPI cuMemRangeGetAttributes(void **data, size_t *dataSizes,
                                         CUmem_range_attribute *attributes,
                                         size_t numAttributes,
                                         CUdeviceptr devPtr, size_t count) {
  using FuncPtr = CUresult(CUDAAPI *)(
      void **, size_t *, CUmem_range_attribute *, size_t, CUdeviceptr, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuMemRangeGetAttributes");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(data, dataSizes, attributes, numAttributes, devPtr, count);
}

CUresult CUDAAPI cuPointerSetAttribute(const void *value,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  using FuncPtr =
      CUresult(CUDAAPI *)(const void *, CUpointer_attribute, CUdeviceptr);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuPointerSetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(value, attribute, ptr);
}

CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes,
                                        CUpointer_attribute *attributes,
                                        void **data, CUdeviceptr ptr) {
  using FuncPtr = CUresult(CUDAAPI *)(unsigned int, CUpointer_attribute *,
                                      void **, CUdeviceptr);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuPointerGetAttributes");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(numAttributes, attributes, data, ptr);
}

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream *, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phStream, Flags);
}

CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream,
                                            unsigned int flags, int priority) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream *, unsigned int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamCreateWithPriority");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phStream, flags, priority);
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamGetPriority");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream, priority);
}

CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream, unsigned int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamGetFlags");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream, flags);
}

CUresult CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream, CUcontext *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamGetCtx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream, pctx);
}

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream, CUevent, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamWaitEvent");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream, hEvent, Flags);
}

CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback, void *userData,
                                     unsigned int flags) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUstream, CUstreamCallback, void *, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamAddCallback");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream, callback, userData, flags);
}

CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream,
                                      CUstreamCaptureMode mode) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream, CUstreamCaptureMode);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamBeginCapture_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream, mode);
}

CUresult CUDAAPI cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstreamCaptureMode *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuThreadExchangeStreamCaptureMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(mode);
}

CUresult CUDAAPI cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream, CUgraph *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamEndCapture");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream, phGraph);
}

CUresult CUDAAPI cuStreamIsCapturing(CUstream hStream,
                                     CUstreamCaptureStatus *captureStatus) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream, CUstreamCaptureStatus *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamIsCapturing");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream, captureStatus);
}

CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream hStream,
                                        CUstreamCaptureStatus *captureStatus,
                                        cuuint64_t *id) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUstream, CUstreamCaptureStatus *, cuuint64_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamGetCaptureInfo");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream, captureStatus, id);
}

CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUstream, CUdeviceptr, size_t, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamAttachMemAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream, dptr, length, flags);
}

CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamQuery");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream);
}

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamSynchronize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream);
}

CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamDestroy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream);
}

CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUevent *, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuEventCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phEvent, Flags);
}

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(CUevent, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuEventRecord");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hEvent, hStream);
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
  using FuncPtr = CUresult(CUDAAPI *)(CUevent);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuEventQuery");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hEvent);
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
  using FuncPtr = CUresult(CUDAAPI *)(CUevent);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuEventSynchronize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hEvent);
}

CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
  using FuncPtr = CUresult(CUDAAPI *)(CUevent);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuEventDestroy_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hEvent);
}

CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart,
                                    CUevent hEnd) {
  using FuncPtr = CUresult(CUDAAPI *)(float *, CUevent, CUevent);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuEventElapsedTime");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pMilliseconds, hStart, hEnd);
}

CUresult CUDAAPI
cuImportExternalMemory(CUexternalMemory *extMem_out,
                       const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
  using FuncPtr = CUresult(CUDAAPI *)(CUexternalMemory *,
                                      const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuImportExternalMemory");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(extMem_out, memHandleDesc);
}

CUresult CUDAAPI cuExternalMemoryGetMappedBuffer(
    CUdeviceptr *devPtr, CUexternalMemory extMem,
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr *, CUexternalMemory,
                                      const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuExternalMemoryGetMappedBuffer");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(devPtr, extMem, bufferDesc);
}

CUresult CUDAAPI cuExternalMemoryGetMappedMipmappedArray(
    CUmipmappedArray *mipmap, CUexternalMemory extMem,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUmipmappedArray *, CUexternalMemory,
                          const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuExternalMemoryGetMappedMipmappedArray");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(mipmap, extMem, mipmapDesc);
}

CUresult CUDAAPI cuDestroyExternalMemory(CUexternalMemory extMem) {
  using FuncPtr = CUresult(CUDAAPI *)(CUexternalMemory);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDestroyExternalMemory");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(extMem);
}

CUresult CUDAAPI cuImportExternalSemaphore(
    CUexternalSemaphore *extSem_out,
    const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
  using FuncPtr = CUresult(CUDAAPI *)(
      CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuImportExternalSemaphore");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(extSem_out, semHandleDesc);
}

CUresult CUDAAPI cuSignalExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
  using FuncPtr = CUresult(CUDAAPI *)(
      const CUexternalSemaphore *,
      const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *, unsigned int, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuSignalExternalSemaphoresAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(extSemArray, paramsArray, numExtSems, stream);
}

CUresult CUDAAPI cuWaitExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
  using FuncPtr = CUresult(CUDAAPI *)(
      const CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *,
      unsigned int, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuWaitExternalSemaphoresAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(extSemArray, paramsArray, numExtSems, stream);
}

CUresult CUDAAPI cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
  using FuncPtr = CUresult(CUDAAPI *)(CUexternalSemaphore);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDestroyExternalSemaphore");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(extSem);
}

CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     cuuint32_t value, unsigned int flags) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamWaitValue32");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(stream, addr, value, flags);
}

CUresult CUDAAPI cuStreamWaitValue64(CUstream stream, CUdeviceptr addr,
                                     cuuint64_t value, unsigned int flags) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamWaitValue64");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(stream, addr, value, flags);
}

CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                      cuuint32_t value, unsigned int flags) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamWriteValue32");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(stream, addr, value, flags);
}

CUresult CUDAAPI cuStreamWriteValue64(CUstream stream, CUdeviceptr addr,
                                      cuuint64_t value, unsigned int flags) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamWriteValue64");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(stream, addr, value, flags);
}

CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams *paramArray,
                                    unsigned int flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream, unsigned int,
                                      CUstreamBatchMemOpParams *, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuStreamBatchMemOp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(stream, count, paramArray, flags);
}

CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                                    CUfunction hfunc) {
  using FuncPtr = CUresult(CUDAAPI *)(int *, CUfunction_attribute, CUfunction);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuFuncGetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pi, attrib, hfunc);
}

CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc,
                                    CUfunction_attribute attrib, int value) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, CUfunction_attribute, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuFuncSetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hfunc, attrib, value);
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, CUfunc_cache);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuFuncSetCacheConfig");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hfunc, config);
}

CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction hfunc,
                                          CUsharedconfig config) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, CUsharedconfig);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuFuncSetSharedMemConfig");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hfunc, config);
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra) {
  using FuncPtr = CUresult(CUDAAPI *)(
      CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
      unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuLaunchKernel");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                  blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

CUresult CUDAAPI cuLaunchCooperativeKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  using FuncPtr = CUresult(CUDAAPI *)(
      CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
      unsigned int, unsigned int, unsigned int, CUstream, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuLaunchCooperativeKernel");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                  blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult CUDAAPI cuLaunchCooperativeKernelMultiDevice(
    CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices,
    unsigned int flags) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUDA_LAUNCH_PARAMS *, unsigned int, unsigned int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuLaunchCooperativeKernelMultiDevice");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(launchParamsList, numDevices, flags);
}

CUresult CUDAAPI cuLaunchHostFunc(CUstream hStream, CUhostFn fn,
                                  void *userData) {
  using FuncPtr = CUresult(CUDAAPI *)(CUstream, CUhostFn, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuLaunchHostFunc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hStream, fn, userData);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuFuncSetBlockShape(CUfunction hfunc, int x,
                                                       int y, int z) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, int, int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuFuncSetBlockShape");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hfunc, x, y, z);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuFuncSetSharedSize(CUfunction hfunc,
                                                       unsigned int bytes) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuFuncSetSharedSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hfunc, bytes);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuParamSetSize(CUfunction hfunc,
                                                  unsigned int numbytes) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuParamSetSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hfunc, numbytes);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuParamSeti(CUfunction hfunc, int offset,
                                               unsigned int value) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, int, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuParamSeti");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hfunc, offset, value);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuParamSetf(CUfunction hfunc, int offset,
                                               float value) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, int, float);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuParamSetf");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hfunc, offset, value);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuParamSetv(CUfunction hfunc, int offset,
                                               void *ptr,
                                               unsigned int numbytes) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, int, void *, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuParamSetv");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hfunc, offset, ptr, numbytes);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuLaunch(CUfunction f) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuLaunch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(f);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuLaunchGrid(CUfunction f, int grid_width,
                                                int grid_height) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuLaunchGrid");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(f, grid_width, grid_height);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuLaunchGridAsync(CUfunction f,
                                                     int grid_width,
                                                     int grid_height,
                                                     CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, int, int, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuLaunchGridAsync");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(f, grid_width, grid_height, hStream);
}

__CUDA_DEPRECATED CUresult CUDAAPI cuParamSetTexRef(CUfunction hfunc,
                                                    int texunit,
                                                    CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfunction, int, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuParamSetTexRef");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hfunc, texunit, hTexRef);
}

CUresult CUDAAPI cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraph *, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phGraph, flags);
}

CUresult CUDAAPI cuGraphAddKernelNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t,
                          const CUDA_KERNEL_NODE_PARAMS *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphAddKernelNode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phGraphNode, hGraph, dependencies, numDependencies,
                  nodeParams);
}

CUresult CUDAAPI cuGraphKernelNodeGetParams(
    CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphKernelNodeGetParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphKernelNodeSetParams(
    CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphKernelNodeSetParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                      const CUgraphNode *dependencies,
                                      size_t numDependencies,
                                      const CUDA_MEMCPY3D *copyParams,
                                      CUcontext ctx) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t,
                          const CUDA_MEMCPY3D *, CUcontext);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphAddMemcpyNode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phGraphNode, hGraph, dependencies, numDependencies,
                  copyParams, ctx);
}

CUresult CUDAAPI cuGraphMemcpyNodeGetParams(CUgraphNode hNode,
                                            CUDA_MEMCPY3D *nodeParams) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode, CUDA_MEMCPY3D *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphMemcpyNodeGetParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphMemcpyNodeSetParams(CUgraphNode hNode,
                                            const CUDA_MEMCPY3D *nodeParams) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode, const CUDA_MEMCPY3D *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphMemcpyNodeSetParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddMemsetNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams,
    CUcontext ctx) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t,
                          const CUDA_MEMSET_NODE_PARAMS *, CUcontext);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphAddMemsetNode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phGraphNode, hGraph, dependencies, numDependencies,
                  memsetParams, ctx);
}

CUresult CUDAAPI cuGraphMemsetNodeGetParams(
    CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode, CUDA_MEMSET_NODE_PARAMS *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphMemsetNodeGetParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphMemsetNodeSetParams(
    CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUgraphNode, const CUDA_MEMSET_NODE_PARAMS *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphMemsetNodeSetParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                    const CUgraphNode *dependencies,
                                    size_t numDependencies,
                                    const CUDA_HOST_NODE_PARAMS *nodeParams) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t,
                          const CUDA_HOST_NODE_PARAMS *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphAddHostNode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phGraphNode, hGraph, dependencies, numDependencies,
                  nodeParams);
}

CUresult CUDAAPI cuGraphHostNodeGetParams(CUgraphNode hNode,
                                          CUDA_HOST_NODE_PARAMS *nodeParams) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode, CUDA_HOST_NODE_PARAMS *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphHostNodeGetParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphHostNodeSetParams(
    CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUgraphNode, const CUDA_HOST_NODE_PARAMS *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphHostNodeSetParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddChildGraphNode(CUgraphNode *phGraphNode,
                                          CUgraph hGraph,
                                          const CUgraphNode *dependencies,
                                          size_t numDependencies,
                                          CUgraph childGraph) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode *, CUgraph,
                                      const CUgraphNode *, size_t, CUgraph);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphAddChildGraphNode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phGraphNode, hGraph, dependencies, numDependencies,
                  childGraph);
}

CUresult CUDAAPI cuGraphChildGraphNodeGetGraph(CUgraphNode hNode,
                                               CUgraph *phGraph) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode, CUgraph *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphChildGraphNodeGetGraph");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, phGraph);
}

CUresult CUDAAPI cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                     const CUgraphNode *dependencies,
                                     size_t numDependencies) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphAddEmptyNode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phGraphNode, hGraph, dependencies, numDependencies);
}

CUresult CUDAAPI cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraph *, CUgraph);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphClone");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phGraphClone, originalGraph);
}

CUresult CUDAAPI cuGraphNodeFindInClone(CUgraphNode *phNode,
                                        CUgraphNode hOriginalNode,
                                        CUgraph hClonedGraph) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode *, CUgraphNode, CUgraph);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphNodeFindInClone");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phNode, hOriginalNode, hClonedGraph);
}

CUresult CUDAAPI cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode, CUgraphNodeType *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphNodeGetType");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, type);
}

CUresult CUDAAPI cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes,
                                 size_t *numNodes) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraph, CUgraphNode *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphGetNodes");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hGraph, nodes, numNodes);
}

CUresult CUDAAPI cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes,
                                     size_t *numRootNodes) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraph, CUgraphNode *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphGetRootNodes");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hGraph, rootNodes, numRootNodes);
}

CUresult CUDAAPI cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from,
                                 CUgraphNode *to, size_t *numEdges) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUgraph, CUgraphNode *, CUgraphNode *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphGetEdges");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hGraph, from, to, numEdges);
}

CUresult CUDAAPI cuGraphNodeGetDependencies(CUgraphNode hNode,
                                            CUgraphNode *dependencies,
                                            size_t *numDependencies) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode, CUgraphNode *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphNodeGetDependencies");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, dependencies, numDependencies);
}

CUresult CUDAAPI cuGraphNodeGetDependentNodes(CUgraphNode hNode,
                                              CUgraphNode *dependentNodes,
                                              size_t *numDependentNodes) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode, CUgraphNode *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphNodeGetDependentNodes");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode, dependentNodes, numDependentNodes);
}

CUresult CUDAAPI cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from,
                                        const CUgraphNode *to,
                                        size_t numDependencies) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraph, const CUgraphNode *,
                                      const CUgraphNode *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphAddDependencies");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hGraph, from, to, numDependencies);
}

CUresult CUDAAPI cuGraphRemoveDependencies(CUgraph hGraph,
                                           const CUgraphNode *from,
                                           const CUgraphNode *to,
                                           size_t numDependencies) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraph, const CUgraphNode *,
                                      const CUgraphNode *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphRemoveDependencies");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hGraph, from, to, numDependencies);
}

CUresult CUDAAPI cuGraphDestroyNode(CUgraphNode hNode) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphNode);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphDestroyNode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hNode);
}

CUresult CUDAAPI cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                                    CUgraphNode *phErrorNode, char *logBuffer,
                                    size_t bufferSize) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphExec *, CUgraph, CUgraphNode *,
                                      char *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphInstantiate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize);
}

CUresult CUDAAPI
cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                               const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphExec, CUgraphNode,
                                      const CUDA_KERNEL_NODE_PARAMS *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphExecKernelNodeSetParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hGraphExec, hNode, nodeParams);
}

CUresult CUDAAPI cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphExec, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphLaunch");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hGraphExec, hStream);
}

CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec hGraphExec) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphExec);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphExecDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hGraphExec);
}

CUresult CUDAAPI cuGraphDestroy(CUgraph hGraph) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraph);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hGraph);
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  using FuncPtr = CUresult(CUDAAPI *)(int *, CUfunction, int, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(numBlocks, func, blockSize, dynamicSMemSize);
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  using FuncPtr =
      CUresult(CUDAAPI *)(int *, CUfunction, int, size_t, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>(
      "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(numBlocks, func, blockSize, dynamicSMemSize, flags);
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit) {
  using FuncPtr = CUresult(CUDAAPI *)(int *, int *, CUfunction,
                                      CUoccupancyB2DSize, size_t, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuOccupancyMaxPotentialBlockSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(minGridSize, blockSize, func, blockSizeToDynamicSMemSize,
                  dynamicSMemSize, blockSizeLimit);
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags) {
  using FuncPtr = CUresult(CUDAAPI *)(
      int *, int *, CUfunction, CUoccupancyB2DSize, size_t, int, unsigned int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuOccupancyMaxPotentialBlockSizeWithFlags");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(minGridSize, blockSize, func, blockSizeToDynamicSMemSize,
                  dynamicSMemSize, blockSizeLimit, flags);
}

CUresult CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                                  unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, CUarray, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetArray");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, hArray, Flags);
}

CUresult CUDAAPI cuTexRefSetMipmappedArray(CUtexref hTexRef,
                                           CUmipmappedArray hMipmappedArray,
                                           unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, CUmipmappedArray, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetMipmappedArray");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, hMipmappedArray, Flags);
}

CUresult CUDAAPI cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef,
                                    CUdeviceptr dptr, size_t bytes) {
  using FuncPtr = CUresult(CUDAAPI *)(size_t *, CUtexref, CUdeviceptr, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetAddress_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ByteOffset, hTexRef, dptr, bytes);
}

CUresult CUDAAPI cuTexRefSetAddress2D(CUtexref hTexRef,
                                      const CUDA_ARRAY_DESCRIPTOR *desc,
                                      CUdeviceptr dptr, size_t Pitch) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, const CUDA_ARRAY_DESCRIPTOR *,
                                      CUdeviceptr, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetAddress2D_v3");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, desc, dptr, Pitch);
}

CUresult CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                                   int NumPackedComponents) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, CUarray_format, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetFormat");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, fmt, NumPackedComponents);
}

CUresult CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim,
                                        CUaddress_mode am) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, int, CUaddress_mode);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetAddressMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, dim, am);
}

CUresult CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, CUfilter_mode);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetFilterMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, fm);
}

CUresult CUDAAPI cuTexRefSetMipmapFilterMode(CUtexref hTexRef,
                                             CUfilter_mode fm) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, CUfilter_mode);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetMipmapFilterMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, fm);
}

CUresult CUDAAPI cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, float);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetMipmapLevelBias");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, bias);
}

CUresult CUDAAPI cuTexRefSetMipmapLevelClamp(CUtexref hTexRef,
                                             float minMipmapLevelClamp,
                                             float maxMipmapLevelClamp) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, float, float);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetMipmapLevelClamp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
}

CUresult CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef,
                                          unsigned int maxAniso) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetMaxAnisotropy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, maxAniso);
}

CUresult CUDAAPI cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, float *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetBorderColor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, pBorderColor);
}

CUresult CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefSetFlags");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef, Flags);
}

CUresult CUDAAPI cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(CUdeviceptr *, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetAddress_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pdptr, hTexRef);
}

CUresult CUDAAPI cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(CUarray *, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetArray");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phArray, hTexRef);
}

CUresult CUDAAPI cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray,
                                           CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(CUmipmappedArray *, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetMipmappedArray");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phMipmappedArray, hTexRef);
}

CUresult CUDAAPI cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef,
                                        int dim) {
  using FuncPtr = CUresult(CUDAAPI *)(CUaddress_mode *, CUtexref, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetAddressMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pam, hTexRef, dim);
}

CUresult CUDAAPI cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfilter_mode *, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetFilterMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pfm, hTexRef);
}

CUresult CUDAAPI cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels,
                                   CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(CUarray_format *, int *, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetFormat");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pFormat, pNumChannels, hTexRef);
}

CUresult CUDAAPI cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm,
                                             CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(CUfilter_mode *, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetMipmapFilterMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pfm, hTexRef);
}

CUresult CUDAAPI cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(float *, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetMipmapLevelBias");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pbias, hTexRef);
}

CUresult CUDAAPI cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp,
                                             float *pmaxMipmapLevelClamp,
                                             CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(float *, float *, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetMipmapLevelClamp");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
}

CUresult CUDAAPI cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(int *, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetMaxAnisotropy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pmaxAniso, hTexRef);
}

CUresult CUDAAPI cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(float *, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetBorderColor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pBorderColor, hTexRef);
}

CUresult CUDAAPI cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(unsigned int *, CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefGetFlags");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pFlags, hTexRef);
}

CUresult CUDAAPI cuTexRefCreate(CUtexref *pTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pTexRef);
}

CUresult CUDAAPI cuTexRefDestroy(CUtexref hTexRef) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexRefDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hTexRef);
}

CUresult CUDAAPI cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray,
                                   unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUsurfref, CUarray, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuSurfRefSetArray");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(hSurfRef, hArray, Flags);
}

CUresult CUDAAPI cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
  using FuncPtr = CUresult(CUDAAPI *)(CUarray *, CUsurfref);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuSurfRefGetArray");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(phArray, hSurfRef);
}

CUresult CUDAAPI
cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                  const CUDA_TEXTURE_DESC *pTexDesc,
                  const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexObject *, const CUDA_RESOURCE_DESC *,
                                      const CUDA_TEXTURE_DESC *,
                                      const CUDA_RESOURCE_VIEW_DESC *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexObjectCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject) {
  using FuncPtr = CUresult(CUDAAPI *)(CUtexObject);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexObjectDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(texObject);
}

CUresult CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                            CUtexObject texObject) {
  using FuncPtr = CUresult(CUDAAPI *)(CUDA_RESOURCE_DESC *, CUtexObject);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexObjectGetResourceDesc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pResDesc, texObject);
}

CUresult CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                           CUtexObject texObject) {
  using FuncPtr = CUresult(CUDAAPI *)(CUDA_TEXTURE_DESC *, CUtexObject);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexObjectGetTextureDesc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pTexDesc, texObject);
}

CUresult CUDAAPI cuTexObjectGetResourceViewDesc(
    CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject) {
  using FuncPtr = CUresult(CUDAAPI *)(CUDA_RESOURCE_VIEW_DESC *, CUtexObject);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuTexObjectGetResourceViewDesc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pResViewDesc, texObject);
}

CUresult CUDAAPI cuSurfObjectCreate(CUsurfObject *pSurfObject,
                                    const CUDA_RESOURCE_DESC *pResDesc) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUsurfObject *, const CUDA_RESOURCE_DESC *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuSurfObjectCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pSurfObject, pResDesc);
}

CUresult CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject) {
  using FuncPtr = CUresult(CUDAAPI *)(CUsurfObject);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuSurfObjectDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(surfObject);
}

CUresult CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                             CUsurfObject surfObject) {
  using FuncPtr = CUresult(CUDAAPI *)(CUDA_RESOURCE_DESC *, CUsurfObject);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuSurfObjectGetResourceDesc");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pResDesc, surfObject);
}

CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                                       CUdevice peerDev) {
  using FuncPtr = CUresult(CUDAAPI *)(int *, CUdevice, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceCanAccessPeer");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(canAccessPeer, dev, peerDev);
}

CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext,
                                       unsigned int Flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext, unsigned int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxEnablePeerAccess");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(peerContext, Flags);
}

CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext) {
  using FuncPtr = CUresult(CUDAAPI *)(CUcontext);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuCtxDisablePeerAccess");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(peerContext);
}

CUresult CUDAAPI cuDeviceGetP2PAttribute(int *value,
                                         CUdevice_P2PAttribute attrib,
                                         CUdevice srcDevice,
                                         CUdevice dstDevice) {
  using FuncPtr =
      CUresult(CUDAAPI *)(int *, CUdevice_P2PAttribute, CUdevice, CUdevice);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuDeviceGetP2PAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(value, attrib, srcDevice, dstDevice);
}

CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphicsResource);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphicsUnregisterResource");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(resource);
}

CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(
    CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex,
    unsigned int mipLevel) {
  using FuncPtr = CUresult(CUDAAPI *)(CUarray *, CUgraphicsResource,
                                      unsigned int, unsigned int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuGraphicsSubResourceGetMappedArray");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pArray, resource, arrayIndex, mipLevel);
}

CUresult CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(
    CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) {
  using FuncPtr = CUresult(CUDAAPI *)(CUmipmappedArray *, CUgraphicsResource);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuGraphicsResourceGetMappedMipmappedArray");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pMipmappedArray, resource);
}

CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(
    CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource) {
  using FuncPtr =
      CUresult(CUDAAPI *)(CUdeviceptr *, size_t *, CUgraphicsResource);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuGraphicsResourceGetMappedPointer_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(pDevPtr, pSize, resource);
}

CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,
                                               unsigned int flags) {
  using FuncPtr = CUresult(CUDAAPI *)(CUgraphicsResource, unsigned int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cuGraphicsResourceSetMapFlags_v2");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(resource, flags);
}

CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource *resources,
                                        CUstream hStream) {
  using FuncPtr =
      CUresult(CUDAAPI *)(unsigned int, CUgraphicsResource *, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphicsMapResources");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(count, resources, hStream);
}

CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource *resources,
                                          CUstream hStream) {
  using FuncPtr =
      CUresult(CUDAAPI *)(unsigned int, CUgraphicsResource *, CUstream);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGraphicsUnmapResources");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(count, resources, hStream);
}

CUresult CUDAAPI cuGetExportTable(const void **ppExportTable,
                                  const CUuuid *pExportTableId) {
  using FuncPtr = CUresult(CUDAAPI *)(const void **, const CUuuid *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cuGetExportTable");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ppExportTable, pExportTableId);
}

}  // extern "C"
