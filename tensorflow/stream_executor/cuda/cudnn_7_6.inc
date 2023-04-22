// Auto-generated, do not edit.

extern "C" {

size_t CUDNNWINAPI cudnnGetVersion(void) {
  using FuncPtr = size_t(CUDNNWINAPI *)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetVersion");
  if (!func_ptr) return 0;
  return func_ptr();
}

size_t CUDNNWINAPI cudnnGetCudartVersion(void) {
  using FuncPtr = size_t(CUDNNWINAPI *)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetCudartVersion");
  if (!func_ptr) return 0;
  return func_ptr();
}

const char *CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status) {
  using FuncPtr = const char *(CUDNNWINAPI *)(cudnnStatus_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetErrorString");
  if (!func_ptr) return "cudnnGetErrorString symbol not found.";
  return func_ptr(status);
}

cudnnStatus_t CUDNNWINAPI cudnnQueryRuntimeError(cudnnHandle_t handle,
                                                 cudnnStatus_t *rstatus,
                                                 cudnnErrQueryMode_t mode,
                                                 cudnnRuntimeTag_t *tag) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnStatus_t *, cudnnErrQueryMode_t, cudnnRuntimeTag_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnQueryRuntimeError");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rstatus, mode, tag);
}

cudnnStatus_t CUDNNWINAPI cudnnGetProperty(libraryPropertyType type,
                                           int *value) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(libraryPropertyType, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetProperty");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(type, value);
}

cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnHandle_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreate");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnHandle_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroy");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle);
}

cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle,
                                         cudaStream_t streamId) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnHandle_t, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetStream");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, streamId);
}

cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t handle,
                                         cudaStream_t *streamId) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnHandle_t, cudaStream_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetStream");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, streamId);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnTensorDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(tensorDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, /* image data type */
    int n,                    /* number of inputs (batch size) */
    int c,                    /* number of input feature maps */
    int h,                    /* height of input section */
    int w) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnTensorDescriptor_t, cudnnTensorFormat_t,
                                   cudnnDataType_t, int, int, int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetTensor4dDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(tensorDesc, format, dataType, n, c, h, w);
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType, /* image data type */
    int n,                    /* number of inputs (batch size) */
    int c,                    /* number of input feature maps */
    int h,                    /* height of input section */
    int w,                    /* width of input section */
    int nStride, int cStride, int hStride, int wStride) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnTensorDescriptor_t, cudnnDataType_t,
                                   int, int, int, int, int, int, int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetTensor4dDescriptorEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride,
                  wStride);
}

cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor(
    const cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t *dataType, /* image data type */
    int *n,                    /* number of inputs (batch size) */
    int *c,                    /* number of input feature maps  */
    int *h,                    /* height of input section */
    int *w,                    /* width of input section */
    int *nStride, int *cStride, int *hStride, int *wStride) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnTensorDescriptor_t, cudnnDataType_t *, int *, int *, int *,
      int *, int *, int *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetTensor4dDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride,
                  wStride);
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims,
    const int dimA[], const int strideA[]) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnTensorDescriptor_t, cudnnDataType_t, int, const int[], const int[]);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetTensorNdDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(tensorDesc, dataType, nbDims, dimA, strideA);
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptorEx(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int nbDims, const int dimA[]) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnTensorDescriptor_t, cudnnTensorFormat_t,
                                   cudnnDataType_t, int, const int[]);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetTensorNdDescriptorEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(tensorDesc, format, dataType, nbDims, dimA);
}

cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(
    const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, int *nbDims, int dimA[], int strideA[]) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(const cudnnTensorDescriptor_t, int,
                                   cudnnDataType_t *, int *, int[], int[]);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetTensorNdDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);
}

cudnnStatus_t CUDNNWINAPI cudnnGetTensorSizeInBytes(
    const cudnnTensorDescriptor_t tensorDesc, size_t *size) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(const cudnnTensorDescriptor_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetTensorSizeInBytes");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(tensorDesc, size);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnTensorDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(tensorDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnInitTransformDest(
    const cudnnTensorTransformDescriptor_t transformDesc,
    const cudnnTensorDescriptor_t srcDesc, cudnnTensorDescriptor_t destDesc,
    size_t *destSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnTensorTransformDescriptor_t, const cudnnTensorDescriptor_t,
      cudnnTensorDescriptor_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnInitTransformDest");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(transformDesc, srcDesc, destDesc, destSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t *transformDesc) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnTensorTransformDescriptor_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnCreateTensorTransformDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(transformDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc, const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat, const int32_t padBeforeA[],
    const int32_t padAfterA[], const uint32_t foldA[],
    const cudnnFoldingDirection_t direction) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnTensorTransformDescriptor_t, const uint32_t,
      const cudnnTensorFormat_t, const int32_t[], const int32_t[],
      const uint32_t[], const cudnnFoldingDirection_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnSetTensorTransformDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(transformDesc, nbDims, destFormat, padBeforeA, padAfterA,
                  foldA, direction);
}

cudnnStatus_t CUDNNWINAPI cudnnGetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc, uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat, int32_t padBeforeA[], int32_t padAfterA[],
    uint32_t foldA[], cudnnFoldingDirection_t *direction) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnTensorTransformDescriptor_t, uint32_t, cudnnTensorFormat_t *,
      int32_t[], int32_t[], uint32_t[], cudnnFoldingDirection_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetTensorTransformDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(transformDesc, nbDimsRequested, destFormat, padBeforeA,
                  padAfterA, foldA, direction);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnTensorTransformDescriptor_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnDestroyTensorTransformDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(transformDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *,
      const void *, const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnTransformTensor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha, xDesc, x, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnTransformTensorEx(
    cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha, const cudnnTensorDescriptor_t srcDesc,
    const void *srcData, const void *beta,
    const cudnnTensorDescriptor_t destDesc, void *destData) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorTransformDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnTransformTensorEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc,
                  destData);
}

cudnnStatus_t CUDNNWINAPI cudnnGetFoldedConvBackwardDataDescriptors(
    const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc,
    cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc,
    cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc,
    cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc,
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnHandle_t, const cudnnFilterDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnTensorFormat_t,
      cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
      cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
      cudnnTensorTransformDescriptor_t, cudnnTensorTransformDescriptor_t,
      cudnnTensorTransformDescriptor_t, cudnnTensorTransformDescriptor_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetFoldedConvBackwardDataDescriptors");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, filterDesc, diffDesc, convDesc, gradDesc,
                  transformFormat, foldedFilterDesc, paddedDiffDesc,
                  foldedConvDesc, foldedGradDesc, filterFoldTransDesc,
                  diffPadTransDesc, gradFoldTransDesc, gradUnfoldTransDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t handle,
                                         const void *alpha,
                                         const cudnnTensorDescriptor_t aDesc,
                                         const void *A, const void *beta,
                                         const cudnnTensorDescriptor_t cDesc,
                                         void *C) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *,
      const void *, const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnAddTensor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha, aDesc, A, beta, cDesc, C);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnOpTensorDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateOpTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(opTensorDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetOpTensorDescriptor(
    cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp,
    cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t,
                                   cudnnDataType_t, cudnnNanPropagation_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetOpTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
}

cudnnStatus_t CUDNNWINAPI cudnnGetOpTensorDescriptor(
    const cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType, cudnnNanPropagation_t *opTensorNanOpt) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t *, cudnnDataType_t *,
      cudnnNanPropagation_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetOpTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnOpTensorDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyOpTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(opTensorDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnOpTensor(
    cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1, const cudnnTensorDescriptor_t aDesc, const void *A,
    const void *alpha2, const cudnnTensorDescriptor_t bDesc, const void *B,
    const void *beta, const cudnnTensorDescriptor_t cDesc, void *C) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnOpTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnOpTensor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B,
                  beta, cDesc, C);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t *reduceTensorDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnReduceTensorDescriptor_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnCreateReduceTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(reduceTensorDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t reduceTensorOp, cudnnDataType_t reduceTensorCompType,
    cudnnNanPropagation_t reduceTensorNanOpt,
    cudnnReduceTensorIndices_t reduceTensorIndices,
    cudnnIndicesType_t reduceTensorIndicesType) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnReduceTensorDescriptor_t, cudnnReduceTensorOp_t, cudnnDataType_t,
      cudnnNanPropagation_t, cudnnReduceTensorIndices_t, cudnnIndicesType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetReduceTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(reduceTensorDesc, reduceTensorOp, reduceTensorCompType,
                  reduceTensorNanOpt, reduceTensorIndices,
                  reduceTensorIndicesType);
}

cudnnStatus_t CUDNNWINAPI cudnnGetReduceTensorDescriptor(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType,
    cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices,
    cudnnIndicesType_t *reduceTensorIndicesType) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnReduceTensorDescriptor_t, cudnnReduceTensorOp_t *,
      cudnnDataType_t *, cudnnNanPropagation_t *, cudnnReduceTensorIndices_t *,
      cudnnIndicesType_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetReduceTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(reduceTensorDesc, reduceTensorOp, reduceTensorCompType,
                  reduceTensorNanOpt, reduceTensorIndices,
                  reduceTensorIndicesType);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t reduceTensorDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnReduceTensorDescriptor_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnDestroyReduceTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(reduceTensorDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnReduceTensorDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetReductionIndicesSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnReduceTensorDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetReductionWorkspaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices, size_t indicesSizeInBytes, void *workspace,
    size_t workspaceSizeInBytes, const void *alpha,
    const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta,
    const cudnnTensorDescriptor_t cDesc, void *C) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnReduceTensorDescriptor_t, void *, size_t,
      void *, size_t, const void *, const cudnnTensorDescriptor_t, const void *,
      const void *, const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnReduceTensor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, reduceTensorDesc, indices, indicesSizeInBytes,
                  workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc,
                  C);
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensor(cudnnHandle_t handle,
                                         const cudnnTensorDescriptor_t yDesc,
                                         void *y, const void *valuePtr) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t, void *, const void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetTensor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, yDesc, y, valuePtr);
}

cudnnStatus_t CUDNNWINAPI cudnnScaleTensor(cudnnHandle_t handle,
                                           const cudnnTensorDescriptor_t yDesc,
                                           void *y, const void *alpha) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t, void *, const void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnScaleTensor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, yDesc, y, alpha);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnFilterDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateFilterDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(filterDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,          /* image data type */
    cudnnTensorFormat_t format, int k, /* number of output feature maps */
    int c,                             /* number of input feature maps */
    int h,                             /* height of each input filter */
    int w) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnFilterDescriptor_t, cudnnDataType_t,
                                   cudnnTensorFormat_t, int, int, int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetFilter4dDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(filterDesc, dataType, format, k, c, h, w);
}

cudnnStatus_t CUDNNWINAPI cudnnGetFilter4dDescriptor(
    const cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t *dataType,           /* image data type */
    cudnnTensorFormat_t *format, int *k, /* number of output feature maps */
    int *c,                              /* number of input feature maps */
    int *h,                              /* height of each input filter */
    int *w) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnFilterDescriptor_t, cudnnDataType_t *, cudnnTensorFormat_t *,
      int *, int *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetFilter4dDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(filterDesc, dataType, format, k, c, h, w);
}

cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType, /* image data type */
    cudnnTensorFormat_t format, int nbDims, const int filterDimA[]) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnFilterDescriptor_t, cudnnDataType_t,
                                   cudnnTensorFormat_t, int, const int[]);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetFilterNdDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(filterDesc, dataType, format, nbDims, filterDimA);
}

cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor(
    const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, /* image data type */
    cudnnTensorFormat_t *format, int *nbDims, int filterDimA[]) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnFilterDescriptor_t, int, cudnnDataType_t *,
      cudnnTensorFormat_t *, int *, int[]);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetFilterNdDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(filterDesc, nbDimsRequested, dataType, format, nbDims,
                  filterDimA);
}

cudnnStatus_t CUDNNWINAPI cudnnGetFilterSizeInBytes(
    const cudnnFilterDescriptor_t filterDesc, size_t *size) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(const cudnnFilterDescriptor_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetFilterSizeInBytes");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(filterDesc, size);
}

cudnnStatus_t CUDNNWINAPI cudnnTransformFilter(
    cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
    const void *alpha, const cudnnFilterDescriptor_t srcDesc,
    const void *srcData, const void *beta,
    const cudnnFilterDescriptor_t destDesc, void *destData) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorTransformDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *, const void *,
      const cudnnFilterDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnTransformFilter");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc,
                  destData);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnFilterDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyFilterDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(filterDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnReorderFilterAndBias(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    cudnnReorderType_t reorderType, const void *filterData,
    void *reorderedFilterData, int reorderBias, const void *biasData,
    void *reorderedBiasData) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnFilterDescriptor_t, cudnnReorderType_t,
      const void *, void *, int, const void *, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnReorderFilterAndBias");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, filterDesc, reorderType, filterData,
                  reorderedFilterData, reorderBias, biasData,
                  reorderedBiasData);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnConvolutionDescriptor_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnCreateConvolutionDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnConvolutionDescriptor_t,
                                               cudnnMathType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetConvolutionMathType");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, mathType);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionMathType(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnConvolutionDescriptor_t,
                                               cudnnMathType_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetConvolutionMathType");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, mathType);
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t convDesc, int groupCount) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnConvolutionDescriptor_t, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetConvolutionGroupCount");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, groupCount);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t convDesc, int *groupCount) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnConvolutionDescriptor_t, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetConvolutionGroupCount");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, groupCount);
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionReorderType(
    cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnConvolutionDescriptor_t,
                                               cudnnReorderType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetConvolutionReorderType");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, reorderType);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionReorderType(
    cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnConvolutionDescriptor_t,
                                               cudnnReorderType_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetConvolutionReorderType");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, reorderType);
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int pad_h, /* zero-padding height */
    int pad_w,                                        /* zero-padding width */
    int u,          /* vertical filter stride */
    int v,          /* horizontal filter stride */
    int dilation_h, /* filter dilation in the vertical dimension */
    int dilation_w, /* filter dilation in the horizontal dimension */
    cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnConvolutionDescriptor_t, int, int, int, int, int, int,
      cudnnConvolutionMode_t, cudnnDataType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetConvolution2dDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode,
                  computeType);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dDescriptor(
    const cudnnConvolutionDescriptor_t convDesc,
    int *pad_h,      /* zero-padding height */
    int *pad_w,      /* zero-padding width */
    int *u,          /* vertical filter stride */
    int *v,          /* horizontal filter stride */
    int *dilation_h, /* filter dilation in the vertical dimension */
    int *dilation_w, /* filter dilation in the horizontal dimension */
    cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnConvolutionDescriptor_t, int *, int *, int *, int *, int *,
      int *, cudnnConvolutionMode_t *, cudnnDataType_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetConvolution2dDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode,
                  computeType);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t,
      const cudnnFilterDescriptor_t, int *, int *, int *, int *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolution2dForwardOutputDim");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, inputTensorDesc, filterDesc, n, c, h, w);
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int arrayLength, /* nbDims-2 size */
    const int padA[], const int filterStrideA[], const int dilationA[],
    cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnConvolutionDescriptor_t, int, const int[], const int[], const int[],
      cudnnConvolutionMode_t, cudnnDataType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetConvolutionNdDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, arrayLength, padA, filterStrideA, dilationA, mode,
                  computeType);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor(
    const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested,
    int *arrayLength, int padA[], int strideA[], int dilationA[],
    cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnConvolutionDescriptor_t, int, int *, int[], int[], int[],
      cudnnConvolutionMode_t *, cudnnDataType_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetConvolutionNdDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, arrayLengthRequested, arrayLength, padA, strideA,
                  dilationA, mode, computeType);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims,
    int tensorOutputDimA[]) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t,
      const cudnnFilterDescriptor_t, int, int[]);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionNdForwardOutputDim");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc, inputTensorDesc, filterDesc, nbDims,
                  tensorOutputDimA);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnConvolutionDescriptor_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnDestroyConvolutionDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(convDesc);
}

cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnHandle_t, int *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionForwardAlgorithmMaxCount");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, count);
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t,
      const cudnnFilterDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnTensorDescriptor_t, const int, int *,
      cudnnConvolutionFwdAlgoPerf_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnFindConvolutionForwardAlgorithm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount,
                  returnedAlgoCount, perfResults);
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, void *y, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, void *,
      const int, int *, cudnnConvolutionFwdAlgoPerf_t *, void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnFindConvolutionForwardAlgorithmEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, xDesc, x, wDesc, w, convDesc, yDesc, y,
                  requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
                  workSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionFwdAlgo_t *algo) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t,
      const cudnnFilterDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnTensorDescriptor_t, cudnnConvolutionFwdPreference_t, size_t,
      cudnnConvolutionFwdAlgo_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionForwardAlgorithm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, xDesc, wDesc, convDesc, yDesc, preference,
                  memoryLimitInBytes, algo);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t,
      const cudnnFilterDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnTensorDescriptor_t, const int, int *,
      cudnnConvolutionFwdAlgoPerf_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionForwardAlgorithm_v7");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, srcDesc, filterDesc, convDesc, destDesc,
                  requestedAlgoCount, returnedAlgoCount, perfResults);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t,
      const cudnnFilterDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnTensorDescriptor_t, cudnnConvolutionFwdAlgo_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionForwardWorkspaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, void *,
      size_t, const void *, const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnConvolutionForward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
                  workSpaceSizeInBytes, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBiasActivationForward(
    cudnnHandle_t handle, const void *alpha1,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *alpha2,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc, void *y) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, void *,
      size_t, const void *, const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnActivationDescriptor_t, const cudnnTensorDescriptor_t, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnConvolutionBiasActivationForward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace,
                  workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias,
                  activationDesc, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
    const cudnnTensorDescriptor_t dbDesc, void *db) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *,
      const void *, const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnConvolutionBackwardBias");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha, dyDesc, dy, beta, dbDesc, db);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
    cudnnHandle_t handle, int *count) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnHandle_t, int *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, count);
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnFilterDescriptor_t, const int, int *,
      cudnnConvolutionBwdFilterAlgoPerf_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnFindConvolutionBackwardFilterAlgorithm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount,
                  returnedAlgoCount, perfResults);
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *y,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnConvolutionDescriptor_t, const cudnnFilterDescriptor_t, void *,
      const int, int *, cudnnConvolutionBwdFilterAlgoPerf_t *, void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnFindConvolutionBackwardFilterAlgorithmEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw,
                  requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
                  workSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t *algo) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnFilterDescriptor_t, cudnnConvolutionBwdFilterPreference_t,
      size_t, cudnnConvolutionBwdFilterAlgo_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionBackwardFilterAlgorithm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, xDesc, dyDesc, convDesc, dwDesc, preference,
                  memoryLimitInBytes, algo);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnFilterDescriptor_t, const int, int *,
      cudnnConvolutionBwdFilterAlgoPerf_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionBackwardFilterAlgorithm_v7");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, srcDesc, diffDesc, convDesc, gradDesc,
                  requestedAlgoCount, returnedAlgoCount, perfResults);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnFilterDescriptor_t, cudnnConvolutionBwdFilterAlgo_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionBackwardFilterWorkspaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnConvolutionDescriptor_t, cudnnConvolutionBwdFilterAlgo_t,
      void *, size_t, const void *, const cudnnFilterDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnConvolutionBackwardFilter");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo,
                  workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle, int *count) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnHandle_t, int *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, count);
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnFilterDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnTensorDescriptor_t, const int, int *,
      cudnnConvolutionBwdDataAlgoPerf_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnFindConvolutionBackwardDataAlgorithm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount,
                  returnedAlgoCount, perfResults);
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnFilterDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, void *,
      const int, int *, cudnnConvolutionBwdDataAlgoPerf_t *, void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnFindConvolutionBackwardDataAlgorithmEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx,
                  requestedAlgoCount, returnedAlgoCount, perfResults, workSpace,
                  workSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdDataAlgo_t *algo) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnFilterDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnTensorDescriptor_t, cudnnConvolutionBwdDataPreference_t,
      size_t, cudnnConvolutionBwdDataAlgo_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionBackwardDataAlgorithm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, wDesc, dyDesc, convDesc, dxDesc, preference,
                  memoryLimitInBytes, algo);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnFilterDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnTensorDescriptor_t, const int, int *,
      cudnnConvolutionBwdDataAlgoPerf_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionBackwardDataAlgorithm_v7");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, filterDesc, diffDesc, convDesc, gradDesc,
                  requestedAlgoCount, returnedAlgoCount, perfResults);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnFilterDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnConvolutionDescriptor_t,
      const cudnnTensorDescriptor_t, cudnnConvolutionBwdDataAlgo_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetConvolutionBackwardDataWorkspaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const void *, const cudnnFilterDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnConvolutionDescriptor_t, cudnnConvolutionBwdDataAlgo_t, void *,
      size_t, const void *, const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnConvolutionBackwardData");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo,
                  workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
}

cudnnStatus_t CUDNNWINAPI
cudnnIm2Col(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
            const void *x, const cudnnFilterDescriptor_t wDesc,
            const cudnnConvolutionDescriptor_t convDesc, void *colBuffer) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnHandle_t, const cudnnTensorDescriptor_t,
                                   const void *, const cudnnFilterDescriptor_t,
                                   const cudnnConvolutionDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnIm2Col");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, xDesc, x, wDesc, convDesc, colBuffer);
}

cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSoftmaxForward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSoftmaxBackward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc,
                  dx);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnPoolingDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreatePoolingDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(poolingDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(
    cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth,
    int verticalPadding, int horizontalPadding, int verticalStride,
    int horizontalStride) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, int,
      int, int, int, int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetPooling2dDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(poolingDesc, mode, maxpoolingNanOpt, windowHeight,
                  windowWidth, verticalPadding, horizontalPadding,
                  verticalStride, horizontalStride);
}

cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescriptor(
    const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding, int *horizontalPadding,
    int *verticalStride, int *horizontalStride) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnPoolingDescriptor_t, cudnnPoolingMode_t *,
      cudnnNanPropagation_t *, int *, int *, int *, int *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetPooling2dDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(poolingDesc, mode, maxpoolingNanOpt, windowHeight,
                  windowWidth, verticalPadding, horizontalPadding,
                  verticalStride, horizontalStride);
}

cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(
    cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
    const int windowDimA[], const int paddingA[], const int strideA[]) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnPoolingDescriptor_t, const cudnnPoolingMode_t,
      const cudnnNanPropagation_t, int, const int[], const int[], const int[]);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetPoolingNdDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA,
                  paddingA, strideA);
}

cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor(
    const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
    cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims, int windowDimA[], int paddingA[], int strideA[]) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnPoolingDescriptor_t, int, cudnnPoolingMode_t *,
      cudnnNanPropagation_t *, int *, int[], int[], int[]);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetPoolingNdDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims,
                  windowDimA, paddingA, strideA);
}

cudnnStatus_t CUDNNWINAPI
cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int nbDims, int outputTensorDimA[]) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(const cudnnPoolingDescriptor_t,
                                   const cudnnTensorDescriptor_t, int, int[]);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetPoolingNdForwardOutputDim");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
}

cudnnStatus_t CUDNNWINAPI
cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int *n, int *c, int *h, int *w) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(const cudnnPoolingDescriptor_t,
                                               const cudnnTensorDescriptor_t,
                                               int *, int *, int *, int *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetPooling2dForwardOutputDim");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(poolingDesc, inputTensorDesc, n, c, h, w);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnPoolingDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyPoolingDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(poolingDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnPoolingDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnPoolingForward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnPoolingDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnPoolingBackward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x,
                  beta, dxDesc, dx);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnActivationDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateActivationDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(activationDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(
    cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt, double coef) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnActivationDescriptor_t,
                                               cudnnActivationMode_t,
                                               cudnnNanPropagation_t, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetActivationDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(activationDesc, mode, reluNanOpt, coef);
}

cudnnStatus_t CUDNNWINAPI
cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t *mode,
                             cudnnNanPropagation_t *reluNanOpt, double *coef) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnActivationDescriptor_t, cudnnActivationMode_t *,
      cudnnNanPropagation_t *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetActivationDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(activationDesc, mode, reluNanOpt, coef);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnActivationDescriptor_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnDestroyActivationDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(activationDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnActivationForward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnActivationDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnActivationForward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnActivationBackward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnActivationDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnActivationBackward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x,
                  beta, dxDesc, dx);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnLRNDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateLRNDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(normDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                                unsigned lrnN, double lrnAlpha,
                                                double lrnBeta, double lrnK) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnLRNDescriptor_t, unsigned int, double, double, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetLRNDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
}

cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                                unsigned *lrnN,
                                                double *lrnAlpha,
                                                double *lrnBeta, double *lrnK) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnLRNDescriptor_t, unsigned int *, double *, double *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetLRNDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnLRNDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyLRNDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(lrnDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnLRNCrossChannelForward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnLRNCrossChannelBackward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc,
                  x, beta, dxDesc, dx);
}

cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, /* same desc for means, temp, temp2 */
    const void *x,
    const void *means, /* if NULL, means are assumed to be zero */
    void *temp, void *temp2, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *, void *, void *,
      const void *, const cudnnTensorDescriptor_t, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnDivisiveNormalizationForward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2,
                  beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t
        xDesc, /* same desc for x, means, dy, temp, temp2 */
    const void *x,
    const void *means, /* if NULL, means are assumed to be zero */
    const void *dy, void *temp, void *temp2, const void *beta,
    const cudnnTensorDescriptor_t dXdMeansDesc, /* same desc for dx, dMeans */
    void *dx,                                   /* output x differential */
    void *dMeans) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *, const void *,
      void *, void *, const void *, const cudnnTensorDescriptor_t, void *,
      void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnDivisiveNormalizationBackward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, normDesc, mode, alpha, xDesc, x, means, dy, temp,
                  temp2, beta, dXdMeansDesc, dx, dMeans);
}

cudnnStatus_t CUDNNWINAPI cudnnDeriveBNTensorDescriptor(
    cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnTensorDescriptor_t,
                                               const cudnnTensorDescriptor_t,
                                               cudnnBatchNormMode_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDeriveBNTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(derivedBnDesc, xDesc, mode);
}

cudnnStatus_t CUDNNWINAPI
cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
      const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
      const cudnnActivationDescriptor_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>(
      "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode, bnOps, xDesc, zDesc, yDesc,
                  bnScaleBiasMeanVarDesc, activationDesc, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
      const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
      const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
      const cudnnActivationDescriptor_t, size_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetBatchNormalizationBackwardExWorkspaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc,
                  dBnScaleBiasDesc, activationDesc, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
      const cudnnActivationDescriptor_t, const cudnnTensorDescriptor_t,
      size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>(
      "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode, bnOps, activationDesc, xDesc, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

    const cudnnTensorDescriptor_t xDesc, const void *x, /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc, void *y,       /* NxCxHxW */

    /* Shared desc for the next 6 tensors in the argument list.
       Data type to be set as follows:
       type = (typeOf(x) == double) ? double : float
       Dimensions for this descriptor depend on normalization mode
       - Spatial Normalization : tensors are expected to have dims 1xCx1x1
        (normalization is performed across NxHxW)
       - Per-Activation Normalization : tensors are expected to have dims of
       1xCxHxW (normalization is performed across N) */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,

    /* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation
     */
    const void *bnScale, const void *bnBias,

    /* MUST use factor=1 in the very first call of a complete training cycle.
       Use a factor=1/(1+n) at N-th call to the function to get
       Cumulative Moving Average (CMA) behavior
       CMA[n] = (x[1]+...+x[n])/n
       Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
       ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
       CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
    double exponentialAverageFactor,

    /* Used in Training phase only.
       runningMean = newMean*factor + runningMean*(1-factor) */
    void *resultRunningMean,
    /* Output in training mode, input in inference. Is the moving average
       of  variance[x] (factor is applied in the same way as for runningMean) */
    void *resultRunningVariance,

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and
       backward functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean, void *resultSaveInvVariance) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnBatchNormMode_t, const void *, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t,
      const void *, const void *, double, void *, void *, double, void *,
      void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnBatchNormalizationForwardTraining");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(
      handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc,
      bnScale, bnBias, exponentialAverageFactor, resultRunningMean,
      resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance);
}

cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t zDesc, const void *zData,
    const cudnnTensorDescriptor_t yDesc, void *yData,

    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias,

    double exponentialAverageFactor, void *resultRunningMean,
    void *resultRunningVariance,

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and
       backward functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean, void *resultSaveInvVariance,

    cudnnActivationDescriptor_t activationDesc, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const void *,
      const void *, const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t,
      const void *, const void *, double, void *, void *, double, void *,
      void *, cudnnActivationDescriptor_t, void *, size_t, void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnBatchNormalizationForwardTrainingEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData,
                  yDesc, yData, bnScaleBiasMeanVarDesc, bnScale, bnBias,
                  exponentialAverageFactor, resultRunningMean,
                  resultRunningVariance, epsilon, resultSaveMean,
                  resultSaveInvVariance, activationDesc, workspace,
                  workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */
    const cudnnTensorDescriptor_t xDesc, const void *x, /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc, void *y,       /* NxCxHxW */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnBatchNormMode_t, const void *, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t,
      const void *, const void *, const void *, const void *, double);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnBatchNormalizationForwardInference");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode, alpha, beta, xDesc, x, yDesc, y,
                  bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean,
                  estimatedVariance, epsilon);
}

cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alphaDataDiff,
    const void *betaDataDiff, const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, /* same desc for x, dx, dy */
    const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    /* Shared tensor desc for the 4 tensors below */
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScale, /* bnBias doesn't affect backpropagation */
    /* scale and bias diff are not backpropagated below this layer */
    void *dBnScaleResult, void *dBnBiasResult,
    /* Same epsilon as forward pass */
    double epsilon,

    /* Optionally cached intermediate results from
       forward pass */
    const void *savedMean, const void *savedInvVariance) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnBatchNormMode_t, const void *, const void *,
      const void *, const void *, const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t,
      const void *, void *, void *, double, const void *, const void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnBatchNormalizationBackward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff,
                  betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx,
                  dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult,
                  epsilon, savedMean, savedInvVariance);
}

cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,

    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t yDesc, const void *yData,
    const cudnnTensorDescriptor_t dyDesc, const void *dyData,
    const cudnnTensorDescriptor_t dzDesc, void *dzData,
    const cudnnTensorDescriptor_t dxDesc, void *dxData,

    /* Shared tensor desc for the 4 tensors below */
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScaleData,
    const void *bnBiasData, /* needed if there is activation */
    void *dBnScaleData, void *dBnBiasData,
    double epsilon, /* Same epsilon as forward pass */

    /* Optionally cached intermediate results from
       forward pass */
    const void *savedMean, const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const void *,
      const void *, const void *, const void *, const cudnnTensorDescriptor_t,
      const void *, const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t,
      void *, const cudnnTensorDescriptor_t, const void *, const void *, void *,
      void *, double, const void *, const void *, cudnnActivationDescriptor_t,
      void *, size_t, void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnBatchNormalizationBackwardEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(
      handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff,
      betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData,
      dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData,
      dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc,
      workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateSpatialTransformerDescriptor(
    cudnnSpatialTransformerDescriptor_t *stDesc) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnSpatialTransformerDescriptor_t *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnCreateSpatialTransformerDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(stDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetSpatialTransformerNdDescriptor(
    cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType, const int nbDims, const int dimA[]) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnSpatialTransformerDescriptor_t, cudnnSamplerType_t, cudnnDataType_t,
      const int, const int[]);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnSetSpatialTransformerNdDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(stDesc, samplerType, dataType, nbDims, dimA);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroySpatialTransformerDescriptor(
    cudnnSpatialTransformerDescriptor_t stDesc) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnSpatialTransformerDescriptor_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnDestroySpatialTransformerDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(stDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorForward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta, void *grid) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnSpatialTransformerDescriptor_t, const void *,
      void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnSpatialTfGridGeneratorForward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, stDesc, theta, grid);
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorBackward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid, void *dtheta) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnSpatialTransformerDescriptor_t, const void *,
      void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnSpatialTfGridGeneratorBackward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, stDesc, dgrid, dtheta);
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerForward(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *grid, const void *beta, cudnnTensorDescriptor_t yDesc,
    void *y) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *, const void *,
      cudnnTensorDescriptor_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSpatialTfSamplerForward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerBackward(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx,
    const void *alphaDgrid, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const void *grid, const void *betaDgrid, void *dgrid) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *,
      const cudnnTensorDescriptor_t, void *, const void *,
      const cudnnTensorDescriptor_t, const void *, const void *, const void *,
      void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSpatialTfSamplerBackward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid,
                  dyDesc, dy, grid, betaDgrid, dgrid);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnDropoutDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateDropoutDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dropoutDesc);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnDropoutDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyDropoutDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dropoutDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle,
                                                    size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnHandle_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDropoutGetStatesSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(
    cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnTensorDescriptor_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDropoutGetReserveSpaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(xdesc, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout,
    void *states, size_t stateSizeInBytes, unsigned long long seed) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnDropoutDescriptor_t, cudnnHandle_t,
                                   float, void *, size_t, unsigned long long);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetDropoutDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
}

cudnnStatus_t CUDNNWINAPI cudnnRestoreDropoutDescriptor(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout,
    void *states, size_t stateSizeInBytes, unsigned long long seed) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnDropoutDescriptor_t, cudnnHandle_t,
                                   float, void *, size_t, unsigned long long);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRestoreDropoutDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
}

cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float *dropout,
    void **states, unsigned long long *seed) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnDropoutDescriptor_t, cudnnHandle_t,
                                   float *, void **, unsigned long long *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetDropoutDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(dropoutDesc, handle, dropout, states, seed);
}

cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t xdesc, const void *x,
    const cudnnTensorDescriptor_t ydesc, void *y, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnDropoutDescriptor_t,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, void *, void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDropoutForward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace,
                  reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnDropoutBackward(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t dydesc, const void *dy,
    const cudnnTensorDescriptor_t dxdesc, void *dx, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnDropoutDescriptor_t,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, void *, void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDropoutBackward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace,
                  reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateRNNDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDesc);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyRNNDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize,
    const int numLayers, cudnnDropoutDescriptor_t dropoutDesc,
    cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction,
    cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnRNNDescriptor_t, const int, const int,
      cudnnDropoutDescriptor_t, cudnnRNNInputMode_t, cudnnDirectionMode_t,
      cudnnRNNMode_t, cudnnRNNAlgo_t, cudnnDataType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetRNNDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc,
                  inputMode, direction, mode, algo, mathPrec);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int *hiddenSize,
    int *numLayers, cudnnDropoutDescriptor_t *dropoutDesc,
    cudnnRNNInputMode_t *inputMode, cudnnDirectionMode_t *direction,
    cudnnRNNMode_t *mode, cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnRNNDescriptor_t, int *, int *,
      cudnnDropoutDescriptor_t *, cudnnRNNInputMode_t *, cudnnDirectionMode_t *,
      cudnnRNNMode_t *, cudnnRNNAlgo_t *, cudnnDataType_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetRNNDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc,
                  inputMode, direction, mode, algo, mathPrec);
}

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDescriptor_t, cudnnMathType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetRNNMatrixMathType");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDesc, mType);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNMatrixMathType(
    cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDescriptor_t, cudnnMathType_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetRNNMatrixMathType");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDesc, mType);
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc,
                                              cudnnRNNBiasMode_t biasMode) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDescriptor_t, cudnnRNNBiasMode_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetRNNBiasMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDesc, biasMode);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc,
                                              cudnnRNNBiasMode_t *biasMode) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDescriptor_t, cudnnRNNBiasMode_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetRNNBiasMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDesc, biasMode);
}

cudnnStatus_t CUDNNWINAPI cudnnRNNSetClip(cudnnHandle_t handle,
                                          cudnnRNNDescriptor_t rnnDesc,
                                          cudnnRNNClipMode_t clipMode,
                                          cudnnNanPropagation_t clipNanOpt,
                                          double lclip, double rclip) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNClipMode_t,
      cudnnNanPropagation_t, double, double);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRNNSetClip");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
}

cudnnStatus_t CUDNNWINAPI cudnnRNNGetClip(cudnnHandle_t handle,
                                          cudnnRNNDescriptor_t rnnDesc,
                                          cudnnRNNClipMode_t *clipMode,
                                          cudnnNanPropagation_t *clipNanOpt,
                                          double *lclip, double *rclip) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNClipMode_t *,
      cudnnNanPropagation_t *, double *, double *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRNNGetClip");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
}

cudnnStatus_t CUDNNWINAPI
cudnnSetRNNProjectionLayers(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                            const int recProjSize, const int outProjSize) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnRNNDescriptor_t, const int, const int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetRNNProjectionLayers");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, recProjSize, outProjSize);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNProjectionLayers(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *recProjSize,
    int *outProjSize) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetRNNProjectionLayers");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, recProjSize, outProjSize);
}

cudnnStatus_t CUDNNWINAPI cudnnCreatePersistentRNNPlan(
    cudnnRNNDescriptor_t rnnDesc, const int minibatch,
    const cudnnDataType_t dataType, cudnnPersistentRNNPlan_t *plan) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDescriptor_t, const int,
                                               const cudnnDataType_t,
                                               cudnnPersistentRNNPlan_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreatePersistentRNNPlan");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDesc, minibatch, dataType, plan);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnPersistentRNNPlan_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyPersistentRNNPlan");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan);
}

cudnnStatus_t CUDNNWINAPI cudnnSetPersistentRNNPlan(
    cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDescriptor_t,
                                               cudnnPersistentRNNPlan_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetPersistentRNNPlan");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDesc, plan);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNWorkspaceSize(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetRNNWorkspaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNTrainingReserveSize(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetRNNTrainingReserveSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI
cudnnGetRNNParamsSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                      const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes,
                      cudnnDataType_t dataType) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const cudnnTensorDescriptor_t,
      size_t *, cudnnDataType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetRNNParamsSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, xDesc, sizeInBytes, dataType);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerMatrixParams(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID,
    cudnnFilterDescriptor_t linLayerMatDesc, void **linLayerMat) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t,
      const void *, const int, cudnnFilterDescriptor_t, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetRNNLinLayerMatrixParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID,
                  linLayerMatDesc, linLayerMat);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerBiasParams(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID,
    cudnnFilterDescriptor_t linLayerBiasDesc, void **linLayerBias) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t,
      const void *, const int, cudnnFilterDescriptor_t, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetRNNLinLayerBiasParams");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID,
                  linLayerBiasDesc, linLayerBias);
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInference(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t *, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnTensorDescriptor_t *, void *, const cudnnTensorDescriptor_t,
      void *, const cudnnTensorDescriptor_t, void *, void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRNNForwardInference");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx,
                  wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace,
                  workSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t *, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnTensorDescriptor_t *, void *, const cudnnTensorDescriptor_t,
      void *, const cudnnTensorDescriptor_t, void *, void *, size_t, void *,
      size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRNNForwardTraining");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx,
                  wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace,
                  workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardData(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                     const int seqLength, const cudnnTensorDescriptor_t *yDesc,
                     const void *y, const cudnnTensorDescriptor_t *dyDesc,
                     const void *dy, const cudnnTensorDescriptor_t dhyDesc,
                     const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
                     const void *dcy, const cudnnFilterDescriptor_t wDesc,
                     const void *w, const cudnnTensorDescriptor_t hxDesc,
                     const void *hx, const cudnnTensorDescriptor_t cxDesc,
                     const void *cx, const cudnnTensorDescriptor_t *dxDesc,
                     void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx,
                     const cudnnTensorDescriptor_t dcxDesc, void *dcx,
                     void *workspace, size_t workSpaceSizeInBytes,
                     void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t *, const void *,
      const cudnnTensorDescriptor_t *, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t *, void *, const cudnnTensorDescriptor_t,
      void *, const cudnnTensorDescriptor_t, void *, void *, size_t, void *,
      size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRNNBackwardData");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc,
                  dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc,
                  dx, dhxDesc, dhx, dcxDesc, dcx, workspace,
                  workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t *yDesc, const void *y, const void *workspace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw,
    const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t *, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t *, const void *, const void *, size_t,
      const cudnnFilterDescriptor_t, void *, const void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRNNBackwardWeights");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y,
                  workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace,
                  reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNPaddingMode(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t paddingMode) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDescriptor_t, cudnnRNNPaddingMode_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetRNNPaddingMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDesc, paddingMode);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNPaddingMode(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t *paddingMode) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDescriptor_t,
                                               cudnnRNNPaddingMode_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetRNNPaddingMode");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDesc, paddingMode);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDataDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateRNNDataDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDataDesc);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnRNNDataDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyRNNDataDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDataDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDataDescriptor(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType,
    cudnnRNNDataLayout_t layout, int maxSeqLength, int batchSize,
    int vectorSize,
    const int seqLengthArray[], /* length of each sequence in the batch */
    void *paddingFill) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnRNNDataDescriptor_t, cudnnDataType_t, cudnnRNNDataLayout_t, int, int,
      int, const int[], void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetRNNDataDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDataDesc, dataType, layout, maxSeqLength, batchSize,
                  vectorSize, seqLengthArray, paddingFill);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNDataDescriptor(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t *dataType,
    cudnnRNNDataLayout_t *layout, int *maxSeqLength, int *batchSize,
    int *vectorSize, int arrayLengthRequested, int seqLengthArray[],
    void *paddingFill) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnRNNDataDescriptor_t, cudnnDataType_t *, cudnnRNNDataLayout_t *,
      int *, int *, int *, int, int[], void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetRNNDataDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDataDesc, dataType, layout, maxSeqLength, batchSize,
                  vectorSize, arrayLengthRequested, seqLengthArray,
                  paddingFill);
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTrainingEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
    const void *keys,                     /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
    void *cAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
    void *iAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
    void *queries,                        /* reserved, should pass NULL */
    void *workSpace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const cudnnRNNDataDescriptor_t,
      const void *, const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnRNNDataDescriptor_t, void *, const cudnnTensorDescriptor_t,
      void *, const cudnnTensorDescriptor_t, void *,
      const cudnnRNNDataDescriptor_t, const void *,
      const cudnnRNNDataDescriptor_t, void *, const cudnnRNNDataDescriptor_t,
      void *, const cudnnRNNDataDescriptor_t, void *, void *, size_t, void *,
      size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRNNForwardTrainingEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
                  yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn,
                  iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes,
                  reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInferenceEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
    const void *keys,                     /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
    void *cAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
    void *iAttn,                          /* reserved, should pass NULL */
    const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
    void *queries,                        /* reserved, should pass NULL */
    void *workSpace, size_t workSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const cudnnRNNDataDescriptor_t,
      const void *, const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnRNNDataDescriptor_t, void *, const cudnnTensorDescriptor_t,
      void *, const cudnnTensorDescriptor_t, void *,
      const cudnnRNNDataDescriptor_t, const void *,
      const cudnnRNNDataDescriptor_t, void *, const cudnnRNNDataDescriptor_t,
      void *, const cudnnRNNDataDescriptor_t, void *, void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRNNForwardInferenceEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
                  yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn,
                  iDesc, iAttn, qDesc, queries, workSpace,
                  workSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardDataEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t yDesc, const void *y,
    const cudnnRNNDataDescriptor_t dyDesc, const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc, /* reserved, should pass NULL */
    const void *dcAttn,                    /* reserved, should pass NULL */
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnRNNDataDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc, /* reserved, should pass NULL */
    void *dkeys,                           /* reserved, should pass NULL */
    void *workSpace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const cudnnRNNDataDescriptor_t,
      const void *, const cudnnRNNDataDescriptor_t, const void *,
      const cudnnRNNDataDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnRNNDataDescriptor_t, void *, const cudnnTensorDescriptor_t,
      void *, const cudnnTensorDescriptor_t, void *,
      const cudnnRNNDataDescriptor_t, void *, void *, size_t, void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRNNBackwardDataEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn,
                  dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx,
                  dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, dkDesc, dkeys,
                  workSpace, workSpaceSizeInBytes, reserveSpace,
                  reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeightsEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnRNNDataDescriptor_t yDesc, const void *y, void *workSpace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw,
    void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const cudnnRNNDataDescriptor_t,
      const void *, const cudnnTensorDescriptor_t, const void *,
      const cudnnRNNDataDescriptor_t, const void *, void *, size_t,
      const cudnnFilterDescriptor_t, void *, void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRNNBackwardWeightsEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace,
                  workSpaceSizeInBytes, dwDesc, dw, reserveSpace,
                  reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNAlgorithmDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnAlgorithmDescriptor_t algoDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnRNNDescriptor_t, cudnnAlgorithmDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetRNNAlgorithmDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, algoDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardInferenceAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, int *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetRNNForwardInferenceAlgorithmMaxCount");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, count);
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardInferenceAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t *, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnTensorDescriptor_t *, void *, const cudnnTensorDescriptor_t,
      void *, const cudnnTensorDescriptor_t, void *, const float, const int,
      int *, cudnnAlgorithmPerformance_t *, void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnFindRNNForwardInferenceAlgorithmEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx,
                  wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity,
                  requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
                  workSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardTrainingAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, int *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetRNNForwardTrainingAlgorithmMaxCount");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, count);
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardTrainingAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t *, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnTensorDescriptor_t *, void *, const cudnnTensorDescriptor_t,
      void *, const cudnnTensorDescriptor_t, void *, const float, const int,
      int *, cudnnAlgorithmPerformance_t *, void *, size_t, void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnFindRNNForwardTrainingAlgorithmEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx,
                  wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity,
                  requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
                  workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, int *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetRNNBackwardDataAlgorithmMaxCount");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, count);
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *yDesc, const void *y,
    const cudnnTensorDescriptor_t *dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnTensorDescriptor_t *dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t *, const void *,
      const cudnnTensorDescriptor_t *, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnFilterDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t *, void *, const cudnnTensorDescriptor_t,
      void *, const cudnnTensorDescriptor_t, void *, const float, const int,
      int *, cudnnAlgorithmPerformance_t *, void *, size_t, void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnFindRNNBackwardDataAlgorithmEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc,
                  dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc,
                  dx, dhxDesc, dhx, dcxDesc, dcx, findIntensity,
                  requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
                  workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, int *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, count);
}

cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardWeightsAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t *yDesc, const void *y,
    const float findIntensity, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults,
    const void *workspace, size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc, void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
      const cudnnTensorDescriptor_t *, const void *,
      const cudnnTensorDescriptor_t, const void *,
      const cudnnTensorDescriptor_t *, const void *, const float, const int,
      int *, cudnnAlgorithmPerformance_t *, const void *, size_t,
      const cudnnFilterDescriptor_t, void *, const void *, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnFindRNNBackwardWeightsAlgorithmEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y,
                  findIntensity, requestedAlgoCount, returnedAlgoCount,
                  perfResults, workspace, workSpaceSizeInBytes, dwDesc, dw,
                  reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnSeqDataDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateSeqDataDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(seqDataDesc);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnSeqDataDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroySeqDataDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(seqDataDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetSeqDataDescriptor(
    cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType, int nbDims,
    const int dimA[], const cudnnSeqDataAxis_t axes[],
    size_t seqLengthArraySize, const int seqLengthArray[], void *paddingFill) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnSeqDataDescriptor_t, cudnnDataType_t, int, const int[],
      const cudnnSeqDataAxis_t[], size_t, const int[], void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetSeqDataDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize,
                  seqLengthArray, paddingFill);
}

cudnnStatus_t CUDNNWINAPI cudnnGetSeqDataDescriptor(
    const cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t *dataType,
    int *nbDims, int nbDimsRequested, int dimA[], cudnnSeqDataAxis_t axes[],
    size_t *seqLengthArraySize, size_t seqLengthSizeRequested,
    int seqLengthArray[], void *paddingFill) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnSeqDataDescriptor_t, cudnnDataType_t *, int *, int, int[],
      cudnnSeqDataAxis_t[], size_t *, size_t, int[], void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetSeqDataDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes,
                  seqLengthArraySize, seqLengthSizeRequested, seqLengthArray,
                  paddingFill);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnAttnDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateAttnDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(attnDesc);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnAttnDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyAttnDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(attnDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, cudnnAttnQueryMap_t queryMap, int nHeads,
    double smScaler, cudnnDataType_t dataType, cudnnDataType_t computePrec,
    cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc, int qSize, int kSize, int vSize,
    int qProjSize, int kProjSize, int vProjSize, int oProjSize,
    int qoMaxSeqLength, int kvMaxSeqLength, int maxBatchSize, int maxBeamSize) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnAttnDescriptor_t, cudnnAttnQueryMap_t, int, double, cudnnDataType_t,
      cudnnDataType_t, cudnnMathType_t, cudnnDropoutDescriptor_t,
      cudnnDropoutDescriptor_t, int, int, int, int, int, int, int, int, int,
      int, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetAttnDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(attnDesc, queryMap, nHeads, smScaler, dataType, computePrec,
                  mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize,
                  vSize, qProjSize, kProjSize, vProjSize, oProjSize,
                  qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);
}

cudnnStatus_t CUDNNWINAPI cudnnGetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, cudnnAttnQueryMap_t *queryMap, int *nHeads,
    double *smScaler, cudnnDataType_t *dataType, cudnnDataType_t *computePrec,
    cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize, int *kSize,
    int *vSize, int *qProjSize, int *kProjSize, int *vProjSize, int *oProjSize,
    int *qoMaxSeqLength, int *kvMaxSeqLength, int *maxBatchSize,
    int *maxBeamSize) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnAttnDescriptor_t, cudnnAttnQueryMap_t *, int *, double *,
      cudnnDataType_t *, cudnnDataType_t *, cudnnMathType_t *,
      cudnnDropoutDescriptor_t *, cudnnDropoutDescriptor_t *, int *, int *,
      int *, int *, int *, int *, int *, int *, int *, int *, int *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetAttnDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(attnDesc, queryMap, nHeads, smScaler, dataType, computePrec,
                  mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize,
                  vSize, qProjSize, kProjSize, vProjSize, oProjSize,
                  qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);
}

cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnBuffers(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    size_t *weightSizeInBytes, size_t *workSpaceSizeInBytes,
    size_t *reserveSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnAttnDescriptor_t, size_t *, size_t *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetMultiHeadAttnBuffers");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes,
                  reserveSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes,
    const void *w, cudnnTensorDescriptor_t wDesc, void **wAddr) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnAttnDescriptor_t,
      cudnnMultiHeadAttnWeightKind_t, size_t, const void *,
      cudnnTensorDescriptor_t, void **);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetMultiHeadAttnWeights");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, attnDesc, wKind, weightSizeInBytes, w, wDesc, wAddr);
}

cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnForward(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, int currIdx,
    const int *loWinIdx, const int *hiWinIdx, const int *seqLengthArrayQRO,
    const int *seqLengthArrayKV, const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t oDesc, void *out, size_t weightSizeInBytes,
    const void *w, size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnAttnDescriptor_t, int, const int *, const int *,
      const int *, const int *, const cudnnSeqDataDescriptor_t, const void *,
      const void *, const cudnnSeqDataDescriptor_t, const void *,
      const cudnnSeqDataDescriptor_t, const void *,
      const cudnnSeqDataDescriptor_t, void *, size_t, const void *, size_t,
      void *, size_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnMultiHeadAttnForward");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, attnDesc, currIdx, loWinIdx, hiWinIdx,
                  seqLengthArrayQRO, seqLengthArrayKV, qDesc, queries,
                  residuals, kDesc, keys, vDesc, values, oDesc, out,
                  weightSizeInBytes, w, workSpaceSizeInBytes, workSpace,
                  reserveSpaceSizeInBytes, reserveSpace);
}

cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    const int *loWinIdx, const int *hiWinIdx, const int *seqLengthArrayDQDO,
    const int *seqLengthArrayDKDV, const cudnnSeqDataDescriptor_t doDesc,
    const void *dout, const cudnnSeqDataDescriptor_t dqDesc, void *dqueries,
    const void *queries, const cudnnSeqDataDescriptor_t dkDesc, void *dkeys,
    const void *keys, const cudnnSeqDataDescriptor_t dvDesc, void *dvalues,
    const void *values, size_t weightSizeInBytes, const void *w,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnAttnDescriptor_t, const int *, const int *,
      const int *, const int *, const cudnnSeqDataDescriptor_t, const void *,
      const cudnnSeqDataDescriptor_t, void *, const void *,
      const cudnnSeqDataDescriptor_t, void *, const void *,
      const cudnnSeqDataDescriptor_t, void *, const void *, size_t,
      const void *, size_t, void *, size_t, void *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnMultiHeadAttnBackwardData");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, attnDesc, loWinIdx, hiWinIdx, seqLengthArrayDQDO,
                  seqLengthArrayDKDV, doDesc, dout, dqDesc, dqueries, queries,
                  dkDesc, dkeys, keys, dvDesc, dvalues, values,
                  weightSizeInBytes, w, workSpaceSizeInBytes, workSpace,
                  reserveSpaceSizeInBytes, reserveSpace);
}

cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnWgradMode_t addGrad, const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t doDesc, const void *dout,
    size_t weightSizeInBytes, const void *w, void *dw,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnAttnDescriptor_t, cudnnWgradMode_t,
      const cudnnSeqDataDescriptor_t, const void *,
      const cudnnSeqDataDescriptor_t, const void *,
      const cudnnSeqDataDescriptor_t, const void *,
      const cudnnSeqDataDescriptor_t, const void *, size_t, const void *,
      void *, size_t, void *, size_t, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnMultiHeadAttnBackwardWeights");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc,
                  values, doDesc, dout, weightSizeInBytes, w, dw,
                  workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes,
                  reserveSpace);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnCTCLossDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateCTCLossDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctcLossDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptor(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnCTCLossDescriptor_t, cudnnDataType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetCTCLossDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctcLossDesc, compType);
}

cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptorEx(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnCTCLossDescriptor_t, cudnnDataType_t, cudnnLossNormalizationMode_t,
      cudnnNanPropagation_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetCTCLossDescriptorEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctcLossDesc, compType, normMode, gradMode);
}

cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptor(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnCTCLossDescriptor_t, cudnnDataType_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetCTCLossDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctcLossDesc, compType);
}

cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptorEx(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnCTCLossDescriptor_t, cudnnDataType_t *,
      cudnnLossNormalizationMode_t *, cudnnNanPropagation_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetCTCLossDescriptorEx");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctcLossDesc, compType, normMode, gradMode);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnCTCLossDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyCTCLossDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(ctcLossDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnCTCLoss(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t
        probsDesc, /* Tensor descriptor for probabilities, the dimensions are
                      T,N,A (T is the timing steps, N is the
                      mini batch size, A is the alphabet size)  */
    const void *probs,       /* probabilities after softmax, in GPU memory */
    const int *labels,       /* labels, in CPU memory */
    const int *labelLengths, /* the length of each label, in CPU memory */
    const int *inputLengths, /* the lengths of timing steps in each batch, in
                                CPU memory */
    void *costs,             /* the returned costs of CTC, in GPU memory */
    const cudnnTensorDescriptor_t
        gradientsDesc, /* Tensor descriptor for gradients, the dimensions are
                          T,N,A */
    const void *gradients,   /* the returned CTC gradients, in GPU memory, to
                                compute costs only, set it to NULL */
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    void *workspace, /* pointer to the workspace, in GPU memory */
    size_t workSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t, const void *, const int *,
      const int *, const int *, void *, const cudnnTensorDescriptor_t,
      const void *, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, void *,
      size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCTCLoss");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, probsDesc, probs, labels, labelLengths, inputLengths,
                  costs, gradientsDesc, gradients, algo, ctcLossDesc, workspace,
                  workSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t
        probsDesc, /* Tensor descriptor for probabilities, the dimensions are
                      T,N,A (T is the
                      timing steps, N is the mini batch size, A is the alphabet
                      size) */
    const cudnnTensorDescriptor_t
        gradientsDesc,       /* Tensor descriptor for gradients, the
                                dimensions are T,N,A. To compute costs
                                only, set it to NULL */
    const int *labels,       /* labels, in CPU memory */
    const int *labelLengths, /* the length of each label, in CPU memory */
    const int *inputLengths, /* the lengths of timing steps in each batch, in
                                CPU memory */
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc, size_t *sizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, const cudnnTensorDescriptor_t,
      const cudnnTensorDescriptor_t, const int *, const int *, const int *,
      cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetCTCLossWorkspaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, probsDesc, gradientsDesc, labels, labelLengths,
                  inputLengths, algo, ctcLossDesc, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI
cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnAlgorithmDescriptor_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateAlgorithmDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(algoDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmDescriptor(
    cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnAlgorithmDescriptor_t,
                                               cudnnAlgorithm_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetAlgorithmDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(algoDesc, algorithm);
}

cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmDescriptor(
    const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(const cudnnAlgorithmDescriptor_t,
                                               cudnnAlgorithm_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetAlgorithmDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(algoDesc, algorithm);
}

cudnnStatus_t CUDNNWINAPI cudnnCopyAlgorithmDescriptor(
    const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(const cudnnAlgorithmDescriptor_t,
                                               cudnnAlgorithmDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCopyAlgorithmDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(src, dest);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnAlgorithmDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyAlgorithmDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(algoDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateAlgorithmPerformance(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnAlgorithmPerformance_t *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateAlgorithmPerformance");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(algoPerf, numberToCreate);
}

cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmPerformance(
    cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t algoDesc,
    cudnnStatus_t status, float time, size_t memory) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnAlgorithmPerformance_t,
                                               cudnnAlgorithmDescriptor_t,
                                               cudnnStatus_t, float, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetAlgorithmPerformance");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(algoPerf, algoDesc, status, time, memory);
}

cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmPerformance(
    const cudnnAlgorithmPerformance_t algoPerf,
    cudnnAlgorithmDescriptor_t *algoDesc, cudnnStatus_t *status, float *time,
    size_t *memory) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnAlgorithmPerformance_t, cudnnAlgorithmDescriptor_t *,
      cudnnStatus_t *, float *, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetAlgorithmPerformance");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(algoPerf, algoDesc, status, time, memory);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyAlgorithmPerformance(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnAlgorithmPerformance_t *, int);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnDestroyAlgorithmPerformance");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(algoPerf, numberToDestroy);
}

cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmSpaceSize(
    cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
    size_t *algoSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnAlgorithmDescriptor_t, size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetAlgorithmSpaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algoDesc, algoSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI
cudnnSaveAlgorithm(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
                   void *algoSpace, size_t algoSpaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnAlgorithmDescriptor_t, void *, size_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSaveAlgorithm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algoDesc, algoSpace, algoSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnRestoreAlgorithm(
    cudnnHandle_t handle, void *algoSpace, size_t algoSpaceSizeInBytes,
    cudnnAlgorithmDescriptor_t algoDesc) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnHandle_t, void *, size_t,
                                               cudnnAlgorithmDescriptor_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnRestoreAlgorithm");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, algoSpace, algoSpaceSizeInBytes, algoDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetCallback(unsigned mask, void *udata,
                                           cudnnCallback_t fptr) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(unsigned int, void *, cudnnCallback_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetCallback");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(mask, udata, fptr);
}

cudnnStatus_t CUDNNWINAPI cudnnGetCallback(unsigned *mask, void **udata,
                                           cudnnCallback_t *fptr) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(unsigned int *, void **, cudnnCallback_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnGetCallback");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(mask, udata, fptr);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsConstParamPack(
    cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnFusedOpsConstParamPack_t *,
                                               cudnnFusedOps_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnCreateFusedOpsConstParamPack");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(constPack, ops);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t constPack) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnFusedOpsConstParamPack_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnDestroyFusedOpsConstParamPack");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(constPack);
}

cudnnStatus_t CUDNNWINAPI cudnnSetFusedOpsConstParamPackAttribute(
    cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel, const void *param) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnFusedOpsConstParamPack_t,
                                               cudnnFusedOpsConstParamLabel_t,
                                               const void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnSetFusedOpsConstParamPackAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(constPack, paramLabel, param);
}

cudnnStatus_t CUDNNWINAPI cudnnGetFusedOpsConstParamPackAttribute(
    const cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel, void *param, int *isNULL) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      const cudnnFusedOpsConstParamPack_t, cudnnFusedOpsConstParamLabel_t,
      void *, int *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetFusedOpsConstParamPackAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(constPack, paramLabel, param, isNULL);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsVariantParamPack(
    cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnFusedOpsVariantParamPack_t *, cudnnFusedOps_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnCreateFusedOpsVariantParamPack");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(varPack, ops);
}

cudnnStatus_t CUDNNWINAPI
cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t varPack) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnFusedOpsVariantParamPack_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnDestroyFusedOpsVariantParamPack");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(varPack);
}

cudnnStatus_t CUDNNWINAPI cudnnSetFusedOpsVariantParamPackAttribute(
    cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnFusedOpsVariantParamPack_t,
                                   cudnnFusedOpsVariantParamLabel_t, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnSetFusedOpsVariantParamPackAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(varPack, paramLabel, ptr);
}

cudnnStatus_t CUDNNWINAPI cudnnGetFusedOpsVariantParamPackAttribute(
    const cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(const cudnnFusedOpsVariantParamPack_t,
                                   cudnnFusedOpsVariantParamLabel_t, void *);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cudnnGetFusedOpsVariantParamPackAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(varPack, paramLabel, ptr);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t *plan,
                                                  cudnnFusedOps_t ops) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnFusedOpsPlan_t *, cudnnFusedOps_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnCreateFusedOpsPlan");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan, ops);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(cudnnFusedOpsPlan_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnDestroyFusedOpsPlan");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(plan);
}

cudnnStatus_t CUDNNWINAPI
cudnnMakeFusedOpsPlan(cudnnHandle_t handle, cudnnFusedOpsPlan_t plan,
                      const cudnnFusedOpsConstParamPack_t constPack,
                      size_t *workspaceSizeInBytes) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnFusedOpsPlan_t, const cudnnFusedOpsConstParamPack_t,
      size_t *);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnMakeFusedOpsPlan");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, plan, constPack, workspaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI
cudnnFusedOpsExecute(cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan,
                     cudnnFusedOpsVariantParamPack_t varPack) {
  using FuncPtr =
      cudnnStatus_t(CUDNNWINAPI *)(cudnnHandle_t, const cudnnFusedOpsPlan_t,
                                   cudnnFusedOpsVariantParamPack_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnFusedOpsExecute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, plan, varPack);
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v6(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize,
    const int numLayers, cudnnDropoutDescriptor_t dropoutDesc,
    cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction,
    cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnHandle_t, cudnnRNNDescriptor_t, const int, const int,
      cudnnDropoutDescriptor_t, cudnnRNNInputMode_t, cudnnDirectionMode_t,
      cudnnRNNMode_t, cudnnRNNAlgo_t, cudnnDataType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetRNNDescriptor_v6");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc,
                  inputMode, direction, mode, algo, mathPrec);
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v5(
    cudnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
    cudnnDataType_t mathPrec) {
  using FuncPtr = cudnnStatus_t(CUDNNWINAPI *)(
      cudnnRNNDescriptor_t, int, int, cudnnDropoutDescriptor_t,
      cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t,
      cudnnDataType_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudnnSetRNNDescriptor_v5");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode,
                  direction, mode, mathPrec);
}

}  // extern "C"
