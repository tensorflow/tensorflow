// Auto-generated, do not edit.

extern "C" {

nvinfer1::IPluginV2* createRPNROIPlugin(int featureStride, int preNmsTop,
                                        int nmsMaxOut, float iouThreshold,
                                        float minBoxSize, float spatialScale,
                                        nvinfer1::DimsHW pooling,
                                        nvinfer1::Weights anchorRatios,
                                        nvinfer1::Weights anchorScales) {
  using FuncPtr = nvinfer1::IPluginV2 * ( *)(int, int, int, float, float, float, nvinfer1::DimsHW, nvinfer1::Weights, nvinfer1::Weights);
  static auto func_ptr = LoadSymbol<FuncPtr>("createRPNROIPlugin");
  if (!func_ptr) LogFatalSymbolNotFound("createRPNROIPlugin");
  return func_ptr(featureStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale, pooling, anchorRatios, anchorScales);
}

nvinfer1::IPluginV2* createNormalizePlugin(const nvinfer1::Weights* scales,
                                           bool acrossSpatial,
                                           bool channelShared, float eps) {
  using FuncPtr = nvinfer1::IPluginV2 * ( *)(const nvinfer1::Weights *, bool, bool, float);
  static auto func_ptr = LoadSymbol<FuncPtr>("createNormalizePlugin");
  if (!func_ptr) LogFatalSymbolNotFound("createNormalizePlugin");
  return func_ptr(scales, acrossSpatial, channelShared, eps);
}

nvinfer1::IPluginV2* createPriorBoxPlugin(
    nvinfer1::plugin::PriorBoxParameters param) {
  using FuncPtr = nvinfer1::IPluginV2 * ( *)(nvinfer1::plugin::PriorBoxParameters);
  static auto func_ptr = LoadSymbol<FuncPtr>("createPriorBoxPlugin");
  if (!func_ptr) LogFatalSymbolNotFound("createPriorBoxPlugin");
  return func_ptr(param);
}

nvinfer1::IPluginV2* createAnchorGeneratorPlugin(
    nvinfer1::plugin::GridAnchorParameters* param, int numLayers) {
  using FuncPtr = nvinfer1::IPluginV2 * ( *)(nvinfer1::plugin::GridAnchorParameters *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("createAnchorGeneratorPlugin");
  if (!func_ptr) LogFatalSymbolNotFound("createAnchorGeneratorPlugin");
  return func_ptr(param, numLayers);
}

nvinfer1::IPluginV2* createNMSPlugin(
    nvinfer1::plugin::DetectionOutputParameters param) {
  using FuncPtr = nvinfer1::IPluginV2 * ( *)(nvinfer1::plugin::DetectionOutputParameters);
  static auto func_ptr = LoadSymbol<FuncPtr>("createNMSPlugin");
  if (!func_ptr) LogFatalSymbolNotFound("createNMSPlugin");
  return func_ptr(param);
}

nvinfer1::IPluginV2* createLReLUPlugin(float negSlope) {
  using FuncPtr = nvinfer1::IPluginV2 * ( *)(float);
  static auto func_ptr = LoadSymbol<FuncPtr>("createLReLUPlugin");
  if (!func_ptr) LogFatalSymbolNotFound("createLReLUPlugin");
  return func_ptr(negSlope);
}

nvinfer1::IPluginV2* createReorgPlugin(int stride) {
  using FuncPtr = nvinfer1::IPluginV2 * ( *)(int);
  static auto func_ptr = LoadSymbol<FuncPtr>("createReorgPlugin");
  if (!func_ptr) LogFatalSymbolNotFound("createReorgPlugin");
  return func_ptr(stride);
}

nvinfer1::IPluginV2* createRegionPlugin(
    nvinfer1::plugin::RegionParameters params) {
  using FuncPtr = nvinfer1::IPluginV2 * ( *)(nvinfer1::plugin::RegionParameters);
  static auto func_ptr = LoadSymbol<FuncPtr>("createRegionPlugin");
  if (!func_ptr) LogFatalSymbolNotFound("createRegionPlugin");
  return func_ptr(params);
}

nvinfer1::IPluginV2* createClipPlugin(const char* layerName, float clipMin,
                                      float clipMax) {
  using FuncPtr = nvinfer1::IPluginV2 * ( *)(const char *, float, float);
  static auto func_ptr = LoadSymbol<FuncPtr>("createClipPlugin");
  if (!func_ptr) LogFatalSymbolNotFound("createClipPlugin");
  return func_ptr(layerName, clipMin, clipMax);
}

nvinfer1::IPluginV2* createBatchedNMSPlugin(
    nvinfer1::plugin::NMSParameters param) {
  using FuncPtr = nvinfer1::IPluginV2 * ( *)(nvinfer1::plugin::NMSParameters);
  static auto func_ptr = LoadSymbol<FuncPtr>("createBatchedNMSPlugin");
  if (!func_ptr) LogFatalSymbolNotFound("createBatchedNMSPlugin");
  return func_ptr(param);
}

bool initLibNvInferPlugins(void* logger, const char* libNamespace) {
  using FuncPtr = bool ( *)(void *, const char *);
  static auto func_ptr = LoadSymbol<FuncPtr>("initLibNvInferPlugins");
  if (!func_ptr) LogFatalSymbolNotFound("initLibNvInferPlugins");
  return func_ptr(logger, libNamespace);
}

}  // extern "C"
