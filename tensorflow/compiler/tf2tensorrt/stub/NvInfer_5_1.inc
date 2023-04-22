// Auto-generated, do not edit.

extern "C" {

void* createInferBuilder_INTERNAL(void* logger, int version) {
  using FuncPtr = void * (*)(void *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("createInferBuilder_INTERNAL");
  if (!func_ptr) LogFatalSymbolNotFound("createInferBuilder_INTERNAL");
  return func_ptr(logger, version);
}

void* createInferRefitter_INTERNAL(void* engine, void* logger, int version) {
  using FuncPtr = void * (*)(void *, void *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("createInferRefitter_INTERNAL");
  if (!func_ptr) LogFatalSymbolNotFound("createInferRefitter_INTERNAL");
  return func_ptr(engine, logger, version);
}

void* createInferRuntime_INTERNAL(void* logger, int version) {
  using FuncPtr = void * (*)(void *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("createInferRuntime_INTERNAL");
  if (!func_ptr) LogFatalSymbolNotFound("createInferRuntime_INTERNAL");
  return func_ptr(logger, version);
}

nvinfer1::ILogger* getLogger() {
  using FuncPtr = nvinfer1::ILogger * (*)();
  static auto func_ptr = LoadSymbol<FuncPtr>("getLogger");
  if (!func_ptr) LogFatalSymbolNotFound("getLogger");
  return func_ptr();
}

int getInferLibVersion() {
  using FuncPtr = int (*)();
  static auto func_ptr = LoadSymbol<FuncPtr>("getInferLibVersion");
  if (!func_ptr) LogFatalSymbolNotFound("getInferLibVersion");
  return func_ptr();
}

nvinfer1::IPluginRegistry* getPluginRegistry() {
  using FuncPtr = nvinfer1::IPluginRegistry * (*)();
  static auto func_ptr = LoadSymbol<FuncPtr>("getPluginRegistry");
  if (!func_ptr) LogFatalSymbolNotFound("getPluginRegistry");
  return func_ptr();
}

}  // extern "C"
