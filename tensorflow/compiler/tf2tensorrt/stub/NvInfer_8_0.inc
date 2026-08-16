// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// Auto-generated, do not edit.

extern "C" {

void* createInferBuilder_INTERNAL(void* logger, int version) noexcept {
  using FuncPtr = void * (*)(void *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("createInferBuilder_INTERNAL");
  if (!func_ptr) LogFatalSymbolNotFound("createInferBuilder_INTERNAL");
  return func_ptr(logger, version);
}

void* createInferRefitter_INTERNAL(void* engine, void* logger,
                                   int version) noexcept {
  using FuncPtr = void* (*)(void*, void*, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("createInferRefitter_INTERNAL");
  if (!func_ptr) LogFatalSymbolNotFound("createInferRefitter_INTERNAL");
  return func_ptr(engine, logger, version);
}

void* createInferRuntime_INTERNAL(void* logger, int version) noexcept {
  using FuncPtr = void * (*)(void *, int);
  static auto func_ptr = LoadSymbol<FuncPtr>("createInferRuntime_INTERNAL");
  if (!func_ptr) LogFatalSymbolNotFound("createInferRuntime_INTERNAL");
  return func_ptr(logger, version);
}

nvinfer1::ILogger* getLogger() noexcept {
  using FuncPtr = nvinfer1::ILogger * (*)();
  static auto func_ptr = LoadSymbol<FuncPtr>("getLogger");
  if (!func_ptr) LogFatalSymbolNotFound("getLogger");
  return func_ptr();
}

int getInferLibVersion() noexcept {
  using FuncPtr = int (*)();
  static auto func_ptr = LoadSymbol<FuncPtr>("getInferLibVersion");
  if (!func_ptr) LogFatalSymbolNotFound("getInferLibVersion");
  return func_ptr();
}

nvinfer1::IPluginRegistry* getPluginRegistry() noexcept {
  using FuncPtr = nvinfer1::IPluginRegistry * (*)();
  static auto func_ptr = LoadSymbol<FuncPtr>("getPluginRegistry");
  if (!func_ptr) LogFatalSymbolNotFound("getPluginRegistry");
  return func_ptr();
}

}  // extern "C"
