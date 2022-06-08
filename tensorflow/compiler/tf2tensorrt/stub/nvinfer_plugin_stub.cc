/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/platform/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "third_party/tensorrt/NvInferPlugin.h"

// Implements the TensorRT API by forwarding to TensorRT loaded from the DSO.

namespace {
// Returns DSO handle or null if loading the DSO fails.
void* GetDsoHandle() {
#ifdef PLATFORM_GOOGLE
  return nullptr;
#else
  static auto handle = []() -> void* {
    auto handle_or =
        stream_executor::internal::DsoLoader::GetNvInferPluginDsoHandle();
    if (!handle_or.ok()) return nullptr;
    return handle_or.ValueOrDie();
  }();
  return handle;
#endif
}

template <typename T>
T LoadSymbol(const char* symbol_name) {
  void* symbol = nullptr;
  if (auto handle = GetDsoHandle()) {
    tensorflow::Env::Default()
        ->GetSymbolFromLibrary(handle, symbol_name, &symbol)
        .IgnoreError();
  }
  return reinterpret_cast<T>(symbol);
}

void LogFatalSymbolNotFound(const char* symbol_name) {
  LOG(FATAL) << symbol_name << " symbol not found.";
}
}  // namespace

#if NV_TENSORRT_MAJOR < 7
#error TensorRT version earlier than 7 is not supported.
#elif NV_TENSORRT_MAJOR == 7 || NV_TENSORRT_MAJOR == 8
#include "tensorflow/compiler/tf2tensorrt/stub/NvInferPlugin_7_0.inc"
#else
#error This version of TensorRT is not supported.
#endif
