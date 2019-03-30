/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "cuda/include/cudnn.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"

// Implements the cuDNN API by forwarding to cuDNN loaded from the DSO.

namespace {
// Returns DSO handle or null if loading the DSO fails.
void* GetDsoHandle() {
#ifdef PLATFORM_GOOGLE
  return nullptr;
#else
  static auto handle = []() -> void* {
    auto handle_or = stream_executor::internal::DsoLoader::GetCudnnDsoHandle();
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
    stream_executor::port::Env::Default()
        ->GetSymbolFromLibrary(handle, symbol_name, &symbol)
        .IgnoreError();
  }
  return reinterpret_cast<T>(symbol);
}

cudnnStatus_t GetSymbolNotFoundError() { return CUDNN_STATUS_INTERNAL_ERROR; }
}  // namespace

#if CUDNN_MAJOR < 6
#error cuDNN version earlier than 6 is not supported.
#elif CUDNN_MAJOR < 7
#include "tensorflow/stream_executor/cuda/cudnn_6_0.inc"
#elif CUDNN_MINOR < 1
#include "tensorflow/stream_executor/cuda/cudnn_7_0.inc"
#elif CUDNN_MINOR < 3
#include "tensorflow/stream_executor/cuda/cudnn_7_1.inc"
#elif CUDNN_MINOR < 4
#include "tensorflow/stream_executor/cuda/cudnn_7_3.inc"
#else
#include "tensorflow/stream_executor/cuda/cudnn_7_4.inc"
#endif
