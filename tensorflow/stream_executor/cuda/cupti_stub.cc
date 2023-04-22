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

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/cuda.h"
// IWYU pragma: no_include "perftools/gputools/executor/stream_executor.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"

// Implements the CUPTI API by forwarding to CUPTI loaded from the DSO.

namespace {
// Returns DSO handle or null if loading the DSO fails.
void* GetDsoHandle() {
#if defined(PLATFORM_GOOGLE) && (CUDA_VERSION > 10000)
  return nullptr;
#else
  static auto handle = []() -> void* {
    auto handle_or = stream_executor::internal::DsoLoader::GetCuptiDsoHandle();
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

CUptiResult GetSymbolNotFoundError() { return CUPTI_ERROR_UNKNOWN; }
}  // namespace

// For now we only need one stub implementation. We will need to generate
// a new file when CUPTI breaks backwards compatibility (has not been the case
// for quite a while) or if we want to use functionality introduced in a new
// version.
//
// Calling a function that is not yet available in the loaded CUPTI version will
// return CUPTI_ERROR_UNKNOWN.
#include "tensorflow/stream_executor/cuda/cupti_10_0.inc"
