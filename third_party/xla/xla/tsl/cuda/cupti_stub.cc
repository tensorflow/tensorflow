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
#include "tsl/platform/dso_loader.h"
#include "tsl/platform/load_library.h"
#include "tsl/platform/logging.h"

// Implements the CUPTI API by forwarding to CUPTI loaded from the DSO.

namespace {
// Returns DSO handle or null if loading the DSO fails.
void* GetDsoHandle() {
#if defined(PLATFORM_GOOGLE) && (CUDA_VERSION > 10000)
  return nullptr;
#else
  static auto handle = []() -> void* {
    auto handle_or = tsl::internal::DsoLoader::GetCuptiDsoHandle();
    if (!handle_or.ok()) return nullptr;
    return handle_or.value();
  }();
  return handle;
#endif
}

void* LoadSymbol(const char* symbol_name) {
  void* symbol = nullptr;
  if (auto handle = GetDsoHandle()) {
    tsl::internal::GetSymbolFromLibrary(handle, symbol_name, &symbol)
        .IgnoreError();
  }
  return symbol;
}

const char* kSymbols[] = {
#include "xla/tsl/cuda/cupti.inc"
};

constexpr size_t kNumSymbols = sizeof(kSymbols) / sizeof(const char*);

}  // namespace

extern "C" {

static CUptiResult GetSymbolNotFoundError() { return CUPTI_ERROR_UNKNOWN; }

extern void* _cupti_tramp_table[];

void _cupti_tramp_resolve(int i) {
  CHECK_LE(0, i);
  CHECK_LT(i, kNumSymbols);
  void* p = LoadSymbol(kSymbols[i]);
  if (!p) {
    p = reinterpret_cast<void*>(&GetSymbolNotFoundError);
  }
  _cupti_tramp_table[i] = p;
}

}  // extern "C"
