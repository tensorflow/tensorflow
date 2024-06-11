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

#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cusparse.h"
#include "tsl/platform/dso_loader.h"
#include "tsl/platform/load_library.h"
#include "tsl/platform/logging.h"

// Implements the cusparse API by forwarding to cusparse loaded from the DSO.

namespace {
// Returns DSO handle or null if loading the DSO fails.
void* GetDsoHandle() {
#ifdef PLATFORM_GOOGLE
  return nullptr;
#else
  static auto handle = []() -> void* {
    auto handle_or = tsl::internal::DsoLoader::GetCusparseDsoHandle();
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
#include "xla/tsl/cuda/cusparse.inc"
};

constexpr size_t kNumSymbols = sizeof(kSymbols) / sizeof(const char*);

absl::flat_hash_set<std::string_view> const& StringErrorSymbols() {
  static auto* syms = new absl::flat_hash_set<std::string_view>{
      "cusparseGetErrorName",
      "cusparseGetErrorString",
  };
  return *syms;
}

}  // namespace

extern "C" {

static cusparseStatus_t GetSymbolNotFoundError() {
  return CUSPARSE_STATUS_INTERNAL_ERROR;
}

static const char* GetErrorString(cusparseStatus_t status) {
  return "The cuSPARSE library was not found.";
}

extern void* _cusparse_tramp_table[];

void _cusparse_tramp_resolve(int i) {
  CHECK_LE(0, i);
  CHECK_LT(i, kNumSymbols);
  void* p = LoadSymbol(kSymbols[i]);
  if (!p) {
    const auto& string_error_symbols = StringErrorSymbols();
    if (string_error_symbols.find(kSymbols[i]) != string_error_symbols.end()) {
      p = reinterpret_cast<void*>(&GetErrorString);
    } else {
      p = reinterpret_cast<void*>(&GetSymbolNotFoundError);
    }
  }
  _cusparse_tramp_table[i] = p;
}

}  // extern "C"
