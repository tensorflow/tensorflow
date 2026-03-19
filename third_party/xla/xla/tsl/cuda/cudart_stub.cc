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

// This file wraps cuda runtime calls with dso loader so that we don't need to
// have explicit linking to libcuda.

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/dso_loader.h"
#include "tsl/platform/load_library.h"

namespace {
void *GetDsoHandle() {
  static auto handle = []() -> void * {
    auto handle_or = tsl::internal::DsoLoader::GetCudaRuntimeDsoHandle();
    if (!handle_or.ok()) {
      LOG(INFO) << "Could not find cuda drivers on your machine, "
                   "GPU will not be used.";
      return nullptr;
    }
    return handle_or.value();
  }();
  return handle;
}

void *LoadSymbol(const char *symbol_name) {
  void *symbol = nullptr;
  tsl::internal::GetSymbolFromLibrary(GetDsoHandle(), symbol_name, &symbol)
      .IgnoreError();
  return symbol;
}

const char *kSymbols[] = {
#include "xla/tsl/cuda/cudart.inc"
};

constexpr size_t kNumSymbols = sizeof(kSymbols) / sizeof(const char *);

absl::flat_hash_set<absl::string_view> const &ErrorStringSymbols() {
  static auto *syms = new absl::flat_hash_set<absl::string_view>{
      "cudaGetErrorName",
      "cudaGetErrorString",
  };
  return *syms;
}

}  // namespace

extern "C" {

static const char *ReturnStringError() {
  return "Error loading CUDA libraries. GPU will not be used.";
}

static cudaError_t GetSymbolNotFoundError() {
  return cudaErrorSharedObjectSymbolNotFound;
}

extern void *_cudart_tramp_table[];

void _cudart_tramp_resolve(int i) {
  CHECK_LE(0, i);
  CHECK_LT(i, kNumSymbols);
  void *p = LoadSymbol(kSymbols[i]);
  if (!p) {
    const auto &error_string_symbols = ErrorStringSymbols();
    if (error_string_symbols.find(kSymbols[i]) != error_string_symbols.end()) {
      p = reinterpret_cast<void *>(&ReturnStringError);
    } else {
      p = reinterpret_cast<void *>(&GetSymbolNotFoundError);
    }
  }
  _cudart_tramp_table[i] = p;
}

}  // extern "C"
