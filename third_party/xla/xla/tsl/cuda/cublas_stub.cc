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

#include "absl/strings/string_view.h"

#if CUBLAS_VER_MAJOR >= 11
#include "third_party/gpus/cuda/include/cublas_v2.h"
#else
#include "third_party/gpus/cuda/include/cublas.h"
#endif

#include "absl/container/flat_hash_set.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/dso_loader.h"
#include "tsl/platform/load_library.h"

// Implements the cuBLAS API by forwarding to cuBLAS loaded from the DSO.
// Note that it does not implement the v1 interface.

namespace {
// Returns DSO handle or null if loading the DSO fails.
void *GetDsoHandle() {
  static auto handle = []() -> void * {
    auto handle_or = tsl::internal::DsoLoader::GetCublasDsoHandle();
    if (!handle_or.ok()) return nullptr;
    return handle_or.value();
  }();
  return handle;
}

void *LoadSymbol(const char *symbol_name) {
  void *symbol = nullptr;
  if (auto handle = GetDsoHandle()) {
    tsl::internal::GetSymbolFromLibrary(handle, symbol_name, &symbol)
        .IgnoreError();
  }
  return symbol;
}

const char *kSymbols[] = {
#include "xla/tsl/cuda/cublas.inc"
};

constexpr size_t kNumSymbols = sizeof(kSymbols) / sizeof(const char *);

absl::flat_hash_set<absl::string_view> const &FatalErrorSymbols() {
  static auto *syms = new absl::flat_hash_set<absl::string_view>{
      "cublasGetCudartVersion",
      "cublasXerbla",
      "cublasSnrm2",
      "cublasDnrm2",
      "cublasScnrm2",
      "cublasDznrm2",
      "cublasSdot",
      "cublasDdot",
      "cublasCdotu",
      "cublasCdotc",
      "cublasZdotu",
      "cublasZdotc",
      "cublasSscal",
      "cublasDscal",
      "cublasCscal",
      "cublasZscal",
      "cublasCsscal",
      "cublasZdscal",
      "cublasSaxpy",
      "cublasDaxpy",
      "cublasCaxpy",
      "cublasZaxpy",
      "cublasScopy",
      "cublasDcopy",
      "cublasCcopy",
      "cublasZcopy",
      "cublasSswap",
      "cublasDswap",
      "cublasCswap",
      "cublasZswap",
      "cublasIsamax",
      "cublasIdamax",
      "cublasIcamax",
      "cublasIzamax",
      "cublasIsamin",
      "cublasIdamin",
      "cublasIcamin",
      "cublasIzamin",
      "cublasSasum",
      "cublasDasum",
      "cublasScasum",
      "cublasDzasum",
      "cublasSrot",
      "cublasDrot",
      "cublasCrot",
      "cublasZrot",
      "cublasCsrot",
      "cublasZdrot",
      "cublasSrotg",
      "cublasDrotg",
      "cublasCrotg",
      "cublasZrotg",
      "cublasSrotm",
      "cublasDrotm",
      "cublasSrotmg",
      "cublasDrotmg",
      "cublasSgemv",
      "cublasDgemv",
      "cublasCgemv",
      "cublasZgemv",
      "cublasSgbmv",
      "cublasDgbmv",
      "cublasCgbmv",
      "cublasZgbmv",
      "cublasStrmv",
      "cublasDtrmv",
      "cublasCtrmv",
      "cublasZtrmv",
      "cublasStbmv",
      "cublasDtbmv",
      "cublasCtbmv",
      "cublasZtbmv",
      "cublasStpmv",
      "cublasDtpmv",
      "cublasCtpmv",
      "cublasZtpmv",
      "cublasStrsv",
      "cublasDtrsv",
      "cublasCtrsv",
      "cublasZtrsv",
      "cublasStpsv",
      "cublasDtpsv",
      "cublasCtpsv",
      "cublasZtpsv",
      "cublasStbsv",
      "cublasDtbsv",
      "cublasCtbsv",
      "cublasZtbsv",
      "cublasSsymv",
      "cublasDsymv",
      "cublasChemv",
      "cublasZhemv",
      "cublasSsbmv",
      "cublasDsbmv",
      "cublasChbmv",
      "cublasZhbmv",
      "cublasSspmv",
      "cublasDspmv",
      "cublasChpmv",
      "cublasZhpmv",
      "cublasSger",
      "cublasDger",
      "cublasCgeru",
      "cublasCgerc",
      "cublasZgeru",
      "cublasZgerc",
      "cublasSsyr",
      "cublasDsyr",
      "cublasCher",
      "cublasZher",
      "cublasSspr",
      "cublasDspr",
      "cublasChpr",
      "cublasZhpr",
      "cublasSsyr2",
      "cublasDsyr2",
      "cublasCher2",
      "cublasZher2",
      "cublasSspr2",
      "cublasDspr2",
      "cublasChpr2",
      "cublasZhpr2",
      "cublasSgemm",
      "cublasDgemm",
      "cublasCgemm",
      "cublasZgemm",
      "cublasSsyrk",
      "cublasDsyrk",
      "cublasCsyrk",
      "cublasZsyrk",
      "cublasCherk",
      "cublasZherk",
      "cublasSsyr2k",
      "cublasDsyr2k",
      "cublasCsyr2k",
      "cublasZsyr2k",
      "cublasCher2k",
      "cublasZher2k",
      "cublasSsymm",
      "cublasDsymm",
      "cublasCsymm",
      "cublasZsymm",
      "cublasChemm",
      "cublasZhemm",
      "cublasStrsm",
      "cublasDtrsm",
      "cublasCtrsm",
      "cublasZtrsm",
      "cublasStrmm",
      "cublasDtrmm",
      "cublasCtrmm",
      "cublasZtrmm",
  };
  return *syms;
}

}  // namespace

extern "C" {

static void CublasLogFatalSymbolNotFound(const char *symbol_name) {
  LOG(FATAL) << symbol_name << " symbol not found.";
}

static cublasStatus_t CublasGetSymbolNotFoundError() {
  return CUBLAS_STATUS_INTERNAL_ERROR;
}

extern void *_cublas_tramp_table[];

void _cublas_tramp_resolve(int i) {
  CHECK_LE(0, i);
  CHECK_LT(i, kNumSymbols);
  void *p = LoadSymbol(kSymbols[i]);
  if (!p) {
    const auto &fatal_error_symbols = FatalErrorSymbols();
    if (fatal_error_symbols.find(kSymbols[i]) != fatal_error_symbols.end()) {
      p = reinterpret_cast<void *>(&CublasLogFatalSymbolNotFound);
    } else {
      p = reinterpret_cast<void *>(&CublasGetSymbolNotFoundError);
    }
  }
  _cublas_tramp_table[i] = p;
}

}  // extern "C"
