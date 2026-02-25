/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_MSAN_EMULATED_TLS_H_
#define XLA_BACKENDS_CPU_CODEGEN_MSAN_EMULATED_TLS_H_

#include <cstdint>

extern "C" {
// Returns the address of the host's MSAN TLS variables.
// See https://github.com/google/sanitizers/wiki/MemorySanitizerJIT.
void* __xla_cpu_emutls_get_address(void* control);
}  // extern "C"

namespace xla::cpu {

// Selectors for __emutls_get_address.
// All of these are needed for msan and msan-track-origins.
enum class MsanTlsSelector : uintptr_t {
  kParamTls = 1,
  kRetvalTls = 2,
  kVaArgTls = 3,
  kVaArgOverflowSizeTls = 4,
  kParamOriginTls = 5,
  kRetvalOriginTls = 6,
  kVaArgOriginTls = 7,
  kOriginTls = 8,
};

inline constexpr char kMsanEmutlsGetAddressBridge[] =
    "__xla_cpu_emutls_get_address";

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_MSAN_EMULATED_TLS_H_
