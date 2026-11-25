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

#include "xla/backends/cpu/runtime/msan_emulated_tls.h"

#include <cstdint>

#include "absl/base/attributes.h"
#include "absl/base/config.h"  // IWYU pragma: keep
#include "absl/log/log.h"

#ifdef ABSL_HAVE_MEMORY_SANITIZER
extern "C" {
// Mark these initial-exec as compiler-rt does.
ABSL_ATTRIBUTE_INITIAL_EXEC extern __thread uint64_t __msan_param_tls[];
ABSL_ATTRIBUTE_INITIAL_EXEC extern __thread uint32_t __msan_param_origin_tls[];
ABSL_ATTRIBUTE_INITIAL_EXEC extern __thread uint64_t __msan_retval_tls[];
ABSL_ATTRIBUTE_INITIAL_EXEC extern __thread uint32_t __msan_retval_origin_tls;
ABSL_ATTRIBUTE_INITIAL_EXEC extern __thread uint64_t __msan_va_arg_tls[];
ABSL_ATTRIBUTE_INITIAL_EXEC extern __thread uint32_t __msan_va_arg_origin_tls[];
ABSL_ATTRIBUTE_INITIAL_EXEC extern __thread uintptr_t
    __msan_va_arg_overflow_size_tls;
ABSL_ATTRIBUTE_INITIAL_EXEC extern __thread uint32_t __msan_origin_tls;
}
#endif  // ABSL_HAVE_MEMORY_SANITIZER

extern "C" {

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void* __xla_cpu_runtime_emutls_get_address(
    void* control) {
  // The control argument is already the selector value (not a pointer to it)
  // because the internal __emutls_get_address wrapper in the LLVM module
  // dereferences the selector before calling this bridge.
  uintptr_t selector = reinterpret_cast<uintptr_t>(control);
#ifdef ABSL_HAVE_MEMORY_SANITIZER
  using xla::cpu::MsanTlsSelector;
  // No default case, so that -Wswitch flags any unhandled selector.
  switch (static_cast<MsanTlsSelector>(selector)) {
    case MsanTlsSelector::kParamTls:
      return __msan_param_tls;
    case MsanTlsSelector::kRetvalTls:
      return __msan_retval_tls;
    case MsanTlsSelector::kVaArgTls:
      return __msan_va_arg_tls;
    case MsanTlsSelector::kVaArgOverflowSizeTls:
      return &__msan_va_arg_overflow_size_tls;
    case MsanTlsSelector::kParamOriginTls:
      return __msan_param_origin_tls;
    case MsanTlsSelector::kRetvalOriginTls:
      return &__msan_retval_origin_tls;
    case MsanTlsSelector::kVaArgOriginTls:
      return __msan_va_arg_origin_tls;
    case MsanTlsSelector::kOriginTls:
      return &__msan_origin_tls;
  }
#endif  // ABSL_HAVE_MEMORY_SANITIZER
  // Reached with an invalid selector, or in a non-msan build, where no
  // msan-instrumented code can run.
  LOG(FATAL) << "Invalid msan TLS selector: " << selector;
}

}  // extern "C"
