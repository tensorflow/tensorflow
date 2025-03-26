/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_RUNTIME_HANDLE_FFI_CALL_H_
#define XLA_SERVICE_CPU_RUNTIME_HANDLE_FFI_CALL_H_

#include <cstdint>

extern "C" {

extern void __xla_cpu_runtime_HandleFfiCall(
    const void* run_options_ptr, const char* target_name_ptr,
    int64_t target_name_len, void** outputs, void** inputs,
    const char* opaque_str_ptr, int64_t opaque_str_len, void* status_opaque,
    int32_t* operand_types, int64_t operand_count, int64_t* operand_dims,
    int32_t* result_types, int64_t result_count, int64_t* result_dims);

}  // extern "C"

#endif  // XLA_SERVICE_CPU_RUNTIME_HANDLE_FFI_CALL_H_
