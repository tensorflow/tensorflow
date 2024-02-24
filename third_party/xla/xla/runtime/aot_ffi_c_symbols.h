/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_RUNTIME_AOT_FFI_C_SYMBOLS_H_
#define XLA_RUNTIME_AOT_FFI_C_SYMBOLS_H_

#include <stdint.h>

extern "C" {

void* GetResultStorage(void* execution_context, int64_t index);
void runtimeSetError(void* execution_context, const char* error);
bool CustomCall(void* execution_context, const char* target, void** args,
                void** attrs, void** rets);

// We use int64_t just to make sure these symbols have reasonable alignment.
// But really all we need is to have these symbols defined; we don't do anything
// with them other than taking their address.
extern int64_t __type_id_opaque;
extern int64_t __type_id_nullopt;
extern int64_t __type_id_string;
extern int64_t __type_id_function_ordinal;

extern int64_t __type_id_bool;
extern int64_t __type_id_int8;
extern int64_t __type_id_int16;
extern int64_t __type_id_int32;
extern int64_t __type_id_int64;
extern int64_t __type_id_uint8;
extern int64_t __type_id_uint16;
extern int64_t __type_id_uint32;
extern int64_t __type_id_uint64;
extern int64_t __type_id_bfloat16;
extern int64_t __type_id_f16;
extern int64_t __type_id_float;
extern int64_t __type_id_double;

extern int64_t __type_id_memref_view;
extern int64_t __type_id_strided_memref_view;
extern int64_t __type_id_empty_array;
extern int64_t __type_id_dictionary;

extern int64_t __type_id_array_int8;
extern int64_t __type_id_array_int16;
extern int64_t __type_id_array_int32;
extern int64_t __type_id_array_int64;
extern int64_t __type_id_array_float;
extern int64_t __type_id_array_double;

extern int64_t __type_id_tensor_int32;
extern int64_t __type_id_tensor_int64;
extern int64_t __type_id_tensor_float;
extern int64_t __type_id_tensor_double;

extern int64_t __type_id_async_bool;
extern int64_t __type_id_async_int8;
extern int64_t __type_id_async_int16;
extern int64_t __type_id_async_int32;
extern int64_t __type_id_async_int64;
extern int64_t __type_id_async_float;
extern int64_t __type_id_async_double;
extern int64_t __type_id_async_memref;
extern int64_t __type_id_async_chain;

}  // extern "C"

#endif  // XLA_RUNTIME_AOT_FFI_C_SYMBOLS_H_
