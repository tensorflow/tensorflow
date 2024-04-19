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

#include "xla/runtime/aot_ffi_c_symbols.h"

#include "xla/runtime/aot_ffi_execution_context.h"

void* GetResultStorage(void* execution_context, int64_t index) {
  auto* ctx =
      static_cast<xla::runtime::aot::ExecutionContext*>(execution_context);
  ctx->error = "AOT uses no result storage";
  return nullptr;
}

void runtimeSetError(void* execution_context, const char* error) {
  auto* ctx =
      static_cast<xla::runtime::aot::ExecutionContext*>(execution_context);
  ctx->error = error;
}

bool CustomCall(void* execution_context, const char* target, void** args,
                void** attrs, void** rets) {
  auto* ctx =
      static_cast<xla::runtime::aot::ExecutionContext*>(execution_context);
  ctx->error = "AOT has no custom call registry";
  return false;
}

int64_t __type_id_opaque;
int64_t __type_id_nullopt;
int64_t __type_id_string;
int64_t __type_id_function_ordinal;

int64_t __type_id_bool;
int64_t __type_id_int8;
int64_t __type_id_int16;
int64_t __type_id_int32;
int64_t __type_id_int64;
int64_t __type_id_uint8;
int64_t __type_id_uint16;
int64_t __type_id_uint32;
int64_t __type_id_uint64;
int64_t __type_id_bfloat16;
int64_t __type_id_f16;
int64_t __type_id_float;
int64_t __type_id_double;

int64_t __type_id_memref_view;
int64_t __type_id_strided_memref_view;
int64_t __type_id_empty_array;
int64_t __type_id_dictionary;

int64_t __type_id_array_int8;
int64_t __type_id_array_int16;
int64_t __type_id_array_int32;
int64_t __type_id_array_int64;
int64_t __type_id_array_float;
int64_t __type_id_array_double;

int64_t __type_id_tensor_int32;
int64_t __type_id_tensor_int64;
int64_t __type_id_tensor_float;
int64_t __type_id_tensor_double;

int64_t __type_id_async_bool;
int64_t __type_id_async_int8;
int64_t __type_id_async_int16;
int64_t __type_id_async_int32;
int64_t __type_id_async_int64;
int64_t __type_id_async_float;
int64_t __type_id_async_double;
int64_t __type_id_async_memref;
int64_t __type_id_async_chain;
