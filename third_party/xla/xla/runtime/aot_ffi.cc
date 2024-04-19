// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/runtime/aot_ffi.h"

#include <iostream>
#include <string>

#include "xla/runtime/aot_ffi_c_symbols.h"
#include "xla/runtime/ffi/ffi_api.h"

// XLA_FFI_Error is forward-declared by the XLA FFI C API,
// and therefore has to be defined in the global namespace.

struct XLA_FFI_Error {
  XLA_FFI_Error_Code errc;
  std::string error;
};

namespace xla {
namespace runtime {
namespace aot {

template <const int64_t* ptr>
static const void* GetAotType() {
  return static_cast<const void*>(ptr);
}

static XLA_FFI_Error* CreateError(XLA_FFI_Error_Create_Args* args) {
  assert(XLA_FFI_Error_Create_Args_STRUCT_SIZE == args->struct_size);
  return new XLA_FFI_Error{args->errc, std::string(args->message)};
}

XLA_FFI_Api FfiApi() {
  XLA_FFI_Api api = {
      /*struct_size=*/XLA_FFI_Api_STRUCT_SIZE,
      /*priv=*/nullptr,

      // Module Registration APIs.
      nullptr,

      // Execution Context APIs.
      nullptr,  // module state
      nullptr,  // stream

      // Error Reporting APIs.
      CreateError,

      // Type table.
      GetAotType<&__type_id_string>,
      GetAotType<&__type_id_float>,
      GetAotType<&__type_id_double>,
      GetAotType<&__type_id_bool>,
      GetAotType<&__type_id_int32>,
      GetAotType<&__type_id_int64>,
      GetAotType<&__type_id_array_float>,
      GetAotType<&__type_id_array_double>,
      GetAotType<&__type_id_array_int32>,
      GetAotType<&__type_id_array_int64>,
      GetAotType<&__type_id_tensor_float>,
      GetAotType<&__type_id_tensor_double>,
      GetAotType<&__type_id_tensor_int32>,
      GetAotType<&__type_id_tensor_int64>,
      GetAotType<&__type_id_memref_view>,
      GetAotType<&__type_id_strided_memref_view>,
      GetAotType<&__type_id_dictionary>,
  };
  return api;
}

XLA_FFI_Function_Args FfiArgs(XLA_FFI_Api* api, void** args, void** attrs,
                              void** rets) {
  XLA_FFI_Function_Args ffi_args;
  ffi_args.api = api;
  ffi_args.ctx = nullptr;
  ffi_args.priv = nullptr;
  ffi_args.struct_size = XLA_FFI_Function_Args_STRUCT_SIZE;
  ffi_args.args = args;
  ffi_args.attrs = attrs;
  ffi_args.rets = rets;
  return ffi_args;
}

bool ProcessErrorIfAny(XLA_FFI_Error* error) {
  if (error == nullptr) {
    return true;
  }
  // XLA has no way of passing errors; print to stderr.
  std::cerr << "XLA FFI error: " << error->error << ".\n";
  delete error;
  return false;
}

}  // namespace aot
}  // namespace runtime
}  // namespace xla
