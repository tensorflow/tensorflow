/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <dlfcn.h>
#include <memory>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

namespace {

template <typename R, typename... Args>
Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                            R (**symbol)(Args...)) {
  Env* env = Env::Default();
  void* symbol_ptr;
  Status status = env->GetSymbolFromLibrary(handle, symbol_name, &symbol_ptr);
  *symbol = reinterpret_cast<R (*)(Args...)>(symbol_ptr);
  return status;
}

}  // namespace

// Load a dynamic library and register the ops and kernels defined in that file.
// Expects the symbols "RegisterOps", "RegisterKernels", and "GetOpList" to be
// defined in the library.
// On success, returns the handle to library in result, copies the serialized
// OpList of OpDefs registered in the library to *buf and the length to *len,
// and returns OK from the function. Otherwise return nullptr in result
// and an error status from the function, leaving buf and len untouched.
Status LoadLibrary(const char* library_filename, void** result,
                   const void** buf, size_t* len) {
  Env* env = Env::Default();
  void* lib;
  TF_RETURN_IF_ERROR(env->LoadLibrary(library_filename, &lib));

  typedef void (*FuncType)(void*);
  FuncType RegisterOps, RegisterKernels, GetOpList;
  TF_RETURN_IF_ERROR(GetSymbolFromLibrary(lib, "RegisterOps", &RegisterOps));
  TF_RETURN_IF_ERROR(
      GetSymbolFromLibrary(lib, "RegisterKernels", &RegisterKernels));
  TF_RETURN_IF_ERROR(GetSymbolFromLibrary(lib, "GetOpList", &GetOpList));

  *buf = nullptr;
  *len = 0;

  RegisterOps(OpRegistry::Global());
  RegisterKernels(GlobalKernelRegistry());
  string str;
  GetOpList(&str);
  char* str_buf = reinterpret_cast<char*>(operator new(str.length()));
  strncpy(str_buf, str.data(), str.length());
  *buf = str_buf;
  *len = str.length();

  *result = lib;
  return Status::OK();
}

}  // namespace tensorflow
