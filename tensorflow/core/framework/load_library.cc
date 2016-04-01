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

// Load a dynamic library.
// On success, returns the handle to library in result, copies the serialized
// OpList of OpDefs registered in the library to *buf and the length to *len,
// and returns OK from the function. Otherwise return nullptr in result
// and an error status from the function, leaving buf and len untouched.
Status LoadLibrary(const char* library_filename, void** result,
                   const void** buf, size_t* len) {
  static mutex mu;
  Env* env = Env::Default();
  void* lib;
  OpList op_list;
  {
    mutex_lock lock(mu);
    TF_RETURN_IF_ERROR(OpRegistry::Global()->SetWatcher(
        [&op_list](const OpDef& opdef) { *op_list.add_op() = opdef; }));
    TF_RETURN_IF_ERROR(env->LoadLibrary(library_filename, &lib));
    TF_RETURN_IF_ERROR(OpRegistry::Global()->SetWatcher(nullptr));
  }
  string str;
  op_list.SerializeToString(&str);
  char* str_buf = reinterpret_cast<char*>(operator new(str.length()));
  memcpy(str_buf, str.data(), str.length());
  *buf = str_buf;
  *len = str.length();

  *result = lib;
  return Status::OK();
}

}  // namespace tensorflow
