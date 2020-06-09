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

#include <memory>
#include <unordered_set>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/tpu/tpu_library_loader.h"
#endif  // IS_MOBILE_PLATFORM

namespace tensorflow {

namespace {

struct Library {
  void* handle = nullptr;
  OpList op_list;
};

}  // namespace

// Load a dynamic library.
// On success, returns the handle to library in result, copies the serialized
// OpList of OpDefs registered in the library to *buf and the length to *len,
// and returns OK from the function. Otherwise return nullptr in result
// and an error status from the function, leaving buf and len untouched.
//
// If `library_filename` has already been loaded, we return a cached handle
// and OpList. Ops and kernels are registered as globals when a library is
// loaded for the first time. Without caching, every subsequent load would not
// perform initialization again, so the OpList would be empty.
Status LoadLibrary(const char* library_filename, void** result,
                   const void** buf, size_t* len) {
  static mutex mu(LINKER_INITIALIZED);
  static std::unordered_map<string, Library> loaded_libs;
  Env* env = Env::Default();
  Library library;
  std::unordered_set<string> seen_op_names;
  {
    mutex_lock lock(mu);
    if (loaded_libs.find(library_filename) != loaded_libs.end()) {
      library = loaded_libs[library_filename];
    } else {
      Status s = OpRegistry::Global()->ProcessRegistrations();
      if (!s.ok()) {
        return s;
      }
      TF_RETURN_IF_ERROR(OpRegistry::Global()->SetWatcher(
          [&library, &seen_op_names](const Status& s,
                                     const OpDef& opdef) -> Status {
            if (errors::IsAlreadyExists(s)) {
              if (seen_op_names.find(opdef.name()) == seen_op_names.end()) {
                // Over writing a registration of an op not in this custom op
                // library. Treat this as not an error.
                return Status::OK();
              }
            }
            if (s.ok()) {
              *library.op_list.add_op() = opdef;
              seen_op_names.insert(opdef.name());
            }
            return s;
          }));
      OpRegistry::Global()->DeferRegistrations();
      s = env->LoadLibrary(library_filename, &library.handle);
      if (s.ok()) {
        s = OpRegistry::Global()->ProcessRegistrations();
      }
      if (!s.ok()) {
        OpRegistry::Global()->ClearDeferredRegistrations();
        TF_RETURN_IF_ERROR(OpRegistry::Global()->SetWatcher(nullptr));
        return s;
      }
      TF_RETURN_IF_ERROR(OpRegistry::Global()->SetWatcher(nullptr));

      loaded_libs[library_filename] = library;
    }
  }
  string str;
  library.op_list.SerializeToString(&str);
  char* str_buf = reinterpret_cast<char*>(port::Malloc(str.length()));
  memcpy(str_buf, str.data(), str.length());
  *buf = str_buf;
  *len = str.length();

#if !defined(IS_MOBILE_PLATFORM)
  // Determine if this library is a TPU library, and if so, calls the TPU
  // initialization functions to populate function tables, etc...
  void* unused_symbol;
  if (env->GetSymbolFromLibrary(library.handle, "TfTpu_Initialize",
                                &unused_symbol)
          .ok()) {
    TF_RETURN_IF_ERROR(tensorflow::tpu::InitializeTpuLibrary(library.handle));
  }
#endif  // IS_MOBILE_PLATFORM

  *result = library.handle;
  return Status::OK();
}

}  // namespace tensorflow
