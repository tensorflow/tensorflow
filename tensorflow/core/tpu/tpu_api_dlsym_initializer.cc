/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/tpu_api_dlsym_initializer.h"

#include <dlfcn.h>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#if !defined(PLATFORM_GOOGLE)
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#endif

#define TFTPU_SET_FN(Struct, FnName)                                         \
  Struct->FnName##Fn =                                                       \
      reinterpret_cast<decltype(FnName)*>(dlsym(library_handle, #FnName));   \
  if (!(Struct->FnName##Fn)) {                                               \
    LOG(ERROR) << #FnName " not available in this library.";                 \
    return errors::Unimplemented(#FnName " not available in this library."); \
  }

// Reminder: Update tpu_library_loader_windows.cc if you are adding new publicly
// visible methods.

namespace tensorflow {
namespace tpu {

#if defined(PLATFORM_GOOGLE)
Status InitializeTpuLibrary(void* library_handle) {
  return errors::Unimplemented("You must statically link in a TPU library.");
}
#else
#include "tensorflow/core/tpu/tpu_library_init_fns.inc"

Status InitializeTpuLibrary(void* library_handle) {
  bool shared_object_loaded = true;
  if (library_handle == nullptr) {
    library_handle = dlopen(nullptr, RTLD_NOW);
    shared_object_loaded = false;
  }

  Status s = InitializeTpuStructFns(library_handle);

  // TPU platform registration must only be performed after the library is
  // loaded. We do not want to register a TPU platform in XLA without the
  // supporting library providing the necessary APIs.
  if (shared_object_loaded && s.ok()) {
    // TODO(frankchn): Make initialization actually work
    // Initialize TPU platform when the platform code is loaded from a library.
    // InitializeApiFn()->TfTpu_InitializeFn();

    RegisterTpuPlatform();
  }

  return s;
}
#endif

}  // namespace tpu
}  // namespace tensorflow
