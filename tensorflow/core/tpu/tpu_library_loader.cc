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

// TODO(frankchn): Rename to `tpu_api_dlsym_initializer` or similar.

#include "tensorflow/core/tpu/tpu_library_loader.h"

#include <dlfcn.h>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/stream_executor/tpu/tpu_node_context_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"

#define TFTPU_SET_FN(Struct, FnName)                                       \
  Struct->FnName##Fn =                                                     \
      reinterpret_cast<decltype(FnName)*>(dlsym(library_handle, #FnName)); \
  if (!(Struct->FnName##Fn)) {                                             \
    LOG(ERROR) << #FnName " not available in this library.";               \
  }

// Reminder: Update tpu_library_loader_windows.cc if you are adding new publicly
// visible methods.

namespace tensorflow {
namespace tpu {

#include "tensorflow/core/tpu/tpu_library_init_fns.inc"

TfTpu_BaseFn* InitializeApiFn() {
  static TfTpu_BaseFn base_fn;
  return &base_fn;
}

TfTpu_ConfigApiFn* ConfigApiFn() {
  static TfTpu_ConfigApiFn config_api_fn;
  return &config_api_fn;
}

TfTpu_MeshStateApiFn* MeshStateApiFn() {
  static TfTpu_MeshStateApiFn mesh_state_api_fn;
  return &mesh_state_api_fn;
}

TfTpu_CompileApiFn* CompileApiFn() {
  static TfTpu_CompileApiFn compile_api_fn;
  return &compile_api_fn;
}

TfTpu_ExecutorApiFn* ExecutorApiFn() {
  static TfTpu_ExecutorApiFn executor_api_fn;
  return &executor_api_fn;
}

TfTpu_NodeContextApiFn* NodeContextApiFn() {
  static TfTpu_NodeContextApiFn node_context_api_fn;
  return &node_context_api_fn;
}

TfTpu_UtilApiFn* UtilApiFn() {
  static TfTpu_UtilApiFn util_api_fn;
  return &util_api_fn;
}

Status InitializeTpuLibrary(void* library_handle) {
  bool shared_object_loaded = true;
  if (library_handle == nullptr) {
    library_handle = dlopen(nullptr, RTLD_NOW);
    shared_object_loaded = false;
  }

  TF_RETURN_IF_ERROR(InitializeTpuStructFns(library_handle));

  if (shared_object_loaded) {
    // TODO(frankchn): Make initialization actually work
    // Initialize TPU platform when the platform code is loaded from a library.
    // InitializeApiFn()->TfTpu_InitializeFn();

    // We should only register the TPU platform when the library is loaded.
    // TODO(frankchn): Resolve the circular dependency and register the platform
    // RegisterTpuPlatform();
  }

  return Status::OK();
}

}  // namespace tpu
}  // namespace tensorflow
