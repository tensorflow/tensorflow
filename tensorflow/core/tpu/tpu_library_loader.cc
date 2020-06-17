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

#include "tensorflow/core/tpu/tpu_library_loader.h"

#include <dlfcn.h>

#define TFTPU_SET_FN(Struct, FnName) \
  Struct->FnName##Fn =               \
      reinterpret_cast<decltype(FnName)*>(dlsym(library_handle, #FnName));

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

// Reminder: Update tpu_library_loader_windows.cc if you are adding new publicly
// visible methods.

namespace tensorflow {
namespace tpu {

Status SetTpuInitializeStructFns(void* library_handle) {
  auto* base_fn = InitializeApiFn();

  TFTPU_SET_FN(base_fn, TfTpu_Initialize);

  return Status::OK();
}

Status SetTpuConfigStructFns(void* library_handle) {
  auto* config_fn = ConfigApiFn();

  TFTPU_SET_FN(config_fn, ConfigureDistributedTpuOp_DoWork);
  TFTPU_SET_FN(config_fn, WaitForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(config_fn, ShutdownDistributedTpuOp_DoWork);
  TFTPU_SET_FN(config_fn, InitializeHostForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(config_fn, SetGlobalTPUArrayOp_DoWork);
  TFTPU_SET_FN(config_fn, DisconnectDistributedTpuChipsOp_DoWork);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_FreeCharArray);
  TFTPU_SET_FN(config_fn, TpuConfigurationApi_FreeInt32Array);

  return Status::OK();
}

TfTpu_BaseFn* InitializeApiFn() {
  static TfTpu_BaseFn base_fn;
  return &base_fn;
}

TfTpu_ConfigApiFn* ConfigApiFn() {
  static TfTpu_ConfigApiFn config_api_fn;
  return &config_api_fn;
}

Status InitializeTpuLibrary(void* library_handle) {
  bool shared_object_loaded = true;
  if (library_handle == nullptr) {
    library_handle = dlopen(nullptr, RTLD_LAZY);
    shared_object_loaded = false;
  }

  TF_RETURN_IF_ERROR(SetTpuInitializeStructFns(library_handle));
  TF_RETURN_IF_ERROR(SetTpuConfigStructFns(library_handle));

  if (shared_object_loaded) {
    // Initialize TPU platform when the platform code is loaded from a library.
    InitializeApiFn()->TfTpu_InitializeFn();
  }

  return Status::OK();
}

}  // namespace tpu
}  // namespace tensorflow
