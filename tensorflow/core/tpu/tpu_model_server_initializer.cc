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

#include "tensorflow/core/tpu/tpu_model_server_initializer.h"

#include <dlfcn.h>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/tpu_api_dlsym_set_fn.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_initializer_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"

namespace tensorflow {
namespace tpu {
namespace {
#if !defined(PLATFORM_GOOGLE)
bool FindAndLoadTpuModelServer() {
  void* library = dlopen("libtpu.so", RTLD_NOW);
  if (library) {
    if (TryAcquireTpuLock()) {
      InitializeTpuLibrary(library);
    }
  }
  OpsApiFn()->TfTpu_InitializeTpuModelServerFn();
  return true;
}

static bool tpu_library_finder = FindAndLoadTpuModelServer();
#endif  // PLATFORM_GOOGLE
}  // namespace
}  // namespace tpu
}  // namespace tensorflow
