/* Copyright 2026 The TensorFlow Metal Plugin Authors. All Rights Reserved.

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

// The out-of-tree entry points.
//
// TensorFlow dlopens this library and looks up two symbols by name. Everything
// behind them is the same code an in-tree build registers through
// PluggableDeviceInit_Api, so this file is the whole difference between the
// two forms: there, TensorFlow holds the function pointers already and calls
// them directly; here, it resolves them out of a shared object.

#include <cstdlib>
#include <cstring>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernels.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_profiler.h"

namespace {

// Set TF_DISABLE_METAL=1 to keep the backend out of the process without
// uninstalling it. Worth having for a device backend: it lets a bug report
// separate "Metal is involved" from "Metal is the cause" in one run.
bool MetalDisabledByEnvironment() {
  const char* value = std::getenv("TF_DISABLE_METAL");
  if (value == nullptr) return false;
  return std::strcmp(value, "0") != 0 && value[0] != '\0';
}

}  // namespace

extern "C" {

void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
  if (MetalDisabledByEnvironment()) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Metal: backend disabled by TF_DISABLE_METAL.");
    return;
  }
  tensorflow::metal::MetalInitPlugin(params, status);
}

void TF_InitKernel() { tensorflow::metal::RegisterAllMetalKernels(); }

// Optional: TensorFlow looks this up and carries on without it if absent. It
// is what puts a Metal row in the trace viewer next to the host's.
void TF_InitProfiler(TF_ProfilerRegistrationParams* params,
                     TF_Status* status) {
  if (MetalDisabledByEnvironment()) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Metal: backend disabled by TF_DISABLE_METAL.");
    return;
  }
  tensorflow::metal::MetalInitProfiler(params, status);
}

}  // extern "C"
