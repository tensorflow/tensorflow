/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
// NOLINTBEGIN(whitespace/line_length)
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include
/// "third_party/tensorflow/lite/experimental/acceleration/configuration/c/gpu_plugin.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
// NOLINTEND(whitespace/line_length)
#ifndef TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_GPU_PLUGIN_H_
#define TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_GPU_PLUGIN_H_

// This header file is for the delegate plugin for GPU.
//
// For the C++ delegate plugin interface, the GPU delegate plugin is added to
// the DelegatePluginRegistry by the side effect of a constructor for a static
// object, so there's no public API needed for this plugin, other than the API
// of tflite::delegates::DelegatePluginRegistry, which is declared in
// delegate_registry.h.
//
// But to provide a C API to access the GPU delegate plugin, we do expose
// some functions, which are declared below.

#include "tensorflow/lite/core/experimental/acceleration/configuration/c/delegate_plugin.h"

#ifdef __cplusplus
extern "C" {
#endif

// C API for the GPU delegate plugin.
// Returns a pointer to a statically allocated table of function pointers.
const TfLiteDelegatePlugin* TfLiteGpuDelegatePluginCApi();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_GPU_PLUGIN_H_
