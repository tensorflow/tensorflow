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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_PLUGIN_INIT_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_PLUGIN_INIT_H_

#include "tensorflow/core/platform/status.h"

namespace tensorflow {

struct PluggableDeviceInit_Api {
  void* init_plugin_fn = nullptr;
  void* init_np_plugin_fn = nullptr;
  void* get_pjrt_api_fn = nullptr;
  void* init_kernel_fn = nullptr;
  void* init_graph_fn = nullptr;
  void* init_profiler_fn = nullptr;
};

absl::Status RegisterPluggableDevicePlugin(void* dso_handle);
absl::Status RegisterPluggableDevicePlugin(const PluggableDeviceInit_Api* api);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_PLUGIN_INIT_H_
