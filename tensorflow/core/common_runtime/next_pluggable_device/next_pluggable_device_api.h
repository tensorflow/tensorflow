/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_API_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_API_H_

#include <string>

#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

// Global TFNPD_Api* singleton.
const TFNPD_Api* TfnpdApi();
void SetTfnpdApi(const TFNPD_Api* api);

typedef const TFNPD_Api* (*TFNPDInitPluginFn)(TFNPD_PluginParams*, TF_Status*);
tsl::StatusOr<TFNPD_PluginParams> InitNextPluggableDevicePlugin(
    TFNPDInitPluginFn init_fn);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_API_H_
