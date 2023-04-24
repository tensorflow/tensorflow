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

#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"

#include <string>

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"
#include "tensorflow/tsl/platform/errors.h"

namespace tensorflow {

static const TFNPD_Api* tfnpd_api;

const TFNPD_Api* TfnpdApi() { return tfnpd_api; }

void SetTfnpdApi(const TFNPD_Api* api) { tfnpd_api = api; }

tsl::StatusOr<TFNPD_PluginParams> InitNextPluggableDevicePlugin(
    TFNPDInitPluginFn init_fn) {
  TFNPD_PluginParams params{TFNPD_PLUGIN_PARAMS_STRUCT_SIZE};
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  const TFNPD_Api* api = init_fn(&params, c_status_ptr.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status_ptr.get()));

  SetTfnpdApi(api);

  return params;
}

}  // namespace tensorflow
