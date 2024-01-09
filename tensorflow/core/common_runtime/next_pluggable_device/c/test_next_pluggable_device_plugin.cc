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

#include "xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"

namespace {
PJRT_Error* PJRT_Plugin_Initialize_NoOp(PJRT_Plugin_Initialize_Args* args) {
  return nullptr;
}
}  // namespace
const TFNPD_Api example_plugin_api = {
    /*struct_size=*/TFNPD_Api_STRUCT_SIZE,
    /*priv=*/nullptr,
};

const TFNPD_Api* GetExamplePluginApi() { return &example_plugin_api; }

const PJRT_Api example_pjrt_api = {
    /*struct_size=*/PJRT_Api_STRUCT_SIZE,
    /*priv=*/nullptr,
    /*pjrt_api_version=*/
    PJRT_Api_Version{/*struct_size=*/PJRT_Api_Version_STRUCT_SIZE,
                     /*priv=*/nullptr,
                     /*major_version=*/PJRT_API_MAJOR,
                     /*minor_version=*/PJRT_API_MINOR},
    /*PJRT_Error_Destroy=*/nullptr,
    /*PJRT_Error_Message=*/nullptr,
    /*PJRT_Error_GetCode=*/nullptr,

    /*PJRT_Plugin_Initialize=*/PJRT_Plugin_Initialize_NoOp,
};

extern "C" {
const TFNPD_Api* TFNPD_InitPlugin(TFNPD_PluginParams* params,
                                  TF_Status* tf_status) {
  params->device_type = "GPU";
  params->compilation_device_name = "GPU";
  return GetExamplePluginApi();
}

const PJRT_Api* GetPjrtApi() { return &example_pjrt_api; }
}
