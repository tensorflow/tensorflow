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

#include "tensorflow/core/common_runtime/next_pluggable_device/c_plugin_coordination_service_agent.h"

#include <string>

#include "tensorflow/c/tf_buffer.h"
#include "tensorflow/c/tf_status_helper.h"

namespace tensorflow {

Status CPluginCoordinationServiceAgent::InsertKeyValue(
    const std::string& key, const std::string& value) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_CoordinationServiceInsertKeyValue(key.data(), value.data(), agent_,
                                       status);
  return StatusFromTF_Status(status);
}

StatusOr<std::string> CPluginCoordinationServiceAgent::GetKeyValue(
    const std::string& key) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_Buffer* result_buf =
      TF_CoordinationServiceGetKeyValue(key.data(), agent_, status);

  if (TF_GetCode(status) != TF_OK) {
    return StatusFromTF_Status(status);
  } else {
    std::string result{static_cast<const char*>(result_buf->data),
                       result_buf->length};
    TF_DeleteBuffer(result_buf);
    return result;
  }
}

Status CPluginCoordinationServiceAgent::DeleteKeyValue(const std::string& key) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_CoordinationServiceDeleteKeyValue(key.data(), agent_, status);
  return StatusFromTF_Status(status);
}

}  // namespace tensorflow
