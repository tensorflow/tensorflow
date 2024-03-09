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
#include <string_view>

#include "absl/time/time.h"
#include "tensorflow/c/experimental/next_pluggable_device/c_api.h"
#include "tensorflow/c/tf_buffer.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {

namespace {
absl::StatusOr<std::string> ProcessGetKeyValueResult(TF_Buffer* result_buf,
                                                     TF_Status* status) {
  if (TF_GetCode(status) != TF_OK) {
    return StatusFromTF_Status(status);
  } else {
    std::string result{static_cast<const char*>(result_buf->data),
                       result_buf->length};
    TF_DeleteBuffer(result_buf);
    return result;
  }
}
}  // namespace

Status CPluginCoordinationServiceAgent::InsertKeyValue(std::string_view key,
                                                       std::string_view value) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_CoordinationServiceInsertKeyValue(key.data(), key.size(), value.data(),
                                       value.size(), agent_, status);
  return StatusFromTF_Status(status);
}

absl::StatusOr<std::string> CPluginCoordinationServiceAgent::GetKeyValue(
    std::string_view key) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_Buffer* result_buf =
      TF_CoordinationServiceGetKeyValue(key.data(), key.size(), agent_, status);
  return ProcessGetKeyValueResult(result_buf, status);
}

absl::StatusOr<std::string> CPluginCoordinationServiceAgent::GetKeyValue(
    std::string_view key, absl::Duration timeout) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_Buffer* result_buf = TF_CoordinationServiceGetKeyValueWithTimeout(
      key.data(), key.size(), absl::ToInt64Seconds(timeout), agent_, status);
  return ProcessGetKeyValueResult(result_buf, status);
}

absl::StatusOr<std::string> CPluginCoordinationServiceAgent::TryGetKeyValue(
    std::string_view key) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_Buffer* result_buf = TF_CoordinationServiceTryGetKeyValue(
      key.data(), key.size(), agent_, status);
  return ProcessGetKeyValueResult(result_buf, status);
}

Status CPluginCoordinationServiceAgent::DeleteKeyValue(std::string_view key) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_CoordinationServiceDeleteKeyValue(key.data(), key.size(), agent_, status);
  return StatusFromTF_Status(status);
}

}  // namespace tensorflow
