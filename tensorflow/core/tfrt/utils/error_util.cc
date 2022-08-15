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
#include "tensorflow/core/tfrt/utils/error_util.h"

#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime

namespace tfrt {

tfrt::ErrorCode ConvertTfErrorCodeToTfrtErrorCode(
    const tensorflow::Status& status) {
  auto tf_error_code = status.code();
  switch (tf_error_code) {
    default:
      LOG(INFO) << "Unsupported TensorFlow error code: " << status.ToString();
      return tfrt::ErrorCode::kUnknown;
#define ERROR_TYPE(TFRT_ERROR, TF_ERROR) \
  case tensorflow::error::TF_ERROR:      \
    return tfrt::ErrorCode::TFRT_ERROR;
#include "tensorflow/core/tfrt/utils/error_type.def"  // NOLINT
  }
}

tensorflow::Status CreateTfErrorStatus(const DecodedDiagnostic& error) {
  auto tf_error_code = tensorflow::error::Code::UNKNOWN;
  switch (error.code) {
    default:
      tf_error_code = tensorflow::error::Code::INTERNAL;
      LOG(INFO) << "Unsupported TFRT error code "
                << ErrorName(error.code).str();
      break;
#define ERROR_TYPE(TFRT_ERROR, TF_ERROR)               \
  case tfrt::ErrorCode::TFRT_ERROR:                    \
    tf_error_code = tensorflow::error::Code::TF_ERROR; \
    break;
#include "tensorflow/core/tfrt/utils/error_type.def"  // NOLINT
  }
  return tensorflow::Status(tf_error_code, error.message);
}

tensorflow::Status ToTfStatus(const tfrt::AsyncValue* av) {
  CHECK(av != nullptr && av->IsAvailable())  // Crash OK
      << "Expected a ready async value.";
  if (av->IsError()) {
    return CreateTfErrorStatus(av->GetError());
  }
  return ::tensorflow::OkStatus();
}

tensorflow::Status TfStatusFromAbslStatus(absl::Status status) {
  if (status.ok()) return tensorflow::OkStatus();
  return tensorflow::Status(
      static_cast<tensorflow::error::Code>(status.raw_code()),
      status.message());
}

}  // namespace tfrt
