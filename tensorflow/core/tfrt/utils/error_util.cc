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

#include "tensorflow/core/platform/status.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime

namespace tfrt {

tfrt::ErrorCode ConvertTfErrorCodeToTfrtErrorCode(
    const tensorflow::Status& status) {
  auto tf_error_code = status.code();
  switch (tf_error_code) {
    default:
      LOG(INFO) << "Unsupported TensorFlow error code: " << status;
      return tfrt::ErrorCode::kUnknown;
#define ERROR_TYPE(TFRT_ERROR, TF_ERROR) \
  case absl::StatusCode::TF_ERROR:       \
    return tfrt::ErrorCode::TFRT_ERROR;
#include "tensorflow/core/tfrt/utils/error_type.def"  // NOLINT
  }
}

tensorflow::Status CreateTfErrorStatus(const DecodedDiagnostic& error) {
  return error.status;
}

tensorflow::Status ToTfStatus(const tfrt::AsyncValue* av) {
  CHECK(av != nullptr && av->IsAvailable())  // Crash OK
      << "Expected a ready async value.";
  if (av->IsError()) {
    return av->GetError();
  }
  return absl::OkStatus();
}

}  // namespace tfrt
