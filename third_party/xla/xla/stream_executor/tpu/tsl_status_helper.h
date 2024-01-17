/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_TSL_STATUS_HELPER_H_
#define XLA_STREAM_EXECUTOR_TPU_TSL_STATUS_HELPER_H_

#include "absl/status/status.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "tsl/c/tsl_status.h"
#include "tsl/c/tsl_status_helper.h"
#include "tsl/platform/status.h"

class TslStatusHelper {
 public:
  TslStatusHelper() : c_status(TSL_NewStatus()) {}

  ~TslStatusHelper() { TSL_DeleteStatus(c_status); }

  static tsl::Status FromC(TF_Status* const c_status) {  // TENSORFLOW_STATUS_OK
    absl::StatusCode code = tsl::StatusCodeFromTSLCode(TSL_GetCode(c_status));
    if (code == absl::StatusCode::kOk) {
      return tsl::OkStatus();
    }
    return tsl::Status(code, TSL_Message(c_status));  // TENSORFLOW_STATUS_OK
  }

  bool ok() const {
    return tsl::StatusCodeFromTSLCode(TSL_GetCode(c_status)) ==
           absl::StatusCode::kOk;
  }

  tsl::Status status() const {  // TENSORFLOW_STATUS_OK
    return FromC(c_status);
  }

  TF_Status* const c_status;  // NOLINT
};

#endif  // XLA_STREAM_EXECUTOR_TPU_TSL_STATUS_HELPER_H_
