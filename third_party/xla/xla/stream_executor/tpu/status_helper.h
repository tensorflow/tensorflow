/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_STATUS_HELPER_H_
#define XLA_STREAM_EXECUTOR_TPU_STATUS_HELPER_H_

#include "absl/status/status.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/tpu_executor_api.h"
#include "xla/stream_executor/tpu/tpu_executor_c_api.h"

class StatusHelper {
 public:
  StatusHelper()
      : c_status(stream_executor::tpu::ExecutorApiFn()->TpuStatus_NewFn()) {}

  ~StatusHelper() {
    stream_executor::tpu::ExecutorApiFn()->TpuStatus_FreeFn(c_status);
  }

  static absl::Status FromC(  // TENSORFLOW_STATUS_OK
      TF_Status* const c_status) {
    if (stream_executor::tpu::ExecutorApiFn()->TpuStatus_OkFn(c_status)) {
      return absl::OkStatus();
    } else {
      return absl::Status(  // TENSORFLOW_STATUS_OK
          absl::StatusCode(
              stream_executor::tpu::ExecutorApiFn()->TpuStatus_CodeFn(
                  c_status)),
          stream_executor::tpu::ExecutorApiFn()->TpuStatus_MessageFn(c_status));
    }
  }

  bool ok() const {
    return stream_executor::tpu::ExecutorApiFn()->TpuStatus_OkFn(c_status);
  }

  absl::Status status() const {  // TENSORFLOW_STATUS_OK
    return FromC(c_status);
  }

  TF_Status* const c_status;  // NOLINT
};

#endif  // XLA_STREAM_EXECUTOR_TPU_STATUS_HELPER_H_
