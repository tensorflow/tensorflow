// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_

#include <optional>

#include "third_party/odml/infra/southbound/sb_api.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/dispatch_api.h"

class LiteRtDispatchInvocationContextT {
 public:
  LiteRtDispatchInvocationContextT(ThrInvocationContext* thr_invocation_context,
                                   LiteRtDispatchDeviceContext device_context,
                                   LiteRtDispatchGraph graph)
      : thr_invocation_context_(thr_invocation_context),
        device_context_(device_context),
        graph_(graph) {}

  ~LiteRtDispatchInvocationContextT() {
    if (exec_handle_) {
      litert::google_tensor::UnloadExecutable(device_context_, *exec_handle_);
    }
  }

  litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LiteRtRankedTensorType& tensor_type);
  litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LiteRtRankedTensorType& tensor_type);

  ThrInvocationContext* thr_invocation_context() {
    return thr_invocation_context_;
  }

  LiteRtDispatchDeviceContext device_context() { return device_context_; }

  LiteRtDispatchGraph graph() { return graph_; }

  void AttachExecutable(LiteRtDispatchExecutableHandle exec_handle) {
    exec_handle_ = exec_handle;
  }

 private:
  ThrInvocationContext* thr_invocation_context_;
  LiteRtDispatchDeviceContext device_context_;
  LiteRtDispatchGraph graph_;
  std::optional<LiteRtDispatchExecutableHandle> exec_handle_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
