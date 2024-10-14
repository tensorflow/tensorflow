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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_PIXEL_DISPATCH_LRT_DISPATCH_INVOCATION_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_PIXEL_DISPATCH_LRT_DISPATCH_INVOCATION_CONTEXT_H_

#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/odml/infra/southbound/sb_api.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"
#include "tensorflow/lite/experimental/lrt/vendors/pixel/dispatch/dispatch_api.h"

class LrtDispatchInvocationContextT {
 public:
  LrtDispatchInvocationContextT(ThrInvocationContext* thr_invocation_context,
                                LrtDispatchDeviceContext device_context,
                                LrtDispatchGraph graph)
      : thr_invocation_context_(thr_invocation_context),
        device_context_(device_context),
        graph_(graph) {}

  ~LrtDispatchInvocationContextT() {
    if (exec_handle_) {
      lrt::pixel::UnloadExecutable(device_context_, *exec_handle_);
    }
  }

  absl::StatusOr<LrtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LrtRankedTensorType& tensor_type);
  absl::StatusOr<LrtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LrtRankedTensorType& tensor_type);

  ThrInvocationContext* thr_invocation_context() {
    return thr_invocation_context_;
  }

  LrtDispatchDeviceContext device_context() { return device_context_; }

  LrtDispatchGraph graph() { return graph_; }

  void AttachExecutable(LrtDispatchExecutableHandle exec_handle) {
    exec_handle_ = exec_handle;
  }

 private:
  ThrInvocationContext* thr_invocation_context_;
  LrtDispatchDeviceContext device_context_;
  LrtDispatchGraph graph_;
  std::optional<LrtDispatchExecutableHandle> exec_handle_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_PIXEL_DISPATCH_LRT_DISPATCH_INVOCATION_CONTEXT_H_
