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

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include "third_party/odml/infra/southbound/sb_api.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/dispatch_api.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/southbound.h"

class LiteRtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  static litert::Expected<Ptr> CreateFromBytecode(
      const litert::google_tensor::Southbound& southbound,
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
      int num_inputs, int num_outputs);

  static litert::Expected<Ptr> CreateFromGraph(
      const litert::google_tensor::Southbound& southbound,
      LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph);

  ~LiteRtDispatchInvocationContextT();

  litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LiteRtRankedTensorType& tensor_type);
  litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LiteRtRankedTensorType& tensor_type);

  litert::Expected<void> AttachInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);
  litert::Expected<void> AttachOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> DetachInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);
  litert::Expected<void> DetachOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> Invoke();
  litert::Expected<void> InvokeAsync(int num_output_events,
                                     LiteRtEvent* output_events);
  litert::Expected<void> StartMetricsCollection(int detail_level);
  litert::Expected<void> StopMetricsCollection(LiteRtDispatchMetrics* metrics);

  litert::Expected<void> AttachInputEvent(int graph_input_index,
                                          LiteRtEvent input_event);

  ThrInvocationContext* thr_invocation_context() {
    return thr_invocation_context_;
  }

  LiteRtDispatchDeviceContext device_context() { return device_context_; }

  LiteRtDispatchGraph graph() { return graph_; }

 private:
  LiteRtDispatchInvocationContextT(
      const litert::google_tensor::Southbound& southbound,
      ThrInvocationContext* thr_invocation_context,
      LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph)
      : southbound_(southbound),
        thr_invocation_context_(thr_invocation_context),
        device_context_(device_context),
        graph_(graph) {}

  void AttachExecutable(LiteRtDispatchExecutableHandle exec_handle) {
    exec_handle_ = exec_handle;
  }

  const litert::google_tensor::Southbound& southbound_;
  ThrInvocationContext* thr_invocation_context_;
  LiteRtDispatchDeviceContext device_context_;
  LiteRtDispatchGraph graph_;
  std::optional<LiteRtDispatchExecutableHandle> exec_handle_;
  std::map<std::string, int> input_sync_fences_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
