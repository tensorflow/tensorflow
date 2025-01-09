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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_

#include <optional>

#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter.h"

class LiteRtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  static litert::Expected<Ptr> Create(
      litert::mediatek::NeuronAdapter& neuron_adapter,
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType exec_type, const void* exec_bytecode_ptr,
      size_t exec_bytecode_size, const char* function_name, int num_inputs,
      int num_outputs);

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

 private:
  class IoRequirementsBuilder {
   public:
    IoRequirementsBuilder(size_t buffer_size,
                          const std::vector<uint32_t>& padded_dimensions);
    litert::Expected<LiteRtTensorBufferRequirements> Create();

   private:
    size_t buffer_size_;
    std::vector<uint32_t> strides_;
  };

  LiteRtDispatchInvocationContextT(
      const litert::mediatek::NeuronAdapter& neuron_adapter,
      LiteRtDispatchDeviceContext device_context,
      litert::mediatek::NeuronModel* model,
      litert::mediatek::NeuronCompilation* compilation,
      litert::mediatek::NeuronExecution* execution, int num_inputs,
      int num_outputs)
      : neuron_adapter_(neuron_adapter),
        device_context_(device_context),
        model_(model),
        compilation_(compilation),
        execution_(execution),
        input_requirements_builders_(num_inputs),
        output_requirements_builders_(num_outputs) {}

  const litert::mediatek::NeuronAdapter& neuron_adapter_;
  LiteRtDispatchDeviceContext device_context_;
  litert::mediatek::NeuronModel* model_;
  litert::mediatek::NeuronCompilation* compilation_;
  litert::mediatek::NeuronExecution* execution_;
  std::vector<std::unique_ptr<IoRequirementsBuilder>>
      input_requirements_builders_;
  std::vector<std::unique_ptr<IoRequirementsBuilder>>
      output_requirements_builders_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
