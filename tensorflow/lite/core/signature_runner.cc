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

#include "tensorflow/lite/core/signature_runner.h"

#include <cstdint>
#include <map>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/internal/signature_def.h"

namespace tflite {
namespace impl {

SignatureRunner::SignatureRunner(const internal::SignatureDef* signature_def,
                                 Subgraph* subgraph)
    : signature_def_(signature_def), subgraph_(subgraph) {
  // Collects the list of input and output tensor names.
  for (const auto& it : signature_def_->inputs) {
    input_names_.push_back(it.first.c_str());
  }
  for (const auto& it : signature_def_->outputs) {
    output_names_.push_back(it.first.c_str());
  }

  // Create the list of input and output tensor names in the order of subgraph
  // inputs and outputs.
  std::map<uint32_t, const char*> inputs_reverse_map;
  for (const auto& it : signature_def_->inputs) {
    inputs_reverse_map[it.second] = it.first.c_str();
  }
  subgraph_input_names_.reserve(subgraph_->inputs().size());
  for (int i = 0; i < subgraph_->inputs().size(); ++i) {
    subgraph_input_names_.push_back(inputs_reverse_map[subgraph_->inputs()[i]]);
  }
  std::map<uint32_t, const char*> outputs_reverse_map;
  for (const auto& it : signature_def_->outputs) {
    outputs_reverse_map[it.second] = it.first.c_str();
  }
  subgraph_output_names_.reserve(subgraph_->outputs().size());
  for (int i = 0; i < subgraph_->outputs().size(); ++i) {
    subgraph_output_names_.push_back(
        outputs_reverse_map[subgraph_->outputs()[i]]);
  }
}

TfLiteTensor* SignatureRunner::input_tensor(const char* input_name) {
  const auto& it = signature_def_->inputs.find(input_name);
  if (it == signature_def_->inputs.end()) {
    subgraph_->ReportError("Input name %s was not found", input_name);
    return nullptr;
  }
  return subgraph_->tensor(it->second);
}

const TfLiteTensor* SignatureRunner::output_tensor(
    const char* output_name) const {
  const auto& it = signature_def_->outputs.find(output_name);
  if (it == signature_def_->outputs.end()) {
    subgraph_->ReportError("Output name %s was not found", output_name);
    return nullptr;
  }
  return subgraph_->tensor(it->second);
}

TfLiteStatus SignatureRunner::SetInputBufferHandle(
    const char* input_name, TfLiteBufferHandle buffer_handle,
    TfLiteDelegate* delegate, bool release_existing_buffer_handle) {
  const auto& it = signature_def_->inputs.find(input_name);
  if (it == signature_def_->inputs.end()) {
    subgraph_->ReportError("Input name %s was not found", input_name);
    return kTfLiteError;
  }
  return subgraph_->SetBufferHandle(it->second, buffer_handle, delegate,
                                    release_existing_buffer_handle);
}

TfLiteStatus SignatureRunner::SetOutputBufferHandle(
    const char* output_name, TfLiteBufferHandle buffer_handle,
    TfLiteDelegate* delegate, bool release_existing_buffer_handle) {
  const auto& it = signature_def_->outputs.find(output_name);
  if (it == signature_def_->outputs.end()) {
    subgraph_->ReportError("Output name %s was not found", output_name);
    return kTfLiteError;
  }
  return subgraph_->SetBufferHandle(it->second, buffer_handle, delegate,
                                    release_existing_buffer_handle);
}

TfLiteStatus SignatureRunner::ResizeInputTensor(
    const char* input_name, const std::vector<int>& new_size) {
  const auto& it = signature_def_->inputs.find(input_name);
  if (it == signature_def_->inputs.end()) {
    subgraph_->ReportError("Input name %s was not found", input_name);
    return kTfLiteError;
  }
  return subgraph_->ResizeInputTensor(it->second, new_size);
}

TfLiteStatus SignatureRunner::ResizeInputTensorStrict(
    const char* input_name, const std::vector<int>& new_size) {
  const auto& it = signature_def_->inputs.find(input_name);
  if (it == signature_def_->inputs.end()) {
    subgraph_->ReportError("Input name %s was not found", input_name);
    return kTfLiteError;
  }
  return subgraph_->ResizeInputTensorStrict(it->second, new_size);
}

TfLiteStatus SignatureRunner::Invoke() {
  // "Resets" cancellation flag so cancellation happens before this invoke will
  // not take effect.
  if (subgraph_->continue_invocation_)
    (void)subgraph_->continue_invocation_->test_and_set();

  TF_LITE_ENSURE_STATUS(subgraph_->Invoke());

  // Makes sure output tensors are readable.
  if (!allow_buffer_handle_output_) {
    for (int tensor_index : subgraph_->outputs()) {
      TF_LITE_ENSURE_STATUS(
          subgraph_->EnsureTensorDataIsReadable(tensor_index));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus SignatureRunner::SetCustomAllocationForInputTensor(
    const char* input_name, const TfLiteCustomAllocation& allocation,
    int64_t flags) {
  const auto& it = signature_def_->inputs.find(input_name);
  if (it == signature_def_->inputs.end()) {
    subgraph_->ReportError("Input name %s was not found", input_name);
    return kTfLiteError;
  }
  return subgraph_->SetCustomAllocationForTensor(it->second, allocation, flags);
}

TfLiteStatus SignatureRunner::SetCustomAllocationForOutputTensor(
    const char* output_name, const TfLiteCustomAllocation& allocation,
    int64_t flags) {
  const auto& it = signature_def_->outputs.find(output_name);
  if (it == signature_def_->outputs.end()) {
    subgraph_->ReportError("Output name %s was not found", output_name);
    return kTfLiteError;
  }
  return subgraph_->SetCustomAllocationForTensor(it->second, allocation, flags);
}

}  // namespace impl
}  // namespace tflite
