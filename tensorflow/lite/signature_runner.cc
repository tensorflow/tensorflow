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

#include "tensorflow/lite/signature_runner.h"

#include "tensorflow/lite/c/c_api_types.h"

namespace tflite {

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

TfLiteStatus SignatureRunner::ResizeInputTensor(
    const char* input_name, const std::vector<int>& new_size) {
  const auto& it = signature_def_->inputs.find(input_name);
  if (it == signature_def_->inputs.end()) {
    subgraph_->ReportError("Input name %s was not found", input_name);
    return kTfLiteError;
  }

  return subgraph_->ResizeInputTensor(it->second, new_size);
}

TfLiteStatus SignatureRunner::Invoke() {
  TF_LITE_ENSURE_STATUS(subgraph_->Invoke());

  // Makes sure output tensors are readable.
  for (int tensor_index : subgraph_->outputs()) {
    TF_LITE_ENSURE_STATUS(subgraph_->EnsureTensorDataIsReadable(tensor_index));
  }
  return kTfLiteOk;
}

}  // namespace tflite
