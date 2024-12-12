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

#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"

namespace litert {

Expected<std::vector<TensorBuffer>> CompiledModel::CreateInputBuffers(
    size_t signature_index) {
  auto signature = model_->GetSignature(signature_index);
  if (!signature) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find signature");
  }
  auto subgraph = model_->Subgraph(signature->Key());
  if (!subgraph) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to get subgraph");
  }
  std::vector<TensorBuffer> input_buffers;
  auto input_tensors = subgraph->Inputs();
  input_buffers.reserve(input_tensors.size());

  for (int i = 0; i < input_tensors.size(); ++i) {
    auto input_buffer_requirements =
        GetInputBufferRequirements(signature_index, i);
    if (!input_buffer_requirements) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        input_buffer_requirements.Error().Message());
    }
    auto tensor_type = input_tensors[i].RankedTensorType();
    LiteRtTensorBufferType tensor_buffer_type =
        (*(*input_buffer_requirements).SupportedTypes())[0];
    auto input_buffer = TensorBuffer::CreateManaged(
        tensor_buffer_type, tensor_type,
        (*input_buffer_requirements).BufferSize().Value());
    if (!input_buffer) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        input_buffer.Error().Message());
    }
    input_buffers.push_back(std::move(*input_buffer));
  }
  return std::move(input_buffers);
}

Expected<std::vector<TensorBuffer>> CompiledModel::CreateOutputBuffers(
    size_t signature_index) {
  auto signature = model_->GetSignature(signature_index);
  if (!signature) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find signature");
  }
  auto subgraph = model_->Subgraph(signature->Key());
  if (!subgraph) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to get subgraph");
  }
  std::vector<TensorBuffer> output_buffers;
  auto output_tensors = subgraph->Outputs();
  output_buffers.reserve(output_tensors.size());
  for (int i = 0; i < output_tensors.size(); ++i) {
    auto output_buffer_requirements =
        GetOutputBufferRequirements(signature_index, i);
    if (!output_buffer_requirements.HasValue()) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        output_buffer_requirements.Error().Message());
    }
    auto tensor_type = output_tensors[i].RankedTensorType();
    LiteRtTensorBufferType tensor_buffer_type =
        (*(*output_buffer_requirements).SupportedTypes())[0];
    auto output_buffer = TensorBuffer::CreateManaged(
        tensor_buffer_type, tensor_type,
        (*output_buffer_requirements).BufferSize().Value());
    if (!output_buffer.HasValue()) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        output_buffer.Error().Message());
    }
    output_buffers.push_back(std::move(*output_buffer));
  }
  return std::move(output_buffers);
}

Expected<void> CompiledModel::Run(
    size_t signature_index, const std::vector<TensorBuffer>& input_buffers,
    const std::vector<TensorBuffer>& output_buffers) {
  auto input_buffers_ptr =
      std::make_unique<LiteRtTensorBuffer[]>(input_buffers.size());
  for (int i = 0; i < input_buffers.size(); ++i) {
    input_buffers_ptr[i] = input_buffers[i].Get();
  }
  auto output_buffers_ptr =
      std::make_unique<LiteRtTensorBuffer[]>(output_buffers.size());
  for (int i = 0; i < output_buffers.size(); ++i) {
    output_buffers_ptr[i] = output_buffers[i].Get();
  }
  if (auto status = LiteRtRunCompiledModel(
          Get(), signature_index, input_buffers.size(), input_buffers_ptr.get(),
          output_buffers.size(), output_buffers_ptr.get());
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to invoke the compiled model");
  }
  return {};
}

}  // namespace litert
