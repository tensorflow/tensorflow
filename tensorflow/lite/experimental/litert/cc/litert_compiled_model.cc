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

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"

namespace litert {

Expected<size_t> CompiledModel::FindInputIndex(
    size_t signature_index, absl::string_view input_name) const {
  LITERT_ASSIGN_OR_RETURN(const Signature& signature,
                          model_.GetSignature(signature_index));
  const std::vector<absl::string_view>& input_names = signature.InputNames();
  auto it = std::find(input_names.begin(), input_names.end(), input_name);
  if (it != input_names.end()) {
    return std::distance(input_names.begin(), it);
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find input");
}

Expected<size_t> CompiledModel::FindOutputIndex(
    size_t signature_index, absl::string_view output_name) const {
  LITERT_ASSIGN_OR_RETURN(const Signature& signature,
                          model_.GetSignature(signature_index));
  const std::vector<absl::string_view>& output_names = signature.OutputNames();
  auto it = std::find(output_names.begin(), output_names.end(), output_name);
  if (it != output_names.end()) {
    return std::distance(output_names.begin(), it);
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find output");
}

Expected<TensorBuffer> CompiledModel::CreateBufferImpl(
    const TensorBufferRequirements& buffer_requirements,
    const RankedTensorType& tensor_type) {
  LITERT_ASSIGN_OR_RETURN(
      const std::vector<LiteRtTensorBufferType>& supported_types,
      buffer_requirements.SupportedTypes());
  if (supported_types.empty()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Input doesn't support any tensor buffer types");
  }
  // For simplicity we just pick the first supported tensor buffer type.
  LiteRtTensorBufferType tensor_buffer_type = supported_types[0];
  LITERT_ASSIGN_OR_RETURN(size_t buffer_size, buffer_requirements.BufferSize());

  LITERT_ASSIGN_OR_RETURN(TensorBuffer buffer,
                          TensorBuffer::CreateManaged(
                              tensor_buffer_type, tensor_type, buffer_size));
  return buffer;
}

Expected<TensorBuffer> CompiledModel::CreateInputOutputBuffer(
    size_t signature_index, absl::string_view tensor_name,
    bool is_input) const {
  LITERT_ASSIGN_OR_RETURN(Signature signature,
                          model_.GetSignature(signature_index));

  LITERT_ASSIGN_OR_RETURN(Subgraph subgraph, model_.Subgraph(signature.Key()));

  Expected<Tensor> tensor_expected =
      is_input ? subgraph.Input(tensor_name) : subgraph.Output(tensor_name);
  Expected<TensorBufferRequirements> buffer_requirements_expected =
      is_input ? GetInputBufferRequirements(signature_index, tensor_name)
               : GetOutputBufferRequirements(signature_index, tensor_name);

  LITERT_ASSIGN_OR_RETURN(const Tensor& tensor, tensor_expected);
  LITERT_ASSIGN_OR_RETURN(const TensorBufferRequirements& buffer_requirements,
                          buffer_requirements_expected);
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType& tensor_type,
                          tensor.RankedTensorType());

  return CreateBufferImpl(buffer_requirements, tensor_type);
}

Expected<std::vector<TensorBuffer>> CompiledModel::CreateInputOutputBuffers(
    size_t signature_index, bool is_input) const {
  LITERT_ASSIGN_OR_RETURN(const Signature& signature,
                          model_.GetSignature(signature_index));
  LITERT_ASSIGN_OR_RETURN(const Subgraph subgraph,
                          model_.Subgraph(signature.Key()));
  std::vector<TensorBuffer> tensor_buffers;
  std::vector<absl::string_view> tensor_names;

  tensor_names = is_input ? signature.InputNames() : signature.OutputNames();
  tensor_buffers.reserve(tensor_names.size());

  for (int i = 0; i < tensor_names.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(
        TensorBuffer tensor_buffer,
        CreateInputOutputBuffer(signature.Key(), tensor_names[i], is_input));
    tensor_buffers.push_back(std::move(tensor_buffer));
  }

  return tensor_buffers;
}

Expected<void> CompiledModel::RunCApiHelper(LiteRtParamIndex signature_index,
                                            size_t num_input_buffers,
                                            LiteRtTensorBuffer* input_buffers,
                                            size_t num_output_buffers,
                                            LiteRtTensorBuffer* output_buffers,
                                            bool& async) const {
  LiteRtStatus status =
      async ? LiteRtRunCompiledModelAsync(
                  Get(), signature_index, num_input_buffers, input_buffers,
                  num_output_buffers, output_buffers, &async)
            : LiteRtRunCompiledModel(Get(), signature_index, num_input_buffers,
                                     input_buffers, num_output_buffers,
                                     output_buffers);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to invoke the compiled model");
  }
  return {};
}

Expected<void> CompiledModel::RunHelper(
    size_t signature_index, const std::vector<TensorBuffer>& input_buffers,
    const std::vector<TensorBuffer>& output_buffers, bool& async) const {
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
  return RunCApiHelper(signature_index, input_buffers.size(),
                       input_buffers_ptr.get(), output_buffers.size(),
                       output_buffers_ptr.get(), async);
}

Expected<void> CompiledModel::RunMapHelper(
    absl::string_view signature_key,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
    bool& async) const {
  auto signature_index = model_.GetSignatureIndex(signature_key);
  if (!signature_index) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get signature_index");
  }
  auto subgraph = model_.Subgraph(signature_key);
  if (!subgraph) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to get subgraph");
  }
  return RunMapWithIndexHelper(*signature_index, *subgraph, input_map,
                               output_map, async);
}

Expected<void> CompiledModel::RunMapWithIndexHelper(
    size_t signature_index, const Subgraph& subgraph,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
    bool& async) const {
  auto input_tensors = subgraph.Inputs();
  size_t num_inputs = input_tensors.size();
  auto input_buffers_ptr = std::make_unique<LiteRtTensorBuffer[]>(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    absl::string_view input_name = input_tensors[i].Name();
    auto it = input_map.find(input_name);
    if (it == input_map.end()) {
      return Unexpected(kLiteRtStatusErrorNotFound,
                        "The given map is missing some input TensorBuffers");
    }
    input_buffers_ptr[i] = it->second.Get();
  }
  auto output_tensors = subgraph.Outputs();
  size_t num_outputs = output_tensors.size();
  auto output_buffers_ptr = std::make_unique<LiteRtTensorBuffer[]>(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    absl::string_view output_name = output_tensors[i].Name();
    auto it = output_map.find(output_name);
    if (it == output_map.end()) {
      return Unexpected(kLiteRtStatusErrorNotFound,
                        "The given map is missing some output TensorBuffers");
    }
    output_buffers_ptr[i] = it->second.Get();
  }
  return RunCApiHelper(signature_index, num_inputs, input_buffers_ptr.get(),
                       num_outputs, output_buffers_ptr.get(), async);
}

}  // namespace litert
