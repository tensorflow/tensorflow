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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"

namespace litert {

Expected<size_t> CompiledModel::FindInputIndex(
    size_t signature_index, absl::string_view input_name) const {
  auto signature = model_.GetSignature(signature_index);
  if (!signature) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find signature");
  }
  for (int i = 0; i < signature->InputNames().size(); ++i) {
    if (signature->InputNames()[i] == input_name) {
      return i;
    }
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find input");
}

Expected<size_t> CompiledModel::FindOutputIndex(
    size_t signature_index, absl::string_view output_name) const {
  auto signature = model_.GetSignature(signature_index);
  if (!signature) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find signature");
  }
  for (int i = 0; i < signature->OutputNames().size(); ++i) {
    if (signature->OutputNames()[i] == output_name) {
      return i;
    }
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find output");
}

Expected<TensorBuffer> CompiledModel::CreateBufferImpl(
    const TensorBufferRequirements& buffer_requirements,
    const RankedTensorType& tensor_type) {
  auto supported_types = buffer_requirements.SupportedTypes();
  if (!supported_types) {
    return supported_types.Error();
  }
  if (supported_types->empty()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Input doesn't support any tensor buffer types");
  }
  // For simplicity we just pick the first supported tensor buffer type.
  LiteRtTensorBufferType tensor_buffer_type = (*supported_types)[0];

  auto buffer =
      TensorBuffer::CreateManaged(tensor_buffer_type, tensor_type,
                                  buffer_requirements.BufferSize().Value());
  if (!buffer) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      buffer.Error().Message());
  }

  return std::move(*buffer);
}

Expected<TensorBuffer> CompiledModel::CreateInputOutputBuffer(
    absl::string_view signature_name, absl::string_view tensor_name,
    bool is_input) const {
  auto signature_index = model_.GetSignatureIndex(signature_name);
  if (!signature_index) {
    return signature_index.Error();
  }
  auto signature = model_.GetSignature(*signature_index);
  if (!signature) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find signature");
  }
  if (!signature) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find signature");
  }
  auto subgraph = model_.Subgraph(signature->Key());
  if (!subgraph) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to get subgraph");
  }

  LiteRtTensor target_litert_tensor;
  LiteRtTensorBufferRequirements litert_buffer_requirements;
  if (is_input) {
    auto input_tensor = subgraph->Input(tensor_name);
    if (!input_tensor) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find input");
    }
    target_litert_tensor = input_tensor->Get();
    auto input_buffer_requirements =
        GetInputBufferRequirements(*signature_index, tensor_name);
    if (!input_buffer_requirements) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        input_buffer_requirements.Error().Message());
    }
    litert_buffer_requirements = input_buffer_requirements->Get();
  } else {
    auto output_tensor = subgraph->Output(tensor_name);
    if (!output_tensor) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find output");
    }
    target_litert_tensor = output_tensor->Get();
    auto output_buffer_requirements =
        GetOutputBufferRequirements(*signature_index, tensor_name);
    if (!output_buffer_requirements) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        output_buffer_requirements.Error().Message());
    }
    litert_buffer_requirements = output_buffer_requirements->Get();
  }

  auto buffer_requirements =
      TensorBufferRequirements(litert_buffer_requirements, /*owned=*/false);
  auto target_tensor = Tensor(target_litert_tensor);
  auto tensor_type = target_tensor.RankedTensorType();
  if (!tensor_type) {
    return tensor_type.Error();
  }
  return CreateBufferImpl(buffer_requirements, *tensor_type);
}

Expected<std::vector<TensorBuffer>> CompiledModel::CreateInputOutputBuffers(
    size_t signature_index, bool is_input) const {
  auto signature = model_.GetSignature(signature_index);
  if (!signature) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find signature");
  }
  auto subgraph = model_.Subgraph(signature->Key());
  if (!subgraph) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to get subgraph");
  }
  std::vector<TensorBuffer> tensor_buffers;
  std::vector<absl::string_view> tensor_names;
  if (is_input) {
    tensor_names = signature->InputNames();
  } else {
    tensor_names = signature->OutputNames();
  }
  tensor_buffers.reserve(tensor_names.size());

  for (int i = 0; i < tensor_names.size(); ++i) {
    LiteRtTensor target_litert_tensor;
    LiteRtTensorBufferRequirements litert_buffer_requirements;
    if (is_input) {
      auto input_buffer_requirements =
          GetInputBufferRequirements(signature_index, i);
      if (!input_buffer_requirements) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          input_buffer_requirements.Error().Message());
      }
      litert_buffer_requirements = input_buffer_requirements->Get();
      auto input_tensor = subgraph->Input(tensor_names[i]);
      if (!input_tensor) {
        return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find input");
      }
      target_litert_tensor = input_tensor->Get();
    } else {
      auto output_buffer_requirements =
          GetOutputBufferRequirements(signature_index, i);
      if (!output_buffer_requirements) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          output_buffer_requirements.Error().Message());
      }
      litert_buffer_requirements = output_buffer_requirements->Get();
      auto output_tensor = subgraph->Output(tensor_names[i]);
      if (!output_tensor) {
        return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find output");
      }
      target_litert_tensor = output_tensor->Get();
    }

    auto buffer_requirements =
        TensorBufferRequirements(litert_buffer_requirements, /*owned=*/false);
    auto target_tensor = Tensor(target_litert_tensor);
    auto tensor_type = target_tensor.RankedTensorType();
    if (!tensor_type) {
      return tensor_type.Error();
    }
    auto tensor_buffer = CreateBufferImpl(buffer_requirements, *tensor_type);
    if (!tensor_buffer) {
      return tensor_buffer.Error();
    }
    tensor_buffers.push_back(std::move(*tensor_buffer));
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

Expected<void> CompiledModel::RunHelper(
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
  auto input_tensors = subgraph->Inputs();
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
  auto output_tensors = subgraph->Outputs();
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
  return RunCApiHelper(*signature_index, num_inputs, input_buffers_ptr.get(),
                       num_outputs, output_buffers_ptr.get(), async);
}

}  // namespace litert
