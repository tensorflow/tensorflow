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

#include "tensorflow/lite/experimental/litert/vendors/mediatek/dispatch/litert_dispatch_invocation_context.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/dispatch/litert_dispatch_device_context.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter.h"

using litert::mediatek::NEURON_NO_ERROR;
using litert::mediatek::NEURON_PREFER_SUSTAINED_SPEED;
using litert::mediatek::NEURON_PRIORITY_HIGH;
using litert::mediatek::NEURON_TENSOR_FLOAT32;
using litert::mediatek::NeuronCompilation;
using litert::mediatek::NeuronExecution;
using litert::mediatek::NeuronModel;
using litert::mediatek::NeuronOperandType;
using litert::mediatek::NeuronOperationType;
using litert::mediatek::NeuronRuntimeVersion;

namespace {

bool LoadFromCachedNetwork(
    const litert::mediatek::NeuronAdapter& neuron_adapter, NeuronModel*& model,
    NeuronCompilation*& compilation, const void* bytecode_addr,
    size_t bytecode_size) {
  return neuron_adapter.api().model_restore_from_compiled_network(
             &model, &compilation, bytecode_addr, bytecode_size) ==
         NEURON_NO_ERROR;
}

uint16_t GetRestoreDlaExtensionOperandType(
    const litert::mediatek::NeuronAdapter& neuron_adapter) {
  NeuronRuntimeVersion version;
  neuron_adapter.api().get_version(&version);
  // The values below were suggested by MTK.
  if (version.major >= 8) {
    return 0x0200;
  } else {
    return 0x0100;
  }
}

bool LoadFromDlaBytecode(const litert::mediatek::NeuronAdapter& neuron_adapter,
                         NeuronModel*& model, NeuronCompilation*& compilation,
                         const void* bytecode_addr, size_t bytecode_size,
                         int num_inputs, int num_outputs) {
  LITERT_LOG(LITERT_INFO, "Creating model...");
  if (neuron_adapter.api().model_create(&model) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to create model");
    return false;
  }

  // fake input, the real outputs are loaded by compiled network.
  constexpr const NeuronOperandType fake_io_operand_type{
      .type = NEURON_TENSOR_FLOAT32,
      .dimensionCount = 0,
      .scale = 0.0f,
      .zeroPoint = 0,
  };

  std::vector<uint32_t> input_op_number;
  input_op_number.reserve(num_inputs);
  for (auto i = 0; i < num_inputs; i++) {
    if (neuron_adapter.api().model_add_operand(model, &fake_io_operand_type) !=
        NEURON_NO_ERROR) {
      LITERT_LOG(LITERT_ERROR, "Failed to add input operand %d", i);
      return false;
    }
    input_op_number.emplace_back(i);
  }

  const uint16_t kNetworkOperandRestoreData =
      GetRestoreDlaExtensionOperandType(neuron_adapter);
  constexpr const uint16_t kRestoreDlaExtensionOperationType = 0;
  constexpr const char* kExtensionRestoreCompiledNetwork =
      "com.mediatek.compiled_network";

  int32_t operand_type;
  if (neuron_adapter.api().model_get_extension_operand_type(
          model, kExtensionRestoreCompiledNetwork, kNetworkOperandRestoreData,
          &operand_type) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to get extension operand");
    return false;
  }

  const NeuronOperandType extension_operand_type{
      .type = operand_type,
      .dimensionCount = 0,
      .scale = 0.0f,
      .zeroPoint = 0,
  };
  if (neuron_adapter.api().model_add_operand(model, &extension_operand_type) !=
      NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to add extension operand");
    return false;
  }
  input_op_number.emplace_back(input_op_number.size());
  if (neuron_adapter.api().model_set_operand_value(
          model, input_op_number.back(), bytecode_addr, bytecode_size) !=
      NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to set extension operand value");
    return false;
  }

  std::vector<uint32_t> output_op_number;
  for (auto i = 0; i < num_outputs; i++) {
    if (neuron_adapter.api().model_add_operand(model, &fake_io_operand_type) !=
        NEURON_NO_ERROR) {
      LITERT_LOG(LITERT_ERROR, "Failed to add output operand %d", i);
      return false;
    }
    output_op_number.emplace_back(input_op_number.size() + i);
  }

  int32_t operation_type;
  if (neuron_adapter.api().model_get_extension_operation_type(
          model, kExtensionRestoreCompiledNetwork,
          kRestoreDlaExtensionOperationType,
          &operation_type) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to get extension operation");
    return false;
  }

  // Add extension operation
  if (neuron_adapter.api().model_add_operation(
          model, static_cast<NeuronOperationType>(operation_type),
          input_op_number.size(), input_op_number.data(),
          output_op_number.size(),
          output_op_number.data()) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to add extension operation");
    return false;
  }

  if (neuron_adapter.api().model_identify_inputs_and_outputs(
          model, input_op_number.size() - 1, input_op_number.data(),
          output_op_number.size(),
          output_op_number.data()) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to identify I/Os");
    return false;
  }

  if (neuron_adapter.api().model_finish(model) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to finish model");
    return false;
  }

  if (neuron_adapter.api().compilation_create(model, &compilation) !=
      NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compilation");
    return false;
  }

  if (neuron_adapter.api().compilation_set_priority(
          compilation, NEURON_PRIORITY_HIGH) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to set compilation priority");
    return false;
  }

  if (neuron_adapter.api().compilation_set_preference(
          compilation, NEURON_PREFER_SUSTAINED_SPEED) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to set compilation preference");
    return false;
  }

  if (neuron_adapter.api().compilation_finish(compilation) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to finish compilation");
    return false;
  }

  return true;
}

bool LoadModelAndCompilation(
    const litert::mediatek::NeuronAdapter& neuron_adapter, NeuronModel*& model,
    NeuronCompilation*& compilation, const void* bytecode_addr,
    size_t bytecode_size, int num_inputs, int num_outputs) {
  if (!LoadFromDlaBytecode(neuron_adapter, model, compilation, bytecode_addr,
                           bytecode_size, num_inputs, num_outputs)) {
    return LoadFromCachedNetwork(neuron_adapter, model, compilation,
                                 bytecode_addr, bytecode_size);
  }
  return true;
}

}  // namespace

litert::Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    litert::mediatek::NeuronAdapter& neuron_adapter,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type, const void* bytecode_ptr,
    size_t bytecode_size, const char* function_name, int num_inputs,
    int num_outputs) {
  NeuronModel* model;
  NeuronCompilation* compilation;
  if (!LoadModelAndCompilation(neuron_adapter, model, compilation, bytecode_ptr,
                               bytecode_size, num_inputs, num_outputs)) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to load compiled model");
  }

  NeuronExecution* execution;
  if (neuron_adapter.api().execution_create(compilation, &execution) !=
      NEURON_NO_ERROR) {
    if (compilation) {
      neuron_adapter.api().compilation_free(compilation);
    }
    if (model) {
      neuron_adapter.api().model_free(model);
    }
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create execution");
  }

  if (neuron_adapter.api().execution_set_boost_hint(execution, 100) !=
      NEURON_NO_ERROR) {
    if (execution) {
      neuron_adapter.api().execution_free(execution);
    }
    if (compilation) {
      neuron_adapter.api().compilation_free(compilation);
    }
    if (model) {
      neuron_adapter.api().model_free(model);
    }
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to set execution boost hint");
  }

  return Ptr(new LiteRtDispatchInvocationContextT(
      neuron_adapter, device_context, model, compilation, execution, num_inputs,
      num_outputs));
}

LiteRtDispatchInvocationContextT::~LiteRtDispatchInvocationContextT() {
  if (execution_) {
    neuron_adapter_.api().execution_free(execution_);
  }
  if (compilation_) {
    neuron_adapter_.api().compilation_free(compilation_);
  }
  if (model_) {
    neuron_adapter_.api().model_free(model_);
  }
}

LiteRtDispatchInvocationContextT::IoRequirementsBuilder::IoRequirementsBuilder(
    size_t buffer_size, const std::vector<uint32_t>& padded_dimensions)
    : buffer_size_(buffer_size) {
  auto rank = padded_dimensions.size();
  strides_.resize(rank);
  strides_[0] = 1;
  for (auto i = 1; i < rank; ++i) {
    strides_[i] = padded_dimensions[i - 1];
  }
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::IoRequirementsBuilder::Create() {
  static constexpr std::array<LiteRtTensorBufferType, 1>
      kSupportedTensorBufferTypes = {
          kLiteRtTensorBufferTypeAhwb,
      };

  LiteRtTensorBufferRequirements requirements;
  if (auto status = LiteRtCreateTensorBufferRequirements(
          kSupportedTensorBufferTypes.size(),
          kSupportedTensorBufferTypes.data(), buffer_size_, strides_.size(),
          strides_.data(), &requirements);
      status != kLiteRtStatusOk) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create tensor buffer requirements");
  }

  return requirements;
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  if (!input_requirements_builders_[input_index]) {
    size_t buffer_size;
    if (neuron_adapter_.api().compilation_get_input_padded_size(
            compilation_, input_index, &buffer_size) != NEURON_NO_ERROR) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "Failed to get input padded size");
    }

    std::vector<uint32_t> padded_dimensions(tensor_type.layout.rank);
    if (neuron_adapter_.api().compilation_get_input_padded_dimensions(
            compilation_, input_index, padded_dimensions.data()) !=
        NEURON_NO_ERROR) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "Failed to get input padded dimensions");
    }

    input_requirements_builders_[input_index] =
        std::make_unique<IoRequirementsBuilder>(buffer_size, padded_dimensions);
  }

  return input_requirements_builders_[input_index]->Create();
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  if (!output_requirements_builders_[output_index]) {
    size_t buffer_size;
    if (neuron_adapter_.api().compilation_get_output_padded_size(
            compilation_, output_index, &buffer_size) != NEURON_NO_ERROR) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "Failed to get output padded size");
    }

    std::vector<uint32_t> padded_dimensions(tensor_type.layout.rank);
    if (neuron_adapter_.api().compilation_get_output_padded_dimensions(
            compilation_, output_index, padded_dimensions.data()) !=
        NEURON_NO_ERROR) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "Failed to get output padded dimensions");
    }

    output_requirements_builders_[output_index] =
        std::make_unique<IoRequirementsBuilder>(buffer_size, padded_dimensions);
  }

  return output_requirements_builders_[output_index]->Create();
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto neuron_memory_info =
      device_context_->GetNeuronMemoryInfo(tensor_buffer_handle);
  if (!neuron_memory_info) {
    return litert::Unexpected(neuron_memory_info.Error());
  }

  if (neuron_adapter_.api().execution_set_input_from_memory(
          execution_, graph_input_index, nullptr,
          neuron_memory_info->neuron_memory, neuron_memory_info->offset,
          neuron_memory_info->size) != NEURON_NO_ERROR) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to set execution input from memory");
  }
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto neuron_memory_info =
      device_context_->GetNeuronMemoryInfo(tensor_buffer_handle);
  if (!neuron_memory_info) {
    return litert::Unexpected(neuron_memory_info.Error());
  }

  if (neuron_adapter_.api().execution_set_output_from_memory(
          execution_, graph_output_index, nullptr,
          neuron_memory_info->neuron_memory, neuron_memory_info->offset,
          neuron_memory_info->size) != NEURON_NO_ERROR) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to set execution output from memory");
  }
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::DetachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  // Nothing to do.
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::DetachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  // Nothing to do.
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::Invoke() {
  if (neuron_adapter_.api().execution_compute(execution_) != NEURON_NO_ERROR) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to execute network");
  }
  return {};
}
