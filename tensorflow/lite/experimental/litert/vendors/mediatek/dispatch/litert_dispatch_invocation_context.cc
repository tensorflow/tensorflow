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
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/dispatch/litert_dispatch_device_context.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/schema/schema_resolver.h"

using litert::Error;
using litert::Expected;
using litert::mediatek::NeuronCompilationPtr;
using litert::mediatek::NeuronExecutionPtr;
using litert::mediatek::NeuronModelPtr;

namespace {

Expected<std::pair<NeuronModelPtr, NeuronCompilationPtr>> LoadFromCachedNetwork(
    const litert::mediatek::NeuronAdapterApi& neuron_adapter_api,
    const void* bytecode_addr, size_t bytecode_size) {
  NeuronModel* model;
  NeuronCompilation* compilation;
  if (neuron_adapter_api.api().model_restore_from_compiled_network(
          &model, &compilation, bytecode_addr, bytecode_size) !=
      NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to restore model from compiled network");
  }
  return std::make_pair(
      NeuronModelPtr{model, neuron_adapter_api.api().model_free},
      NeuronCompilationPtr{compilation,
                           neuron_adapter_api.api().compilation_free});
}

uint16_t GetRestoreDlaExtensionOperandType(
    const litert::mediatek::NeuronAdapterApi& neuron_adapter_api) {
  NeuronRuntimeVersion version;
  neuron_adapter_api.api().get_version(&version);
  // The values below were suggested by MTK.
  if (version.major >= 8) {
    return 0x0200;
  } else {
    return 0x0100;
  }
}

Expected<std::pair<NeuronModelPtr, NeuronCompilationPtr>> LoadFromDlaBytecode(
    const litert::mediatek::NeuronAdapterApi& neuron_adapter_api,
    const void* bytecode_addr, size_t bytecode_size, int num_inputs,
    int num_outputs) {
  Expected<NeuronModelPtr> model = neuron_adapter_api.CreateModel();
  if (!model) {
    return model.Error();
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
    if (neuron_adapter_api.api().model_add_operand(
            model->get(), &fake_io_operand_type) != NEURON_NO_ERROR) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to add input operand");
    }
    input_op_number.emplace_back(i);
  }

  const uint16_t kNetworkOperandRestoreData =
      GetRestoreDlaExtensionOperandType(neuron_adapter_api);
  constexpr const uint16_t kRestoreDlaExtensionOperationType = 0;
  constexpr const char* kExtensionRestoreCompiledNetwork =
      "com.mediatek.compiled_network";

  int32_t operand_type;
  if (neuron_adapter_api.api().model_get_extension_operand_type(
          model->get(), kExtensionRestoreCompiledNetwork,
          kNetworkOperandRestoreData, &operand_type) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to getextension operand");
  }

  const NeuronOperandType extension_operand_type{
      .type = operand_type,
      .dimensionCount = 0,
      .scale = 0.0f,
      .zeroPoint = 0,
  };
  if (neuron_adapter_api.api().model_add_operand(
          model->get(), &extension_operand_type) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to add extension operand");
  }
  input_op_number.emplace_back(input_op_number.size());
  if (neuron_adapter_api.api().model_set_operand_value(
          model->get(), input_op_number.back(), bytecode_addr, bytecode_size) !=
      NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set extension operand value");
  }

  std::vector<uint32_t> output_op_number;
  for (auto i = 0; i < num_outputs; i++) {
    if (neuron_adapter_api.api().model_add_operand(
            model->get(), &fake_io_operand_type) != NEURON_NO_ERROR) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to add output operand");
    }
    output_op_number.emplace_back(input_op_number.size() + i);
  }

  int32_t operation_type;
  if (neuron_adapter_api.api().model_get_extension_operation_type(
          model->get(), kExtensionRestoreCompiledNetwork,
          kRestoreDlaExtensionOperationType,
          &operation_type) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to get extension operation");
  }

  // Add extension operation
  if (neuron_adapter_api.api().model_add_operation(
          model->get(), static_cast<NeuronOperationType>(operation_type),
          input_op_number.size(), input_op_number.data(),
          output_op_number.size(),
          output_op_number.data()) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to add extension operation");
  }

  if (neuron_adapter_api.api().model_identify_inputs_and_outputs(
          model->get(), input_op_number.size() - 1, input_op_number.data(),
          output_op_number.size(),
          output_op_number.data()) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to identify I/Os");
  }

  if (neuron_adapter_api.api().model_finish(model->get()) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to finish model");
  }

  auto compilation = neuron_adapter_api.CreateCompilation(model->get());
  if (!compilation) {
    return compilation.Error();
  }

  if (neuron_adapter_api.api().compilation_set_priority(
          compilation->get(), NEURON_PRIORITY_HIGH) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set compilation priority");
  }

  if (neuron_adapter_api.api().compilation_set_preference(
          compilation->get(), NEURON_PREFER_SUSTAINED_SPEED) !=
      NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set compilation preference");
  }

  // We use AOT compile options since the DLA file was compiled ahead of time.
  const auto compile_options =
      std::string(neuron_adapter_api.AotCompileOptions());
  if (!compile_options.empty()) {
    if (neuron_adapter_api.api().compilation_set_optimization_string(
            compilation->get(), compile_options.c_str()) != NEURON_NO_ERROR) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to set optimization string");
    }
  }

  if (neuron_adapter_api.api().compilation_finish(compilation->get()) !=
      NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to finish compilation");
  }

  return std::make_pair(std::move(*model), std::move(*compilation));
}

Expected<std::pair<NeuronModelPtr, NeuronCompilationPtr>>
LoadModelAndCompilation(
    const litert::mediatek::NeuronAdapterApi& neuron_adapter_api,
    const void* bytecode_addr, size_t bytecode_size, int num_inputs,
    int num_outputs) {
  if (auto result = LoadFromDlaBytecode(neuron_adapter_api, bytecode_addr,
                                        bytecode_size, num_inputs, num_outputs);
      !result) {
    return LoadFromCachedNetwork(neuron_adapter_api, bytecode_addr,
                                 bytecode_size);
  } else {
    return result;
  }
}

}  // namespace

Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    litert::mediatek::NeuronAdapterApi& neuron_adapter_api,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs) {
  neuron::SchemaResolver resolver;

  const void* exec_bytecode_ptr =
      static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr) +
      exec_bytecode_buffer->offset;
  auto exec_bytecode_size = exec_bytecode_buffer->size;
  auto res = resolver.Initialize((const uint8_t*)exec_bytecode_ptr,
                                 exec_bytecode_size);
  if (res.HasValue() && res.Value()) {
    std::string func = function_name != nullptr ? function_name : "";
    auto graph = resolver.GetCompiledGraph(func);
    if (!graph.has_value()) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "Couldn't find the subgraph");
    }
    auto compile_graph = graph.value().GetCompiledNetwork();
    if (!compile_graph) {
      return compile_graph.Error();
    }
    std::tie(exec_bytecode_ptr, exec_bytecode_size) = compile_graph.Value();
  }

  auto model_and_compilation =
      LoadModelAndCompilation(neuron_adapter_api, exec_bytecode_ptr,
                              exec_bytecode_size, num_inputs, num_outputs);
  if (!model_and_compilation) {
    return model_and_compilation.Error();
  }

  auto& model = model_and_compilation->first;
  auto& compilation = model_and_compilation->second;

  auto execution = neuron_adapter_api.CreateExecution(compilation.get());
  if (!execution) {
    return execution.Error();
  }

  if (neuron_adapter_api.api().execution_set_boost_hint(
          execution->get(), 100) != NEURON_NO_ERROR) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to set execution boost hint");
  }

  return Ptr(new LiteRtDispatchInvocationContextT(
      neuron_adapter_api, device_context, model.release(),
      compilation.release(), execution->release(), num_inputs, num_outputs));
}

LiteRtDispatchInvocationContextT::~LiteRtDispatchInvocationContextT() {
  if (execution_) {
    neuron_adapter_api_.api().execution_free(execution_);
  }
  if (compilation_) {
    neuron_adapter_api_.api().compilation_free(compilation_);
  }
  if (model_) {
    neuron_adapter_api_.api().model_free(model_);
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

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::IoRequirementsBuilder::Create() {
  static constexpr std::array kSupportedTensorBufferTypes = {
#if defined(__ANDROID__)
      kLiteRtTensorBufferTypeAhwb,
#endif  // __ANDROID__
      kLiteRtTensorBufferTypeDmaBuf,
  };

  LiteRtTensorBufferRequirements requirements;
  if (auto status = LiteRtCreateTensorBufferRequirements(
          kSupportedTensorBufferTypes.size(),
          kSupportedTensorBufferTypes.data(), buffer_size_, strides_.size(),
          strides_.data(), &requirements);
      status != kLiteRtStatusOk) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to create tensor buffer requirements");
  }

  return requirements;
}

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  if (!input_requirements_builders_[input_index]) {
    size_t buffer_size;
    if (neuron_adapter_api_.api().compilation_get_input_padded_size(
            compilation_, input_index, &buffer_size) != NEURON_NO_ERROR) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "Failed to get input padded size");
    }

    std::vector<uint32_t> padded_dimensions(tensor_type.layout.rank);
    if (neuron_adapter_api_.api().compilation_get_input_padded_dimensions(
            compilation_, input_index, padded_dimensions.data()) !=
        NEURON_NO_ERROR) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "Failed to get input padded dimensions");
    }

    input_requirements_builders_[input_index] =
        std::make_unique<IoRequirementsBuilder>(buffer_size, padded_dimensions);
  }

  return input_requirements_builders_[input_index]->Create();
}

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  if (!output_requirements_builders_[output_index]) {
    size_t buffer_size;
    if (neuron_adapter_api_.api().compilation_get_output_padded_size(
            compilation_, output_index, &buffer_size) != NEURON_NO_ERROR) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "Failed to get output padded size");
    }

    std::vector<uint32_t> padded_dimensions(tensor_type.layout.rank);
    if (neuron_adapter_api_.api().compilation_get_output_padded_dimensions(
            compilation_, output_index, padded_dimensions.data()) !=
        NEURON_NO_ERROR) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "Failed to get output padded dimensions");
    }

    output_requirements_builders_[output_index] =
        std::make_unique<IoRequirementsBuilder>(buffer_size, padded_dimensions);
  }

  return output_requirements_builders_[output_index]->Create();
}

Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto neuron_memory_info =
      device_context_->GetNeuronMemoryInfo(tensor_buffer_handle);
  if (!neuron_memory_info) {
    return litert::Error(neuron_memory_info.Error());
  }

  if (neuron_adapter_api_.api().execution_set_input_from_memory(
          execution_, graph_input_index, nullptr,
          neuron_memory_info->neuron_memory, neuron_memory_info->offset,
          neuron_memory_info->size) != NEURON_NO_ERROR) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to set execution input from memory");
  }
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto neuron_memory_info =
      device_context_->GetNeuronMemoryInfo(tensor_buffer_handle);
  if (!neuron_memory_info) {
    return litert::Error(neuron_memory_info.Error());
  }

  if (neuron_adapter_api_.api().execution_set_output_from_memory(
          execution_, graph_output_index, nullptr,
          neuron_memory_info->neuron_memory, neuron_memory_info->offset,
          neuron_memory_info->size) != NEURON_NO_ERROR) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to set execution output from memory");
  }
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::DetachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  // Nothing to do.
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::DetachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  // Nothing to do.
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::Invoke() {
  if (neuron_adapter_api_.api().execution_compute(execution_) !=
      NEURON_NO_ERROR) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to execute network");
  }
  return {};
}
