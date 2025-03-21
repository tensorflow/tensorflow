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

#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"

#include <dlfcn.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_shared_library.h"

#define LOAD_SYMB(S, H)                                                   \
  if (auto maybe_H = dlib_.LookupSymbol<void*>(#S); maybe_H.HasValue()) { \
    H = reinterpret_cast<decltype(&S)>(std::move(maybe_H).Value());       \
  } else {                                                                \
    LITERT_LOG(LITERT_WARNING, "Failed to load symbol %s: %s", #S,        \
               dlib_.DlError());                                          \
  }

namespace litert {
namespace mediatek {

NeuronAdapterApi::NeuronAdapterApi() : api_(new Api) {}

litert::Expected<NeuronAdapterApi::Ptr> NeuronAdapterApi::Create(
    std::optional<std::string> shared_library_dir) {
  std::unique_ptr<NeuronAdapterApi> neuron_adapter_api(new NeuronAdapterApi);
  if (auto status = neuron_adapter_api->LoadSymbols(shared_library_dir);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to load NeuronAdapter shared library: %s",
               status.Error().Message().c_str());
    return status.Error();
  }

  return neuron_adapter_api;
}

litert::Expected<void> NeuronAdapterApi::LoadSymbols(
    std::optional<std::string> shared_library_dir) {
  constexpr auto kLibNeuronAdapterLib = "libneuron_adapter.so";

  const std::vector<std::string> so_paths = {
      // The following preinstalled library is for system partition
      // applications.
      "libneuronusdk_adapter.mtk.so", "libneuron_adapter_mgvi.so",
      kLibNeuronAdapterLib,
      // Finally, the app may want to provide their own version of the library.
      shared_library_dir.has_value()
          ? absl::StrCat(*shared_library_dir, "/", kLibNeuronAdapterLib)
          : kLibNeuronAdapterLib};
  for (auto& so_path : so_paths) {
    auto maybe_dlib = SharedLibrary::Load(so_path, RtldFlags::Default());
    if (maybe_dlib.HasValue()) {
      dlib_ = std::move(maybe_dlib).Value();
    }
  }

  if (!dlib_.Loaded()) {
    return litert::Error(kLiteRtStatusErrorDynamicLoading,
                         "Failed to load NeuronAdapter shared library");
  }

  LITERT_LOG(LITERT_INFO, "Loaded NeuronAdapter shared library.");

  // Binds all supported symbols from the shared library to the function
  // pointers.
  LOAD_SYMB(NeuronCompilation_create, api_->compilation_create);
  LOAD_SYMB(NeuronCompilation_createWithOptions,
            api_->compilation_create_with_options);
  LOAD_SYMB(NeuronCompilation_finish, api_->compilation_finish);
  LOAD_SYMB(NeuronCompilation_free, api_->compilation_free);
  LOAD_SYMB(NeuronCompilation_getInputPaddedDimensions,
            api_->compilation_get_input_padded_dimensions);
  LOAD_SYMB(NeuronCompilation_getInputPaddedSize,
            api_->compilation_get_input_padded_size);
  LOAD_SYMB(NeuronCompilation_getOutputPaddedDimensions,
            api_->compilation_get_output_padded_dimensions);
  LOAD_SYMB(NeuronCompilation_getOutputPaddedSize,
            api_->compilation_get_output_padded_size);
  LOAD_SYMB(NeuronCompilation_setOptimizationString,
            api_->compilation_set_optimization_string);
  LOAD_SYMB(NeuronCompilation_setPreference, api_->compilation_set_preference);
  LOAD_SYMB(NeuronCompilation_setPriority, api_->compilation_set_priority);
  LOAD_SYMB(NeuronExecution_compute, api_->execution_compute);
  LOAD_SYMB(NeuronExecution_create, api_->execution_create);
  LOAD_SYMB(NeuronExecution_free, api_->execution_free);
  LOAD_SYMB(NeuronCompilation_getCompiledNetworkSize,
            api_->compilation_get_compiled_network_size);
  LOAD_SYMB(NeuronCompilation_storeCompiledNetwork,
            api_->compilation_store_compiled_network);
  LOAD_SYMB(NeuronExecution_setBoostHint, api_->execution_set_boost_hint);
  LOAD_SYMB(NeuronExecution_setInputFromMemory,
            api_->execution_set_input_from_memory);
  LOAD_SYMB(NeuronExecution_setOutputFromMemory,
            api_->execution_set_output_from_memory);
  LOAD_SYMB(NeuronMemory_createFromAHardwareBuffer,
            api_->memory_create_from_ahwb);
  LOAD_SYMB(NeuronMemory_createFromFd, api_->memory_create_from_fd);
  LOAD_SYMB(NeuronMemory_free, api_->memory_free);
  LOAD_SYMB(NeuronModel_addOperand, api_->model_add_operand);
  LOAD_SYMB(NeuronModel_addOperation, api_->model_add_operation);
  LOAD_SYMB(NeuronModel_create, api_->model_create);
  LOAD_SYMB(NeuronModel_finish, api_->model_finish);
  LOAD_SYMB(NeuronModel_free, api_->model_free);
  LOAD_SYMB(NeuronModel_getExtensionOperandType,
            api_->model_get_extension_operand_type);
  LOAD_SYMB(NeuronModel_getExtensionOperationType,
            api_->model_get_extension_operation_type);
  LOAD_SYMB(NeuronModel_identifyInputsAndOutputs,
            api_->model_identify_inputs_and_outputs);
  LOAD_SYMB(NeuronModel_restoreFromCompiledNetwork,
            api_->model_restore_from_compiled_network);
  LOAD_SYMB(NeuronModel_setName, api_->model_set_name);
  LOAD_SYMB(NeuronModel_setOperandValue, api_->model_set_operand_value);
  LOAD_SYMB(Neuron_getVersion, api_->get_version);

  LITERT_LOG(LITERT_INFO, "NeuronAdapter symbols loaded");
  return {};
}

Expected<NeuronModelPtr> NeuronAdapterApi::CreateModel() const {
  NeuronModel* model;
  if (api().model_create(&model) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create NeuroModel");
  }
  return NeuronModelPtr{model, api().model_free};
}

Expected<NeuronCompilationPtr> NeuronAdapterApi::CreateCompilation(
    NeuronModel* model) const {
  NeuronCompilation* compilation;
  if (api().compilation_create(model, &compilation) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create NeuronCompilation");
  }
  return NeuronCompilationPtr{compilation, api().compilation_free};
}

Expected<NeuronCompilationPtr> NeuronAdapterApi::CreateCompilation(
    NeuronModel* model, const std::string& compile_options) const {
  NeuronCompilation* compilation;
  if (auto status = api().compilation_create_with_options(
          model, &compilation, compile_options.c_str());
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR,
               "NeuronCompilation_createWithOptions failed with error %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create NeuronCompilation");
  }
  return NeuronCompilationPtr{compilation, api().compilation_free};
}

Expected<NeuronExecutionPtr> NeuronAdapterApi::CreateExecution(
    NeuronCompilation* compilation) const {
  NeuronExecution* execution;
  if (api().execution_create(compilation, &execution) != NEURON_NO_ERROR) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to create execution");
  }
  return NeuronExecutionPtr{execution, api().execution_free};
}

}  // namespace mediatek
}  // namespace litert
