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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_NEURON_ADAPTER_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_NEURON_ADAPTER_API_H_

#include <memory>
#include <optional>
#include <string>

#include "neuron/api/NeuronAdapter.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_shared_library.h"

#if LITERT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#else
struct AHardwareBuffer {};
#endif

namespace litert::mediatek {

using NeuronModelPtr = std::unique_ptr<NeuronModel, void (*)(NeuronModel*)>;
using NeuronCompilationPtr =
    std::unique_ptr<NeuronCompilation, void (*)(NeuronCompilation*)>;
using NeuronExecutionPtr =
    std::unique_ptr<NeuronExecution, void (*)(NeuronExecution*)>;

class NeuronAdapterApi {
 public:
  using Ptr = std::unique_ptr<NeuronAdapterApi>;
  struct Api;

  NeuronAdapterApi(NeuronAdapterApi&) = delete;
  NeuronAdapterApi(NeuronAdapterApi&&) = delete;
  NeuronAdapterApi& operator=(const NeuronAdapterApi&) = delete;
  NeuronAdapterApi& operator=(NeuronAdapterApi&&) = delete;

  static Expected<Ptr> Create(std::optional<std::string> shared_library_dir);

  const Api& api() const { return *api_; }

  absl::string_view AotCompileOptions() const {
    // Option `import_forever` has been recommended by MediaTek to reduce memory
    // footprint when using the same I/O buffers across multiple invocations.
    return "--apusys-config \"{ \\\"import_forever\\\": true }\"";
  }

  absl::string_view JitCompileOptions() const { return ""; }

  Expected<NeuronModelPtr> CreateModel() const;

  Expected<NeuronCompilationPtr> CreateCompilation(NeuronModel* model) const;

  Expected<NeuronCompilationPtr> CreateCompilation(
      NeuronModel* model, const std::string& compile_options) const;

  Expected<NeuronExecutionPtr> CreateExecution(
      NeuronCompilation* compilation) const;

 private:
  NeuronAdapterApi();
  litert::Expected<void> LoadSymbols(
      std::optional<std::string> shared_library_dir);

  // Handle to the shared library that implements the Neuron API.
  //
  // This will keep the shared library open until the NeuronAdapterApi object is
  // destroyed.
  SharedLibrary dlib_;
  std::unique_ptr<Api> api_;
};

// This is not part of the provided NeuronAdapter header for some reason.
int NeuronCompilation_createWithOptions(NeuronModel* model,
                                        NeuronCompilation** compilation,
                                        const char* options);

// A convenient struct for holding function pointers to NeuronAdapter API
// symbols. These function pointers will be loaded to the shared library on
// device during runtime.
struct NeuronAdapterApi::Api {
  decltype(&NeuronCompilation_create) compilation_create = nullptr;
  decltype(&NeuronCompilation_createWithOptions)
      compilation_create_with_options = nullptr;
  decltype(&NeuronCompilation_finish) compilation_finish = nullptr;
  decltype(&NeuronCompilation_free) compilation_free = nullptr;
  decltype(&NeuronCompilation_getCompiledNetworkSize)
      compilation_get_compiled_network_size = nullptr;
  decltype(&NeuronCompilation_getInputPaddedDimensions)
      compilation_get_input_padded_dimensions = nullptr;
  decltype(&NeuronCompilation_getInputPaddedSize)
      compilation_get_input_padded_size = nullptr;
  decltype(&NeuronCompilation_getOutputPaddedDimensions)
      compilation_get_output_padded_dimensions = nullptr;
  decltype(&NeuronCompilation_getOutputPaddedSize)
      compilation_get_output_padded_size = nullptr;
  decltype(&NeuronCompilation_setOptimizationString)
      compilation_set_optimization_string = nullptr;
  decltype(&NeuronCompilation_setPreference) compilation_set_preference =
      nullptr;
  decltype(&NeuronCompilation_setPriority) compilation_set_priority = nullptr;
  decltype(&NeuronCompilation_storeCompiledNetwork)
      compilation_store_compiled_network = nullptr;
  decltype(&NeuronExecution_compute) execution_compute = nullptr;
  decltype(&NeuronExecution_create) execution_create = nullptr;
  decltype(&NeuronExecution_free) execution_free = nullptr;
  decltype(&NeuronExecution_setBoostHint) execution_set_boost_hint = nullptr;
  decltype(&NeuronExecution_setInputFromMemory)
      execution_set_input_from_memory = nullptr;
  decltype(&NeuronExecution_setOutputFromMemory)
      execution_set_output_from_memory = nullptr;
  decltype(&NeuronMemory_createFromAHardwareBuffer) memory_create_from_ahwb =
      nullptr;
  decltype(&NeuronMemory_createFromFd) memory_create_from_fd = nullptr;
  decltype(&NeuronMemory_free) memory_free = nullptr;
  decltype(&NeuronModel_addOperand) model_add_operand = nullptr;
  decltype(&NeuronModel_addOperation) model_add_operation = nullptr;
  decltype(&NeuronModel_create) model_create = nullptr;
  decltype(&NeuronModel_finish) model_finish = nullptr;
  decltype(&NeuronModel_free) model_free = nullptr;
  decltype(&NeuronModel_getExtensionOperandType)
      model_get_extension_operand_type = nullptr;
  decltype(&NeuronModel_getExtensionOperationType)
      model_get_extension_operation_type = nullptr;
  decltype(&NeuronModel_identifyInputsAndOutputs)
      model_identify_inputs_and_outputs = nullptr;
  decltype(&NeuronModel_restoreFromCompiledNetwork)
      model_restore_from_compiled_network = nullptr;
  decltype(&NeuronModel_setName) model_set_name = nullptr;
  decltype(&NeuronModel_setOperandValue) model_set_operand_value = nullptr;
  decltype(&Neuron_getVersion) get_version = nullptr;
};

}  // namespace litert::mediatek

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_NEURON_ADAPTER_API_H_
