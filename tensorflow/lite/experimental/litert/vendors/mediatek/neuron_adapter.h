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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_NEURON_ADAPTER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_NEURON_ADAPTER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

#if LITERT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#else
struct AHardwareBuffer {};
#endif

namespace litert::mediatek {

// /////////////////////////////////////////////////////////////////////////////
//
// A minimal set of definitions for the NeuronAdapter API, from public domain
// sources.
//
// /////////////////////////////////////////////////////////////////////////////

struct NeuronRuntimeVersion {
  uint8_t major;
  uint8_t minor;
  uint8_t patch;
};

enum NeuronOperationType {
  NEURON_ADD = 0,
};

struct NeuronOperandType {
  int32_t type;
  // NOLINTNEXTLINE
  uint32_t dimensionCount;
  const uint32_t* dimensions;
  float scale;
  // NOLINTNEXTLINE
  int32_t zeroPoint;
};

struct NeuronModel;
struct NeuronCompilation;
struct NeuronExecution;
struct NeuronMemory;

static constexpr int NEURON_NO_ERROR = 0;
static constexpr int NEURON_FLOAT32 = 0;
static constexpr int NEURON_INT32 = 1;
static constexpr int NEURON_BOOL = 6;
static constexpr int NEURON_TENSOR_FLOAT32 = 3;
static constexpr int NEURON_TENSOR_INT32 = 4;
static constexpr int NEURON_PRIORITY_HIGH = 110;
static constexpr int NEURON_PREFER_SUSTAINED_SPEED = 2;

int NeuronCompilation_create(NeuronModel* model,
                             NeuronCompilation** compilation);
int NeuronCompilation_createWithOptions(NeuronModel* model,
                                        NeuronCompilation** compilation,
                                        const char* options);
int NeuronCompilation_finish(NeuronCompilation* compilation);
int NeuronCompilation_getCompiledNetworkSize(NeuronCompilation* compilation,
                                             size_t* size);
int NeuronCompilation_getInputPaddedDimensions(NeuronCompilation* compilation,
                                               int32_t index,
                                               uint32_t* dimensions);
int NeuronCompilation_getInputPaddedSize(NeuronCompilation* compilation,
                                         int32_t index, size_t* size);
int NeuronCompilation_getOutputPaddedDimensions(NeuronCompilation* compilation,
                                                int32_t index,
                                                uint32_t* dimensions);
int NeuronCompilation_getOutputPaddedSize(NeuronCompilation* compilation,
                                          int32_t index, size_t* size);
int NeuronCompilation_setOptimizationString(NeuronCompilation* compilation,
                                            const char* optimizationString);
int NeuronCompilation_setPreference(NeuronCompilation* compilation,
                                    int32_t preference);
int NeuronCompilation_setPriority(NeuronCompilation* compilation, int priority);
int NeuronCompilation_storeCompiledNetwork(NeuronCompilation* compilation,
                                           void* buffer, size_t size);
int NeuronExecution_compute(NeuronExecution* execution);
int NeuronExecution_create(NeuronCompilation* compilation,
                           NeuronExecution** execution);
int NeuronExecution_setBoostHint(NeuronExecution* execution,
                                 uint8_t boostValue);
int NeuronExecution_setInputFromMemory(NeuronExecution* execution,
                                       uint32_t index,
                                       const NeuronOperandType* type,
                                       const NeuronMemory* memory,
                                       size_t offset, size_t length);
int NeuronExecution_setOutputFromMemory(NeuronExecution* execution,
                                        uint32_t index,
                                        const NeuronOperandType* type,
                                        const NeuronMemory* memory,
                                        size_t offset, size_t length);
int NeuronMemory_createFromAHardwareBuffer(const AHardwareBuffer* ahwb,
                                           NeuronMemory** memory);
int NeuronModel_addOperand(NeuronModel* model, const NeuronOperandType* type);
int NeuronModel_addOperation(NeuronModel* model, NeuronOperationType type,
                             uint32_t inputCount, const uint32_t* inputs,
                             uint32_t outputCount, const uint32_t* outputs);
int NeuronModel_create(NeuronModel** model);
int NeuronModel_finish(NeuronModel* model);
int NeuronModel_getExtensionOperandType(NeuronModel* model,
                                        const char* extensionName,
                                        uint16_t operandCodeWithinExtension,
                                        int32_t* type);
int NeuronModel_getExtensionOperationType(NeuronModel* model,
                                          const char* extensionName,
                                          uint16_t operationCodeWithinExtension,
                                          int32_t* type);
int NeuronModel_identifyInputsAndOutputs(NeuronModel* model,
                                         uint32_t inputCount,
                                         const uint32_t* inputs,
                                         uint32_t outputCount,
                                         const uint32_t* outputs);
int NeuronModel_restoreFromCompiledNetwork(NeuronModel** model,
                                           NeuronCompilation** compilation,
                                           const void* buffer, size_t size);
int NeuronModel_setName(NeuronModel* model, const char* name);
int NeuronModel_setOperandValue(NeuronModel* model, int32_t index,
                                const void* buffer, size_t length);
int Neuron_getVersion(NeuronRuntimeVersion* version);
void NeuronCompilation_free(NeuronCompilation* compilation);
void NeuronExecution_free(NeuronExecution* execution);
void NeuronMemory_free(NeuronMemory* memory);
void NeuronModel_free(NeuronModel* model);

// /////////////////////////////////////////////////////////////////////////////

using NeuronModelPtr = std::unique_ptr<NeuronModel, void (*)(NeuronModel*)>;
using NeuronCompilationPtr =
    std::unique_ptr<NeuronCompilation, void (*)(NeuronCompilation*)>;
using NeuronExecutionPtr =
    std::unique_ptr<NeuronExecution, void (*)(NeuronExecution*)>;

class NeuronAdapter {
 public:
  using Ptr = std::unique_ptr<NeuronAdapter>;
  struct Api;

  NeuronAdapter(NeuronAdapter&) = delete;
  NeuronAdapter(NeuronAdapter&&) = delete;
  NeuronAdapter& operator=(const NeuronAdapter&) = delete;
  NeuronAdapter& operator=(NeuronAdapter&&) = delete;

  ~NeuronAdapter();

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
  NeuronAdapter();
  litert::Expected<void> LoadSymbols(
      std::optional<std::string> shared_library_dir);

  void* dlib_handle_ = nullptr;
  std::unique_ptr<Api> api_;
};

// A convenient struct for holding function pointers to NeuronAdapter API
// symbols. These function pointers will be loaded to the shared library on
// device during runtime.
struct NeuronAdapter::Api {
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

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_NEURON_ADAPTER_H_
