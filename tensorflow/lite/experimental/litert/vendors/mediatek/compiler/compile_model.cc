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

#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/compile_model.h"

#include <optional>
#include <string>

#include "neuron/api/NeuronAdapter.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

Expected<NeuronCompilationPtr> CompileModel(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    std::optional<std::string> soc_model) {
#if defined(__ANDROID__)
  if (soc_model) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "JIT compilation for a specific SoC is not supported");
  }
#endif

  // Per MediaTek recommendation, Compilation_create,
  // Compilation_createWithOptions, and Compilation_setOptimizationString
  // should be used as follow:
  // - AOT Compilation: Compilation_createWithOptions only
  // - JIT Compilation: Compilation_create and Compilation_setOptimizationString
  // The code below takes care of those conditions.

  // NOLINTBEGIN
  const auto compile_options =
#if __ANDROID__
      std::string(neuron_adapter_api.JitCompileOptions());
#else
      std::string(neuron_adapter_api.AotCompileOptions());
#endif
  // NOLINTEND

  auto compilation =
#if __ANDROID__
      neuron_adapter_api.CreateCompilation(model);
#else
      neuron_adapter_api.CreateCompilation(model, compile_options);
#endif
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

#if __ANDROID__
  if (!compile_options.empty()) {
    if (auto status =
            neuron_adapter_api.api().compilation_set_optimization_string(
                compilation->get(), compile_options.c_str());
        status != NEURON_NO_ERROR) {
      LITERT_LOG(LITERT_INFO,
                 "NeuronCompilation_setOptimizationString failed with error %d",
                 status);
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to set optimization string");
    }
  }
#endif

  if (auto status =
          neuron_adapter_api.api().compilation_finish(compilation->get());
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_INFO, "NeuronCompilation_finish failed with error %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to finish compilation");
  }

  return compilation;
}

}  // namespace litert::mediatek
