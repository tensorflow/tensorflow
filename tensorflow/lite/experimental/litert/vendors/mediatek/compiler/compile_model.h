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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_COMPILE_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_COMPILE_MODEL_H_

#include <optional>
#include <string>

#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter.h"

namespace litert::mediatek {

Expected<NeuronCompilationPtr> CompileModel(
    const NeuronAdapter& neuron_adapter, NeuronModel* model,
    std::optional<std::string> soc_model);

}  // namespace litert::mediatek

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_COMPILE_MODEL_H_
