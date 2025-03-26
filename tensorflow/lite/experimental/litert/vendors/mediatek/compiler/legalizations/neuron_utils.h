// Copyright (c) 2025 MediaTek Inc.
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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_NEURON_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_NEURON_UTILS_H_

#include <cstdint>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {
using NeuronTensorType = int32_t;
using NeuronReturnCode = int32_t;

Expected<NeuronTensorType> GetNeuronTensorType(const Tensor& t);

Expected<uint32_t> GetNeuronDataSize(NeuronTensorType type);

Expected<bool> IsQuantizedType(NeuronTensorType type);

NeuronReturnCode ModelAddOperation(const NeuronAdapterApi& api,
                                   NeuronModel* model, NeuronOperationType type,
                                   std::vector<uint32_t> input,
                                   std::vector<uint32_t> output);

}  // namespace litert::mediatek

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_NEURON_UTILS_H_
