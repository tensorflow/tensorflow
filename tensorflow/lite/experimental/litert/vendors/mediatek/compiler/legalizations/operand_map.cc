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

#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/operand_map.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

Expected<uint32_t> OperandMap::Register(const NeuronOperandType& operand_type) {
  if (neuron_adapter_api_.api().model_add_operand(model_, &operand_type) !=
      NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to register model operand");
  }
  return AllocateOperandIndex();
}

Expected<uint32_t> OperandMap::Register(const Tensor& t) {
  auto operand_type = OperandType::Create(t);
  if (!operand_type) {
    return operand_type.Error();
  }

  auto operand_index =
      Register(static_cast<const NeuronOperandType&>(*operand_type));
  if (!operand_index) {
    return operand_index.Error();
  }
  LITERT_LOG(LITERT_INFO, "\nOperandIndex: %d", operand_index.Value());
  operand_type->Info();

  if (t.HasWeights()) {
    auto weights = t.Weights().Bytes();
    if (t.QTypeId() == kLiteRtQuantizationPerChannel) {
      auto quant_param = operand_type->GetPerChannelQuantParams().Value();
      if (neuron_adapter_api_.api().model_set_symm_per_channel_quant_params(
              model_, *operand_index, &quant_param) != NEURON_NO_ERROR) {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Failed to set param of per channel quant params");
      }
    }
    if (neuron_adapter_api_.api().model_set_operand_value(
            model_, *operand_index, weights.data(), weights.size()) !=
        NEURON_NO_ERROR) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to set value of tensor weights");
    }
  }

  map_[t.Get()] = *operand_index;
  return *operand_index;
}

}  // namespace litert::mediatek
