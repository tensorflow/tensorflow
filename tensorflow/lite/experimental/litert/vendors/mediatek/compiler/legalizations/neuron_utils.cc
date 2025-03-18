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

#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/neuron_utils.h"

namespace litert::mediatek {

Expected<NeuronTensorType> GetNeuronTensorType(const Tensor& t) {
  auto ranked_tensor_type = t.RankedTensorType();
  if (!ranked_tensor_type) {
    return ranked_tensor_type.Error();
  }

  int32_t mtk_type;
  switch (ranked_tensor_type->ElementType()) {
    case ElementType::Float32:
      mtk_type = NEURON_TENSOR_FLOAT32;
      break;
    case ElementType::Float16:
      mtk_type = NEURON_TENSOR_FLOAT16;
      break;
    case ElementType::Int32:
      mtk_type = NEURON_TENSOR_INT32;
      break;
    case ElementType::Int16:
      if (t.QTypeId() == kLiteRtQuantizationPerTensor) {
        mtk_type = NEURON_TENSOR_QUANT16_SYMM;
      } else {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Int16 is not supported.");
      }
      break;
    case ElementType::Int8:
      if (t.QTypeId() == kLiteRtQuantizationPerTensor) {
        mtk_type = NEURON_TENSOR_QUANT8_SYMM;
      } else if (t.QTypeId() == kLiteRtQuantizationPerChannel) {
        mtk_type = NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL;
      } else {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Int8 is not supported.");
      }
      break;
    default:
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   absl::StrFormat("Unsupported element type: %d",
                                   ranked_tensor_type->ElementType()));
  }
  return mtk_type;
}

Expected<uint32_t> GetNeuronDataSize(NeuronTensorType type) {
  switch (type) {
    case NEURON_FLOAT32:
    case NEURON_TENSOR_FLOAT32:
    case NEURON_INT32:
    case NEURON_TENSOR_INT32:
      return 4;
    case NEURON_FLOAT16:
    case NEURON_TENSOR_FLOAT16:
    case NEURON_EXT_TENSOR_QUANT16_ASYMM_SIGNED:
      return 2;
    case NEURON_BOOL:
    case NEURON_TENSOR_BOOL8:
    case NEURON_TENSOR_QUANT8_ASYMM:
    case NEURON_TENSOR_QUANT8_ASYMM_SIGNED:
      return 1;
    default:
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Get Data Size fail for Neuron Type");
  }
  return Error(kLiteRtStatusErrorRuntimeFailure, "Unexpected neuron type");
}

Expected<bool> IsQuantizedType(NeuronTensorType type) {
  switch (type) {
    case NEURON_TENSOR_QUANT16_SYMM:
    case NEURON_TENSOR_QUANT16_ASYMM:
    case NEURON_TENSOR_QUANT8_ASYMM:
    case NEURON_TENSOR_QUANT8_ASYMM_SIGNED:
      return true;
  }
  return false;
}

NeuronReturnCode ModelAddOperation(const NeuronAdapterApi& api,
                                   NeuronModel* model, NeuronOperationType type,
                                   std::vector<uint32_t> input,
                                   std::vector<uint32_t> output) {
  return api.api().model_add_operation(model, type, input.size(), input.data(),
                                       output.size(), output.data());
};

}  // namespace litert::mediatek
