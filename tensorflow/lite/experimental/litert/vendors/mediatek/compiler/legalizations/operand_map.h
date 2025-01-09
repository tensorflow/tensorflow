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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_OPERAND_MAP_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_OPERAND_MAP_H_

#include <cstdint>
#include <map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter.h"

namespace litert::mediatek {

// This class takes care of registering Tensors and scalars with a given
// NeuronModel and returing their "operand index", which is how the MTK SDK
// handles them.
class OperandMap {
 public:
  OperandMap(const NeuronAdapter& neuron_adapter, NeuronModel* model)
      : neuron_adapter_(neuron_adapter), model_(model) {}

  // Add a scalar operand to the model.
  Expected<uint32_t> AddScalarBool(bool value) {
    return AddScalar(NEURON_BOOL, value);
  }
  Expected<uint32_t> AddScalarInt32(int32_t value) {
    return AddScalar(NEURON_INT32, value);
  }
  Expected<uint32_t> AddScalarFloat32(float value) {
    return AddScalar(NEURON_FLOAT32, value);
  }

  // Find the operand index for a given tensor and, if not done already, add the
  // tensor as an operand in the model.
  Expected<uint32_t> GetOperandIndex(const Tensor& t) {
    auto i = map_.find(t.Get());
    if (i != map_.end()) {
      return i->second;
    } else {
      return Register(t);
    }
  }

 private:
  Expected<uint32_t> Register(const Tensor& t);
  Expected<uint32_t> Register(const NeuronOperandType& operand_type);
  uint32_t AllocateOperandIndex() { return next_operand_index_++; }

  template <typename T>
  Expected<uint32_t> AddScalar(int32_t mtk_type, T value) {
    const NeuronOperandType scalar_type = {
        .type = mtk_type,
        .dimensionCount = 0,
        .dimensions = nullptr,
    };
    auto operand_index = Register(scalar_type);
    if (!operand_index) {
      return operand_index.Error();
    }
    if (neuron_adapter_.api().model_set_operand_value(
            model_, *operand_index, &value, sizeof(value)) != NEURON_NO_ERROR) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to set value of scalar operand");
    }
    return operand_index;
  }

  const NeuronAdapter& neuron_adapter_;
  NeuronModel* model_;
  int next_operand_index_ = 0;
  absl::flat_hash_map<LiteRtTensor, uint32_t> map_;
};

}  // namespace litert::mediatek

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_OPERAND_MAP_H_
