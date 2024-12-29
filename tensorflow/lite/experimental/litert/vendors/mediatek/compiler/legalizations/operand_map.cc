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
#include <utility>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter.h"

namespace litert::mediatek {

namespace {

class OperandType : public NeuronOperandType {
 public:
  static Expected<OperandType> Create(const Tensor& t) {
    auto ranked_tensor_type = t.RankedTensorType();
    if (!ranked_tensor_type) {
      return ranked_tensor_type.Error();
    }

    auto tensor_dimensions = ranked_tensor_type->Layout().Dimensions();
    std::vector<uint32_t> mtk_dimensions;
    mtk_dimensions.reserve(tensor_dimensions.size());
    std::copy(tensor_dimensions.begin(), tensor_dimensions.end(),
              std::back_inserter(mtk_dimensions));

    int32_t mtk_type;
    switch (ranked_tensor_type->ElementType()) {
      case ElementType::Float32:
        mtk_type = NEURON_TENSOR_FLOAT32;
        break;
      case ElementType::Int32:
        mtk_type = NEURON_TENSOR_INT32;
        break;
      default:
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Unsupported element type");
    }

    return OperandType(mtk_type, std::move(mtk_dimensions));
  }

  OperandType(const OperandType&) = delete;

  OperandType(OperandType&& other) : dimensions_(std::move(other.dimensions_)) {
    // Copy all the scalar fields from other.
    *static_cast<NeuronOperandType*>(this) =
        *static_cast<NeuronOperandType*>(&other);
    // Reset the pointer fields by using own data.
    dimensions = dimensions_.data();
  };

  OperandType& operator=(const OperandType&) = delete;
  OperandType& operator=(OperandType&& other) = delete;

 private:
  explicit OperandType(int32_t mtk_type, std::vector<uint32_t>&& mtk_dimensions)
      : dimensions_(std::move(mtk_dimensions)) {
    this->type = mtk_type;
    this->dimensionCount = dimensions_.size();
    this->dimensions = dimensions_.data();
  };

  std::vector<uint32_t> dimensions_;
};

}  // namespace

// /////////////////////////////////////////////////////////////////////////////

Expected<uint32_t> OperandMap::Register(const NeuronOperandType& operand_type) {
  if (neuron_adapter_.api().model_add_operand(model_, &operand_type) !=
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

  if (t.HasWeights()) {
    auto weights = t.Weights().Bytes();
    if (neuron_adapter_.api().model_set_operand_value(
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
