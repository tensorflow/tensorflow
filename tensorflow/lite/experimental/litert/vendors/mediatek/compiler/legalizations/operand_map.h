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
#include <numeric>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/neuron_utils.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

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

    // tensor type dimensions couldn't be zero.
    if (mtk_dimensions.size() == 0) {
      mtk_dimensions = {
          1,
      };
    }

    // BlockWise Quantize is not supported now.
    if (t.HasQuantization() && t.QTypeId() == kLiteRtQuantizationBlockWise) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Doesn't support BlockWise quantize now");
    }

    auto mtk_type = GetNeuronTensorType(t);
    if (!mtk_type) {
      return mtk_type.Error();
    }

    if (t.QTypeId() == kLiteRtQuantizationPerTensor) {
      auto quant_info = t.PerTensorQuantization();
      LITERT_LOG(LITERT_INFO, "zeroPoint: %d, scale: %f", quant_info.zero_point,
                 quant_info.scale);
      return OperandType(*mtk_type, std::move(mtk_dimensions), quant_info.scale,
                         quant_info.zero_point, std::nullopt);
    } else if (t.QTypeId() == kLiteRtQuantizationPerChannel) {
      auto quant_info = t.PerChannelQuantization();
      NeuronSymmPerChannelQuantParams params;
      params.scaleCount = quant_info.num_channels;
      params.scales = quant_info.scales;
      params.channelDim = quant_info.quantized_dimension;
      LITERT_LOG(LITERT_INFO, "quantized_dimension: %d",
                 quant_info.quantized_dimension);
      LITERT_LOG(LITERT_INFO, "params.channelDim: %d", params.channelDim);
      return OperandType(*mtk_type, std::move(mtk_dimensions), 0, 0, params);
    } else {
      return OperandType(*mtk_type, std::move(mtk_dimensions), /*scale*/ 0,
                         /*zero_point*/ 0, std::nullopt);
    }
  }

  void Info() {
    std::string vector = "[";
    for (int i = 0; i < dimensionCount; i++) {
      vector += std::to_string(dimensions_[i]);
      vector += ",";
    }
    vector += "]";
    LITERT_LOG(LITERT_INFO,
               "\n[Type] %d"
               "\n[zeroPoint]%d"
               "\n[scale]%f"
               "\n[dimensionCount]%u"
               "\n[dimensions]%s\n",
               type, zeroPoint, scale, dimensionCount, vector.c_str());
  }

  OperandType(const OperandType&) = delete;

  OperandType(OperandType&& other)
      : dimensions_(std::move(other.dimensions_)),
        neuron_per_channel_params_(other.neuron_per_channel_params_) {
    // Copy all the scalar fields from other.
    *static_cast<NeuronOperandType*>(this) =
        *static_cast<NeuronOperandType*>(&other);
    // Reset the pointer fields by using own data.
    dimensions = dimensions_.data();
  };

  Expected<void> Reshape(std::vector<uint32_t>& shape) {
    auto elements = GetElementCount();
    if (elements != std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<uint32_t>())) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "the elements is not the same");
    }
    this->dimensions_ = shape;
    this->dimensionCount = this->dimensions_.size();
    this->dimensions = this->dimensions_.data();
    return {};
  }

  Expected<NeuronSymmPerChannelQuantParams> GetPerChannelQuantParams() {
    if (!neuron_per_channel_params_.has_value()) {
      return Error(kLiteRtStatusErrorRuntimeFailure, "No quant param is set");
    }
    return neuron_per_channel_params_.value();
  }

  int32_t GetNeuronType() const { return this->type; }

  std::vector<uint32_t> GetDimension() { return this->dimensions_; }

  uint32_t GetElementCount() {
    return std::accumulate(dimensions_.begin(), dimensions_.end(), 1,
                           std::multiplies<uint32_t>());
  }

  uint32_t GetRank() { return this->dimensions_.size(); }

  OperandType& operator=(const OperandType&) = delete;
  OperandType& operator=(OperandType&& other) = delete;

 private:
  explicit OperandType(int32_t mtk_type, std::vector<uint32_t>&& mtk_dimensions,
                       float scale, int32_t zero_point,
                       std::optional<NeuronSymmPerChannelQuantParams> pararms)
      : dimensions_(std::move(mtk_dimensions)),
        neuron_per_channel_params_(pararms) {
    this->scale = scale;
    this->zeroPoint = zero_point;
    this->type = mtk_type;
    this->dimensionCount = dimensions_.size();
    this->dimensions = dimensions_.data();
  }

  std::vector<uint32_t> dimensions_;

  std::optional<NeuronSymmPerChannelQuantParams> neuron_per_channel_params_ =
      std::nullopt;
};

// This class takes care of registering Tensors and scalars with a given
// NeuronModel and returing their "operand index", which is how the MTK SDK
// handles them.
class OperandMap {
 public:
  OperandMap(const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model)
      : neuron_adapter_api_(neuron_adapter_api), model_(model) {}

  // Add a scalar operand to the model.
  Expected<uint32_t> AddScalarBool(bool value) {
    return AddScalar(NEURON_BOOL, value);
  }
  Expected<uint32_t> AddScalarInt32(int32_t value) {
    return AddScalar(NEURON_INT32, value);
  }
  Expected<uint32_t> AddScalarUInt32(uint32_t value) {
    return AddScalar(NEURON_UINT32, value);
  }
  Expected<uint32_t> AddScalarFloat32(float value) {
    return AddScalar(NEURON_FLOAT32, value);
  }

  // Add a tensor operand to the model
  Expected<uint32_t> AddTensorInt32(std::vector<uint32_t>& shape,
                                    const void* data, const size_t data_size) {
    return AddTensor(NEURON_TENSOR_INT32, shape, data, data_size);
  }

  // Add a tensor operand to the model
  Expected<uint32_t> AddTensorByType(int mtk_type, std::vector<uint32_t>& shape,
                                     const void* data, const size_t data_size) {
    return AddTensor(mtk_type, shape, data, data_size);
  }

  Expected<uint32_t> AddOperand(const NeuronOperandType& operand) {
    return Register(operand);
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
    if (neuron_adapter_api_.api().model_set_operand_value(
            model_, *operand_index, &value, sizeof(value)) != NEURON_NO_ERROR) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to set value of scalar operand");
    }
    return operand_index;
  }

  Expected<uint32_t> AddTensor(int32_t mtk_type,
                               const std::vector<uint32_t>& shape,
                               const void* data, const size_t data_size) {
    const NeuronOperandType scalar_type = {
        .type = mtk_type,
        .dimensionCount = (uint32_t)shape.size(),
        .dimensions = shape.data(),
    };
    auto operand_index = Register(scalar_type);
    if (!operand_index) {
      return operand_index.Error();
    }
    if (neuron_adapter_api_.api().model_set_operand_value(
            model_, *operand_index, data, data_size) != NEURON_NO_ERROR) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to set value of tensor operand");
    }
    return operand_index;
  }

  const NeuronAdapterApi& neuron_adapter_api_;
  NeuronModel* model_;
  int next_operand_index_ = 0;
  absl::flat_hash_map<LiteRtTensor, uint32_t> map_;
};

}  // namespace litert::mediatek

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_OPERAND_MAP_H_
