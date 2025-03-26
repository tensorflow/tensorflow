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

#include "tensorflow/lite/experimental/litert/core/model/model_file_test_util.h"

#include <algorithm>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/core/model/flatbuffer_to_litert.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"

namespace litert::internal {

namespace {

template <class LiteRtQType>
bool EqualsFbQuantizationDetail(LiteRtQType litert_quantization,
                                const TflQuantization* tfl_quantization) {
  return false;
}

template <>
bool EqualsFbQuantizationDetail<LiteRtQuantizationPerTensor>(
    LiteRtQuantizationPerTensor litert_quantization,
    const TflQuantization* tfl_quantization) {
  auto tfl_q_params = AsPerTensorQparams(tfl_quantization);
  if (!tfl_q_params) return false;
  return litert_quantization.zero_point == tfl_q_params->first &&
         litert_quantization.scale == tfl_q_params->second;
}

template <>
bool EqualsFbQuantizationDetail<LiteRtQuantizationPerChannel>(
    LiteRtQuantizationPerChannel litert_quantization,
    const TflQuantization* tfl_quantization) {
  auto tfl_q_params = AsPerChannelQparams(tfl_quantization);
  if (!tfl_q_params) return false;
  const auto& [quantized_dimension, num_channels, zero_points, scales] =
      *tfl_q_params;
  const auto qd_eq =
      litert_quantization.quantized_dimension == quantized_dimension;
  const auto num_chan_eq = litert_quantization.num_channels == num_channels;
  const auto zeros_eq = std::equal(zero_points.begin(), zero_points.end(),
                                   litert_quantization.zero_points);
  const auto scales_eq =
      std::equal(scales.begin(), scales.end(), litert_quantization.scales);
  return qd_eq && num_chan_eq && zeros_eq && scales_eq;
}
template <class LiteRtTenzorType>
bool EqualsFbTensorTypeDetail(LiteRtTenzorType litert_tensor_type,
                              const TflTensorType& tfl_tensor) {
  LITERT_LOG(LITERT_ERROR, "LiteRtTensorType not supported");
  return false;
}

template <>
bool EqualsFbTensorTypeDetail<LiteRtRankedTensorType>(
    LiteRtRankedTensorType litert_tensor_type,
    const TflTensorType& tfl_tensor_type) {
  auto tfl_shape = AsDynamicShape(tfl_tensor_type.second);
  if (!tfl_shape) {
    LITERT_LOG(LITERT_ERROR, "Not ranked shape");
    return false;
  }

  if (MapElementType(tfl_tensor_type.first) !=
      static_cast<LiteRtElementType>(litert_tensor_type.element_type)) {
    LITERT_LOG(LITERT_ERROR, "Element type not equal");
    return false;
  }

  auto same_or_both_dyn = [](auto l, auto r) {
    const auto same_static = l >= 0 && l == r;
    const auto both_dyn = l < 0 && r < 0;
    return same_static || both_dyn;
  };

  auto& layout = litert_tensor_type.layout;
  const bool shape_eq =
      AllZip(*tfl_shape, absl::MakeConstSpan(layout.dimensions, layout.rank),
             same_or_both_dyn);
  if (!shape_eq) {
    LITERT_LOG(LITERT_ERROR, "Shapes are not equal");
    return false;
  }

  return true;
}

}  // namespace

bool EqualsFbQuantization(const Quantization& litert_quantization,
                          const TflQuantization* tfl_quantization) {
  switch (litert_quantization.first) {
    case kLiteRtQuantizationPerTensor:
      return EqualsFbQuantizationDetail(litert_quantization.second.per_tensor,
                                        tfl_quantization);
    case kLiteRtQuantizationPerChannel:
      return EqualsFbQuantizationDetail(litert_quantization.second.per_channel,
                                        tfl_quantization);
    case kLiteRtQuantizationNone:
      return !IsQuantized(tfl_quantization);
    default:
      // Not implemented yet.
      return false;
  }
}

// Compare tensor type within litert tensor to the type within flatbuffer
// tensor.
bool EqualsFbTensorType(const TensorType& litert_tensor_type,
                        const TflTensorType& tfl_tensor_type) {
  switch (litert_tensor_type.first) {
    case kLiteRtRankedTensorType:
      return EqualsFbTensorTypeDetail(
          litert_tensor_type.second.ranked_tensor_type, tfl_tensor_type);
    default:
      LITERT_LOG(LITERT_ERROR, "Tensor kind not supported");
      // Not implemented yet.
      return false;
  }
}

bool EqualsFbTensor(const LiteRtTensorT& litert_tensor,
                    const TflTensor& tfl_tensor) {
  if (!EqualsFbTensorType(litert_tensor.Type(),
                          {tfl_tensor.type, TflShapeInfo(tfl_tensor)})) {
    LITERT_LOG(LITERT_ERROR, "Tensor not same type");
    return false;
  }

  if (!EqualsFbQuantization(litert_tensor.Qparams(),
                            tfl_tensor.quantization.get())) {
    LITERT_LOG(LITERT_ERROR, "Tensor not same quantization");
    return false;
  }

  return true;
}

bool EqualsFbOp(const LiteRtOpT& litert_op, const TflOp& tfl_op,
                GetTflTensor get_tfl_tensor) {
  auto check_tensors = [&](auto& litert_tensors, auto& tfl_tensors) {
    if (litert_tensors.size() != tfl_tensors.size()) {
      LITERT_LOG(LITERT_ERROR, "Tensors not same size");
      return false;
    }

    for (auto i = 0; i < litert_tensors.size(); ++i) {
      const auto& fb_tensor = get_tfl_tensor(tfl_tensors.at(i)).get();
      const auto& litert_tensor = *litert_tensors.at(i);

      if (!EqualsFbTensor(litert_tensor, fb_tensor)) {
        LITERT_LOG(LITERT_ERROR, "Tensor %d not same", i);
        return false;
      }
    }

    return true;
  };

  return check_tensors(litert_op.Inputs(), tfl_op.inputs) &&
         check_tensors(litert_op.Outputs(), tfl_op.outputs);
}

}  // namespace litert::internal
