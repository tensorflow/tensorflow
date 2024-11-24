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

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_util.h"
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

template <class LiteRtTenzorType>
bool EqualsFbTensorTypeDetail(LiteRtTenzorType litert_tensor_type,
                              const TflTensorType& tfl_tensor) {
  return false;
}

template <>
bool EqualsFbTensorTypeDetail<LiteRtRankedTensorType>(
    LiteRtRankedTensorType litert_tensor_type,
    const TflTensorType& tfl_tensor_type) {
  auto tfl_shape = AsStaticShape(tfl_tensor_type.second);
  if (!tfl_shape) {
    // Not supported yet.
    return false;
  }

  const bool element_type_eq =
      MapElementType(tfl_tensor_type.first) ==
      static_cast<LiteRtElementType>(litert_tensor_type.element_type);
  auto& layout = litert_tensor_type.layout;
  const bool shape_eq =
      AllZip(*tfl_shape, absl::MakeConstSpan(layout.dimensions, layout.rank),
             [](auto l, auto r) { return l == r; });

  return element_type_eq && shape_eq;
}

}  // namespace

bool EqualsFbQuantization(const Quantization& litert_quantization,
                          const TflQuantization* tfl_quantization) {
  switch (litert_quantization.first) {
    case kLiteRtQuantizationPerTensor:
      return EqualsFbQuantizationDetail(litert_quantization.second.per_tensor,
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
      // Not implemented yet.
      return false;
  }
}

// Compare litert op to flatbuffer op along with their input/output tensors
// types and quantization. Takes a callback to lookup tfl tensors the indices
// within the tfl op.
bool EqualsFbOp(const Op& litert_op, const TflOp& tfl_op,
                GetTflTensor get_tfl_tensor) {
  auto litert_inputs = litert_op.Inputs();
  auto litert_outputs = litert_op.Outputs();

  auto check_tensors = [&](auto& litert_tensors, auto& tfl_tensors) {
    if (litert_tensors.size() != tfl_tensors.size()) {
      return false;
    }

    for (auto i = 0; i < litert_tensors.size(); ++i) {
      const auto& fb_tensor = get_tfl_tensor(tfl_tensors.at(i)).get();
      const auto& litert_tensor = *litert_tensors.at(i).Get();

      const auto type_eq =
          EqualsFbTensorType({litert_tensor.type_id, litert_tensor.type_detail},
                             {fb_tensor.type, TflShapeInfo(fb_tensor)});
      const auto quant_eq = EqualsFbQuantization(
          {litert_tensor.q_type_id, litert_tensor.q_type_detail},
          fb_tensor.quantization.get());

      if (!type_eq || !quant_eq) {
        return false;
      }
    }

    return true;
  };

  return check_tensors(litert_inputs, tfl_op.inputs) &&
         check_tensors(litert_outputs, tfl_op.outputs);
}

}  // namespace litert::internal
