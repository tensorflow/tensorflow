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

#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
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
  auto tfl_q_params = GetPerTensorQparams(tfl_quantization);
  if (!tfl_q_params) return false;
  return litert_quantization.zero_point == tfl_q_params->first &&
         litert_quantization.scale == tfl_q_params->second;
}

template <class LiteRtTenzorType>
bool EqualsFbTensorTypeDetail(LiteRtTenzorType litert_tensor_type,
                              const TflTensor& tfl_tensor) {
  return false;
}

template <>
bool EqualsFbTensorTypeDetail<RankedTensorType>(
    RankedTensorType litert_tensor_type, const TflTensor& tfl_tensor) {
  auto tfl_type_info = GetStaticTensorTypeInfo(tfl_tensor);
  if (!tfl_type_info) {
    return false;
  }
  const bool element_type_eq =
      MapElementType(tfl_type_info->first) ==
      static_cast<LiteRtElementType>(litert_tensor_type.ElementType());
  const bool shape_eq =
      tfl_type_info->second == litert_tensor_type.Layout().Dimensions();
  return element_type_eq && shape_eq;
}

}  // namespace

// Compare q-params within litert tensor to flatbuffer q-params for having the
// same type and values.
bool EqualsFbQuantization(const Tensor& litert_tensor,
                          const TflQuantization* tfl_quantization) {
  switch (litert_tensor.QTypeId()) {
    case kLiteRtQuantizationPerTensor:
      return EqualsFbQuantizationDetail(litert_tensor.PerTensorQuantization(),
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
bool EqualsFbTensorType(const Tensor& litert_tensor,
                        const TflTensor& tfl_tensor) {
  switch (litert_tensor.TypeId()) {
    case kLiteRtRankedTensorType:
      return EqualsFbTensorTypeDetail(litert_tensor.RankedTensorType(),
                                      tfl_tensor);
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
      const auto& litert_tensor = litert_tensors.at(i);

      const auto type_eq = EqualsFbTensorType(litert_tensor, fb_tensor);
      const auto quant_eq =
          EqualsFbQuantization(litert_tensor, fb_tensor.quantization.get());

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
