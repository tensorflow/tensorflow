/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <memory>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool GetQuantizedDataTypeNumericalRange(ArrayDataType data_type,
                                        double* out_min_value,
                                        double* out_max_value) {
  switch (data_type) {
    case ArrayDataType::kUint8:
      *out_min_value = 0;
      *out_max_value = 255;
      return true;
    case ArrayDataType::kInt16:
      *out_min_value = -32768;
      *out_max_value = 32767;
      return true;
    default:
      return false;
  }
}

ArrayDataType GetQuantizedDataType(const Array& array,
                                   ArrayDataType default_type) {
  switch (array.final_data_type) {
    case ArrayDataType::kInt8:
    case ArrayDataType::kUint8:
    case ArrayDataType::kInt16:
    case ArrayDataType::kUint16:
    case ArrayDataType::kInt32:
    case ArrayDataType::kUint32:
    case ArrayDataType::kInt64:
    case ArrayDataType::kUint64:
      return array.final_data_type;
    case ArrayDataType::kFloat:
    case ArrayDataType::kNone:
      return default_type;
    default:
      LOG(FATAL) << "Unhandled final quantization type "
                 << static_cast<int>(array.final_data_type);
  }
}

void GetQuantizationParams(ArrayDataType data_type, const MinMax& minmax,
                           QuantizationParams* quantization_params) {
  switch (data_type) {
    case ArrayDataType::kInt8:
      GetQuantizationParamsFromMinMax<ArrayDataType::kInt8>(
          minmax, quantization_params);
      break;
    case ArrayDataType::kUint8:
      GetQuantizationParamsFromMinMax<ArrayDataType::kUint8>(
          minmax, quantization_params);
      break;
    case ArrayDataType::kInt16:
      GetQuantizationParamsFromMinMax<ArrayDataType::kInt16>(
          minmax, quantization_params);
      break;
    case ArrayDataType::kUint16:
      GetQuantizationParamsFromMinMax<ArrayDataType::kUint16>(
          minmax, quantization_params);
      break;
    case ArrayDataType::kInt32:
      GetQuantizationParamsFromMinMax<ArrayDataType::kInt32>(
          minmax, quantization_params);
      break;
    case ArrayDataType::kUint32:
      GetQuantizationParamsFromMinMax<ArrayDataType::kUint32>(
          minmax, quantization_params);
      break;
    case ArrayDataType::kInt64:
      GetQuantizationParamsFromMinMax<ArrayDataType::kInt64>(
          minmax, quantization_params);
      break;
    case ArrayDataType::kUint64:
      GetQuantizationParamsFromMinMax<ArrayDataType::kUint64>(
          minmax, quantization_params);
      break;
    case ArrayDataType::kFloat:
    case ArrayDataType::kNone:
    default:
      LOG(FATAL) << "Unhandled final quantization type "
                 << static_cast<int>(data_type);
  }
}

bool IsArrayQuantizedRangeSubset(GraphTransformation* transformation,
                                 const Array& array, double clamp_min,
                                 double clamp_max) {
  ArrayDataType quantized_data_type =
      GetQuantizedDataType(array, array.data_type);
  if (quantized_data_type == ArrayDataType::kNone ||
      quantized_data_type == ArrayDataType::kFloat) {
    // The array is not (or never will be) quantized.
    return false;
  }

  QuantizationParams quantization_params;
  if (!array.quantization_params) {
    if (!array.minmax) {
      transformation->AddMessageF("No quantization params and no minmax");
      return false;
    } else {
      // Work around cases where we are asking for this prior to the Quantize
      // transformation having added the quantization_params.
      GetQuantizationParams(quantized_data_type, *array.minmax,
                            &quantization_params);
      transformation->AddMessageF(
          "No quantization params - infering from data type %s with minmax "
          "%g,%g as zero_point=%g, scale=%g",
          ArrayDataTypeName(quantized_data_type), array.minmax->min,
          array.minmax->max, quantization_params.zero_point,
          quantization_params.scale);
    }
  } else {
    quantization_params = array.GetQuantizationParams();
  }

  double quantized_min, quantized_max;
  CHECK(GetQuantizedDataTypeNumericalRange(quantized_data_type, &quantized_min,
                                           &quantized_max))
      << "Type is not quantized";

  bool has_nontrivial_min_bound = false;
  bool has_nontrivial_max_bound = false;

  double lowest_representable_output =
      (quantized_min - quantization_params.zero_point) *
      quantization_params.scale;
  if (lowest_representable_output < clamp_min) {
    has_nontrivial_min_bound = true;
    transformation->AddMessageF(
        "Quantized activation function is not trivial: "
        "the lowest representable output value %g"
        " less than the clamp min bound %g.",
        lowest_representable_output, clamp_min);
  }

  double highest_representable_output =
      (quantized_max - quantization_params.zero_point) *
      quantization_params.scale;
  if (highest_representable_output > clamp_max) {
    has_nontrivial_max_bound = true;
    transformation->AddMessageF(
        "Quantized activation function is not trivial: "
        "the highest representable output value %g"
        " is greater than the clamp max bound %g.",
        highest_representable_output, clamp_max);
  }

  return !has_nontrivial_min_bound && !has_nontrivial_max_bound;
}

}  // namespace toco
