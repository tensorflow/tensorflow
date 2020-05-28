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

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool InferQuantizedDataTypeFromFakeQuant(
    const FakeQuantOperator& op, ArrayDataType* out_quantized_data_type) {
  if (op.num_bits <= 8) {
    *out_quantized_data_type = ArrayDataType::kUint8;
    return true;
  } else if (op.num_bits <= 16) {
    *out_quantized_data_type = ArrayDataType::kInt16;
    return true;
  } else {
    *out_quantized_data_type = ArrayDataType::kNone;
    return false;
  }
}

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

template <ArrayDataType A>
void ChooseQuantizationParamsForArrayAndQuantizedDataType(
    const Array& array, QuantizationParams* quantization_params) {
  *quantization_params = ::tflite::ChooseQuantizationParams<DataType<A>>(
      array.minmax->min, array.minmax->max, array.narrow_range);
}

void ChooseQuantizationParamsForArrayAndQuantizedDataType(
    const Array& array, ArrayDataType quantized_data_type,
    QuantizationParams* quantization_params) {
  switch (quantized_data_type) {
    case ArrayDataType::kInt8:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kInt8>(array, quantization_params);
      break;
    case ArrayDataType::kUint8:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kUint8>(array, quantization_params);
      break;
    case ArrayDataType::kInt16:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kInt16>(array, quantization_params);
      break;
    case ArrayDataType::kUint16:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kUint16>(array, quantization_params);
      break;
    case ArrayDataType::kInt32:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kInt32>(array, quantization_params);
      break;
    case ArrayDataType::kUint32:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kUint32>(array, quantization_params);
      break;
    case ArrayDataType::kInt64:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kInt64>(array, quantization_params);
      break;
    case ArrayDataType::kUint64:
      ChooseQuantizationParamsForArrayAndQuantizedDataType<
          ArrayDataType::kUint64>(array, quantization_params);
      break;
    case ArrayDataType::kFloat:
    case ArrayDataType::kComplex64:
    case ArrayDataType::kNone:
    default:
      LOG(FATAL) << "Unhandled final quantization type "
                 << static_cast<int>(quantized_data_type);
  }
}

namespace {

template <ArrayDataType A>
std::unique_ptr<GenericBuffer> QuantizeBuffer(
    const Array& array, const QuantizationParams& quantization_params) {
  const GenericBuffer& buffer = *array.buffer;
  const auto inverse_scale = 1. / quantization_params.scale;
  CHECK(buffer.type == ArrayDataType::kFloat);
  const auto& float_buffer =
      static_cast<const Buffer<ArrayDataType::kFloat>&>(buffer);
  auto* quantized_buffer = new Buffer<A>;
  quantized_buffer->data.resize(float_buffer.data.size());
  for (std::size_t i = 0; i < float_buffer.data.size(); i++) {
    const float src_val = float_buffer.data[i];
    double scaled_val;  // Astonishingly, using 'float' degrades accuracy just
                        // enough to make a few tests fail!
    if (quantization_params.scale == 0) {
      CHECK_EQ(src_val, 0) << "The quantization scale for this array is 0, "
                           << "so all its values should be 0.";
      scaled_val = quantization_params.zero_point;
    } else {
      scaled_val = quantization_params.zero_point + inverse_scale * src_val;
    }
    auto integer_val = tflite::SafeCast<DataType<A>>(std::round(scaled_val));
    // In addition to its effect on the choice of quantization params upstream
    // of here, narrow_range also means nudge the min quantized value by +1,
    // so e.g. uint8 values get constrained to [1, 255].
    if (integer_val == std::numeric_limits<DataType<A>>::min() &&
        array.narrow_range) {
      integer_val++;
    }
    quantized_buffer->data[i] = integer_val;
  }
  return std::unique_ptr<GenericBuffer>(quantized_buffer);
}

template <ArrayDataType A>
void QuantizeArray(GraphTransformation* transformation, Model* model,
                   const string& name,
                   const QuantizationParams& quantization_params) {
  auto& array = model->GetArray(name);
  CHECK(array.data_type == ArrayDataType::kFloat);
  CHECK(!array.quantization_params);
  array.GetOrCreateQuantizationParams() = quantization_params;
  if (array.buffer) {
    array.buffer = QuantizeBuffer<A>(array, quantization_params);
  }
  array.data_type = A;
  array.final_data_type = A;
  transformation->AddMessageF(
      "Quantized array %s to %s zero_point=%g, scale=%g", name,
      ArrayDataTypeName(array.data_type), quantization_params.zero_point,
      quantization_params.scale);
}

}  // namespace

void QuantizeArray(GraphTransformation* transformation, Model* model,
                   const string& name, ArrayDataType quantized_data_type,
                   const QuantizationParams& quantization_params) {
  ArrayDataType adjusted_data_type = quantized_data_type;
  auto& array = model->GetArray(name);
  if (array.final_data_type == ArrayDataType::kInt16) {
    adjusted_data_type = array.final_data_type;
  }

  switch (adjusted_data_type) {
    case ArrayDataType::kUint8:
      return QuantizeArray<ArrayDataType::kUint8>(transformation, model, name,
                                                  quantization_params);
    case ArrayDataType::kInt16:
      return QuantizeArray<ArrayDataType::kInt16>(transformation, model, name,
                                                  quantization_params);
    case ArrayDataType::kInt32:
      return QuantizeArray<ArrayDataType::kInt32>(transformation, model, name,
                                                  quantization_params);
    default:
      LOG(FATAL) << "Unhandled case.";
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
      ChooseQuantizationParamsForArrayAndQuantizedDataType(
          array, quantized_data_type, &quantization_params);
      transformation->AddMessageF(
          "No quantization params - inferring from data type %s with minmax "
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
