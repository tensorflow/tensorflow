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
#ifndef TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_QUANTIZATION_UTIL_H_
#define TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_QUANTIZATION_UTIL_H_

#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"

namespace toco {

// Gets the target quantized data type of an array based on the fake quant op.
// For example, if the num_bits is 8 the data type will be kUint8.
bool InferQuantizedDataTypeFromFakeQuant(
    const FakeQuantOperator& op, ArrayDataType* out_quantized_data_type);

// Gets the min/max numerical range for the given quantized data type.
// For example, kUint8 will return [0,255].
// Returns true if the ranges were set and false if the type is not quantized.
bool GetQuantizedDataTypeNumericalRange(ArrayDataType data_type,
                                        double* out_min_value,
                                        double* out_max_value);

// Returns the quantized data type of an array, falling back to the provided
// default data type.
ArrayDataType GetQuantizedDataType(const Array& array,
                                   ArrayDataType default_type);

// Chooses the quantization params for a given array and a given target
// quantized data type (which may not be the array's current data type).
void ChooseQuantizationParamsForArrayAndQuantizedDataType(
    const Array& array, ArrayDataType quantized_data_type,
    QuantizationParams* quantization_params);

// Quantizes an array by setting its data type and (if constant) quantizing
// all values in the array.
void QuantizeArray(GraphTransformation* transformation, Model* model,
                   const std::string& name, ArrayDataType quantized_data_type,
                   const QuantizationParams& quantization_params);

// Returns true if the given array, when quantized, contains only values between
// the provided clamp min/max.
// Either clamp_min or clamp_max may be +/-infinity to indicate that the value
// is unbounded on that side.
bool IsArrayQuantizedRangeSubset(GraphTransformation* transformation,
                                 const Array& array, double clamp_min,
                                 double clamp_max);

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_QUANTIZATION_UTIL_H_
