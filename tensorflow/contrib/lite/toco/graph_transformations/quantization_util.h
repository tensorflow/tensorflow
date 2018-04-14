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
#ifndef TENSORFLOW_CONTRIB_LITE_TOCO_GRAPH_TRANSFORMATIONS_QUANTIZATION_UTIL_H_
#define TENSORFLOW_CONTRIB_LITE_TOCO_GRAPH_TRANSFORMATIONS_QUANTIZATION_UTIL_H_

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"

namespace toco {

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

// Gets the quantization params for the array with the given data type and
// minmax.
void GetQuantizationParams(ArrayDataType data_type, const MinMax& minmax,
                           QuantizationParams* quantization_params);

// Returns true if the given array, when quantized, contains only values between
// the provided clamp min/max.
// Either clamp_min or clamp_max may be +/-infinity to indicate that the value
// is unbounded on that side.
bool IsArrayQuantizedRangeSubset(GraphTransformation* transformation,
                                 const Array& array, double clamp_min,
                                 double clamp_max);

}  // namespace toco

#endif  // TENSORFLOW_CONTRIB_LITE_TOCO_GRAPH_TRANSFORMATIONS_QUANTIZATION_UTIL_H_
