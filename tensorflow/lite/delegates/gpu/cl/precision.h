/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_PRECISION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_PRECISION_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"

namespace tflite {
namespace gpu {
namespace cl {

enum class CalculationsPrecision { F32, F32_F16, F16 };
// F32 - all data and all math ops in F32
// F16 - all data and all math ops in F16
// F32_F16 - as F16, but some operations (Convolution,
// DepthWiseConvolution, FullyConnected, ConvolutionTransposed)
// have accumulator in F32 and usually it calculates 4 mads in F16, sum them,
// than converts this partial sum to F32 and add to acumulator.

DataType DeduceDataTypeFromPrecision(CalculationsPrecision precision);

std::string ToString(CalculationsPrecision precision);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_PRECISION_H_
