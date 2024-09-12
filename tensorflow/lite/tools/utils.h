/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_UTILS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace utils {
// Implement type erasure with unique_ptr with custom deleter.
using VoidUniquePtr = std::unique_ptr<void, void (*)(void*)>;

// Represents any input tensor data to TfLite model.
struct InputTensorData {
  InputTensorData() : data(nullptr, nullptr) {}

  VoidUniquePtr data;
  size_t bytes;
};

// Returns random test data that can be used for testing purposes only.
// Data returned should be between 'low_range' and 'high_range'.
InputTensorData CreateRandomTensorData(const TfLiteTensor& tensor,
                                       float low_range, float high_range);

// Fills out params 'low_range' and 'high_range' with range for tensor type
// 'type'. Note that these ranges returned are just dummy used only for
// benchmarking and/or testing purposes.
void GetDataRangesForType(TfLiteType type, float* low_range, float* high_range);

// Converts TfLiteTensor to float array. Returns an error if the tensor type is
// not supported or the values size is not equal to the tensor dimension.
TfLiteStatus TfLiteTensorToFloat32Array(const TfLiteTensor& tensor,
                                        absl::Span<float> values);

// Same as above, but converts to int64_t array.
TfLiteStatus TfLiteTensorToInt64Array(const TfLiteTensor& tensor,
                                      absl::Span<int64_t> values);
}  // namespace utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_UTILS_H_
