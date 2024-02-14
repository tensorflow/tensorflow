/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_TENSOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_TENSOR_H_

#include <cstdint>
#include <variant>

#include "tensorflow/lite/experimental/shlo/data_type.h"

namespace shlo_ref {

// A non owning view of tensor data.
struct Tensor {
  DataType type;     // Type of the underlying buffer.
  void* data;        // Non owning pointer to the underlying buffer.
  int64_t rank;      // Number of dimensions.
  int64_t* dims;     // Size of each dimension, in elements. Must hold `rank`
                     // elements.
  int64_t* strides;  // Stride between two elements of each dimension, in bytes.
                     // Must hold `rank` elements.

  struct NoQuantization {};

  // The size of the arrays must be equal to the channel count.
  struct PerChannelAffineQuantization {
    float* scales;
    float* zero_points;
  };

  struct PerTensorAffineQuantization {
    float scales;
    float zero_points;
  };

  std::variant<NoQuantization, PerChannelAffineQuantization,
               PerTensorAffineQuantization>
      quantization;
};

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_TENSOR_H_
