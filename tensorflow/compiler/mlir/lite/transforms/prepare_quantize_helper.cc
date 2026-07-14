/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/transforms/prepare_quantize_helper.h"

#include <cmath>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir {
namespace TFL {

double PowerOfTwoBound(double value) {
  return std::pow(2, std::ceil(std::log2(value)));
}

tensorflow::DataType GetQuantizedInferenceType(bool is_signed,
                                               int number_of_bits) {
  if (is_signed && number_of_bits == 8) {
    return tensorflow::DT_QINT8;
  } else if (!is_signed && number_of_bits == 8) {
    return tensorflow::DT_QUINT8;
  } else if (is_signed && number_of_bits == 16) {
    return tensorflow::DT_QINT16;
  } else {
    return tensorflow::DT_INVALID;
  }
}

}  // namespace TFL
}  // namespace mlir
