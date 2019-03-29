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

#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

#include <cstdint>
#include <vector>

#include <fp16.h>
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace metal {

/// Helper function to convert buffer's content into stream of bytes
std::vector<uint8_t> VectorFloatToHalf(const std::vector<float>& input_vector) {
  std::vector<HalfBits> result;
  result.reserve(input_vector.size());
  for (const float v : input_vector) {
    result.push_back(fp16_ieee_from_fp32_value(v));
  }
  return VectorToUint8Vector(result);
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
