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

#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"

namespace tflite {
namespace gpu {

// Size of object, that covers both input objects (2-dimensional case).
bool IsCoveringObject(const uint2& first_object, const uint2& second_object) {
  return first_object.x >= second_object.x && first_object.y >= second_object.y;
}

// Size of object, that covers both input objects (3-dimensional case).
bool IsCoveringObject(const uint3& first_object, const uint3& second_object) {
  return first_object.x >= second_object.x &&
         first_object.y >= second_object.y && first_object.z >= second_object.z;
}

// Difference between two objects in elements count (2-dimensional case).
size_t AbsDiffInElements(const uint2& first_size, const uint2& second_size) {
  const size_t first_elements_cnt = first_size.y * first_size.x;
  const size_t second_elements_cnt = second_size.y * second_size.x;
  return first_elements_cnt >= second_elements_cnt
             ? first_elements_cnt - second_elements_cnt
             : second_elements_cnt - first_elements_cnt;
}

// Difference between two objects in elements count (3-dimensional case).
size_t AbsDiffInElements(const uint3& first_size, const uint3& second_size) {
  const size_t first_elements_cnt = first_size.z * first_size.y * first_size.x;
  const size_t second_elements_cnt =
      second_size.z * second_size.y * second_size.x;
  return first_elements_cnt >= second_elements_cnt
             ? first_elements_cnt - second_elements_cnt
             : second_elements_cnt - first_elements_cnt;
}

}  // namespace gpu
}  // namespace tflite
