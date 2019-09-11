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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_MICRO_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_MICRO_UTILS_H_
namespace tflite {
namespace ops {
namespace micro {

// Same as gtl::Greater but defined here to reduce dependencies and
// binary size for micro environment.
struct Greater {
  template <typename T>
  bool operator()(const T& x, const T& y) const {
    return x > y;
  }
};

struct Less {
  template <typename T>
  bool operator()(const T& x, const T& y) const {
    return x < y;
  }
};

}  // namespace micro
}  // namespace ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_MICRO_UTILS_H_
