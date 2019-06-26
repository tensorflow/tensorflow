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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_SPI_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_SPI_H_

#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

// Contains only service provider-related interfaces. Users should not use them
// directly.

namespace tflite {
namespace gpu {

// Converts a tensor object into another one.
class TensorObjectConverter {
 public:
  virtual ~TensorObjectConverter() = default;

  virtual Status Convert(const TensorObject& input,
                         const TensorObject& output) = 0;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_SPI_H_
