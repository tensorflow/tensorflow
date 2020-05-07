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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_RUNTIME_OPTIONS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_RUNTIME_OPTIONS_H_

namespace tflite {
namespace gpu {
namespace metal {

struct RuntimeOptions {
  enum class Precision {
    FP16,
    FP32,
  };
  // Buffer storage format. If FP32 then accumulator must be FP32.
  Precision storage_precision = Precision::FP32;
  // Accumulator precision. Defines the precision for convolutions.
  Precision accumulator_precision = Precision::FP32;
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_RUNTIME_OPTIONS_H_
