/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_COMMON_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_COMMON_H_

namespace tflite {
namespace gpu {
namespace cl {

enum class ConvWeightsLayout {
  kUnknown,
  kOHWIOGroupI4O4,
};

struct ConvWeightsDescription {
  ConvWeightsLayout layout;
  int output_group_size;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_COMMON_H_
