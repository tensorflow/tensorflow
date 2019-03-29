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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_CONVOLUTION1X1_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_CONVOLUTION1X1_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

// Convolution for kernel 1x1
// require:
//   kernel_size = 1x1;
//   padding prepended and appended = 0x0
//   dilation = 1x1;
//   stride = 1x1;
// Works very good on A12 (IPhoneXS, etc).
// Works good on A9/A10/A11 (IPhone6S, IPhone7, IPhoneX, etc).
// Works bad on A7/A8 (IPhone5S, IPhone6, etc).
std::vector<ComputeTaskDescriptorPtr> Convolution1x1(
    int id, ValueId input_id, ValueId output_id,
    const Convolution2DAttributes& params,
    const RuntimeOptions& options);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_CONVOLUTION1X1_H_
