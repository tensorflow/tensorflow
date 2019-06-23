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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_DEPTHWISE_CONV_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_DEPTHWISE_CONV_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<ComputeTaskDescriptorPtr> DepthWiseConvolution(
    int id, ValueId input_id, ValueId output_id,
    const DepthwiseConvolution2DAttributes& attr,
    const RuntimeOptions& options);

// Depth Wise Convolution for kernel 3x3
// require:
//   channels_multiplier = 1;
//   kernel_size = 3x3;
//   dilation = 1x1;
//   stride = 1x1;
std::vector<ComputeTaskDescriptorPtr> DepthWiseConv3x3Stride1x1(
    int id, ValueId input_id, ValueId output_id,
    const DepthwiseConvolution2DAttributes& attr,
    const RuntimeOptions& options);

// TODO(impjdi): Move it inside module.
bool CheckDepthWiseConv3x3Stride1x1Support(
    const DepthwiseConvolution2DAttributes& attr);

// Depth Wise Convolution for kernel 3x3
// require:
//   channels_multiplier = 1;
//   kernel_size = 3x3;
//   dilation.y = 1;
//   stride.y = 2;
std::vector<ComputeTaskDescriptorPtr> DepthWiseConv3x3Stride2(
    int id, ValueId input_id, ValueId output_id,
    const DepthwiseConvolution2DAttributes& attr,
    const RuntimeOptions& options);

// TODO(impjdi): Move it inside module.
bool CheckDepthWiseConv3x3Stride2Support(
    const DepthwiseConvolution2DAttributes& attr);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_DEPTHWISE_CONV_H_
