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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_CONV_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_CONV_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<ComputeTaskDescriptorPtr> Convolution(
    int id, ValueId input_id, ValueId output_id,
    const Convolution2DAttributes& params,
    const metal::RuntimeOptions& options);

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
    const Convolution2DAttributes& params, const RuntimeOptions& options);

// TODO(impjdi): Move it inside module.
bool CheckConvolution1x1Support(const Convolution2DAttributes& attr);

// This convolution pass all conv parameters (beside output_channels)
// as dynamic arguments (uniform buffer) to kernel.
// Depending on output_channels can be generated different kernels
// Kernel can proceed 4/8/12/16 output channels per one thread.
// 16 channels output is the fastest but the least flexible.
std::vector<ComputeTaskDescriptorPtr> ConvolutionGeneric(
    int id, ValueId input_id, ValueId output_id,
    const Convolution2DAttributes& params, const RuntimeOptions& options);

// This convolution makes more precise mapping of threads on elements.
// For example, if we have output tensor 12x7 and work group = 8x4,
// then we need 4 workgroups to cover this tensor in usual case.
// But in general we have only 84 elements(12*7), and we can cover it with 3
// workgroups of size 32. So this version of convolution use this precise
// mapping.
// But this convolution, due to some hardware limitations, doesn't work better
// always. In general it works good on A12.
std::vector<ComputeTaskDescriptorPtr> ConvolutionPrecise(
    int id, ValueId input_id, ValueId output_id,
    const Convolution2DAttributes& params, const RuntimeOptions& options);

// This function calculates amount of threads that should be launched for
// ConvolutionGeneric or Convolution1x1 (threads_count1) and amount of threads
// that should be launched for ConvolutionPrecise (threads_count2) and returns
// threads_count1 / threads_count2.
float GetThreadsRatioUsualToPreciseConvolution(const BHWC& dst_shape);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_CONV_H_
