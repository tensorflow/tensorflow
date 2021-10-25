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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SPECIAL_DEPTHWISE_CONV_PLUS_1X1_CONV_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SPECIAL_DEPTHWISE_CONV_PLUS_1X1_CONV_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

bool IsDepthwiseConvPlus1x1ConvSupported(
    const OperationDef& definition, const GpuInfo& gpu_info,
    const DepthwiseConvolution2DAttributes& dw_attr,
    const Convolution2DAttributes& conv_attr);

GPUOperation CreateDepthwiseConvPlus1x1Conv(
    const OperationDef& definition, const GpuInfo& gpu_info,
    const DepthwiseConvolution2DAttributes& dw_attr,
    const Convolution2DAttributes& conv_attr,
    ReLUAttributes* relu_attr_ptr = nullptr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SPECIAL_DEPTHWISE_CONV_PLUS_1X1_CONV_H_
