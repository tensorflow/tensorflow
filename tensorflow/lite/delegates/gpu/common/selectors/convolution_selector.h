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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_CONVOLUTION_SELECTOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_CONVOLUTION_SELECTOR_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"

namespace tflite {
namespace gpu {

std::unique_ptr<GPUOperation> SelectConvolution(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def, ModelHints hints);

std::unique_ptr<GPUOperation> SelectConvolutionForWinograd(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def, ModelHints hints);

std::unique_ptr<GPUOperation> SelectConvolutionWithDynamicWeights(
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC& dst_shape, const GpuInfo& gpu_info, const OperationDef& op_def,
    ModelHints hints, WeightsDescription* weights_desc);

std::unique_ptr<GPUOperation> SelectConvolutionBatchedMatMul(
    const OHWI& weights_shape, const BHWC& dst_shape, const GpuInfo& gpu_info,
    const OperationDef& op_def, ModelHints hints,
    WeightsDescription* weights_desc);

std::unique_ptr<GPUOperation> SelectConverterToConvWeights(
    const WeightsDescription& weights_desc, const OperationDef& op_def,
    ModelHints hints);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_CONVOLUTION_SELECTOR_H_
