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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_WINOGRAD_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_WINOGRAD_H_

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

struct Winograd4x4To36Attributes {
  Padding2D padding;
};

std::vector<ComputeTaskDescriptorPtr> Winograd4x4To36(
    int id, ValueId input_id, ValueId output_id,
    const Winograd4x4To36Attributes& attr);

std::vector<ComputeTaskDescriptorPtr> Winograd4x4To36TileX6(
    int id, ValueId input_id, ValueId output_id,
    const Winograd4x4To36Attributes& attr, const RuntimeOptions& options);

struct Winograd36To4x4Attributes {
  BHWC output_shape;
  tflite::gpu::Tensor<Linear, DataType::FLOAT32> biases;
};

std::vector<ComputeTaskDescriptorPtr> Winograd36To4x4(
    int id, ValueId input_id, ValueId output_id, const RuntimeOptions& options,
    const Winograd36To4x4Attributes& attr);

std::vector<ComputeTaskDescriptorPtr> Winograd36To4x4Tile4x1(
    int id, ValueId input_id, ValueId output_id, const RuntimeOptions& options,
    const Winograd36To4x4Attributes& attr);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_WINOGRAD_H_
