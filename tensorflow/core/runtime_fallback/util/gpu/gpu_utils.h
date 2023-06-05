/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file declares gpu related utility functions.

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_GPU_GPU_UTILS_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_GPU_GPU_UTILS_H_

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tfrt/gpu/device/gpu_config.h"  // from @tf_runtime
#include "tfrt/gpu/gpu_types.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

// Lookup GPU platform (CUDA vs ROCm) from a given tensorflow::TensorHandle.
tfrt::gpu::wrapper::Platform GetTfrtGpuPlatform(tensorflow::TensorHandle* th);

tfrt::gpu::wrapper::Platform GetTfrtGpuPlatform(
    tensorflow::BaseGPUDevice* device);

// Moves one ref on GpuBuffer to tensorflow::Tensor.
tfrt::Expected<tensorflow::Tensor> MoveGpuBufferToTFTensor(
    tfrt::AsyncValueRef<tfrt::gpu::GpuBuffer> gpu_buffer, tfrt::DType dtype,
    tfrt::TensorShape shape);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_GPU_GPU_UTILS_H_
