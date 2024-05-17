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

// This file implements gpu related utility functions.

#include "tensorflow/core/runtime_fallback/util/gpu/gpu_utils.h"

#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "xla/stream_executor/cuda/cuda_driver.h"
#include "xla/stream_executor/platform.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/runtime_fallback/util/tensor_util.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"

namespace tensorflow {
namespace tfd {

// Helper to lookup GPU platform (CUDA vs ROCm) from a given TensorHandle.
static tfrt::gpu::wrapper::Platform GetTfrtGpuPlatformHelper(
    tensorflow::TensorHandle* th) {
  auto device = th->op_device();
  auto gpu_device = static_cast<tensorflow::BaseGPUDevice*>(device);
  return GetTfrtGpuPlatform(gpu_device);
}

tfrt::gpu::wrapper::Platform GetTfrtGpuPlatform(
    tensorflow::BaseGPUDevice* device) {
  auto platform_kind = device->executor()->platform_kind();
  if (platform_kind == stream_executor::PlatformKind::kCuda) {
    return tfrt::gpu::wrapper::Platform::CUDA;
  } else if (platform_kind == stream_executor::PlatformKind::kROCm) {
    return tfrt::gpu::wrapper::Platform::ROCm;
  }
  return tfrt::gpu::wrapper::Platform::NONE;
}

// Lookup GPU platform (CUDA vs ROCm) from a given TensorHandle.
tfrt::gpu::wrapper::Platform GetTfrtGpuPlatform(tensorflow::TensorHandle* th) {
  // Cache lookup result assuming TF does not mix CUDA and ROCm tensor handles.
  static auto gpu_platform = GetTfrtGpuPlatformHelper(th);
  return gpu_platform;
}

namespace {
struct TFManagedBufferDeleter {
  void operator()(TF_ManagedBuffer* p) const { p->Unref(); }
};
using OwnedTFManagedBuffer =
    std::unique_ptr<TF_ManagedBuffer, TFManagedBufferDeleter>;
}  // namespace

// Moves one ref on GpuBuffer to tensorflow::Tensor.
tfrt::Expected<tensorflow::Tensor> MoveGpuBufferToTFTensor(
    tfrt::AsyncValueRef<tfrt::gpu::GpuBuffer> gpu_buffer, tfrt::DType dtype,
    tfrt::TensorShape shape) {
  auto deallocator = [](void* data, size_t len, void* arg) {
    auto* gpu_buffer = reinterpret_cast<tfrt::AsyncValue*>(arg);
    gpu_buffer->DropRef();
  };

  // `owns_memory` is used by tensorflow::Tensor::RefCountIsOne.
  // One ref on `gpu_buffer` is transfered here to TF_ManagedBuffer.
  OwnedTFManagedBuffer tf_managed_buffer{
      new TF_ManagedBuffer(gpu_buffer->pointer().raw(), gpu_buffer->size(),
                           deallocator, gpu_buffer.release(),
                           /*owns_memory=*/false)};
  tensorflow::Tensor tensor(GetTfDataType(dtype), GetTfShape(shape),
                            tf_managed_buffer.get());
  return std::move(tensor);
}

}  // namespace tfd
}  // namespace tensorflow
