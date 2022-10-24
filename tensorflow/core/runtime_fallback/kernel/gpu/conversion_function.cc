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

// This file implements conversion function between TFKernelFallback and Gpu
// tensors.

#include "tensorflow/core/runtime_fallback/kernel/gpu/conversion_function.h"

#include "absl/strings/match.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/kernel/tensor_util.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/runtime_fallback/util/gpu/gpu_utils.h"
#include "tensorflow/core/runtime_fallback/util/tensor_util.h"
#include "tfrt/gpu/device/conversion_function.h"  // from @tf_runtime
#include "tfrt/gpu/device/device.h"  // from @tf_runtime
#include "tfrt/gpu/device/device_util.h"  // from @tf_runtime
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_utils.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

static llvm::Expected<tfrt::gpu::DenseGpuTensor>
ConvertKernelFallbackTensorOnGpuToDenseGpuTensor(
    const KernelFallbackTensor& tensor, const tfrt::gpu::GpuDevice& src,
    const tfrt::gpu::GpuDevice& dst, const tfrt::ExecutionContext& exec_ctx) {
  assert(&src == &dst);
  auto expected_device = GetTfDevice(exec_ctx, src);
  if (!expected_device) {
    return expected_device.takeError();
  }
  auto gpu_device = static_cast<BaseGPUDevice*>(expected_device.get());

  tensorflow::TensorShape shape;

  auto platform = tensorflow::tfd::GetTfrtGpuPlatform(gpu_device);

  const Tensor* t = tensor.GetTensor();

  void* data = t->data();
  size_t size = t->TotalBytes();

  // The underlying Tensorflow buffer is held alive by GpuOneShotAllocator.
  auto allocator =
      tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuOneShotAllocator<Tensor>>(
          tfrt::gpu::wrapper::Pointer<void>(data, platform), *t);
  TFRT_ASSIGN_OR_RETURN(
      tfrt::gpu::GpuBuffer gpu_buffer,
      tfrt::gpu::GpuBuffer::Allocate(std::move(allocator), size));

  // create DenseGpuTensor.
  return tfrt::gpu::DenseGpuTensor{
      tensor.shape(), tensor.dtype(),
      tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuBuffer>(
          std::move(gpu_buffer))};
}

static llvm::Expected<KernelFallbackTensor>
ConvertDenseGpuTensorToKernelFallbackTensor(
    const tfrt::gpu::DenseGpuTensor& tensor, const tfrt::gpu::GpuDevice& src,
    const tfrt::gpu::GpuDevice& dst, const tfrt::ExecutionContext& exec_ctx) {
  assert(&src == &dst);
  auto expected_device = GetTfDevice(exec_ctx, src);
  if (!expected_device) {
    return expected_device.takeError();
  }

  auto expected_tf_tensor = MoveGpuBufferToTFTensor(
      tensor.CopyBufferRef(), tensor.dtype(), tensor.shape());
  if (!expected_tf_tensor) {
    return expected_tf_tensor.takeError();
  }

  return KernelFallbackTensor{tensor.shape(), tensor.dtype(),
                              expected_tf_tensor.get()};
}

void RegisterTFKernelFallbackTensorToGpuConversionFn(
    tfrt::TensorConversionFnRegistry* registry) {
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertKernelFallbackTensorOnGpuToDenseGpuTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertDenseGpuTensorToKernelFallbackTensor));
}

}  // namespace tfd
}  // namespace tensorflow
