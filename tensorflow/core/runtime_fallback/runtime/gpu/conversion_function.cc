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

// This file implements conversion function between TFRuntimeFallback and Gpu
// tensors.

#include "tensorflow/core/runtime_fallback/runtime/gpu/conversion_function.h"

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"
#include "tensorflow/core/runtime_fallback/util/gpu/gpu_utils.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/gpu/device/conversion_function.h"  // from @tf_runtime
#include "tfrt/gpu/device/device.h"  // from @tf_runtime
#include "tfrt/gpu/device/device_util.h"  // from @tf_runtime
#include "tfrt/gpu/gpu_types.h"  // from @tf_runtime
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_utils.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

static tfrt::Expected<RuntimeFallbackTensor>
CopyRefGpuTensorToRuntimeFallbackTensor(
    const tfrt::gpu::DenseGpuTensor& gpu_tensor, Device* device,
    Device* op_device, EagerContext* eager_ctx) {
  // Do not copy the gpu buffer content, CopyRef on the buffer instead.
  tfrt::AsyncValueRef<tfrt::gpu::GpuBuffer> gpu_buffer =
      gpu_tensor.CopyBufferRef();
  tfrt::Expected<tensorflow::Tensor> tensor = MoveGpuBufferToTFTensor(
      std::move(gpu_buffer), gpu_tensor.dtype(), gpu_tensor.shape());
  if (!tensor) return tensor.takeError();

  OwnedTensorHandle tensor_handle{tensorflow::TensorHandle::CreateLocalHandle(
      std::move(tensor.get()), device, op_device, eager_ctx)};
  return RuntimeFallbackTensor(gpu_tensor.shape(), gpu_tensor.dtype(),
                               std::move(tensor_handle));
}

// Convert the RuntimeFallbackTensor to a GpuTensor (currently DenseGpuTensor
// only). If the source tensor is on CPU, copy the data to GPU. If the source
// tensor is already on GPU, just do type conversion.
// TODO(b/167254525): For TFRuntimeFallback tensor, create separate tensor
// types for different devices.
static tfrt::AsyncValueRef<tfrt::gpu::DenseGpuTensor>
ConvertRuntimeFallbackTensorToDenseGpuTensor(
    const RuntimeFallbackTensor& tensor, const tfrt::Device& src,
    const tfrt::gpu::GpuDevice& dst, const tfrt::ExecutionContext& exec_ctx) {
  auto* host_ctx = exec_ctx.host();

  auto tf_tensor_handle = tensor.GetTensorHandle();

  tensorflow::Status status;
  const char* device_name = tf_tensor_handle->DeviceName(&status);

  auto tensor_device_ref =
      host_ctx->GetDeviceManager()->GetDeviceRef<tfrt::Device>(device_name);

  if (!tensor_device_ref) {
    tensor_device_ref =
        host_ctx->GetDeviceManager()->GetDeviceRef<tfrt::Device>(
            ConvertTfDeviceNameToTfrtDefault(device_name));
  }

  if (!tensor_device_ref)
    return tfrt::EmitErrorAsync(
        exec_ctx,
        tfrt::StrCat("Failed to find a device with name: ", device_name));

  if (!status.ok()) {
    return EmitErrorAsync(
        exec_ctx, tfrt::StrCat("error getting device name from TensorHandle: ",
                               status.error_message()));
  }

  // Check if the underlying tensorflow::TensorHandle is already on GPU.
  // If so, just convert the RuntimeFallbackTensor to GpuTensor.
  if (tensor_device_ref.get() == &dst) {
    tensorflow::TensorShape shape;
    tensorflow::Status status = tf_tensor_handle->Shape(&shape);
    if (!status.ok()) {
      return EmitErrorAsync(
          exec_ctx, tfrt::StrCat("error getting shape from TF tensor handle: ",
                                 status.error_message()));
    }

    auto tf_shape = shape.dim_sizes();
    DataType dtype = tf_tensor_handle->DataType();
    // Note that GPU tensor might not be available yet. But since TF
    // and TFRT share the same stream, this is ok.
    const tensorflow::Tensor* tf_tensor = nullptr;
    status = tf_tensor_handle->Tensor(&tf_tensor);
    if (!status.ok()) {
      return EmitErrorAsync(exec_ctx,
                            tfrt::StrCat("error calling TensorHandle::Tensor: ",
                                         status.error_message()));
    }

    auto platform = tensorflow::tfd::GetTfrtGpuPlatform(tf_tensor_handle);

    void* data = tf_tensor->data();
    size_t size = tf_tensor->TotalBytes();

    // Need to add a reference here since we are transferring the ownership
    // of the Tensorflow::TensorHandle and the underlying GPU buffer to
    // tfrt::DenseGpuTensor. Otherwise, the TensorHandle will be released
    // when he RuntimeFallbackTensor goes out of scope after the tensor
    // conversion. The GPU buffer will be deleted as well.
    OwnedTensorHandle owned_tf_tensor_handle =
        OwnedTensorHandle{TensorHandleFromInterface(tf_tensor_handle->Copy())};

    // The OwnedTensorHandle holds a reference on underlying Tensorflow buffer
    // and is held alive by GpuOneShotAllocator.
    auto allocator = tfrt::MakeAvailableAsyncValueRef<
        tfrt::gpu::GpuOneShotAllocator<OwnedTensorHandle>>(
        tfrt::gpu::wrapper::Pointer<void>(data, platform),
        std::move(owned_tf_tensor_handle));
    llvm::Expected<tfrt::gpu::GpuBuffer> gpu_buffer =
        tfrt::gpu::GpuBuffer::Allocate(std::move(allocator), size);
    if (!gpu_buffer) {
      return tfrt::MakeErrorAsyncValueRef(tfrt::StrCat(gpu_buffer.takeError()));
    }

    // create DenseGpuTensor.
    tfrt::gpu::DenseGpuTensor gpu_tensor{
        tfrt::TensorShape(
            std::vector<tfrt::Index>(tf_shape.begin(), tf_shape.end())),
        GetTfrtDtype(dtype),
        tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuBuffer>(
            std::move(*gpu_buffer))};

    return tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::DenseGpuTensor>(
        std::move(gpu_tensor));
  } else {
    // TODO(chuanhao): clean up the branch after cl/325503773. Currently this
    // branch is needed since we don't know what type of tensor that
    // RuntimeFallbackTensor holds.
    // tensorflow::TensorHandle is on host CPU.
    assert(tensor_device_ref.get() == &host_ctx->GetHostDevice());

    // Convert the TFRuntimeFallbackTensor to DenseHostTensor.
    auto host_tensor_ref = tfrt::ConvertTensor(
        exec_ctx, tensor, src, src, tfrt::DenseHostTensor::kTensorType);

    if (!host_tensor_ref.get().IsTensorType(tfrt::DenseHostTensor::kTensorType))
      return EmitErrorAsync(exec_ctx,
                            "TFRuntimeFallbackTensor not converted to "
                            "DenseHostTensor.");
    llvm::Expected<tfrt::gpu::wrapper::CurrentContext> current_context =
        dst.SetCurrentContext();
    if (!current_context) {
      return tfrt::MakeErrorAsyncValueRef(
          tfrt::StrCat(current_context.takeError()));
    }

    auto expected_gpu_tensor =
        tfrt::gpu::ConvertDenseHostTensorToDenseGpuTensor(
            std::move(current_context.get()), dst.stream(), dst.allocator(),
            llvm::cast<tfrt::DenseHostTensor>(host_tensor_ref.get()), host_ctx);
    if (!expected_gpu_tensor) {
      return EmitErrorAsync(
          exec_ctx,
          absl::InternalError(toString(expected_gpu_tensor.takeError())));
    }
    return tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::DenseGpuTensor>(
        std::move(expected_gpu_tensor.get()));
  }
}

static tfrt::AsyncValueRef<RuntimeFallbackTensor>
ConvertDenseGpuTensorToRuntimeFallbackTensor(
    const tfrt::gpu::DenseGpuTensor& tensor, const tfrt::gpu::GpuDevice& src,
    const tfrt::gpu::GpuDevice& dst, const tfrt::ExecutionContext& exec_ctx) {
  tfrt::ResourceContext* resource_context = exec_ctx.resource_context();
  tensorflow::tfd::EagerContextResource* eager_context_resource =
      resource_context
          ->GetOrCreateResource<tensorflow::tfd::EagerContextResource>(
              tensorflow::tfd::kEagerContextResourceName);

  tfrt::Expected<EagerContext*> eager_ctx_expected =
      eager_context_resource->GetTFEagerContext();
  if (!eager_ctx_expected)
    return EmitErrorAsync(exec_ctx, eager_ctx_expected.takeError());

  EagerContext* eager_ctx = eager_ctx_expected.get();

  assert(&src == &dst);
  Device* device;
  Status status = eager_ctx->local_device_mgr()->LookupDevice(
      ToAbslStringView(dst.name()), &device);
  if (!status.ok())
    return EmitErrorAsync(exec_ctx,
                          absl::InternalError(tfrt::StrCat(
                              "error looking up gpu device from EagerContext: ",
                              status.error_message())));

  auto fallback_tensor = CopyRefGpuTensorToRuntimeFallbackTensor(
      tensor, device, device, eager_ctx);
  if (fallback_tensor) {
    return tfrt::MakeAvailableAsyncValueRef<RuntimeFallbackTensor>(
        std::move(*fallback_tensor));
  } else {
    return EmitErrorAsync(
        exec_ctx, absl::InternalError(toString(fallback_tensor.takeError())));
  }
}

void RegisterTFRuntimeFallbackTensorToGpuConversionFn(
    tfrt::TensorConversionFnRegistry* registry) {
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertRuntimeFallbackTensorToDenseGpuTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertDenseGpuTensorToRuntimeFallbackTensor));
}

}  // namespace tfd
}  // namespace tensorflow
