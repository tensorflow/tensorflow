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
#include "tensorflow/core/runtime_fallback/kernel/tensor_util.h"

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

tfrt::AsyncValueRef<KernelFallbackTensor> TransferTensorToDevice(
    const tfrt::ExecutionContext& exec_ctx, const KernelFallbackTensor& tensor,
    const tfrt::Device& src_device, const tfrt::Device& dst_device) {
  bool is_same_device =
      (&src_device == &dst_device) || (src_device.name() == dst_device.name());

  const tensorflow::Tensor* src = tensor.GetTensor();

  if (is_same_device) {
    return tfrt::MakeAvailableAsyncValueRef<KernelFallbackTensor>(*src);
  }

  auto expected_src = GetTfDevice(exec_ctx, src_device);
  if (!expected_src) {
    return tfrt::MakeErrorAsyncValueRef(tfrt::StrCat(expected_src.takeError()));
  }
  auto expected_dst = GetTfDevice(exec_ctx, dst_device);
  if (!expected_dst) {
    return tfrt::MakeErrorAsyncValueRef(tfrt::StrCat(expected_dst.takeError()));
  }
  tensorflow::Device* srcd = expected_src.get();
  tensorflow::Device* dstd = expected_dst.get();
  const bool src_cpu = srcd->tensorflow_gpu_device_info() == nullptr;
  const bool dst_cpu = dstd->tensorflow_gpu_device_info() == nullptr;

  if (!dst_cpu && (src->dtype() != tensorflow::DT_VARIANT &&
                   !tensorflow::DataTypeCanUseMemcpy(src->dtype()))) {
    return tfrt::MakeErrorAsyncValueRef(
        tfrt::StrCat("Can't copy Tensor with type ",
                     tensorflow::DataTypeString(src->dtype()), " to device ",
                     dstd->name(), "."));
  }
  tensorflow::AllocatorAttributes attr;
  if (src->dtype() == tensorflow::DT_VARIANT) {
    attr.set_on_host(true);
  }
  tensorflow::Tensor dst(dstd->GetAllocator(attr), src->dtype(), src->shape());
  if (src->shape().num_elements() == 0) {
    return tfrt::MakeAvailableAsyncValueRef<KernelFallbackTensor>(*src);
  }

  auto result = tfrt::MakeUnconstructedAsyncValueRef<KernelFallbackTensor>();
  bool enqueued = tfrt::EnqueueBlockingWork(
      exec_ctx.host(), [result = result.CopyRef(), src_cpu, dst_cpu, srcd, dstd,
                        src = *src, dst = std::move(dst)]() mutable {
        tensorflow::DeviceContext* src_device_context = nullptr;
        if (!src_cpu) {
          src_device_context =
              srcd->tensorflow_gpu_device_info()->default_context;
        }
        tensorflow::DeviceContext* dst_device_context = nullptr;
        if (!dst_cpu) {
          dst_device_context =
              dstd->tensorflow_gpu_device_info()->default_context;
        }
        // TODO(tfrt-devs): The Sync() call below may be more aggressive than
        // necessary. It is based on knowledge of implementation details - that
        // GPU devices are implemented using 3 streams - one for host->device
        // copies, one for device->host copies and one for sending operations to
        // the GPU. With that setup, Sync()ing across all 3 streams should be
        // sufficient but more than necessary (since it waits for operations
        // that might have nothing to do with this tensor to complete).
        Status s = srcd->Sync();
        if (!s.ok()) {
          result.SetError(s.error_message());
          return;
        }
        tensorflow::Notification n;
        tensorflow::Status status;
        tensorflow::CopyTensor::ViaDMA(
            "copy", src_device_context, dst_device_context, srcd, dstd,
            tensorflow::AllocatorAttributes(),
            tensorflow::AllocatorAttributes(), &src, &dst,
            0 /*dev_to_dev_stream_index*/,
            [&status, &n](const tensorflow::Status& s) {
              status = s;
              n.Notify();
            });
        n.WaitForNotification();
        if (status.ok()) {
          result.emplace(std::move(dst));
        }
      });

  if (!enqueued) {
    return tfrt::MakeErrorAsyncValueRef(
        "Failed to enqueu blocking task to transfer tensor");
  }
  return result;
}

llvm::Expected<Device*> GetTfDevice(const tfrt::ExecutionContext& exec_ctx,
                                    const tfrt::Device& device) {
  auto eager_context_expected =
      exec_ctx.resource_context()
          ->GetOrCreateResource<tfd::EagerContextResource>(
              tfd::kEagerContextResourceName)
          ->GetTFEagerContext();
  if (!eager_context_expected) {
    return eager_context_expected.takeError();
  }
  Device* tf_device;
  Status s = eager_context_expected.get()->FindDeviceFromName(
      device.name().data(), &tf_device);
  if (!s.ok()) {
    return tfrt::MakeStringError(s.error_message());
  }
  return tf_device;
}

}  // namespace tfd
}  // namespace tensorflow
