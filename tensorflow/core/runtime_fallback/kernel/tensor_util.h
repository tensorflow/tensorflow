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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_TENSOR_UTIL_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_TENSOR_UTIL_H_

#include <utility>

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/framework/device.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime

namespace tfrt {
class Device;
}  // namespace tfrt

namespace tensorflow {
class KernelFallbackTensor;
namespace tfd {

// Transfers tensor `src` from `src_device` to `dst_device`.
// Returns the transferred tensor on `dst_device` wrapped as
// `TensorWrapperType`.
template <typename TensorWrapperType>
tfrt::AsyncValueRef<TensorWrapperType> TransferTensorToDevice(
    const tfrt::ExecutionContext& exec_ctx, const Tensor& src,
    Device* src_device, Device* dst_device) {
  const bool is_same_device =
      (src_device == dst_device) || (src_device->name() == dst_device->name());

  // Note: source and destination CPU devices are expected to be on the same
  // host. Currently TFRT doesn't support checking if a CPU is remote CPU,
  // we may consider adding a remote CPU device type in the future.
  const bool src_cpu =
      src_device->tensorflow_accelerator_device_info() == nullptr;
  const bool dst_cpu =
      dst_device->tensorflow_accelerator_device_info() == nullptr;
  const bool is_between_cpu_devices = dst_cpu && src_cpu;

  if (is_same_device || is_between_cpu_devices) {
    return tfrt::MakeAvailableAsyncValueRef<TensorWrapperType>(src);
  }

  if (!dst_cpu && (src.dtype() != tensorflow::DT_VARIANT &&
                   !tensorflow::DataTypeCanUseMemcpy(src.dtype()))) {
    return tfrt::MakeErrorAsyncValueRef(absl::InternalError(tfrt::StrCat(
        "Can't copy Tensor with type ", tensorflow::DataTypeString(src.dtype()),
        " to device ", dst_device->name(), ".")));
  }
  tensorflow::AllocatorAttributes attr;
  if (src.dtype() == tensorflow::DT_VARIANT) {
    attr.set_on_host(true);
  }
  tensorflow::Tensor dst(dst_device->GetAllocator(attr), src.dtype(),
                         src.shape());
  if (src.shape().num_elements() == 0) {
    return tfrt::MakeAvailableAsyncValueRef<TensorWrapperType>(dst);
  }

  auto result = tfrt::MakeUnconstructedAsyncValueRef<TensorWrapperType>();
  bool enqueued = tfrt::EnqueueBlockingWork(
      exec_ctx.host(), [result = result.CopyRef(), src_cpu, dst_cpu, src_device,
                        dst_device, src, dst = std::move(dst)]() mutable {
        tensorflow::DeviceContext* src_device_context = nullptr;
        if (!src_cpu) {
          src_device_context =
              src_device->tensorflow_accelerator_device_info()->default_context;
        }
        tensorflow::DeviceContext* dst_device_context = nullptr;
        if (!dst_cpu) {
          dst_device_context =
              dst_device->tensorflow_accelerator_device_info()->default_context;
        }
        // TODO(tfrt-devs): The Sync() call below may be more aggressive than
        // necessary. It is based on knowledge of implementation details - that
        // GPU devices are implemented using 3 streams - one for host->device
        // copies, one for device->host copies and one for sending operations to
        // the GPU. With that setup, Sync()ing across all 3 streams should be
        // sufficient but more than necessary (since it waits for operations
        // that might have nothing to do with this tensor to complete).
        absl::Status s = src_device->Sync();
        if (!s.ok()) {
          result.SetError(absl::InternalError(s.message()));
          return;
        }
        tensorflow::Notification n;
        absl::Status status;
        tensorflow::CopyTensor::ViaDMA(
            "copy", src_device_context, dst_device_context, src_device,
            dst_device, tensorflow::AllocatorAttributes(),
            tensorflow::AllocatorAttributes(), &src, &dst,
            0 /*dev_to_dev_stream_index*/,
            [&status, &n](const absl::Status& s) {
              status = s;
              n.Notify();
            });
        n.WaitForNotification();
        if (status.ok()) {
          result.emplace(std::move(dst));
        }
      });

  if (!enqueued) {
    return tfrt::MakeErrorAsyncValueRef(absl::InternalError(
        "Failed to enqueue blocking task to transfer tensor."));
  }
  return result;
}

tfrt::AsyncValueRef<KernelFallbackTensor> TransferTensorToDevice(
    const tfrt::ExecutionContext& exec_ctx, const KernelFallbackTensor& tensor,
    const tfrt::Device& src_device, const tfrt::Device& dst_device);

llvm::Expected<Device*> GetTfDevice(const tfrt::ExecutionContext& exec_ctx,
                                    const tfrt::Device& device);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_TENSOR_UTIL_H_
