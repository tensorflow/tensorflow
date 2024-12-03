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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"

#include <cstdint>

#include "absl/status/status.h"
#include "xla/stream_executor/device_memory.h"
#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_context.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"

// IMPLEMENTATION NOTE:
//
// 1. Within this module, we intentionally LOG(FATAL) if any stream
//    involved in memcpy becomes !stream->ok(), because TF process
//    today (3/2021) can not properly recover from such an error.
//
// 2. When 0-size tensor is being copied, we should not schedule a
//    copy ThenMemcpy since there is no byte to move. However, we must
//    ensure the causal ordering by arranging the copy done callback
//    to happen after all activities scheduled on the given stream being
//    finished.

namespace tensorflow {

using se::DeviceMemoryBase;

static absl::Status PrepareCopy(
    Device* device, const DeviceContext* ctx, const Tensor& src,
    const Tensor* dst, const DeviceBase::AcceleratorDeviceInfo** dev_info,
    se::Stream** stream) {
  if (device == nullptr) {
    return errors::Internal("Unexpected null device.");
  }
  auto di = device->tensorflow_accelerator_device_info();
  if (di == nullptr) {
    return errors::Internal("Unexpected null device info.");
  }

  *dev_info = di;
  if (ctx == nullptr) {
    return errors::Internal("Unexpected null device context.");
  }
  auto device_stream =
      static_cast<const PluggableDeviceContext*>(ctx)->stream();
  if (device_stream == nullptr) {
    return errors::Internal("No PluggableDevice stream is available.");
  }
  *stream = device_stream;
  if (dst != nullptr) {
    if (src.dtype() != dst->dtype()) {
      return errors::Internal("Can't copy a tensor of ",
                              DataTypeString(src.dtype()), " into a tensor of ",
                              DataTypeString(dst->dtype()));
    }
    if (src.TotalBytes() != dst->TotalBytes()) {
      return errors::Internal("Can't copy ", src.TotalBytes(),
                              " bytes of a tensor into another with ",
                              dst->TotalBytes(), " bytes buffer.");
    }
    if ((src.TotalBytes() > 0) && !src.IsInitialized()) {
      return errors::Internal("Src tensor is not initialized.");
    }
    if ((dst->TotalBytes() > 0) && !dst->IsInitialized()) {
      return errors::Internal("Dst tensor is not initialized.");
    }
  }
  if (!DMAHelper::CanUseDMA(&src)) {
    return errors::Internal("PluggableDevice copy from non-DMA",
                            DataTypeString(src.dtype()), " tensor.");
  }
  return absl::OkStatus();
}

static void* GetBase(const Tensor* src) {
  return const_cast<void*>(DMAHelper::base(src));
}

static void* GetBase(Tensor* dst) { return DMAHelper::base(dst); }

// static
void PluggableDeviceUtil::DeviceToDeviceCopy(
    DeviceContext* send_dev_context, DeviceContext* recv_dev_context,
    Device* src, Device* dst, AllocatorAttributes src_alloc_attr,
    AllocatorAttributes dst_alloc_attr, const Tensor* input, Tensor* output,
    int dev_to_dev_stream_index, StatusCallback done) {
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  absl::Status s = PrepareCopy(src, send_dev_context, *input, output, &dev_info,
                               &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto send_device_to_device_stream =
      static_cast<const PluggableDeviceContext*>(send_dev_context)
          ->device_to_device_stream(dev_to_dev_stream_index);
  if (send_device_to_device_stream == nullptr) {
    done(errors::Internal(
        "No send PluggableDevice copy-out-stream is available."));
    return;
  }
  // Wait for the main stream on the sender to make sure the result is
  // available.
  s = send_device_to_device_stream->WaitFor(send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  const int64_t total_bytes = input->TotalBytes();
  if (total_bytes > 0) {
    void* src_ptr = GetBase(input);
    DeviceMemoryBase device_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(output);
    DeviceMemoryBase device_dst_ptr(dst_ptr, total_bytes);
    auto recv_stream =
        static_cast<const PluggableDeviceContext*>(recv_dev_context)->stream();
    if (recv_stream == nullptr) {
      done(errors::Internal("No recv PluggableDevice stream is available."));
      return;
    }
    // Since we want to use the memory from recv_stream in the
    // send_device_to_host_stream, add a dependency to make sure the memory is
    // truly free.
    s = send_device_to_device_stream->WaitFor(recv_stream);
    if (!s.ok()) {
      done(s);
      return;
    }

    VLOG(2) << "src_ptr " << src_ptr << " dst_ptr " << dst_ptr;
    s = send_device_to_device_stream->Memcpy(&device_dst_ptr, device_src_ptr,
                                             total_bytes);
    if (!s.ok()) {
      done(s);
      return;
    }
  }
  // Use of input may outlive stack scope, so keep a ref.
  TensorReference input_ref(*input);
  dev_info->event_mgr->ThenExecute(
      send_device_to_device_stream,
      [done, send_device_to_device_stream, input_ref]() {
        input_ref.Unref();
        if (!send_device_to_device_stream->ok()) {
          LOG(FATAL) << "PluggableDevice->PluggableDevice Memcpy "  // Crash OK
                     << "failed.";
        }
        done(absl::OkStatus());
      });
  send_dev_context->MaintainLifetimeOnStream(input,
                                             send_device_to_device_stream);
}

// static
void PluggableDeviceUtil::CopyPluggableDeviceTensorToCPU(
    Device* device, const DeviceContext* device_context,
    const Tensor* device_tensor, Tensor* cpu_tensor, StatusCallback done) {
  VLOG(1) << "CopyPluggableDeviceTensorToCPU";
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  absl::Status s = PrepareCopy(device, device_context, *device_tensor,
                               cpu_tensor, &dev_info, &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto send_device_to_host_stream =
      static_cast<const PluggableDeviceContext*>(device_context)
          ->device_to_host_stream();
  if (send_device_to_host_stream == nullptr) {
    done(errors::Internal(
        "No send PluggableDevice copy-out-stream is available."));
    return;
  }
  // Wait for the sender's main stream to make sure that the data are available.
  s = send_device_to_host_stream->WaitFor(send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  const int64_t total_bytes = device_tensor->TotalBytes();
  if (total_bytes > 0) {
    void* src_ptr = GetBase(device_tensor);
    DeviceMemoryBase device_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(cpu_tensor);
    s = send_device_to_host_stream->Memcpy(dst_ptr, device_src_ptr,
                                           total_bytes);
    if (!s.ok()) {
      done(s);
      return;
    }
  }

  // Use of the input may outlive stack scope, so keep a ref.
  TensorReference input_ref(*device_tensor);
  dev_info->event_mgr->ThenExecute(
      send_device_to_host_stream,
      [send_device_to_host_stream, done, input_ref]() {
        if (!send_device_to_host_stream->ok()) {
          LOG(FATAL) << "PluggableDevice->CPU Memcpy failed.";  // Crash OK
        }
        input_ref.Unref();
        done(absl::OkStatus());
      });
}

// static
void PluggableDeviceUtil::CopyCPUTensorToPluggableDevice(
    const Tensor* cpu_tensor, const DeviceContext* device_context,
    Device* device, Tensor* device_tensor, StatusCallback done,
    bool sync_dst_compute) {
  VLOG(1) << "CopyCPUTensorToPluggableDevice";
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* recv_stream = nullptr;
  absl::Status s = PrepareCopy(device, device_context, *cpu_tensor,
                               device_tensor, &dev_info, &recv_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto recv_host_to_device_stream =
      static_cast<const PluggableDeviceContext*>(device_context)
          ->host_to_device_stream();
  if (recv_host_to_device_stream == nullptr) {
    done(errors::Internal(
        "No send PluggableDevice copy-out-stream is available."));
    return;
  }
  // Wait for the recv-stream to make sure the buffer is truly available.
  if (sync_dst_compute) {
    s = recv_host_to_device_stream->WaitFor(recv_stream);
    if (!s.ok()) {
      done(s);
      return;
    }
  }
  const int64_t total_bytes = cpu_tensor->TotalBytes();
  // Note that 0-size tensors have no backing buffer.
  if (total_bytes > 0) {
    void* src_ptr = GetBase(cpu_tensor);
    void* dst_ptr = GetBase(device_tensor);
    DeviceMemoryBase device_dst_ptr(dst_ptr, total_bytes);
    s = recv_host_to_device_stream->Memcpy(&device_dst_ptr, src_ptr,
                                           total_bytes);
    if (!s.ok()) {
      done(s);
      return;
    }
  }
  // Use of cpu_tensor may outlive stack scope, so keep a ref.
  TensorReference input_ref(*cpu_tensor);
  dev_info->event_mgr->ThenExecute(
      recv_host_to_device_stream,
      [recv_host_to_device_stream, done, input_ref]() {
        input_ref.Unref();
        if (!recv_host_to_device_stream->ok()) {
          LOG(FATAL) << "CPU->PluggableDevice Memcpy failed.";  // Crash OK
        }
        done(absl::OkStatus());
      });
}

absl::Status PluggableDeviceUtil::Sync(Device* device) {
  VLOG(1) << "PluggableDeviceUtil::Sync";
  auto* dev_info = device->tensorflow_accelerator_device_info();
  if (!dev_info) {
    return errors::Internal("Failed to find dest device GPUDeviceInfo.");
  }
  return dev_info->stream->BlockHostUntilDone();
}

absl::Status PluggableDeviceUtil::SyncAll(Device* device) {
  VLOG(1) << "PluggableDeviceUtil::SyncAll";
  auto* dev_info = device->tensorflow_accelerator_device_info();
  if (!dev_info) {
    return errors::Internal("Failed to find dest device GPUDeviceInfo.");
  }
  if (!dev_info->stream->parent()->SynchronizeAllActivity() ||
      !dev_info->stream->ok()) {
    return errors::Internal("PluggableDevice SyncAll failed.");
  }
  return absl::OkStatus();
}

// static
void PluggableDeviceUtil::CopyPluggableDeviceTensorToSameDevice(
    Device* device, const DeviceContext* device_context,
    const Tensor* src_device_tensor, Tensor* dst_device_tensor,
    StatusCallback done) {
  VLOG(1) << "CopyPluggableDeviceTensorToSameDevice";
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  absl::Status s = PrepareCopy(device, device_context, *src_device_tensor,
                               dst_device_tensor, &dev_info, &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  const int64_t total_bytes = src_device_tensor->TotalBytes();
  if (total_bytes > 0) {
    void* src_ptr = GetBase(src_device_tensor);
    DeviceMemoryBase device_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(dst_device_tensor);
    DeviceMemoryBase device_dst_ptr(dst_ptr, total_bytes);
    auto status =
        send_stream->Memcpy(&device_dst_ptr, device_src_ptr, total_bytes);
    if (!status.ok()) {
      done(status);
      return;
    }
  }

  done(absl::OkStatus());
}

}  // namespace tensorflow
