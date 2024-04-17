/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_util.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/status/status.h"

// TODO(b/282059652): Merge google internal and open-source code path once TF
// dependency issue is resolved.
#if (defined(PLATFORM_GOOGLE) && defined(TF_PLATFORM_LINUX_X86_64))
#define TF_GPU_USE_PJRT
#endif  // PLATFORM_GOOGLE && TF_PLATFORM_LINUX_X86_64

#ifdef TF_GPU_USE_PJRT
#include "tensorflow/compiler/jit/pjrt_tensor_buffer.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_future.h"
#endif  // TF_GPU_USE_PJRT

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/util.h"

// IMPLEMENTATION NOTE:
//
// 1. Within this module, we intentionally LOG(FATAL) if any stream
//    involved in memcpy becomes !stream->ok(), because TF process
//    today (1/2016) can not properly recover from such an error.
//
// 2. When 0-size tensor is being copied, we should not schedule a
//    copy Memcpy since there is no byte to move. However, we must
//    ensure the causal ordering by arranging the copy done callback
//    happens-after all activities scheduled on the given stream being
//    finished.

// If this need to be runtime configurable, consider adding options to
// ConfigProto.
const int64_t FLAGS_brain_gpu_util_debug_string_maxlen = 128;
extern bool FLAGS_brain_gpu_record_mem_types;

namespace tensorflow {

using se::DeviceMemoryBase;
using se::Stream;

Status PrepareCopy(Device* device, const DeviceContext* ctx, const Tensor& src,
                   const Tensor* dst,
                   const DeviceBase::AcceleratorDeviceInfo** dev_info,
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
  auto gs = static_cast<const GPUDeviceContext*>(ctx)->stream();
  if (gs == nullptr) {
    return errors::Internal("No gpu stream is available.");
  }
  *stream = gs;
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
    return errors::Internal("GPU copy from non-DMA ",
                            DataTypeString(src.dtype()), " tensor");
  }
  return absl::OkStatus();
}

void* GetBase(const Tensor* src) {
  return const_cast<void*>(DMAHelper::base(src));
}

void* GetBase(Tensor* dst) { return DMAHelper::base(dst); }

/*static*/
void GPUUtil::SetProtoFromGPU(const Tensor& tensor, Device* dev,
                              const DeviceContext* device_context,
                              TensorProto* proto, bool is_dead,
                              StatusCallback done) {
  VLOG(1) << "SetProtoFromGPU device_context " << device_context;
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(dev, device_context, tensor, nullptr, &dev_info,
                         &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto send_device_to_host_stream =
      static_cast<const GPUDeviceContext*>(device_context)
          ->device_to_host_stream();
  if (send_device_to_host_stream == nullptr) {
    done(errors::Internal("No send gpu copy-out-stream is available."));
    return;
  }
  // Wait for the sender's main stream to make sure the data are available.
  s = send_device_to_host_stream->WaitFor(send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  // Tensor values need to be copied from GPU to CPU ram so that
  // we can build the protobuf response for a RecvTensor RPC.
  // "device context" identifies the stream where the _Send op executed.
  proto->set_dtype(tensor.dtype());
  tensor.shape().AsProto(proto->mutable_tensor_shape());

  // Prepare a proto with the right data buf size, and DMA the data
  // over from the GPU buffer.  Note that 0-size tensors do not have a
  // backing buffer.
  Allocator* alloc = nullptr;
  char* buf = nullptr;
  const int64_t total_bytes = is_dead ? 0 : tensor.TotalBytes();
  if (total_bytes > 0) {
    tsl::profiler::ScopedAnnotation annotation("SetProtoFromGPU");
    alloc =
        GPUProcessState::singleton()->GetGpuHostAllocator(/*options=*/{}, 0);
    buf = static_cast<char*>(
        alloc->AllocateRaw(Allocator::kAllocatorAlignment, total_bytes));
    if (LogMemory::IsEnabled()) {
      LogMemory::RecordRawAllocation("SetProtoFromGPU",
                                     LogMemory::PROTO_BUFFER_STEP_ID,
                                     total_bytes, buf, alloc);
    }
    void* src_ptr = GetBase(&tensor);
    DeviceMemoryBase gpu_src_ptr(src_ptr, total_bytes);
    s = send_device_to_host_stream->Memcpy(buf, gpu_src_ptr, total_bytes);
    if (!s.ok()) {
      done(s);
      return;
    }
  }
  // Use of tensor may outlive stack scope, so keep a ref.
  TensorReference tensor_ref(tensor);
  dev_info->event_mgr->ThenExecute(
      send_device_to_host_stream, [send_device_to_host_stream, done, proto, buf,
                                   total_bytes, alloc, tensor_ref]() {
        if (!send_device_to_host_stream->ok()) {
          LOG(FATAL) << "SetProtoFromGPU: GPU Memcpy failed";
        }
        tensor_ref.Unref();
        if (total_bytes > 0) {
          port::CopyFromArray(proto->mutable_tensor_content(), buf,
                              total_bytes);
          if (LogMemory::IsEnabled()) {
            LogMemory::RecordRawDeallocation("SetProtoFromGPU",
                                             LogMemory::PROTO_BUFFER_STEP_ID,
                                             buf, alloc, false);
          }
          alloc->DeallocateRaw(buf);
        }
        done(absl::OkStatus());
      });
}

// static
void GPUUtil::DeviceToDeviceCopy(
    DeviceContext* send_dev_context, DeviceContext* recv_dev_context,
    Device* src, Device* dst, AllocatorAttributes src_alloc_attr,
    AllocatorAttributes dst_alloc_attr, const Tensor* input, Tensor* output,
    int dev_to_dev_stream_index, StatusCallback done) {
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(src, send_dev_context, *input, output, &dev_info,
                         &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }
  auto send_device_to_device_stream =
      static_cast<const GPUDeviceContext*>(send_dev_context)
          ->device_to_device_stream(dev_to_dev_stream_index);
  if (send_device_to_device_stream == nullptr) {
    done(errors::Internal("No send gpu copy-out-stream is available."));
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
    DeviceMemoryBase gpu_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(output);
    DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);
    // For GpuDevice, always gets receive stream from
    // dst->tensorflow_accelerator_device_info()->default_context which is
    // GPUDeviceContext.
    stream_executor::Stream* recv_stream =
        static_cast<const GPUDeviceContext*>(
            dst->tensorflow_accelerator_device_info()->default_context)
            ->stream();
    if (recv_stream == nullptr) {
      done(errors::Internal("No recv gpu stream is available."));
      return;
    }
    // Since we want to use the memory from recv_stream in the
    // send_device_to_device_stream, add a dependency to make sure the memory is
    // truly free.
    // TODO(zhengxq): remove this dependency when we switch to a better way
    // to make sure the memory is free.
    s = send_device_to_device_stream->WaitFor(recv_stream);
    if (!s.ok()) {
      done(s);
      return;
    }

    VLOG(2) << "src_ptr " << src_ptr << " dst_ptr " << dst_ptr;
    s = send_device_to_device_stream->Memcpy(&gpu_dst_ptr, gpu_src_ptr,
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
          LOG(FATAL) << "GPU->GPU Memcpy failed";
        }
        done(absl::OkStatus());
      });
  send_dev_context->MaintainLifetimeOnStream(input,
                                             send_device_to_device_stream);
}

static CopyTensor::Registration register_gpu_gpu_copy(
    DEVICE_GPU, DEVICE_GPU, GPUUtil::DeviceToDeviceCopy);

namespace {

// Returns whether staging is needed based on tensor buffer's memory type.
bool NeedStaging(const Tensor* tensor) {
  // Only stage data if the host tensor is on pageable memory.
  // So if the memory type is unknown, it will fallback to GPU driver to handle
  // the staging if needed.
  return tensor->GetMemoryType() == AllocatorMemoryType::kHostPageable;
}

}  // namespace

void GPUUtil::CopyGPUTensorToCPU(Device* gpu_device,
                                 const DeviceContext* device_context,
                                 const Tensor* gpu_tensor, Tensor* cpu_tensor,
                                 StatusCallback done) {
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(gpu_device, device_context, *gpu_tensor, cpu_tensor,
                         &dev_info, &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto send_device_to_host_stream =
      static_cast<const GPUDeviceContext*>(device_context)
          ->device_to_host_stream();
  if (send_device_to_host_stream == nullptr) {
    done(absl::InternalError("No send gpu copy-out-stream is available."));
    return;
  }
  // Wait for the sender's main stream to make sure the data are available.
  s = send_device_to_host_stream->WaitFor(send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

#ifdef TF_GPU_USE_PJRT
  // The above `WaitFor(send_stream)` for the PjRt case eliminates race
  // conditions caused by either non-XLA ops that have not finished or a case in
  // the PJRT client implementation where an event on the buffer is not sent
  // properly. A possible future improvement is to specifically handle the
  // relevant case(s) and move this TF_GPU_USE_PJRT codeblock to the start of
  // this function (avoiding the need for `WaitFor(send_stream)` for the
  // PjRt case).
  const PjRtTensorBuffer* pjrt_tensor_buffer =
      dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(gpu_tensor));
  if (pjrt_tensor_buffer != nullptr) {
    VLOG(1) << "CopyGPUTensorToCPU using PjRtTensorBuffer";
    auto literal = std::make_unique<xla::MutableBorrowingLiteral>();
    auto status = tensorflow::HostTensorToMutableBorrowingLiteral(
        cpu_tensor, literal.get());
    xla::PjRtFuture<> future =
        pjrt_tensor_buffer->pjrt_buffer()->ToLiteral(literal.get());
    future.OnReady([literal = std::move(literal),
                    done](const tensorflow::Status& status) { done(status); });
    return;
  }
#endif  // TF_GPU_USE_PJRT

  VLOG(1) << "CopyGPUTensorToCPU using AcceleratorDeviceInfo";
  const int64_t total_bytes = gpu_tensor->TotalBytes();
  if (total_bytes > 0) {
    void* src_ptr = GetBase(gpu_tensor);
    DeviceMemoryBase gpu_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(cpu_tensor);
    s = send_device_to_host_stream->Memcpy(dst_ptr, gpu_src_ptr, total_bytes);
    if (!s.ok()) {
      done(s);
      return;
    }
  }
  // Use of the input may outlive stack scope, so keep a ref.
  TensorReference input_ref(*gpu_tensor);
  dev_info->event_mgr->ThenExecute(
      send_device_to_host_stream,
      [send_device_to_host_stream, done, input_ref]() {
        if (!send_device_to_host_stream->ok()) {
          LOG(FATAL) << "GPU->CPU Memcpy failed";  // Crash OK
        }
        input_ref.Unref();
        done(absl::OkStatus());
      });
}

void GPUUtil::CopyCPUTensorToGPU(const Tensor* cpu_tensor,
                                 const DeviceContext* device_context,
                                 Device* gpu_device, Tensor* gpu_tensor,
                                 StatusCallback done, bool sync_dst_compute) {
  VLOG(1) << "CopyCPUTensorToGPU";
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* recv_stream = nullptr;
  Status s = PrepareCopy(gpu_device, device_context, *cpu_tensor, gpu_tensor,
                         &dev_info, &recv_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto recv_host_to_device_stream =
      static_cast<const GPUDeviceContext*>(device_context)
          ->host_to_device_stream();
  if (recv_host_to_device_stream == nullptr) {
    done(errors::Internal("No send gpu copy-out-stream is available."));
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

  bool do_staging = false;
  void* staging_buffer = nullptr;
  Allocator* host_memory_allocator = device_context->host_memory_allocator();

  // Use of cpu_tensor may outlive stack scope, so keep a ref.
  TensorReference input_ref(*cpu_tensor);

  // Note that 0-size tensors have no backing buffer.
  if (total_bytes > 0) {
    void* src_ptr = GetBase(cpu_tensor);
    void* dst_ptr = GetBase(gpu_tensor);
    DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);

    if (NeedStaging(cpu_tensor)) {
      if (host_memory_allocator == nullptr) {
        LOG_FIRST_N(WARNING, 1)
            << "No host memory allocator is available to "
               "stage data for CPU->GPU transfer. Staging will be skipped.";
      } else {
        do_staging = true;
      }
    }

    if (do_staging) {
      staging_buffer = host_memory_allocator->AllocateRaw(
          tensorflow::Allocator::kAllocatorAlignment, total_bytes);
      std::memcpy(staging_buffer, src_ptr, total_bytes);
      input_ref.Unref();

      s = recv_host_to_device_stream->Memcpy(&gpu_dst_ptr, staging_buffer,
                                             total_bytes);
    } else {
      s = recv_host_to_device_stream->Memcpy(&gpu_dst_ptr, src_ptr,
                                             total_bytes);
    }
    if (!s.ok()) {
      done(s);
      return;
    }
  }

  dev_info->event_mgr->ThenExecute(
      recv_host_to_device_stream,
      [recv_host_to_device_stream, done, input_ref, do_staging, staging_buffer,
       host_memory_allocator]() {
        if (do_staging) {
          host_memory_allocator->DeallocateRaw(staging_buffer);
        } else {
          input_ref.Unref();
        }
        if (!recv_host_to_device_stream->ok()) {
          LOG(FATAL) << "CPU->GPU Memcpy failed";
        }
        done(absl::OkStatus());
      });
}

Status GPUUtil::Sync(Device* gpu_device) {
  VLOG(1) << "GPUUtil::Sync";
  auto* dev_info = gpu_device->tensorflow_accelerator_device_info();
  if (!dev_info) {
    return errors::Internal("Failed to find dest device GPUDeviceInfo");
  }
  return dev_info->stream->BlockHostUntilDone();
}

Status GPUUtil::SyncAll(Device* gpu_device) {
  VLOG(1) << "GPUUtil::SyncAll";
  auto* dev_info = gpu_device->tensorflow_accelerator_device_info();
  if (!dev_info) {
    return errors::Internal("Failed to find dest device GPUDeviceInfo");
  }
  if (!dev_info->stream->parent()->SynchronizeAllActivity() ||
      !dev_info->stream->ok()) {
    return errors::Internal("GPU sync failed");
  }
  return absl::OkStatus();
}

string GPUUtil::MemoryDebugString(const Device* device, Tensor* tensor) {
  string ret;
  CHECK(tensor);
  const int64_t num_bytes = std::min<int64_t>(
      FLAGS_brain_gpu_util_debug_string_maxlen, tensor->TotalBytes());
  void* ptr = (num_bytes > 0) ? GetBase(tensor) : nullptr;
  strings::Appendf(&ret, "%p:", ptr);
  if (num_bytes > 0) {
    auto* dev_info = device->tensorflow_accelerator_device_info();
    if (!dev_info) {
      strings::StrAppend(
          &ret, PrintMemory(reinterpret_cast<const char*>(ptr), num_bytes));
    } else {
      string buf;
      buf.resize(num_bytes);
      DeviceMemoryBase gpu_ptr(ptr, num_bytes);
      auto s = dev_info->stream->parent()->SynchronousMemcpyD2H(
          gpu_ptr, num_bytes, &*buf.begin());
      strings::StrAppend(&ret, PrintMemory(&*buf.begin(), num_bytes));
    }
  }
  return ret;
}

// TODO(pbar) Checksum is called from places without a valid device context.
uint64 GPUUtil::Checksum(Device* gpu_device,
                         const DeviceContext* device_context,
                         const Tensor& tensor) {
  Tensor copy(tensor.dtype(), tensor.shape());
  Status s;
  Notification n;
  CopyGPUTensorToCPU(gpu_device, device_context, &tensor, &copy,
                     [&s, &n](Status status) {
                       s.Update(status);
                       n.Notify();
                     });
  n.WaitForNotification();
  CHECK(s.ok()) << s;
  return Checksum(copy);
}

uint64 GPUUtil::Checksum(const Tensor& tensor) {
  const float* fptr = reinterpret_cast<const float*>(GetBase(&tensor));
  size_t num_bytes = tensor.TotalBytes();
  size_t num_floats = num_bytes / sizeof(float);
  for (size_t i = 0; i < num_floats; ++i) {
    CHECK(!std::isnan(fptr[i])) << " i " << i;
  }
  // TODO(tucker): consider using crc32c instead.
  return Hash64(reinterpret_cast<const char*>(GetBase(&tensor)),
                tensor.TotalBytes(), 0);
}

// static
void GPUUtil::CopyGPUTensorToSameGPU(Device* gpu_device,
                                     const DeviceContext* device_context,
                                     const Tensor* src_gpu_tensor,
                                     Tensor* dst_gpu_tensor,
                                     StatusCallback done) {
  VLOG(1) << "CopyGPUTensorToSameGPU";
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(gpu_device, device_context, *src_gpu_tensor,
                         dst_gpu_tensor, &dev_info, &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  const int64_t total_bytes = src_gpu_tensor->TotalBytes();
  if (total_bytes > 0) {
    void* src_ptr = GetBase(src_gpu_tensor);
    DeviceMemoryBase gpu_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(dst_gpu_tensor);
    DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);
    s = send_stream->Memcpy(&gpu_dst_ptr, gpu_src_ptr, total_bytes);
    if (!s.ok()) {
      done(s);
      return;
    }
  }

  done(absl::OkStatus());
}

}  // namespace tensorflow
