/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_STREAM_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_STREAM_H_

#include <cstdint>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_executor_api.h"
#include "xla/stream_executor/tpu/tpu_executor_c_api.h"
#include "xla/stream_executor/tpu/tpu_platform.h"
#include "xla/stream_executor/tpu/tpu_stream_interface.h"

namespace tensorflow {
namespace tpu {

class TpuStream : public tensorflow::tpu::TpuStreamInterface {
 public:
  explicit TpuStream(SE_Stream* stream,
                     stream_executor::StreamExecutor* executor,
                     SE_StreamExecutor* se_executor,
                     tensorflow::tpu::TpuPlatform* tpu_platform)
      : TpuStreamInterface(executor),
        stream_(stream),
        se_executor_(se_executor),
        tpu_platform_(tpu_platform) {}
  ~TpuStream() override {
    BlockHostUntilDone().IgnoreError();
    parent()->DeallocateStream(this);
    stream_executor::tpu::ExecutorApiFn()->TpuStream_FreeFn(stream_);
  }

  bool IsSameSharedMemoryLocation(
      tensorflow::tpu::TpuStreamInterface* other) override {
    return stream_executor::tpu::ExecutorApiFn()
        ->TpuStream_IsSameSharedMemoryLocationFn(
            stream_, static_cast<TpuStream*>(other)->stream_);
  }

  absl::Status EnqueueTransferHostToDevice(
      stream_executor::DeviceMemoryBase device_dst, const void* host_src,
      uint64_t size) {
    StatusHelper status;
    stream_executor::tpu::ExecutorApiFn()
        ->TpuStream_EnqueueTransferHostToDeviceFn(
            stream_, ApiConverter::ToC(device_dst), const_cast<void*>(host_src),
            size, status.c_status);
    return status.status();
  }

  absl::Status EnqueueTransferDeviceToHost(
      stream_executor::DeviceMemoryBase device_src, void* host_dst,
      uint64_t size) {
    StatusHelper status;
    stream_executor::tpu::ExecutorApiFn()
        ->TpuStream_EnqueueTransferDeviceToHostFn(
            stream_, ApiConverter::ToC(device_src), host_dst, size,
            status.c_status);
    return status.status();
  }

  absl::Status EnqueueOnTpuDeviceSendRecvLocal(
      stream_executor::DeviceMemoryBase send_buffer,
      stream_executor::DeviceMemoryBase recv_buffer) override {
    StatusHelper status;
    stream_executor::tpu::ExecutorApiFn()
        ->TpuStream_TpuEnqueueOnDeviceSendRecvLocalFn(
            stream_, ApiConverter::ToC(send_buffer),
            ApiConverter::ToC(recv_buffer), status.c_status);
    return status.status();
  }

  absl::Status WaitFor(stream_executor::Stream* stream) override {
    if (stream_executor::tpu::ExecutorApiFn()
            ->TpuExecutor_CreateStreamDependencyFn(
                se_executor_, stream_, tpu_platform_->LookupStream(stream))) {
      return absl::OkStatus();
    }
    return absl::InternalError("Failed to create stream dependency");
  }

  absl::Status WaitFor(stream_executor::Event* event) override {
    StatusHelper status;
    auto se_event = tpu_platform_->LookupEvent(event);
    stream_executor::tpu::ExecutorApiFn()->TpuExecutor_WaitForEventFn(
        se_executor_, stream_, se_event, status.c_status);
    return status.status();
  }

  absl::Status RefreshStatus() override {
    StatusHelper status;
    stream_executor::tpu::ExecutorApiFn()->TpuExecutor_GetStatusFn(
        se_executor_, stream_, status.c_status);
    CheckStatus(status.status());
    return status.status();
  }

  absl::Status RecordEvent(stream_executor::Event* event) override {
    StatusHelper status;
    auto se_event = tpu_platform_->LookupEvent(event);
    stream_executor::tpu::ExecutorApiFn()->TpuExecutor_RecordEventFn(
        se_executor_, stream_, se_event, status.c_status);
    return status.status();
  }

  absl::Status Memcpy(stream_executor::DeviceMemoryBase* device_dst,
                      const void* host_src, uint64_t size) override {
    StatusHelper status;
    SE_DeviceMemoryBase se_base = ApiConverter::ToC(*device_dst);
    stream_executor::tpu::ExecutorApiFn()->TpuExecutor_MemcpyFromHostFn(
        se_executor_, stream_, &se_base, host_src, size, status.c_status);
    return status.status();
  }
  absl::Status Memcpy(stream_executor::DeviceMemoryBase* device_dst,
                      const stream_executor::DeviceMemoryBase& device_src,
                      uint64_t size) override {
    return absl::UnimplementedError(
        "Memcpy from device to deviceis not implemented for TPU");
  }
  absl::Status Memcpy(void* host_dst,
                      const stream_executor::DeviceMemoryBase& device_src,
                      uint64_t size) override {
    StatusHelper status;
    SE_DeviceMemoryBase se_base = ApiConverter::ToC(device_src);
    stream_executor::tpu::ExecutorApiFn()->TpuExecutor_MemcpyToHostFn(
        se_executor_, stream_, host_dst, &se_base, size, status.c_status);
    return status.status();
  }
  struct HostCallbackContext {
    absl::AnyInvocable<absl::Status() &&> callback;
  };
  static TSL_Status* HostCallbackTrampoline(void* ctx) {
    HostCallbackContext* host_ctx = reinterpret_cast<HostCallbackContext*>(ctx);
    absl::Status status = std::move(host_ctx->callback)();
    TSL_Status* c_status =
        stream_executor::tpu::ExecutorApiFn()->TpuStatus_CreateFn(
            status.raw_code(), absl::StatusMessageAsCStr(status));
    delete host_ctx;
    return c_status;
  }
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override {
    HostCallbackContext* ctx = new HostCallbackContext{std::move(callback)};
    if (stream_executor::tpu::ExecutorApiFn()->TpuExecutor_HostCallbackFn(
            se_executor_, stream_, &HostCallbackTrampoline, ctx)) {
      return absl::OkStatus();
    }
    return absl::InternalError("Failed to  host callback.");
  }

  SE_Stream* se_stream() const { return stream_; }

 private:
  mutable SE_Stream* stream_;
  SE_StreamExecutor* se_executor_;
  tensorflow::tpu::TpuPlatform* tpu_platform_;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_STREAM_H_
