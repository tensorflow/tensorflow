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
// Classes and utilities that work with StreamExecutor C API for internal use.
// This includes functions used for device registration and interfaces needed
// for testing.
#ifndef TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
#define TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

// Plugin initialization function that a device plugin
// must define.
typedef void (*SEInitPluginFn)(SE_PlatformRegistrationParams* const,
                               TF_Status* const);

// Registers StreamExecutor platform. `device_type` and `platform_name` are
// output parameters.
absl::Status InitStreamExecutorPlugin(void* dso_handle,
                                      std::string* device_type,
                                      std::string* platform_name);

// Allow registering a StreamExecutor plugin using a function (used for
// testing).
absl::Status InitStreamExecutorPlugin(SEInitPluginFn init_fn,
                                      std::string* device_type,
                                      std::string* platform_name);

// Converts DeviceMemoryBase to a C struct.
inline SP_DeviceMemoryBase DeviceMemoryBaseToC(const DeviceMemoryBase* mem) {
  SP_DeviceMemoryBase device_memory_base{SP_DEVICE_MEMORY_BASE_STRUCT_SIZE};
  // `opaque` field inside SP_DeviceMemoryBase is not const.
  // Therefore, we need to cast away the constness before setting it.
  device_memory_base.opaque = const_cast<void*>(mem->opaque());
  device_memory_base.size = mem->size();
  device_memory_base.payload = mem->payload();
  return device_memory_base;
}

// This file implements core stream executor base classes in terms of
// the C API defined in stream_executor.h. A class "CSomething" represents a
// "Something" that can be manipulated via calls in the C interface.
class CPlatform : public Platform {
 public:
  explicit CPlatform(SP_Platform platform,
                     void (*destroy_platform)(SP_Platform*),
                     SP_PlatformFns platform_fns,
                     void (*destroy_platform_fns)(SP_PlatformFns*),
                     SP_DeviceFns device_fns, SP_StreamExecutor stream_executor,
                     SP_TimerFns timer_fns);
  ~CPlatform() override;

  Id id() const override { return const_cast<int*>(&plugin_id_value_); }
  const std::string& Name() const override { return name_; }
  int VisibleDeviceCount() const override {
    int visible_device_count = 0;
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    platform_fns_.get_device_count(&platform_, &visible_device_count,
                                   c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return 0;
    }
    return visible_device_count;
  }
  bool UseBfcAllocator() const { return platform_.use_bfc_allocator; }
  bool ForceMemoryGrowth() const { return platform_.force_memory_growth; }
  absl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;
  absl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;
  absl::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override;
  absl::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) override;

  void DestroyAllExecutors() { executor_cache_.DestroyAllExecutors(); }

 private:
  SP_Platform platform_;
  void (*destroy_platform_)(SP_Platform*);
  SP_PlatformFns platform_fns_;
  void (*destroy_platform_fns_)(SP_PlatformFns*);
  SP_DeviceFns device_fns_;
  SP_StreamExecutor stream_executor_;
  SP_TimerFns timer_fns_;
  const std::string name_;
  int plugin_id_value_;
  stream_executor::ExecutorCache executor_cache_;
};

class CEvent : public Event {
 public:
  CEvent(SP_Device* device, SP_StreamExecutor* stream_executor)
      : device_(device),
        stream_executor_(stream_executor),
        event_handle_(nullptr) {}
  ~CEvent() override { Destroy(); }

  Event::Status PollForStatus() override {
    SE_EventStatus event_status =
        stream_executor_->get_event_status(device_, event_handle_);

    switch (event_status) {
      case SE_EVENT_ERROR:
        return Event::Status::kError;
      case SE_EVENT_PENDING:
        return Event::Status::kPending;
      case SE_EVENT_COMPLETE:
        return Event::Status::kComplete;
      default:
        return Event::Status::kUnknown;
    }
  }

  absl::Status Create() {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->create_event(device_, &event_handle_, c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }

  absl::Status Record(SP_Stream stream_handle) {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->record_event(device_, stream_handle, event_handle_,
                                   c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }

  void Destroy() {
    if (event_handle_ != nullptr) {
      stream_executor_->destroy_event(device_, event_handle_);
      event_handle_ = nullptr;
    }
  }

  SP_Event Handle() { return event_handle_; }

 private:
  SP_Device* device_;
  SP_StreamExecutor* stream_executor_;
  SP_Event event_handle_;
};

class CStream : public StreamCommon {
 public:
  CStream(SP_Device* device, SP_StreamExecutor* stream_executor,
          StreamExecutor* executor)
      : StreamCommon(executor),
        device_(device),
        stream_executor_(stream_executor),
        stream_handle_(nullptr) {}
  ~CStream() override {
    parent()->BlockHostUntilDone(this).IgnoreError();
    parent()->DeallocateStream(this);
    Destroy();
  }

  absl::Status Create() {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->create_stream(device_, &stream_handle_, c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }

  void Destroy() {
    if (stream_handle_ != nullptr) {
      stream_executor_->destroy_stream(device_, stream_handle_);
      stream_handle_ = nullptr;
    }
  }
  absl::Status RefreshStatus() override {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->get_stream_status(device_, stream_handle_,
                                        c_status.get());
    absl::Status status = tensorflow::StatusFromTF_Status(c_status.get());
    CheckStatus(status);
    return status;
  }

  absl::Status RecordEvent(Event* event) override {
    return static_cast<CEvent*>(event)->Record(stream_handle_);
  }

  absl::Status WaitFor(Stream* other) override {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    SP_Stream other_handle = static_cast<CStream*>(other)->Handle();
    stream_executor_->create_stream_dependency(device_, stream_handle_,
                                               other_handle, c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }
  absl::Status WaitFor(Event* event) override {
    SP_Event event_handle = static_cast<CEvent*>(event)->Handle();
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->wait_for_event(device_, stream_handle_, event_handle,
                                     c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }
  absl::Status MemZero(DeviceMemoryBase* location, uint64_t size) override {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_mem = DeviceMemoryBaseToC(location);
    stream_executor_->mem_zero(device_, stream_handle_, &device_mem, size,
                               c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }
  absl::Status Memset32(DeviceMemoryBase* location, uint32_t pattern,
                        uint64_t size) override {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_mem = DeviceMemoryBaseToC(location);
    stream_executor_->memset32(device_, stream_handle_, &device_mem, pattern,
                               size, c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }
  absl::Status Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                      uint64_t size) override {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_mem_dst = DeviceMemoryBaseToC(gpu_dst);
    stream_executor_->memcpy_htod(device_, stream_handle_, &device_mem_dst,
                                  host_src, size, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
    }
    return tensorflow::StatusFromTF_Status(c_status.get());
  }
  absl::Status Memcpy(DeviceMemoryBase* gpu_dst,
                      const DeviceMemoryBase& gpu_src, uint64_t size) override {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_mem_dst = DeviceMemoryBaseToC(gpu_dst);
    SP_DeviceMemoryBase device_mem_src = DeviceMemoryBaseToC(&gpu_src);
    stream_executor_->memcpy_dtod(device_, stream_handle_, &device_mem_dst,
                                  &device_mem_src, size, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
    }
    return tensorflow::StatusFromTF_Status(c_status.get());
  }
  absl::Status Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                      uint64_t size) override {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_mem_src = DeviceMemoryBaseToC(&gpu_src);
    stream_executor_->memcpy_dtoh(device_, stream_handle_, host_dst,
                                  &device_mem_src, size, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
    }
    return tensorflow::StatusFromTF_Status(c_status.get());
  }
  SP_Stream Handle() { return stream_handle_; }

 private:
  SP_Device* device_;
  SP_StreamExecutor* stream_executor_;
  SP_Stream stream_handle_;
};

}  // namespace stream_executor
#endif  // TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
