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

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {

// Plugin initialization function that a device plugin
// must define.
typedef void (*SEInitPluginFn)(SE_PlatformRegistrationParams* const,
                               TF_Status* const);

// Registers StreamExecutor platform. `device_type` and `platform_name` are
// output parameters.
port::Status InitStreamExecutorPlugin(void* dso_handle,
                                      std::string* device_type,
                                      std::string* platform_name);

// Allow registering a StreamExecutor plugin using a function (used for
// testing).
port::Status InitStreamExecutorPlugin(SEInitPluginFn init_fn,
                                      std::string* device_type,
                                      std::string* platform_name);

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
  port::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;
  port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;
  port::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& plugin_config) override;
  port::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override;
  port::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) override;

  // Trace listener is not supported
  void RegisterTraceListener(std::unique_ptr<TraceListener> listener) override {
    LOG(FATAL) << "RegisterTraceListener is not supported by pluggable device";
  }
  void UnregisterTraceListener(TraceListener* listener) override {}

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

class CStream : public internal::StreamInterface {
 public:
  CStream(SP_Device* device, SP_StreamExecutor* stream_executor)
      : device_(device),
        stream_executor_(stream_executor),
        stream_handle_(nullptr) {}
  ~CStream() override { Destroy(); }

  port::Status Create() {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->create_stream(device_, &stream_handle_, c_status.get());
    port::Status s = tensorflow::StatusFromTF_Status(c_status.get());
    return s;
  }

  void Destroy() {
    if (stream_handle_ != nullptr) {
      stream_executor_->destroy_stream(device_, stream_handle_);
      stream_handle_ = nullptr;
    }
  }

  SP_Stream Handle() { return stream_handle_; }

 private:
  SP_Device* device_;
  SP_StreamExecutor* stream_executor_;
  SP_Stream stream_handle_;
};

class CEvent : public internal::EventInterface {
 public:
  CEvent(SP_Device* device, SP_StreamExecutor* stream_executor)
      : device_(device),
        stream_executor_(stream_executor),
        event_handle_(nullptr) {}
  ~CEvent() override { Destroy(); }

  port::Status Create() {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->create_event(device_, &event_handle_, c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }

  port::Status Record(SP_Stream stream_handle) {
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

class CTimer : public internal::TimerInterface {
 public:
  CTimer(SP_Device* device, SP_StreamExecutor* stream_executor,
         SP_TimerFns* timer_fns)
      : device_(device),
        stream_executor_(stream_executor),
        timer_handle_(nullptr),
        timer_fns_(timer_fns) {}
  ~CTimer() override { Destroy(); }

  port::Status Create() {
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->create_timer(device_, &timer_handle_, c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }

  void Destroy() {
    if (timer_handle_ != nullptr) {
      stream_executor_->destroy_timer(device_, timer_handle_);
      timer_handle_ = nullptr;
    }
  }

  SP_Timer Handle() { return timer_handle_; }

  uint64 Microseconds() const override {
    return timer_fns_->nanoseconds(timer_handle_) / 1000;
  }

  uint64 Nanoseconds() const override {
    return timer_fns_->nanoseconds(timer_handle_);
  }

 private:
  SP_Device* device_;
  SP_StreamExecutor* stream_executor_;
  SP_Timer timer_handle_;
  SP_TimerFns* timer_fns_;
};

}  // namespace stream_executor
#endif  // TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
