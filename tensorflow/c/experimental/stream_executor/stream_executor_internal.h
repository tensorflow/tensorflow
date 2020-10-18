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
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {

// Plugin initialization function that a device plugin
// must define.
typedef void (*SEInitPluginFn)(SE_PlatformRegistrationParams* const,
                               TF_Status* const);

// Registers StreamExecutor platform.
port::Status InitStreamExecutorPlugin(void* dso_handle);

// Allow registering a StreamExecutor plugin using a function (used for
// testing).
port::Status InitStreamExecutorPlugin(SEInitPluginFn init_fn);

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
    return platform_.visible_device_count;
  }
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

}  // namespace stream_executor
#endif  // TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
