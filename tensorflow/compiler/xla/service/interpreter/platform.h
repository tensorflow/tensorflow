/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_PLATFORM_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_PLATFORM_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/interpreter/platform_id.h"
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/plugin.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/trace_listener.h"

namespace stream_executor {
namespace interpreter {

class XlaInterpreterPlatform : public Platform {
 public:
  XlaInterpreterPlatform()
      : XlaInterpreterPlatform("Interpreter", kXlaInterpreterPlatformId) {}
  XlaInterpreterPlatform(const std::string& name, const Platform::Id& id);
  ~XlaInterpreterPlatform() override;

  Platform::Id id() const override;

  int VisibleDeviceCount() const override;

  const std::string& Name() const override;

  port::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

  port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;

  port::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& config) override;

  port::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override;

  port::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) override;

  void RegisterTraceListener(std::unique_ptr<TraceListener> listener) override;

  void UnregisterTraceListener(TraceListener* listener) override;

 private:
  // This platform's name.
  std::string name_;
  // This platform's id.
  Platform::Id id_;

  // Cache of created StreamExecutors.
  ExecutorCache executor_cache_;

  SE_DISALLOW_COPY_AND_ASSIGN(XlaInterpreterPlatform);
};

}  // namespace interpreter
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INTERPRETER_PLATFORM_H_
