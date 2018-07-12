/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PLATFORM_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PLATFORM_H_

#include <list>
#include <memory>
#include <string>

#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/trace_listener.h"

#include "tensorflow/core/platform/mutex.h"

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace se = stream_executor;

namespace tensorflow {
class IpuTraceEvent;
}

namespace xla {
namespace poplarplugin {

class PoplarPlatform : public se::Platform {
 public:
  PoplarPlatform();
  ~PoplarPlatform() override;

  Platform::Id id() const override;

  // Device count is less clear-cut for CPUs than accelerators. This call
  // currently returns the number of thread units in the host, as reported by
  // base::NumCPUs().
  int VisibleDeviceCount() const override;

  const std::string& Name() const override;

  StatusOr<se::StreamExecutor*> ExecutorForDevice(int ordinal) override;

  StatusOr<se::StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const se::PluginConfig& config) override;

  StatusOr<se::StreamExecutor*> GetExecutor(
      const se::StreamExecutorConfig& config) override;

  StatusOr<std::unique_ptr<se::StreamExecutor>> GetUncachedExecutor(
      const se::StreamExecutorConfig& config) override;

  void RegisterTraceListener(std::unique_ptr<se::TraceListener>) override;

  void UnregisterTraceListener(se::TraceListener* listener) override;

  // Poplar specific interface

  Status ConfigurePoplarDevice(int, const tensorflow::IPUOptions& opts);

  Status GetCompilerEvents(std::list<tensorflow::IpuTraceEvent>& out);

 private:
  // This platform's name.
  std::string name_;

  // Cache of created StreamExecutors.
  se::ExecutorCache executor_cache_;

  // The number of poplar devices to be created
  int num_devices_;

  SE_DISALLOW_COPY_AND_ASSIGN(PoplarPlatform);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
