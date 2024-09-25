/* Copyright 2017 The OpenXLA Authors.

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
#ifndef XLA_BACKENDS_INTERPRETER_PLATFORM_H_
#define XLA_BACKENDS_INTERPRETER_PLATFORM_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "xla/backends/interpreter/platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/statusor.h"

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

  absl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

  absl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;

  absl::StatusOr<StreamExecutor*> FindExisting(int ordinal) override;

  // Returns a device constructed with ordinal without
  // looking in or storing to the Platform's executor cache.
  // Ownership IS transferred to the caller.
  absl::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      int ordinal);

 private:
  // This platform's name.
  std::string name_;
  // This platform's id.
  Platform::Id id_;

  // Cache of created StreamExecutors.
  ExecutorCache executor_cache_;

  XlaInterpreterPlatform(const XlaInterpreterPlatform&) = delete;
  void operator=(const XlaInterpreterPlatform&) = delete;
};

}  // namespace interpreter
}  // namespace stream_executor

#endif  // XLA_BACKENDS_INTERPRETER_PLATFORM_H_
