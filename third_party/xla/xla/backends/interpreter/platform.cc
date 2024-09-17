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

#include "xla/backends/interpreter/platform.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/interpreter/executor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "tsl/platform/status.h"

namespace stream_executor {
namespace interpreter {

XlaInterpreterPlatform::XlaInterpreterPlatform(const std::string& name,
                                               const Platform::Id& id)
    : name_(name), id_(id) {}

XlaInterpreterPlatform::~XlaInterpreterPlatform() {}

Platform::Id XlaInterpreterPlatform::id() const { return id_; }

int XlaInterpreterPlatform::VisibleDeviceCount() const { return 1; }

const std::string& XlaInterpreterPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
XlaInterpreterPlatform::DescriptionForDevice(int ordinal) const {
  return XlaInterpreterExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> XlaInterpreterPlatform::FindExisting(
    int ordinal) {
  return executor_cache_.Get(ordinal);
}

absl::StatusOr<StreamExecutor*> XlaInterpreterPlatform::ExecutorForDevice(
    int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
XlaInterpreterPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<XlaInterpreterExecutor>(ordinal, this);
  auto init_status = executor->Init();
  if (!init_status.ok()) {
    return absl::Status{
        absl::StatusCode::kInternal,
        absl::StrFormat(
            "failed initializing StreamExecutor for device ordinal %d: %s",
            ordinal, init_status.ToString())};
  }

  return std::move(executor);
}

static void InitializeXlaInterpreterPlatform() {
  std::unique_ptr<Platform> platform(new XlaInterpreterPlatform);
  TF_CHECK_OK(PlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace interpreter
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    interpreter_platform,
    stream_executor::interpreter::InitializeXlaInterpreterPlatform());
