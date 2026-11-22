/* Copyright 2026 The OpenXLA Authors.

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

#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/target_config/cudnn_device_props_test_util.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

TEST(CudnnDevicePropsTest, MatchesLiveDevice) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  std::string name = absl::AsciiStrToUpper(platform_name);
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(name));

  bool any_compared = false;
  for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
    ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                         platform->ExecutorForDevice(i));
    const se::DeviceDescription& desc = executor->GetDeviceDescription();
    SCOPED_TRACE(absl::StrCat("device ", i, ": ", desc.name()));

    ExpectDevicePropsMatchLiveDevice(desc, i);
    if (IsSkipped()) {
      return;
    }
    any_compared = true;
  }

  if (!any_compared) {
    GTEST_SKIP() << "No visible GPU devices to compare against.";
  }
}

// Make sure the cudnn_frontend::DeviceProperties created from our checked in
// target config still matches the one from the live device. This should catch
// cases where CuDNN requires a new field to be serialized, which may not have
// been mapped in our target config.
TEST(CudnnDevicePropsTest, SpecMatchesLiveDevice) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  std::string name = absl::AsciiStrToUpper(platform_name);
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(name));
  ASSERT_GT(platform->VisibleDeviceCount(), 0);
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  const se::DeviceDescription& live_desc = executor->GetDeviceDescription();

  std::optional<se::GpuTargetConfigProto> spec;
  // GpuModel values are contiguous from 0, and GetGpuTargetConfig returns
  // NotFound past the last one.
  for (int i = 0;; ++i) {
    absl::StatusOr<se::GpuTargetConfigProto> proto =
        GetGpuTargetConfig(static_cast<GpuModel>(i));
    if (absl::IsNotFound(proto.status())) {
      break;
    }
    ASSERT_OK(proto);
    // Name alone doesn't distinguish MIG partitions from the full GPU.
    if (proto->gpu_device_info().name() == live_desc.name() &&
        proto->gpu_device_info().core_count() == live_desc.core_count()) {
      spec = *std::move(proto);
      break;
    }
  }
  if (!spec.has_value()) {
    GTEST_SKIP() << "No spec for " << live_desc.name();
  }
  SCOPED_TRACE(live_desc.name());

  ASSERT_OK_AND_ASSIGN(GpuTargetConfig spec_config,
                       GpuTargetConfig::FromProto(*spec));
  ExpectDevicePropsMatchLiveDevice(spec_config.device_description);
}

}  // namespace
}  // namespace xla::gpu
