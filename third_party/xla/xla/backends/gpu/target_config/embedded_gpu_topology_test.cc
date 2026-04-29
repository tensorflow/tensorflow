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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/debug_options_flags.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::testing::Optional;
using ::tsl::proto_testing::EqualsProto;

TEST(EmbeddedGpuTopologyTest, GetGpuTopologyForPlatform) {
  ASSERT_OK_AND_ASSIGN(std::string name,
                       xla::PlatformUtil::CanonicalPlatformName("gpu"));
  ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                       stream_executor::PlatformManager::PlatformWithName(
                           absl::AsciiStrToUpper(name)));

  bool all_skipped = true;
  for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
    ASSERT_OK_AND_ASSIGN(stream_executor::StreamExecutor * executor,
                         platform->ExecutorForDevice(i));

    GpuTargetConfig gpu_target_config(executor);

    const auto compare_platform = [&](absl::string_view platform_version) {
      ASSERT_OK_AND_ASSIGN(
          GpuTopology gpu_topology,
          GetGpuTopologyForPlatform(platform_version,
                                    /*num_partitions=*/1,
                                    /*num_hosts_per_partition=*/1,
                                    /*num_devices_per_host=*/1));
      EXPECT_TRUE(gpu_topology.gpu_target_config().device_description.EqualsTo(
          gpu_target_config.device_description,
          {stream_executor::DeviceDescription::CompareOptions::kPortable,
           stream_executor::DeviceDescription::CompareOptions::
               kIgnoreVersionNumbers}));
      ASSERT_TRUE(gpu_topology.host_target_machine_options().has_value());
      cpu::TargetMachineOptions host_target_machine_options{
          DefaultDebugOptionsIgnoringFlags()};
      // Comparing the protos leads to better diagnostics in case of a mismatch.
      EXPECT_THAT(gpu_topology.host_target_machine_options()->ToProto(),
                  EqualsProto(host_target_machine_options.ToProto()));
      EXPECT_THAT(gpu_topology.host_target_machine_options(),
                  Optional(host_target_machine_options));
    };

    if (gpu_target_config.device_description_str == "NVIDIA B200") {
      compare_platform("umbriel_b200");
      all_skipped = false;
      continue;
    }

    if (gpu_target_config.device_description_str == "NVIDIA GB200") {
      compare_platform("oberon_b200");
      all_skipped = false;
      continue;
    }
  }

  if (all_skipped) {
    FAIL() << "No matching GPU found for topology test.";
  }
}

}  // namespace
}  // namespace xla::gpu
