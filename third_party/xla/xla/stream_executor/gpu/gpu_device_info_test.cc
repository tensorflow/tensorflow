/* Copyright 2024 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace {

TEST(DeviceInfoTest, DeviceInfoMatches) {
  absl::flat_hash_map<std::string, GpuDeviceInfoProto> gpu_specs;
  for (const std::string file_name :
       {"a100_pcie_80", "a100_sxm_40", "a100_sxm_80", "a6000", "h100_pcie",
        "h100_sxm", "p100", "v100", "mi200"}) {
    GpuTargetConfigProto proto;
    std::string spec_string;
    TF_ASSERT_OK(tsl::ReadFileToString(
        tsl::Env::Default(),
        tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools", "hlo_opt",
                          "gpu_specs", absl::StrCat(file_name, ".txtpb")),
        &spec_string));
    ASSERT_TRUE(
        tsl::protobuf::TextFormat::ParseFromString(spec_string, &proto));
    gpu_specs[proto.device_description_str()] = proto.gpu_device_info();
  }
  auto name = absl::AsciiStrToUpper(
      xla::PlatformUtil::CanonicalPlatformName("gpu").value());
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName(name));
  bool all_skipped = false;
  for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                            platform->ExecutorForDevice(i));
    const DeviceDescription& physical_device_description =
        executor->GetDeviceDescription();
    auto it = gpu_specs.find(physical_device_description.name());
    const GpuDeviceInfoProto physical_device_info =
        physical_device_description.ToGpuProto();
    VLOG(1) << physical_device_description.name();
    VLOG(1) << physical_device_info.DebugString();
    if (it == gpu_specs.end()) {
      LOG(WARNING) << "No spec file for " << physical_device_description.name();
      continue;
    }
    all_skipped = false;
    const GpuDeviceInfoProto& stored_device_info = it->second;
    tsl::protobuf::util::MessageDifferencer diff;
    diff.IgnoreField(GpuDeviceInfoProto::GetDescriptor()->FindFieldByName(
        "device_memory_size"));
    diff.set_message_field_comparison(
        tsl::protobuf::util::MessageDifferencer::EQUIVALENT);
    std::string result;
    diff.ReportDifferencesToString(&result);
    EXPECT_TRUE(diff.Compare(physical_device_info, stored_device_info))
        << result;
    EXPECT_NEAR(physical_device_info.device_memory_size(),
                stored_device_info.device_memory_size(), 1024 * 1024 * 1024);
  }
  if (all_skipped) {
    GTEST_SKIP() << "No spec files for any of the available GPUs.";
  }
}

}  // namespace
}  // namespace stream_executor
