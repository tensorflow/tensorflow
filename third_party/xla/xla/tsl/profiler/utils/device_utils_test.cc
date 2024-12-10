/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/profiler/utils/device_utils.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/platform/test.h"

namespace tsl {
namespace profiler {
namespace {

tensorflow::profiler::XPlane CreateXPlane(absl::string_view name) {
  tensorflow::profiler::XPlane plane;
  plane.set_name(name.data(), name.size());
  return plane;
}

TEST(DeviceUtilsTest, GetDeviceType) {
  EXPECT_EQ(GetDeviceType(CreateXPlane(kHostThreadsPlaneName)),
            DeviceType::kCpu);
  EXPECT_EQ(GetDeviceType(CreateXPlane(absl::StrCat(kTpuPlanePrefix, 0))),
            DeviceType::kTpu);
  EXPECT_EQ(GetDeviceType(CreateXPlane(absl::StrCat(kGpuPlanePrefix, 0))),
            DeviceType::kGpu);
  EXPECT_EQ(GetDeviceType(CreateXPlane("unknown")), DeviceType::kUnknown);
}

TEST(DeviceUtilsTest, GetDeviceTypeFromName) {
  EXPECT_EQ(GetDeviceType(kHostThreadsPlaneName), DeviceType::kCpu);
  EXPECT_EQ(GetDeviceType(absl::StrCat(kTpuPlanePrefix, 0)), DeviceType::kTpu);
  EXPECT_EQ(GetDeviceType(absl::StrCat(kGpuPlanePrefix, 0)), DeviceType::kGpu);
  EXPECT_EQ(GetDeviceType("unknown"), DeviceType::kUnknown);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
