/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/compatibility/canonicalize_value.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/acceleration/compatibility/variables.h"

namespace tflite::acceleration {
namespace {

TEST(CanonicalizeValue, CharactersAreLowercased) {
  EXPECT_EQ(CanonicalizeValue("hElLo"), "hello");
}

TEST(CanonicalizeValue, HyphensAreReplaced) {
  EXPECT_EQ(CanonicalizeValue("-"), "_");
}

TEST(CanonicalizeValue, SpacesAreReplaced) {
  EXPECT_EQ(CanonicalizeValue(" "), "_");
}

TEST(CanonicalizeValue, OtherSpecialCharactersAreUnaffected) {
  for (unsigned char c = 0; c < 65; ++c) {
    if (c == ' ' || c == '-') continue;
    std::string s = {1, static_cast<char>(c)};
    EXPECT_EQ(CanonicalizeValue(s), s);
  }
}

TEST(CanonicalizeValue, SamsungXclipseGpuNormalized) {
  EXPECT_EQ(CanonicalizeValueWithKey(
                kGPUModel, "ANGLE (Samsung Xclipse 920) on Vulkan 1.1.179"),
            "angle_(samsung_xclipse_920)_on_vulkan");
}

TEST(CanonicalizeValue, SamsungXclipseDoubleParenGpuNormalized) {
  EXPECT_EQ(CanonicalizeValueWithKey(
                kGPUModel, "ANGLE ((Samsung Xclipse 960) on Vulkan 1.4.304)"),
            "angle_(samsung_xclipse_960)_on_vulkan");
}

TEST(CanonicalizeValue, PowerVRGpuNormalized) {
  EXPECT_EQ(CanonicalizeValueWithKey(
                kGPUModel,
                "ANGLE (Imagination Technologies, Vulkan 1.4.317 (PowerVR "
                "C-Series CXTP-48-1536 MC1 (0x70061042)), PowerVR C-Series "
                "Vulkan Driver 1.662.3024)"),
            "angle_(powervr_c_series_cxtp_48_1536_mc1)");
  EXPECT_EQ(CanonicalizeValueWithKey(
                kGPUModel,
                "ANGLE (Imagination Technologies, Vulkan 1.3.288 (PowerVR "
                "D-Series DXT-48-1536 MC1 (0x71061212)), PowerVR D-Series "
                "Vulkan Driver 1.602.400)"),
            "angle_(powervr_d_series_dxt_48_1536_mc1)");
}

TEST(CanonicalizeValue, VirtioVenusGpuNormalized) {
  EXPECT_EQ(CanonicalizeValueWithKey(
                kGPUModel,
                "ANGLE (Intel, Vulkan 1.3.269 (VirtIO GPU Venus (Intel(R) "
                "Graphics (ADL N)) (0x000046d2)), Venus 24.2.8)"),
            "angle_(virtio_gpu_venus_(intel(r)_graphics_(adl_n)))");
  EXPECT_EQ(CanonicalizeValueWithKey(
                kGPUModel,
                "ANGLE (ARM, Vulkan 1.1.255 (VirtIO GPU Venus (Mali-G72) "
                "(0x62210030)), Venus 40.0.0)"),
            "angle_(virtio_gpu_venus_(mali_g72))");
}
}  // namespace
}  // namespace tflite::acceleration
