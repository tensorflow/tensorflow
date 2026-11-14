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
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
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

// Golden corpus of real renderer strings. The expected values were produced by
// the regular expressions that canonicalize_value.cc used before switching to
// plain string scans (to drop the RE2 dependency); the compatibility databases
// were generated with those, so these keys must stay stable. Pathological
// inputs are intentionally out of scope: they do not match any database entry
// either way.
TEST(CanonicalizeValue, RealRendererStringsMatchGoldenKeys) {
  struct Case {
    absl::string_view renderer;
    absl::string_view expected;
  };
  const std::vector<Case> cases = {
      // Non-ANGLE renderers pass through unchanged.
      {"Adreno (TM) 740", "adreno_(tm)_740"},
      {"Mali-G715-Immortalis MC11", "mali_g715_immortalis_mc11"},
      {"PowerVR Rogue GE8320", "powervr_rogue_ge8320"},
      // Samsung Xclipse, with and without the outer paren and driver version.
      {"ANGLE (Samsung Xclipse 920) on Vulkan 1.1.179",
       "angle_(samsung_xclipse_920)_on_vulkan"},
      {"ANGLE (Samsung Xclipse 940) on Vulkan",
       "angle_(samsung_xclipse_940)_on_vulkan"},
      {"ANGLE ((Samsung Xclipse 960) on Vulkan 1.4.304)",
       "angle_(samsung_xclipse_960)_on_vulkan"},
      // Imagination PowerVR.
      {"ANGLE (Imagination Technologies, Vulkan 1.4.317 (PowerVR C-Series "
       "CXTP-48-1536 MC1 (0x70061042)), PowerVR C-Series Vulkan Driver "
       "1.662.3024)",
       "angle_(powervr_c_series_cxtp_48_1536_mc1)"},
      {"ANGLE (Imagination Technologies, Vulkan 1.3.288 (PowerVR D-Series "
       "DXT-48-1536 MC1 (0x71061212)), PowerVR D-Series Vulkan Driver "
       "1.602.400)",
       "angle_(powervr_d_series_dxt_48_1536_mc1)"},
      {"ANGLE (Imagination Technologies, Vulkan 1.3.275 (PowerVR B-Series "
       "BXM-8-256 (0x36001312)), PowerVR B-Series Vulkan Driver 1.19.6345)",
       "angle_(powervr_b_series_bxm_8_256)"},
      // VirtIO Venus across vendors, including nested parens in the model.
      {"ANGLE (Intel, Vulkan 1.3.269 (VirtIO GPU Venus (Intel(R) Graphics "
       "(ADL N)) (0x000046d2)), Venus 24.2.8)",
       "angle_(virtio_gpu_venus_(intel(r)_graphics_(adl_n)))"},
      {"ANGLE (Intel, Vulkan 1.3.255 (VirtIO GPU Venus (Intel(R) UHD Graphics "
       "(TGL GT2)) (0x00009a49)), Venus 23.1.4)",
       "angle_(virtio_gpu_venus_(intel(r)_uhd_graphics_(tgl_gt2)))"},
      {"ANGLE (ARM, Vulkan 1.1.255 (VirtIO GPU Venus (Mali-G72) (0x62210030)), "
       "Venus 40.0.0)",
       "angle_(virtio_gpu_venus_(mali_g72))"},
      {"ANGLE (AMD, Vulkan 1.3.255 (VirtIO GPU Venus (AMD Radeon Graphics "
       "(RADV RAVEN2)) (0x000015d8)), Venus 23.1.4)",
       "angle_(virtio_gpu_venus_(amd_radeon_graphics_(radv_raven2)))"},
      // Unknown ANGLE backends pass through unchanged.
      {"ANGLE (NVIDIA, Vulkan 1.3.0 (NVIDIA GeForce RTX 3080 (0x00002206)), "
       "NVIDIA 535.0.0)",
       "angle_(nvidia,_vulkan_1.3.0_(nvidia_geforce_rtx_3080_(0x00002206)),"
       "_nvidia_535.0.0)"},
      {"ANGLE (Qualcomm, Adreno (TM) 750, OpenGL ES 3.2)",
       "angle_(qualcomm,_adreno_(tm)_750,_opengl_es_3.2)"},
      // Known vendors on a non-Vulkan ANGLE backend pass through unchanged,
      // even when a device id is present: the database keys are Vulkan-only.
      {"ANGLE (Imagination Technologies, OpenGL ES 3.2 (PowerVR B-Series "
       "BXM-8-256 (0x36001312)), PowerVR B-Series GLES Driver 1.19.6345)",
       "angle_(imagination_technologies,_opengl_es_3.2_(powervr_b_series_"
       "bxm_8_256_(0x36001312)),_powervr_b_series_gles_driver_1.19.6345)"},
      {"ANGLE (Intel, OpenGL ES 3.2 (VirtIO GPU Venus (Mali-G72) "
       "(0x62210030)), Venus 40.0.0)",
       "angle_(intel,_opengl_es_3.2_(virtio_gpu_venus_(mali_g72)_"
       "(0x62210030)),_venus_40.0.0)"},
  };
  for (const Case& c : cases) {
    EXPECT_EQ(CanonicalizeValueWithKey(kGPUModel, c.renderer), c.expected)
        << "renderer: " << c.renderer;
  }
}

TEST(CanonicalizeValue, NormalizedGpuModelsAreStable) {
  // Database keys are already normalized; normalizing them again must be a
  // no-op, otherwise lookups could never hit them.
  for (absl::string_view key : {
           "angle_(samsung_xclipse_960)_on_vulkan",
           "angle_(powervr_d_series_dxt_48_1536_mc1)",
           "angle_(virtio_gpu_venus_(intel(r)_graphics_(adl_n)))",
           "angle_(virtio_gpu_venus_(mali_g72))",
           "powervr_rogue_ge8320",
       }) {
    EXPECT_EQ(CanonicalizeValueWithKey(kGPUModel, key), key);
  }
}

}  // namespace
}  // namespace tflite::acceleration
