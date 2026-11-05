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

#include <cstddef>
#include <iterator>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/variables.h"

namespace tflite::acceleration {

namespace {

inline char ascii_normalise(const unsigned char c) {
  if (c == ' ' || c == '-') {
    return '_';
  }
  return absl::ascii_tolower(c);
}

// ANGLE reports Vulkan-backed GPUs with a verbose renderer string that embeds
// driver and API versions, e.g. (after CanonicalizeValue):
//
//   angle_(samsung_xclipse_920)_on_vulkan_1.1.179
//   angle_(imagination_technologies,_vulkan_1.4.317_(powervr_c_series_cxtp_48
//     _1536_mc1_(0x70061042)),_powervr_c_series_vulkan_driver_1.662.3024)
//   angle_(intel,_vulkan_1.3.269_(virtio_gpu_venus_(intel(r)_graphics_(adl_n))
//     _(0x000046d2)),_venus_24.2.8)
//
// The helpers below strip the volatile parts and keep only the GPU model, so
// that the compatibility database keys stay stable across driver updates. They
// are deliberately plain string scans rather than regular expressions: this
// library is linked into size-sensitive binaries (e.g. the LiteRT runtime), and
// a regex engine costs far more than the normalization is worth. When adding
// support for a new renderer format, extend these helpers instead of
// reintroducing a regex dependency.

// Returns the substring of `s` starting right after `marker` and ending right
// before the next `_(0x<device id>` suffix, or an empty string_view if either
// delimiter is missing. For example, with marker "(powervr_", the string
// "..._(powervr_rogue_ge8320_(0x22105230)),_..." yields "rogue_ge8320".
absl::string_view ExtractModelBeforeDeviceId(absl::string_view s,
                                             absl::string_view marker) {
  const size_t begin = s.find(marker);
  if (begin == absl::string_view::npos) return {};
  const size_t model_begin = begin + marker.size();
  const size_t end = s.find("_(0x", model_begin);
  if (end == absl::string_view::npos) return {};
  return s.substr(model_begin, end - model_begin);
}

// Consumes the "angle_(<vendor>,_vulkan_" prefix that ANGLE's Vulkan backend
// puts in front of the renderer details (the "_" before "vulkan" is optional).
// Requiring it keeps the Vulkan-only database keys below from matching
// renderer strings produced by other ANGLE backends.
bool ConsumeAngleVulkanPrefix(absl::string_view* s, absl::string_view vendor) {
  absl::string_view rest = *s;
  if (!absl::ConsumePrefix(&rest, "angle_(") ||
      !absl::ConsumePrefix(&rest, vendor) || !absl::ConsumePrefix(&rest, ",")) {
    return false;
  }
  absl::ConsumePrefix(&rest, "_");
  if (!absl::ConsumePrefix(&rest, "vulkan_")) return false;
  *s = rest;
  return true;
}

// Samsung Xclipse: "angle_(samsung_xclipse_920)_on_vulkan[...]" or, with newer
// drivers, "angle_((samsung_xclipse_960)_on_vulkan[...])".
bool NormalizeSamsungXclipse(absl::string_view s, std::string* result) {
  if (!absl::ConsumePrefix(&s, "angle_(")) return false;
  absl::ConsumePrefix(&s, "(");
  constexpr absl::string_view kModelPrefix = "samsung_xclipse_";
  if (!absl::ConsumePrefix(&s, kModelPrefix)) return false;
  size_t digits = 0;
  while (digits < s.size() && absl::ascii_isdigit(s[digits])) ++digits;
  if (!absl::StartsWith(s.substr(digits), ")_on_vulkan")) return false;
  *result =
      absl::StrCat("angle_(", kModelPrefix, s.substr(0, digits), ")_on_vulkan");
  return true;
}

// Imagination PowerVR: keep the "powervr_..." model that precedes the device
// id.
bool NormalizePowerVR(absl::string_view s, std::string* result) {
  if (!ConsumeAngleVulkanPrefix(&s, "imagination_technologies")) return false;
  const absl::string_view model = ExtractModelBeforeDeviceId(s, "(powervr_");
  if (model.empty()) return false;
  *result = absl::StrCat("angle_(powervr_", model, ")");
  return true;
}

// VirtIO Venus (virtualized GPUs, e.g. emulators and ChromeOS): keep the
// "virtio_gpu_venus_(...)" model, which may itself contain parentheses, that
// precedes the device id.
bool NormalizeVirtioVenus(absl::string_view s, std::string* result) {
  if (!ConsumeAngleVulkanPrefix(&s, "intel") &&
      !ConsumeAngleVulkanPrefix(&s, "arm") &&
      !ConsumeAngleVulkanPrefix(&s, "amd")) {
    return false;
  }
  const absl::string_view model =
      ExtractModelBeforeDeviceId(s, "(virtio_gpu_venus_(");
  if (model.empty()) return false;
  *result = absl::StrCat("angle_(virtio_gpu_venus_(", model, ")");
  return true;
}

}  // namespace

std::string CanonicalizeValue(absl::string_view value) {
  std::string output;
  absl::c_transform(value, std::back_inserter(output),
                    tflite::acceleration::ascii_normalise);
  return output;
}

std::string CanonicalizeValueWithKey(absl::string_view key,
                                     absl::string_view value) {
  std::string output = CanonicalizeValue(value);
  if (key != kGPUModel) {
    return output;
  }

  std::string gpu_model;
  if (NormalizeSamsungXclipse(output, &gpu_model) ||
      NormalizePowerVR(output, &gpu_model) ||
      NormalizeVirtioVenus(output, &gpu_model)) {
    return gpu_model;
  }
  return output;
}
}  // namespace tflite::acceleration
