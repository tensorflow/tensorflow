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

#include <iterator>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/variables.h"

namespace tflite::acceleration {

namespace {

inline char ascii_normalise(const unsigned char c) {
  if (c == ' ' || c == '-') {
    return '_';
  }
  return absl::ascii_tolower(c);
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
  std::string gpu_output;
  return key == kGPUModel &&
                 RE2::FullMatch(
                     output,
                     R"((angle_\(samsung_xclipse_[0-9]*\)_on_vulkan).*$)",
                     &gpu_output)
             ? gpu_output
             : output;
}
}  // namespace tflite::acceleration
