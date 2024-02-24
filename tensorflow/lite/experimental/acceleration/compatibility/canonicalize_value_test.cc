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

#include <gmock/gmock.h>
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
}  // namespace
}  // namespace tflite::acceleration
