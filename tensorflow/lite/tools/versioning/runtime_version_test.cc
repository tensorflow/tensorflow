/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/versioning/runtime_version.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {

TEST(OpVersionTest, CompareRuntimeVersion) {
  EXPECT_TRUE(CompareRuntimeVersion("1.9", "1.13"));
  EXPECT_FALSE(CompareRuntimeVersion("1.13", "1.13"));
  EXPECT_TRUE(CompareRuntimeVersion("1.14", "1.14.1"));
  EXPECT_FALSE(CompareRuntimeVersion("1.14.1", "1.14"));
  EXPECT_FALSE(CompareRuntimeVersion("1.14.1", "1.9"));
  EXPECT_FALSE(CompareRuntimeVersion("1.0.9", "1.0.8"));
  EXPECT_FALSE(CompareRuntimeVersion("2.1.0", "1.2.0"));
  EXPECT_TRUE(CompareRuntimeVersion("", "1.13"));
  EXPECT_FALSE(CompareRuntimeVersion("", ""));
}

}  // namespace tflite
