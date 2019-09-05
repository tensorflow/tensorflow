/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/context.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/platform.h"

namespace ruy {
namespace {

TEST(ContextTest, EnabledPathsGeneral) {
  ruy::Context ruy_context;
  const auto ruy_paths = ruy_context.GetRuntimeEnabledPaths();
  const auto ruy_paths_repeat = ruy_context.GetRuntimeEnabledPaths();
  ASSERT_EQ(ruy_paths, ruy_paths_repeat);
  EXPECT_NE(ruy_paths, Path::kNone);
  EXPECT_EQ(ruy_paths & Path::kReference, Path::kReference);
  EXPECT_EQ(ruy_paths & Path::kStandardCpp, Path::kStandardCpp);
}

#if RUY_PLATFORM(X86)
TEST(ContextTest, EnabledPathsX86) {
  ruy::Context ruy_context;
  ruy_context.SetRuntimeEnabledPaths(Path::kAvx2 | Path::kAvx512);
  const auto ruy_paths = ruy_context.GetRuntimeEnabledPaths();
  EXPECT_EQ(ruy_paths & Path::kReference, Path::kNone);
  EXPECT_EQ(ruy_paths & Path::kStandardCpp, Path::kNone);
}
#endif  // RUY_PLATFORM(X86)

#if RUY_PLATFORM(ARM)
TEST(ContextTest, EnabledPathsArm) {
  ruy::Context ruy_context;
  ruy_context.SetRuntimeEnabledPaths(Path::kNeon | Path::kNeonDotprod);
  const auto ruy_paths = ruy_context.GetRuntimeEnabledPaths();
  EXPECT_EQ(ruy_paths & Path::kReference, Path::kNone);
  EXPECT_EQ(ruy_paths & Path::kStandardCpp, Path::kNone);
  EXPECT_EQ(ruy_paths & Path::kNeon, Path::kNeon);
}
#endif  // RUY_PLATFORM(ARM)

}  // namespace
}  // namespace ruy

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
