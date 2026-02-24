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

#include "xla/service/gpu/autotuning/triton_configs.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace xla::gpu {
namespace {

using ::testing::SizeIs;

TEST(TritonConfigsTest, PlatformsReturnNonEmptyConfig) {
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kAmpere),
              SizeIs(36));
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kBlackwell),
              SizeIs(57));
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultCuda),
              SizeIs(32));
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultRocm),
              SizeIs(6));
  EXPECT_THAT(GetTritonConfigsForPlatform(TritonConfigsPlatform::kHopper),
              SizeIs(63));
}

}  // namespace
}  // namespace xla::gpu
