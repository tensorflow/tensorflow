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

#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"

#include <gtest/gtest.h>
#include "absl/strings/numbers.h"

namespace xla {
namespace {

TEST(GpuAllocatorConfigCompatibilityTest, NumericMemoryFraction) {
  GpuAllocatorConfig config;
  double fraction = config.memory_fraction;
  EXPECT_EQ(fraction, 0.75);
  config.memory_fraction = 0.5;
  EXPECT_TRUE(absl::SimpleAtod("0.25", &config.memory_fraction));
  EXPECT_EQ(config.memory_fraction, 0.25);
  EXPECT_EQ(MemFractionStart(config.GetMemoryFraction()), 0.25);
}

}  // namespace
}  // namespace xla
