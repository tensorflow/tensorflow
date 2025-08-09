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

#include "tsl/platform/cpu_info.h"

#include "xla/tsl/platform/test.h"

namespace tsl {

TEST(CPUInfo, CommonX86CPU) {
  // CPUs from 1999 onwards support SSE.
  if (port::TestCPUFeature(port::CPUFeature::SSE)) {
    EXPECT_TRUE(port::IsX86CPU());
  }
}

TEST(CPUInfo, Aarch64NeoverseV1CPU) {
  if (port::TestAarch64CPU(port::Aarch64CPU::ARM_NEOVERSE_V1)) {
    EXPECT_TRUE(port::IsAarch64CPU());
  }
}

TEST(CPUInfo, Aarch64Bf16) {
  if (port::TestCPUFeature(port::CPUFeature::AARCH64_BF16)) {
    EXPECT_TRUE(port::IsAarch64CPU());
  }
}

}  // namespace tsl
