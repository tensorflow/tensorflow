/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/tests/gpu_codegen_test.h"

namespace xla {
namespace gpu {
namespace {

class UnknownCustomCallFails : public GpuCodegenTest {};

TEST_F(UnknownCustomCallFails, UnknownCustomCallFails) {
  const char* hlo_text = R"(
    HloModule Test1

    ENTRY Test1 {
      a = f32[128] parameter(0)
      ROOT r1 = f32[128] custom-call(a), custom_call_target="my_custom_call"
    }
  )";

  EXPECT_FALSE(Run(hlo_text));
}

class MockedCustomCall : public GpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions opts;
    opts.set_xla_gpu_mock_custom_calls(true);
    return opts;
  }
};

TEST_F(MockedCustomCall, CustomCallIgnored) {
  const char* hlo_text = R"(
    HloModule Test1

    ENTRY Test1 {
      a = f32[128] parameter(0)
      ROOT r1 = f32[128] custom-call(a), custom_call_target="my_custom_call"
    }
  )";

  EXPECT_TRUE(Run(hlo_text));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
