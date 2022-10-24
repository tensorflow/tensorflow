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

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

class KernelLaunchTest : public GpuCodegenTest {};

TEST_F(KernelLaunchTest, Basic) {
  const char* hlo_text = R"(
    HloModule Test1

    add_F32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY Test1 {
      a = f32[2, 2]{1,0} parameter(0)
      b = f32[2, 2]{1,0} parameter(1)
      ROOT r1 = f32[2, 2]{1,0} map(a, b), dimensions={0, 1}, to_apply=add_F32
    }
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(KernelLaunchTest, KernelWithConstants) {
  const char* hlo_text = R"(
    HloModule Test2

    add_F32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY Test2 {
      a = f32[2, 2]{1,0} constant({{1, 2}, {3, 4}})
      b = f32[2, 2]{1,0} constant({{5, 6}, {7, 8}})
      ROOT r1 = f32[2, 2]{1,0} map(a, b), dimensions={0, 1}, to_apply=add_F32
    }
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
