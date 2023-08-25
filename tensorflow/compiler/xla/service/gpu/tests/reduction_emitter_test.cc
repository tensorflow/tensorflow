/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

class ReductionEmitterTest : public gpu::GpuCodegenTest {};

TEST_F(ReductionEmitterTest, ProperShmemAllocation) {
  const char* const kHloString = R"(
  HloModule m

  add {
    a = f64[] parameter(0)
    b = f64[] parameter(1)
    ROOT out = f64[] add(a, b)
  }

  fused_computation {
    p1 = f64[1024,1024]{1,0} parameter(0)
    p2 = f64[1024,1024]{1,0} parameter(1)
    s = pred[1024,1024]{1,0} parameter(2)
    p = f64[1024,1024]{1,0} select(s, p1, p2)
    z = f64[] constant(0)
    ROOT out = f64[1024]{0} reduce(p, z), to_apply=add, dimensions={0}
  }

  ENTRY e {
    p1 = f64[1024,1024]{1,0} parameter(0)
    p2 = f64[1024,1024]{1,0} parameter(1)
    s = pred[1024,1024]{1,0} parameter(2)
    ROOT f = f64[1024]{0} fusion(p1, p2, s), kind=kInput, calls=fused_computation
  })";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

}  // namespace
}  // namespace xla
