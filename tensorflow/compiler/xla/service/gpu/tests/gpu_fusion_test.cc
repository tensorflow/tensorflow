/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuFusionTest : public GpuCodegenTest {};

TEST_F(GpuFusionTest, FusedReshape) {
  const char* hlo_text = R"(
    HloModule test_module

    fused_computation {
      p0.param_0 = f32[4,1,1]{2,1,0} parameter(0)
      p1.param_1 = f32[4,1]{1,0} parameter(1)
      reshape = f32[4,1]{1,0} reshape(p0.param_0)
      ROOT add = f32[4,1] add(reshape, p1.param_1)
    }

    ENTRY BroadcastIntoAdd {
      p0 = f32[4,1,1]{2,1,0} parameter(0)
      p1 = f32[4,1]{1,0} parameter(1)
      ROOT fusion = f32[4,1]{1,0} fusion(p0, p1), kind=kLoop,
                                                  calls=fused_computation
    }
)";

  CompileAndVerifyIr(hlo_text,
                     R"(
; CHECK-LABEL: @fusion
; CHECK: fadd
; CHECK: }
      )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
