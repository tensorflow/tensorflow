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

#include <utility>

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

namespace {

class ReductionVectorizationTest : public GpuCodegenTest {};

class ReductionVectorizationNoOptTest : public GpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // The test MultiOutputStore contain a MOF fusion and XLA optimizer pass
    // doesn't like this.
    debug_options.set_xla_disable_all_hlo_passes(true);
    return debug_options;
  }
};

TEST_F(ReductionVectorizationNoOptTest, MultiOutputStore) {
  const char* hlo_text = R"(
HloModule MultiOutputStore

%add_f32 {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

%fused_computation {
  %param_0 = f32[2,384,1024] parameter(0)
  %param_1 = f32[2,384] parameter(1)
  %constant0 = f32[] constant(0.0009765625)
  %broadcast0 = f32[2,384] broadcast(%constant0), dimensions={}
  %multiply0 = f32[2,384] multiply(%param_1, %broadcast0)
  %broadcast1 = f32[2,384,1024] broadcast(%multiply0), dimensions={0,1}
  %subtract = f32[2,384,1024] subtract(%param_0, %broadcast1)
  %multiply1 = f32[2,384,1024] multiply(%subtract, %subtract)
  %constant1 = f32[] constant(0)
  %reduce = f32[2,384] reduce(%multiply1, %constant1), dimensions={2}, to_apply=%add_f32
  ROOT %tuple = (f32[2,384], f32[2,384,1024], f32[2,384,1024]) tuple(%reduce, %subtract, %broadcast1)
}

ENTRY %cluster {
  %param0 = f32[2,384,1024] parameter(0)
  %param1 =  f32[2,384] parameter(1)
  ROOT %fusion = (f32[2,384], f32[2,384,1024], f32[2,384,1024]) fusion(%param0, %param1), kind=kInput, calls=%fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> optimized_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  std::string expected = R"(
CHECK: ld.global.nc.v2.f32
CHECK: st.global.v2.f32
CHECK: st.global.v2.f32
CHECK: ld.global.nc.v2.f32
CHECK: st.global.v2.f32
CHECK: st.global.v2.f32
CHECK: ld.global.nc.v2.f32
CHECK: st.global.v2.f32
CHECK: st.global.v2.f32
CHECK: ld.global.nc.v2.f32
CHECK: st.global.v2.f32
CHECK: st.global.v2.f32
)";
  CompileAndOptionallyVerifyPtx(std::move(optimized_module), expected);

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
