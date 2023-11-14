/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla::gpu {

class CutlassFusionTest : public HloTestBase {
  // Custom fusions are not supported by XLA runtime.
  DebugOptions GetDebugOptionsForTest() override {
    auto debug_options = GetDebugOptionsFromFlags();
    debug_options.set_xla_gpu_enable_xla_runtime_executable(false);
    return debug_options;
  }
};

TEST_F(CutlassFusionTest, SimpleF32Gemm) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_text_cublas = R"(
  HloModule cublas

  ENTRY e {
    arg0 = f32[32, 64]{1,0} parameter(0)
    arg1 = f32[64, 16]{1,0} parameter(1)
    gemm = (f32[32,16]{1,0}, s8[0]{0}) custom-call(arg0, arg1),
      custom_call_target="__cublas$gemm",
      backend_config={"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[0],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}
    ROOT get-tuple-element = f32[32,16]{1,0} get-tuple-element((f32[32,16]{1,0}, s8[0]{0}) gemm), index=0
  })";

  const char* hlo_text_custom_fusion = R"(
  HloModule cutlass

  cutlass_gemm {
    arg0 = f32[32,64]{1,0} parameter(0)
    arg1 = f32[64,16]{1,0} parameter(1)
    ROOT dot = f32[32,16]{1,0} dot(arg0, arg1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY e {
    arg0 = f32[32, 64]{1,0} parameter(0)
    arg1 = f32[64, 16]{1,0} parameter(1)
    ROOT _ = f32[32,16]{1,0} fusion(arg0, arg1), kind=kCustom, calls=cutlass_gemm,
      backend_config={kind: "__custom_fusion", custom_fusion_config: {"name":"cutlass_gemm"}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_cublas, hlo_text_custom_fusion,
                                      error_spec, /*run_hlo_passes=*/false));
}

}  // namespace xla::gpu
