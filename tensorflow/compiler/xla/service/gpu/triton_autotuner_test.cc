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

#include "tensorflow/compiler/xla/service/gpu/triton_autotuner.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {

namespace {

class TritonAutotunerTest : public HloTestBase {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(true);
    return debug_options;
  }
  void CheckTritonAutotuning(const char* hlo, absl::string_view expected) {
    HloPassPipeline pipeline("gemm_rewrite");
    pipeline.AddPass<GemmRewriterTriton>(backend()
                                             .default_stream_executor()
                                             ->GetDeviceDescription()
                                             .cuda_compute_capability());
    pipeline.AddPass<TritonAutotuner>(backend().default_stream_executor(),
                                      backend().memory_allocator(),
                                      tsl::port::MaxParallelism());

    RunAndFilecheckHloRewrite(hlo, std::move(pipeline), expected);
  }
};

TEST_F(TritonAutotunerTest, Int8FusedGemm) {
  const char* hlo = R"(
HloModule module

ENTRY e {
  x = s8[128,64] parameter(0)
  c = f16[128,64] convert(x)

  y = f16[64,6144] parameter(1)

  ROOT out = f16[128,6144] dot(c, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  CheckTritonAutotuning(hlo, R"(
// CHECK: HloModule module
// CHECK: %out
// CHECK:   %parameter_0 = s8[128,64]{1,0} parameter(0)
// CHECK:   %c.1 = f16[128,64]{1,0} convert(%parameter_0)
// CHECK:   %parameter_1 = f16[64,6144]{1,0} parameter(1)
// CHECK:   ROOT %out.1 = f16[128,6144]{1,0} dot(%c.1, %parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
// CHECK: }
// CHECK: ENTRY %e (x: s8[128,64], y: f16[64,6144]) -> f16[128,6144] {
// CHECK:   %x = s8[128,64]{1,0} parameter(0)
// CHECK:   %y = f16[64,6144]{1,0} parameter(1)
// CHECK:   ROOT %custom-call = f16[128,6144]{1,0} custom-call(%x, %y), custom_call_target="__triton", called_computations={%out}, backend_config="{\"block_m
)");

  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{5e-3, 5e-3}));
}

TEST_F(TritonAutotunerTest, Int8FusedGemm256) {
  const char* hlo = R"(
HloModule module

ENTRY e {
  x = s8[128,256] parameter(0)
  c = f16[128,256] convert(x)

  y = f16[256,6144] parameter(1)

  ROOT out = f16[128,6144] dot(c, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  CheckTritonAutotuning(hlo, R"(
// CHECK: %out
// CHECK-NEXT:   %parameter_0 = s8[128,256]{1,0} parameter(0)
// CHECK-NEXT:   %c.1 = f16[128,256]{1,0} convert(%parameter_0)
// CHECK-NEXT:   %parameter_1 = f16[256,6144]{1,0} parameter(1)
// CHECK-NEXT:   ROOT %out.1 = f16[128,6144]{1,0} dot(%c.1, %parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
// CHECK-NEXT: }
// CHECK: ENTRY %e (x: s8[128,256], y: f16[256,6144]) -> f16[128,6144] {
// CHECK-NEXT:   %x = s8[128,256]{1,0} parameter(0)
// CHECK-NEXT:   %y = f16[256,6144]{1,0} parameter(1)
// CHECK-NEXT:   ROOT %custom-call = f16[128,6144]{1,0} custom-call(%x, %y), custom_call_target="__triton", called_computations={%out}, backend_config="{\"block_m
)");

  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{1e-2, 1e-2}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
