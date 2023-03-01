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

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/protobuf/autotuning.pb.h"

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

  void CheckTritonAutotuning(absl::string_view hlo,
                             absl::string_view expected) {
    HloPassPipeline pipeline("gemm_rewrite");
    pipeline.AddPass<GemmRewriterTriton>(backend()
                                             .default_stream_executor()
                                             ->GetDeviceDescription()
                                             .cuda_compute_capability());
    pipeline.AddPass<TritonAutotuner>(
        DeviceConfig{backend().default_stream_executor(),
                     backend().memory_allocator()},
        tsl::port::MaxParallelism());

    RunAndFilecheckHloRewrite(
        hlo, std::move(pipeline), expected, [](const HloModule* m) {
          CHECK_GT(
              m->entry_computation()
                  ->root_instruction()
                  ->backend_config<tensorflow::AutotuneResult::TritonGemmKey>()
                  .value()
                  .block_m(),
              0);
        });
  }
};

TEST_F(TritonAutotunerTest, Int8FusedGemm) {
  const std::string hlo = R"(
HloModule module

ENTRY e {
  x = s8[128,64] parameter(0)
  c = f16[128,64] convert(x)

  y = f16[64,6144] parameter(1)

  ROOT out = f16[128,6144] dot(c, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  CheckTritonAutotuning(hlo, R"(
// CHECK:   %triton_gemm_out
// CHECK:   ROOT %out.1 = f16[128,6144]{1,0} dot(%c.1, %parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
// CHECK:   ROOT %fusion = f16[128,6144]{1,0} fusion(%x, %y), kind=kCustom, calls=%triton_gemm_out, backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{5e-3, 5e-3}));
}

TEST_F(TritonAutotunerTest, Int8FusedGemm256) {
  const std::string hlo = R"(
HloModule module

ENTRY e {
  x = s8[128,256] parameter(0)
  c = f16[128,256] convert(x)

  y = f16[256,6144] parameter(1)

  ROOT out = f16[128,6144] dot(c, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  CheckTritonAutotuning(hlo, R"(
// CHECK:   %triton_gemm_out (
// CHECK:   ROOT %out.1 = f16[128,6144]{1,0} dot(%c.1, %parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
// CHECK:   ROOT %fusion = f16[128,6144]{1,0} fusion(%x, %y), kind=kCustom, calls=%triton_gemm_out, backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{1e-2, 1e-2}));
}

TEST_F(TritonAutotunerTest, KnownBestConfig) {
  const std::string hlo = R"(
HloModule t

ENTRY e {
  p0 = f16[16,12288]{1,0} parameter(0)
  p1 = s8[2304,12288]{1,0} parameter(1)
  c = f16[2304,12288]{1,0} convert(p1)
  ROOT _ = f16[2304,16]{1,0} dot(c, p0), lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

  // This is the fastest config amongst the currently probed ones
  // at least on RTX A6000 and V100; feel free to modify on related changes.
  const se::DeviceDescription& device_description =
      GetTestPlatform()->ExecutorForDevice(0).value()->GetDeviceDescription();
  const std::string& name = device_description.name();
  if (name == "NVIDIA RTX A6000" || name == "Tesla V100-SXM2-16GB") {
    CheckTritonAutotuning(hlo, R"(
// CHECK: backend_config="{\"block_m\":\"32\",\"block_n\":\"32\",\"block_k\":\"256\",\"split_k\":\"1\",\"num_stages\":\"1\",\"num_warps\":\"4\"}"
  )");
  } else {
    VLOG(1) << "Not tested on " << name;
  }

  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{0.02, 0.01}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
