/* Copyright 2024 The OpenXLA Authors.

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

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/gpu/fusions/emitter_loc_op_builder.h"
#include "xla/service/gpu/fusions/triton/triton_fusion_emitter.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

#if defined(PLATFORM_GOOGLE)
#else

#endif
namespace xla::gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

class AnnotationsTest : public GpuCodegenTest {
 public:
  const stream_executor::GpuComputeCapability& GpuComputeComp() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability();
  }
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_unsupported_annotate_with_emitter_loc(true);
    return debug_options;
  }
};

TEST_F(AnnotationsTest, Annotations) {
  static constexpr absl::string_view kHloText = R"(
    HloModule Annotations

    triton_dot {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      ROOT dot = f32[8,8] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0},
        algorithm=dot_bf16_bf16_f32_x3
    }

    ENTRY e {
      p0 = f32[8,8]{1, 0} parameter(0)
      p1 = f32[8,8]{1, 0} parameter(1)
      ROOT _ = f32[8,8] fusion(p0, p1), kind=kCustom, calls=triton_dot,
        backend_config={"fusion_backend_config": {kind: "__triton_gemm",
          triton_gemm_config:
          {
            "block_m":32,
            "block_n":32,
            "block_k":32,
            "split_k":1,
            "num_stages":1,
            "num_warps":1,
            "num_ctas":1
          }
        }
      }
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  auto* comp = module->GetComputationWithName("triton_dot");
  EXPECT_NE(comp, nullptr);
  auto fusion_backend_config = comp->FusionInstruction()
                                   ->backend_config<GpuBackendConfig>()
                                   ->fusion_backend_config();
  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          fusion_backend_config.block_level_fusion_config());

  auto* fusion = Cast<HloFusionInstruction>(comp->FusionInstruction());

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      auto triton_module,
      CreateTritonModule("triton_fn", fusion,
                         TestGpuDeviceInfo::RTXA6000DeviceInfo(),
                         block_level_parameters, context));

  std::string annotated_ir = DumpTritonIR(triton_module.get(), true);

  if constexpr (EmitterLocOpBuilder::kSourceLocationSupported) {
    EXPECT_THAT(RunFileCheck(annotated_ir, R"(
      CHECK:  [[SOMETHING:.*]] "triton_dot -> [[FILE_LINE:triton_fusion_emitter.*:.*]]"
    )"),
                IsOkAndHolds(true));
  } else {
    EXPECT_THAT(RunFileCheck(annotated_ir, R"(
      CHECK:  [[SOMETHING:.*]] "triton_dot"
    )"),
                IsOkAndHolds(true));
  }
}

}  // namespace
}  // namespace xla::gpu
