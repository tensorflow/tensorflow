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

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

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

  std::string annotated_ir = DumpTritonIR(triton_module.module.get(), true);

  if constexpr (EmitterLocOpBuilder::kSourceLocationSupported) {
    EXPECT_THAT(RunFileCheck(annotated_ir, R"(
      CHECK:  [[SOMETHING:.*]] "triton_dot -> [[FILE_LINE:fusion_emitter.*:.*]]"
    )"),
                IsOkAndHolds(true));
  } else {
    EXPECT_THAT(RunFileCheck(annotated_ir, R"(
      CHECK:  [[SOMETHING:.*]] "triton_dot"
    )"),
                IsOkAndHolds(true));
  }
}

using TritonEmitterDevicelessTest = GpuCodegenTest;

TEST_F(TritonEmitterDevicelessTest, FailsGracefullyIfNumWarpsIsMissing) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  ROOT add = f32[10,10] add(p0, p1)
}

ENTRY entry {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  ROOT r = f32[10,10] fusion(p0, p1),
    kind=kCustom, calls=triton_computation,
    backend_config={"fusion_backend_config": {
      "kind":"__triton",
      "block_level_fusion_config": {"output_tiles":[{"sizes": ["1","1"]}]}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloText));
  const HloFusionInstruction* triton_fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{1, 1}};
  block_level_parameters.num_warps = 0;

  EXPECT_THAT(TritonWrapper("test_fn", triton_fusion,
                            se::CudaComputeCapability::Hopper(), dev_info,
                            block_level_parameters, &llvm_module, mlir_context),
              tsl::testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  ::testing::HasSubstr(
                      "(num_warps, num_ctas, num_stages) must be positive")));
}

}  // namespace
}  // namespace xla::gpu
