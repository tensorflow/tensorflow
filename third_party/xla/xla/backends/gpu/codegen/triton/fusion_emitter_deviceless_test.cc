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
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using TritonEmitterDevicelessTest = HloHardwareIndependentTestBase;

class WarpSpecializationTritonEmitterTest : public TritonEmitterDevicelessTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        TritonEmitterDevicelessTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_enable_triton_warp_specialization(
        true);
    return debug_options;
  }
};

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
  llvm::Triple triple(nvptx::TargetTriple());
  std::string data_layout = nvptx::DataLayout();
  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{1, 1}};
  block_level_parameters.num_warps = 0;

  EXPECT_THAT(TritonWrapper(
                  "test_fn", triton_fusion,
                  se::GpuComputeCapability{se::CudaComputeCapability::Hopper()},
                  dev_info, block_level_parameters, triple, data_layout,
                  llvm_ctx, mlir_context),
              absl_testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  ::testing::HasSubstr(
                      "(num_warps, num_ctas, num_stages) must be positive")));
}

TEST_F(TritonEmitterDevicelessTest,
       BitcastReshapeDifferentTotalSizeRegressionTest) {
  // This is a regression test for a bug where indexing analysis would fail to
  // correctly preserve trivial dimensions, causing symbolic tile analysis to
  // produce an incorrect reshape.
  const std::string kHloText = R"(
parameter0 {
  p0 = bf16[1,5,4] parameter(0)
  convert = f32[1,5,4] convert(p0)
  slice = f32[1,5,2] slice(convert), slice={[0:1], [0:5], [2:4]}
  ROOT bitcast = f32[5,2] bitcast(slice)
}

parameter1 {
  ROOT p0 = f32[5,20]{0,1} parameter(0)
}

fusion {
  p0 = bf16[1,5,4] parameter(0)
  p1 = f32[5,20]{0,1} parameter(1)
  fusion0 = f32[5,2] fusion(p0), kind=kCustom, calls=parameter0,
    backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion",
      "block_level_fusion_config":{
        "num_warps":"2",
        "output_tiles":[{"sizes":["16","16"]}],
        "num_ctas":1,
        "num_stages":1,
        "is_tma_allowed":false}}}
  fusion1 = f32[5,20]{0,1} fusion(p1), kind=kCustom, calls=parameter1,
    backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion",
      "block_level_fusion_config":{
        "num_warps":"2",
        "output_tiles":[{"sizes":["16","16"]}],
        "num_ctas":1,
        "num_stages":1,
        "is_tma_allowed":false}}}
  ROOT dot = f32[2,20] dot(fusion0, fusion1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = bf16[1,5,4] parameter(0)
  p1 = f32[5,20]{0,1} parameter(1)
  ROOT root = f32[2,20] fusion(p0, p1), kind=kCustom, calls=fusion,
    backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion",
      "block_level_fusion_config":{
        "num_warps":"2",
        "output_tiles":[{"sizes":["16","16"]}],
        "num_ctas":1,
        "num_stages":1,
        "is_tma_allowed":false}}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloText));
  const HloFusionInstruction* triton_fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);

  EXPECT_OK(
      CreateTritonModule("test_fn", triton_fusion, dev_info,
                         BlockLevelParameters::FromBlockLevelFusionConfig(
                             triton_fusion->backend_config<GpuBackendConfig>()
                                 ->fusion_backend_config()
                                 .block_level_fusion_config()),
                         mlir_context));
}

TEST_F(WarpSpecializationTritonEmitterTest,
       ExtraWarpsAreRequestedForWarpSpecialization) {
  const std::string hlo_text = R"(
flhs {
  ROOT flhs.p0 = f16[256,256] parameter(0)
}

frhs {
  ROOT frhs.p0 = f16[256,256] parameter(0)
}

fdot {
  fdot.p0 = f16[256,256] parameter(0)
  fdot.p1 = f16[256,256] parameter(1)
  fdot.lhs = f16[256,256] fusion(fdot.p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["128", "64"]}],
        "is_tma_allowed":"1",
        "is_warp_specialization_allowed":"1"
      }
    }
  }
  fdot.rhs = f16[256,256]{1,0} fusion(fdot.p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64", "128"]}],
        "is_tma_allowed":"1",
        "is_warp_specialization_allowed":"1"
      }
    }
  }
  ROOT fdot.root = f16[256,256]{1,0} dot(fdot.lhs, fdot.rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_f16_f16_f32
}

ENTRY entry {
  entry.p0 = f16[256,256] parameter(0)
  entry.p1 = f16[256,256] parameter(1)
  ROOT fusion = f16[256,256] fusion(entry.p0, entry.p1),
    kind=kCustom, calls=fdot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128", "128"]}],
          "num_warps":"8",
          "num_ctas":"1",
          "num_stages":"1",
          "is_tma_allowed":"1",
          "is_warp_specialization_allowed":"1"}}}
})";

  // Check that we extract the launch configuration correctly when warp
  // specialization is used.
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  const se::DeviceDescription dev_info = TestGpuDeviceInfo::B200SXMDeviceInfo();
  llvm::LLVMContext llvm_ctx;
  llvm::Triple triple(nvptx::TargetTriple());
  std::string data_layout = nvptx::DataLayout();
  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);
  TF_ASSERT_OK_AND_ASSIGN(
      TritonWrapperResult result,
      TritonWrapper("test_fn", fusion, se::CudaComputeCapability::Blackwell(),
                    dev_info,
                    BlockLevelParameters::FromBlockLevelFusionConfig(
                        fusion->backend_config<GpuBackendConfig>()
                            ->fusion_backend_config()
                            .block_level_fusion_config()),
                    triple, data_layout, llvm_ctx, mlir_context));

  // Warp specialization influences the total number of threads we end up
  // using. Usually we would expect num_warps * warp_size threads per block, but
  // Triton allocates extra "worker warps" when WS is used.
  //
  // NOTE: The value used here is based on inspecting the value in the IR.
  // Hopefully this is stable across different Triton versions. If it starts
  // failing, we could modify the value here to match and try to understand why
  // it changed.
  EXPECT_EQ(result.thread_dims.x, 384);
  EXPECT_EQ(result.thread_dims.y, 1);
  EXPECT_EQ(result.thread_dims.z, 1);
}

}  // namespace
}  // namespace xla::gpu
