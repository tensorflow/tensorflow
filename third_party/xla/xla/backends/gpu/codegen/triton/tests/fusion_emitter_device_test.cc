/* Copyright 2023 The OpenXLA Authors.

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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "Eigen/Core"
#include "llvm/IR/LLVMContext.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/codegen/triton/test_utils.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/codegen/triton/xtile_test_base.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/error_spec.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace gpu {
namespace {

const HloFusionInstruction& GetFusionInstruction(
    const HloModule& hlo_module, absl::string_view fusion_name) {
  return *Cast<HloFusionInstruction>(
      hlo_module.GetComputationWithName(fusion_name)->FusionInstruction());
}

constexpr ErrorSpec kExactMatch{/*aabs=*/0, /*arel=*/0};

class TritonEmitterTest
    : public HloPjRtInterpreterReferenceMixin<GpuPjRtCodegenTest>,
      public XTileTestBase {
 public:
  const stream_executor::GpuComputeCapability& GpuComputeCapability() {
    return device_description().gpu_compute_capability();
  }
  stream_executor::CudaComputeCapability GetCudaComputeCapability() {
    return device_description().cuda_compute_capability();
  }
  absl::StatusOr<
      std::pair<mlir::OwningOpRef<mlir::ModuleOp>, std::unique_ptr<HloModule>>>
  CreateXTileIrAndFileCheck(absl::string_view hlo_text,
                            absl::string_view triton_fusion_name,
                            absl::string_view filecheck_pattern) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                        ParseAndReturnVerifiedModule(hlo_text));
    return XTileTestBase::CreateXTileIrAndFileCheck(
        std::move(module), triton_fusion_name, filecheck_pattern);
  }
  absl::Status CreateTritonIrFromHloTextAndFileCheck(
      absl::string_view hlo_text, absl::string_view triton_fusion_name,
      absl::string_view filecheck_pattern) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                        ParseAndReturnVerifiedModule(hlo_text));
    return CreateTritonIrAndFileCheck(module.get(), triton_fusion_name,
                                      filecheck_pattern);
  }
  absl::Status CreateTritonIrFromHloTextAndFileCheckForDot(
      absl::string_view hlo_text, absl::string_view triton_fusion_name,
      absl::string_view filecheck_pattern) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                        ParseAndReturnVerifiedModule(hlo_text));
    return CreateTritonIrAndFileCheckForDot(module.get(), triton_fusion_name,
                                            filecheck_pattern);
  }
};

class TmaParameterizedTritonEmitterTest
    : public TritonEmitterTest,
      public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(TmaParameterizedTritonEmitterTestSuite,
                         TmaParameterizedTritonEmitterTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "tma_allowed" : "tma_disabled";
                         });

class WarpSpecializationTritonEmitterTest : public TritonEmitterTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = TritonEmitterTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_enable_triton_warp_specialization(
        true);
    return debug_options;
  }
};

TEST_F(TritonEmitterTest, BitcastReduceWithStride4Tiling) {
  constexpr absl::string_view kHloText = R"(
HloModule m

region {
  param_0.1 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT add = f32[] add(param_0.1, param_1)
}

fused_computation {
  param_0.2 = f32[64] parameter(0)
  abs = f32[64] abs(param_0.2)
  bitcast = f32[4,4,4] bitcast(abs)
  constant = f32[] constant(0)
  reduce = f32[4,4] reduce(bitcast, constant), dimensions={1}, to_apply=region
  ROOT tuple = (f32[4,4], f32[64]) tuple(reduce, abs)
}

ENTRY entry_computation {
  param_0.3 = f32[64] parameter(0)
  ROOT fusion = (f32[4,4], f32[64]) fusion(param_0.3), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1", "1"]},{"sizes":["4"]}],
          "num_warps":"2",
          "num_ctas":"1",
          "num_stages":"1"}}}

})";
  auto status =
      CreateTritonIrFromHloTextAndFileCheck(kHloText, "fused_computation", "");
  EXPECT_THAT(
      status,
      absl_testing::StatusIs(
          tsl::error::UNIMPLEMENTED,
          ::testing::HasSubstr("Unsupported case of multi-output fusion")));
}

class TritonEmitterTestWithOffsetParam
    : public TritonEmitterTest,
      public ::testing::WithParamInterface<int32_t> {};

using EmitDynamicSliceTest = TritonEmitterTestWithOffsetParam;

INSTANTIATE_TEST_SUITE_P(DynamicSliceSuite, EmitDynamicSliceTest,
                         ::testing::Values(0, 1, 10, 100),
                         [](const ::testing::TestParamInfo<int32_t>& info) {
                           return absl::StrCat("offset_", info.param);
                         });

TEST_P(EmitDynamicSliceTest, LowerDynamicSliceWithSingleDimension) {
  int32_t offset = GetParam();
  constexpr absl::string_view kHloText = R"(
f {
  p0 = f32[64] parameter(0)
  c0 = s32[] parameter(1)
  ROOT r = f32[10] dynamic-slice(p0, c0), dynamic_slice_sizes={10}
}

ENTRY entry_computation {
  p0 = f32[64] parameter(0)
  p1 = s32[] parameter(1)
  ROOT fusion = f32[10] fusion(p0, p1), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{
      "output_tiles":[{"sizes":["32"]}],
        "num_warps":1,"num_ctas":1,"num_stages":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> parameters,
                          MakeFakeArguments(module.get()));
  parameters[1].Set<int32_t>({}, offset);
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), LiteralUtil::MakePointers(parameters), kExactMatch));
}

TEST_P(EmitDynamicSliceTest, LowerDynamicSliceOfWithDimensions) {
  constexpr absl::string_view kHloText = R"(
HloModule m

f {
  p0 = s32[64, 32] parameter(0)
  off0 = s32[] parameter(1)
  off1 = s32[] parameter(2)
  ROOT r = s32[10, 5] dynamic-slice(p0, off0, off1), dynamic_slice_sizes={10, 5}
}

ENTRY entry_computation {
  p0 = s32[64, 32] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  ROOT fusion = s32[10, 5] fusion(p0, p1, p2), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{
      "output_tiles":[{"sizes":["32", "8"]}],
        "num_warps":1,"num_ctas":1,"num_stages":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> parameters,
                          MakeFakeArguments(module.get()));
  int32_t offset = GetParam();
  parameters[1].Set<int32_t>({}, offset);
  parameters[2].Set<int32_t>({}, offset);
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), LiteralUtil::MakePointers(parameters), kExactMatch));
}

TEST_P(EmitDynamicSliceTest, LowerDynamicSliceCoveringWholeInput) {
  constexpr absl::string_view kHloText = R"(
f {
  p0 = s32[64, 32] parameter(0)
  c0 = s32[] parameter(1)
  c1 = s32[] parameter(2)
  ROOT r = s32[10, 32] dynamic-slice(p0, c0, c1), dynamic_slice_sizes={10, 32}
}

ENTRY entry_computation {
  p0 = s32[64, 32] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  ROOT fusion = s32[10, 32] fusion(p0, p1, p2), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{
      "output_tiles":[{"sizes":["32", "8"]}],
        "num_warps":1,"num_ctas":1,"num_stages":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> parameters,
                          MakeFakeArguments(module.get()));
  int32_t offset = GetParam();
  parameters[1].Set<int32_t>({}, offset);
  parameters[2].Set<int32_t>({}, offset);
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), LiteralUtil::MakePointers(parameters), kExactMatch));
}

TEST_P(EmitDynamicSliceTest, LowerDynamicSliceWithConstantOffset) {
  int32_t offset = GetParam();
  std::string kHloText =
      absl::StrReplaceAll(R"(
f {
  p0 = s32[64, 32, 16] parameter(0)
  c50 = s32[] constant(50)
  p1 = s32[] parameter(1)
  c0 = s32[] constant(_offset_)
  ROOT r = s32[10, 10, 8] dynamic-slice(p0, c50, p1, c0), dynamic_slice_sizes={10, 10, 8}
}

ENTRY entry_computation {
  p0 = s32[64, 32, 16] parameter(0)
  p1 = s32[] parameter(1)
  ROOT fusion = s32[10, 10, 8] fusion(p0, p1), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{
      "output_tiles":[{"sizes":["32", "8", "8"]}],
        "num_warps":1,"num_ctas":1,"num_stages":1}}}
})",
                          {{"_offset_", absl::StrCat(offset)}});
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_P(EmitDynamicSliceTest, LowerDynamicSliceOfDot) {
  const std::string kHloText = R"(
fdot {
  p0 = f32[64,32] parameter(0)
  p1 = f32[64,512] parameter(1)
  p2 = s32[] parameter(2)
  gemm = f32[32,512] dot(p0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0},
    algorithm=dot_f32_f32_f32, backend_config={sizes:[32]}
  c0 = s32[] constant(0)
  ROOT d = f32[8,16] dynamic-slice(gemm, c0, p2), dynamic_slice_sizes={8,16}
}

ENTRY entry {
  p0 = pred[64,32] parameter(0)
  p1 = pred[64,512] parameter(1)
  p2 = s32[] parameter(2)
  p0_f32 = f32[64,32] convert(p0)
  p1_f32 = f32[64,512] convert(p1)
  ROOT fusion = f32[8,16] fusion(p0_f32, p1_f32, p2),
    kind=kCustom, calls=fdot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16", "64"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> parameters,
                          MakeFakeArguments(module.get()));
  int32_t offset = GetParam();
  parameters[2].Set<int32_t>({}, offset);
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), LiteralUtil::MakePointers(parameters), kExactMatch));
}

TEST_F(TritonEmitterTest, LowerDynamicSliceOfAdd) {
  constexpr absl::string_view kHloText = R"(
HloModule m

f {
  p0 = s32[64] parameter(0)
  p1 = s32[] parameter(1)
  c7 = s32[] constant(7)
  add = s32[] add(c7, p1)
  ROOT r = s32[10] dynamic-slice(p0, add), dynamic_slice_sizes={10}
}

ENTRY entry_computation {
  p0 = s32[64] parameter(0)
  p1 = s32[] parameter(1)
  ROOT fusion = s32[10] fusion(p0, p1), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{
      "output_tiles":[{"sizes":["32"]}],
        "num_warps":1,"num_ctas":1,"num_stages":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> parameters,
                          MakeFakeArguments(module.get()));
  parameters[1].Set<int32_t>({}, 13);
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), LiteralUtil::MakePointers(parameters), kExactMatch));
}

class TritonDevicelessTest : public HloHardwareIndependentTestBase,
                             public XTileTestBase {};

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonDevicelessTest, TestGenericEmitterWithSoftMaxSingleParameter) {
  constexpr absl::string_view kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0),
    kind=kCustom, calls=triton_softmax_computation, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1", "128"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(
      auto xtile_module_and_hlo_module,
      CreateXTileIrAndFileCheck(std::move(module), "triton_softmax_computation",
                                R"(
CHECK:        xtile.entry_func @xtile_dialect_fn(%[[P0:.*]]: {{.*}}, %[[P1:.*]]: {{.*}}, %[[PID:.*]]: index)
CHECK-DAG:        %[[EXTRACT_IDX_0:.*]] = xla.apply_indexing #indexing_map(%[[PID]])
CHECK-NEXT:       xtile.extract %[[P0]]
CHECK-SAME:       [%[[PID]], %[[EXTRACT_IDX_0]]] [1, 128] [1, 1]
CHECK:            stablehlo.reduce{{.*}} applies stablehlo.add
CHECK:            stablehlo.multiply
CHECK-SAME:       tensor<1x128xf32>
CHECK:            xtile.insert {{.*}}[%[[PID]], %{{.*}}] [1, 128] [1, 1]
CHECK:            return
CHECK:        }
)"));

  TF_EXPECT_OK(LowerXTileIrToTritonAndFileCheck(
      xtile_module_and_hlo_module.first.get(), R"(
CHECK:        xtile.entry_func @xtile_dialect_fn(%[[P0:.*]]: {{.*}}, %[[P1:.*]]: {{.*}}, %[[PID:.*]]: index)
CHECK-DAG:        %[[C_0:.*]] = arith.constant 0 : index
CHECK-NEXT:       xtile.extract %[[P0]]
CHECK-SAME:       [%[[PID]], %[[C_0]]] [1, 128] [1, 1]
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG2:[^:]*]]: f32, %[[ARG3:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG2]], %[[ARG3]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<1x128xf32>) -> tensor<1xf32>
CHECK:            arith.mulf
CHECK-SAME:       tensor<1x128xf32>
CHECK:            xtile.insert {{.*}}[%[[PID]], %[[C_0]]] [1, 128] [1, 1]
CHECK:            return
CHECK:        }
)",
      GetFusionInstruction(*xtile_module_and_hlo_module.second,
                           "triton_softmax_computation")));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonDevicelessTest, TestGenericEmitterWithMultipleParameters) {
  constexpr absl::string_view kHloText = R"(
HloModule t

add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[127]{0} parameter(1)
  broadcast_0 = f32[125,127]{1,0} broadcast(param_1), dimensions={1}
  multiply_0 = f32[125,127]{1,0} multiply(param_0, broadcast_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[127]{0} parameter(1)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0, param_1),
    kind=kCustom, calls=triton_softmax_computation,
    backend_config={"fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1", "128"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(
      auto xtile_module_and_hlo_module,
      CreateXTileIrAndFileCheck(std::move(module), "triton_softmax_computation",
                                R"(
CHECK:         xtile.entry_func @xtile_dialect_fn(
CHECK-SAME:                      %[[P0:[A-Za-z0-9_]*]]: memref<125x127xf32>
CHECK-SAME:                      %[[P1:[A-Za-z0-9_]*]]: memref<127xf32>
CHECK-SAME:                      %[[P2:[A-Za-z0-9_]*]]: memref<125x127xf32>
CHECK-SAME:                      %[[TID:[A-Za-z0-9_]*]]: index)
CHECK-DAG:        %[[EXTRACT_IDX_0:.*]] = xla.apply_indexing #indexing_map(%[[TID]])
CHECK-DAG:        xtile.extract %[[P0]][%[[TID]], %[[EXTRACT_IDX_0]]] [1, 128] [1, 1] : {{.*}} -> tensor<1x128xf32>
CHECK-DAG:        %[[EXTRACT_IDX_1:.*]] = xla.apply_indexing #indexing_map(%[[TID]])
CHECK-DAG:        xtile.extract %[[P1]][%[[EXTRACT_IDX_1]]] [128] [1] : {{.*}} -> tensor<128xf32>
CHECK:            stablehlo.reduce{{.*}} applies stablehlo.add
CHECK:            stablehlo.multiply
CHECK-DAG:        xtile.insert {{.*}} into %[[P2]]
CHECK-SAME:       [%[[TID]], %{{.*}}] [1, 128] [1, 1] : tensor<1x128xf32>
)"));

  TF_EXPECT_OK(LowerXTileIrToTritonAndFileCheck(
      xtile_module_and_hlo_module.first.get(), R"(
CHECK:         xtile.entry_func @xtile_dialect_fn(
CHECK-SAME:                      %[[P0:[A-Za-z0-9_]*]]: memref<125x127xf32>
CHECK-SAME:                      %[[P1:[A-Za-z0-9_]*]]: memref<127xf32>
CHECK-SAME:                      %[[P2:[A-Za-z0-9_]*]]: memref<125x127xf32>
CHECK-SAME:                      %[[TID:[A-Za-z0-9_]*]]: index)
CHECK-DAG:        %[[C_0:.*]] = arith.constant 0 : index
CHECK-DAG:        xtile.extract %[[P0]][%[[TID]], %[[C_0]]] [1, 128] [1, 1] : {{.*}} -> tensor<1x128xf32>
CHECK-DAG:        xtile.extract %[[P1]][%[[C_0]]] [128] [1] : {{.*}} -> tensor<128xf32>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG3:[^:]*]]: f32, %[[ARG4:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG3]], %[[ARG4]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<1x128xf32>) -> tensor<1xf32>
CHECK:            arith.mulf
CHECK-DAG:        xtile.insert {{.*}} into %[[P2]]
CHECK-SAME:       [%[[TID]], %[[C_0]]] [1, 128] [1, 1] : tensor<1x128xf32>
)",
      GetFusionInstruction(*xtile_module_and_hlo_module.second,
                           "triton_softmax_computation")));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(HloHardwareIndependentTestBase,
       EmitterFailsIfComputeCapabilityIsBelowAmpere) {
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
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","1"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloText));
  const HloFusionInstruction* triton_fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(se::CudaComputeCapability(7, 0));
  llvm::LLVMContext llvm_ctx;
  mlir::MLIRContext mlir_context;
  llvm::Triple target_triple(nvptx::TargetTriple());
  std::string data_layout(nvptx::DataLayout());

  EXPECT_THAT(
      TritonWrapper("test_fn", *triton_fusion,
                    se::CudaComputeCapability{se::CudaComputeCapability::kVolta,
                                              /*minor=*/0},
                    dev_info, BlockLevelParameters(), target_triple,
                    data_layout, llvm_ctx, mlir_context),
      absl_testing::StatusIs(
          absl::StatusCode::kFailedPrecondition,
          ::testing::HasSubstr("Triton support is only enabled for Ampere GPUs "
                               "(compute capability 8.0) and up, but got")));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(HloHardwareIndependentTestBase,
       EmitterFailsIfFusionBackendConfigDoesNotSatisfyConstraints) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

max_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0, param_1)
}

fused_computation {
  param_0 = f32[8192,50304] parameter(0)
  constant = f32[] constant(-inf)
  reduce = f32[8192] reduce(param_0, constant), dimensions={1}, to_apply=max_computation
  broadcast = f32[8192,50304] broadcast(reduce), dimensions={0}
  ROOT subtract = f32[8192,50304] subtract(param_0, broadcast)
}

ENTRY entry_computation {
  param_0 = f32[8192,50304] parameter(0)
  ROOT fusion = f32[8192,50304] fusion(param_0),
    kind=kCustom, calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1024","1"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})"));
  const HloFusionInstruction* triton_fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto compute_capability = se::CudaComputeCapability{
      se::CudaComputeCapability::kHopper, /*minor=*/0};
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(compute_capability);
  llvm::LLVMContext llvm_ctx;
  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);
  llvm::Triple target_triple(nvptx::TargetTriple());
  std::string data_layout(nvptx::DataLayout());

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{1024, 1}};
  block_level_parameters.num_warps = 1;

  // Because of reduce, we need to load full rows from param_0 and the load tile
  // will be 1024 * 65536 = 67108864 elements, that is larger than the limit of
  // 1048576.
  EXPECT_THAT(
      TritonWrapper("test_fn", *triton_fusion, compute_capability, dev_info,
                    block_level_parameters, target_triple, data_layout,
                    llvm_ctx, mlir_context),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr("Tiling does not satisfy constraints.")));
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should b
// moved to deviceless test file.
TEST_F(TritonDevicelessTest, TestGenericEmitterReductionFusion) {
  constexpr absl::string_view kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_reduction_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  parameter_1 = f32[125]{0} parameter(1)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  ROOT multiply = f32[125]{0} multiply(parameter_1, reduce_0)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[125]{0} parameter(1)
  ROOT triton_reduction = f32[125]{0} fusion(param_0, param_1),
    kind=kCustom, calls=triton_reduction_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_warps":"1",
        "num_ctas":"1",
          "num_stages":"1"}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto xtile_module_and_hlo_module,
                          CreateXTileIrAndFileCheck(
                              std::move(module), "triton_reduction_computation",
                              R"(
CHECK:        xtile.entry_func @xtile_dialect_fn(%[[P0:[A-Za-z0-9_]*]]: memref<125x127xf32>
CHECK-SAME:                               %[[P1:[A-Za-z0-9_]*]]: memref<125xf32>
CHECK-SAME:                               %[[P2:[A-Za-z0-9_]*]]: memref<125xf32>
CHECK-DAG:        xtile.extract {{.*}} -> tensor<1xf32>
CHECK-DAG:        xtile.extract {{.*}} -> tensor<1x128xf32>
CHECK: %[[REDUCE:.*]] = stablehlo.reduce(%[[REDUCE_ARG:.*]] init: %{{.*}}) applies stablehlo.add across dimensions = [1] : (tensor<1x128xf32>,    tensor<f32>) -> tensor<1xf32>
CHECK:            stablehlo.multiply {{.*}} tensor<1xf32>
CHECK:            xtile.insert {{.*}} : tensor<1xf32>
)"));

  TF_EXPECT_OK(LowerXTileIrToTritonAndFileCheck(
      xtile_module_and_hlo_module.first.get(), R"(
CHECK:        xtile.entry_func @xtile_dialect_fn(%[[P0:[A-Za-z0-9_]*]]: memref<125x127xf32>
CHECK-SAME:                               %[[P1:[A-Za-z0-9_]*]]: memref<125xf32>
CHECK-SAME:                               %[[P2:[A-Za-z0-9_]*]]: memref<125xf32>
CHECK-DAG:        xtile.extract {{.*}} -> tensor<1xf32>
CHECK-DAG:        xtile.extract {{.*}} -> tensor<1x128xf32>
CHECK:            tt.reduce
CHECK:              (tensor<1x128xf32>) -> tensor<1xf32>
CHECK:            arith.mulf {{.*}} tensor<1xf32>
CHECK:            xtile.insert {{.*}} : tensor<1xf32>
)",
      GetFusionInstruction(*xtile_module_and_hlo_module.second,
                           "triton_reduction_computation")));
}

TEST_F(TritonEmitterTest,
       TestGenericEmitterWithReductionAndMultidimensionalTile) {
  constexpr absl::string_view kHloText = R"(
HloModule t
max {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT max = f32[] maximum(Arg_0, Arg_1)
}

triton_reduction_computation {
  parameter_0 = f32[4,12,125,127]{3,2,1,0} parameter(0)
  constant_0 = f32[] constant(-inf)
  ROOT reduce = f32[4,12,125]{2,1,0} reduce(parameter_0, constant_0), dimensions={3}, to_apply=max
}

ENTRY main {
  param_0 = f32[4,12,125,127]{3,2,1,0} parameter(0)
  ROOT triton_reduce = f32[4,12,125]{2,1,0} fusion(param_0),
    kind=kCustom, calls=triton_reduction_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2","8","16"]}],
        "num_warps":"4",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TritonEmitterTest, TestSoftMaxWithTileElementsNotAllContiguous) {
  constexpr absl::string_view kHloText = R"(
HloModule m

region {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(param_0, param_1)
}

triton_softmax_computation {
  constant.1 = f32[] constant(0)
  broadcast.2 = f32[4,4,8] broadcast(constant.1), dimensions={}
  param_0.1 = f32[4,4,8] parameter(0)
  constant = f32[] constant(0)
  reduce = f32[4,4] reduce(param_0.1, constant), dimensions={2}, to_apply=region
  broadcast = f32[4,4,8] broadcast(reduce), dimensions={0,1}
  multiply = f32[4,4,8] multiply(broadcast.2, broadcast)
  ROOT add.2 = f32[4,4,8] add(multiply, broadcast)
}

ENTRY entry_computation {
  param_0.2 = f32[4,4,8] parameter(0)
  ROOT fusion = f32[4,4,8] fusion(param_0.2), kind=kCustom,
    calls=triton_softmax_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2","2","8"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}

})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                           /*arel=*/1e-6}));
}

// Parameterized to make sure that slices are also handled correctly when TMA is
// enabled.
TEST_P(TmaParameterizedTritonEmitterTest, TestSliceWithTileThatNeedsMasking) {
  constexpr absl::string_view kHloTextTemplate = R"(
HloModule m

fused_computation {
  p = f32[128,32] parameter(0)
  ROOT slice = f32[12,5] slice(p), slice={[116:128], [20:25]}
}

ENTRY entry_computation {
  p = f32[128,32] parameter(0)
  ROOT fusion = f32[12,5] fusion(p), kind=kCustom, calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8","4"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1",
        "is_tma_allowed":"$0"}}}
})";

  const bool is_tma_allowed = GetParam();
  const std::string hlo_text =
      absl::Substitute(kHloTextTemplate, is_tma_allowed);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, kExactMatch));
}

// Parameterized to make sure that tile strides are handled correctly when TMA
// is enabled.
TEST_P(TmaParameterizedTritonEmitterTest, TestSliceWithNonMinorDimStrides) {
  constexpr absl::string_view kHloTextTemplate = R"(
HloModule m

fused_computation {
  p = f32[128,64,32] parameter(0)
  ROOT slice = f32[12,16,16] slice(p), slice={[102:126:2], [6:38:2], [16:32]}
}

ENTRY entry_computation {
  p = f32[128,64,32] parameter(0)
  ROOT fusion = f32[12,16,16] fusion(p), kind=kCustom, calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4","2","4"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1",
        "is_tma_allowed":"$0"}}}
})";
  const bool is_tma_allowed = GetParam();
  const std::string hlo_text =
      absl::Substitute(kHloTextTemplate, is_tma_allowed);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, kExactMatch));
}

TEST_F(TritonEmitterTest, TestSliceWithTileElementsNotAllContiguous) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0 = f32[16,16,32] parameter(0)
  slice = f32[4,4,8] slice(param_0), slice={[2:10:2], [2:6], [3:11]}
  slice.1 = f32[4,4,8] slice(param_0), slice={[4:8], [8:16:2], [13:21]}
  ROOT add = f32[4,4,8] add(slice, slice.1)
}

ENTRY entry_computation {
  param_0 = f32[16,16,32] parameter(0)
  ROOT fusion = f32[4,4,8] fusion(param_0), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2","2","8"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_P(TmaParameterizedTritonEmitterTest,
       TestSlice2DWithTileElementsNotAllContiguous) {
  constexpr absl::string_view kHloTextTemplate = R"(
HloModule m

fused_computation {
  param_0 = f32[16,32] parameter(0)
  slice = f32[4,16] slice(param_0), slice={[2:6], [3:19]}
  slice.1 = f32[4,16] slice(param_0), slice={[4:8], [13:29]}
  ROOT add = f32[4,16] add(slice, slice.1)
}

ENTRY entry_computation {
  param_0 = f32[16,32] parameter(0)
  ROOT fusion = f32[4,16] fusion(param_0), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2","8"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1",
        "is_tma_allowed":"$0"}}}
})";

  const bool is_tma_allowed = GetParam();
  const std::string hlo_text =
      absl::Substitute(kHloTextTemplate, is_tma_allowed);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, kExactMatch));
}

TEST_F(TritonEmitterTest, TestSliceWithTileElementsNotAllContiguousUnaligned) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  p = f32[7,7,75] parameter(0)
  ROOT slice = f32[3,2,14] slice(p), slice={[1:6:2], [2:6:3], [35:75:3]}
}

ENTRY entry_computation {
  p = f32[7,7,75] parameter(0)
  ROOT fusion = f32[3,2,14] fusion(p),
    kind=kCustom, calls=fused_computation, backend_config={
      "fusion_backend_config": {
        "kind":"__triton",
        "block_level_fusion_config": {
          "output_tiles":[{"sizes":["2","2","8"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// Parameterized to make sure that when TMA flag is enabled, this test would
// correctly fall back to normal loads. This is due to the fact that the most
// minor dimension is not contiguous.
// TODO(b/419025213): To FileCheck here, we need to create a test wrapper that
// also invokes TritonXlaExtractInsertPass to check TTIR.
TEST_P(TmaParameterizedTritonEmitterTest,
       TestSlice2DWithTileElementsNotAllContiguousUnaligned) {
  constexpr absl::string_view kHloTextTemplate = R"(
HloModule m

fused_computation {
  p = f32[7,80] parameter(0)
  ROOT slice = f32[2,14] slice(p), slice={[2:6:3], [35:75:3]}
}

ENTRY entry_computation {
  p = f32[7,80] parameter(0)
  ROOT fusion = f32[2,14] fusion(p),
    kind=kCustom, calls=fused_computation, backend_config={
      "fusion_backend_config": {
        "kind":"__triton",
        "block_level_fusion_config": {
          "output_tiles":[{"sizes":["2","8"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1",
          "is_tma_allowed":"$0"}}}
})";

  const bool is_tma_allowed = GetParam();
  const std::string hlo_text =
      absl::Substitute(kHloTextTemplate, is_tma_allowed);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, kExactMatch));
}

// Parameterized to test TMA with various dimensionalities for loads/stores.
TEST_P(TmaParameterizedTritonEmitterTest,
       ReshapeIntoBroadcastIsLoweredCorrectly) {
  constexpr absl::string_view kHloTextTemplate = R"(
triton_computation {
  param_0 = f32[128,256]{1,0} parameter(0)
  reshape = f32[64,2,256]{2,1,0} reshape(param_0)
  ROOT broadcast = f32[64,2,256,16]{3,2,1,0} broadcast(reshape), dimensions={0,1,2}
}

ENTRY main {
  param_0 = f32[128,256]{1,0} parameter(0)
  ROOT triton_fusion = f32[64,2,256,16]{3,2,1,0} fusion(param_0), kind=kCustom,
    calls=triton_computation, backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["2","2","4","4"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1",
          "is_tma_allowed":"$0"}}}
})";

  const bool is_tma_allowed = GetParam();
  const std::string hlo_text =
      absl::Substitute(kHloTextTemplate, is_tma_allowed);

  TF_ASSERT_OK_AND_ASSIGN(
      auto xtile_module_and_hlo_module,
      CreateXTileIrAndFileCheck(hlo_text, "triton_computation", R"(
CHECK: stablehlo.reshape
)"));

  TF_ASSERT_OK(LowerXTileIrToTritonAndFileCheck(
      xtile_module_and_hlo_module.first.get(), R"(
CHECK: tt.reshape
)",
      GetFusionInstruction(*xtile_module_and_hlo_module.second,
                           "triton_computation")));

  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, kExactMatch));
}

TEST_F(TritonEmitterTest, BitcastIntoBroadcastIsLoweredCorrectly) {
  constexpr absl::string_view kHloText = R"(
triton_computation {
  param_0 = f32[128,256]{1,0} parameter(0)
  bitcast = f32[64,2,256]{2,1,0} bitcast(param_0)
  ROOT broadcast = f32[64,2,256,2]{3,2,1,0} broadcast(bitcast), dimensions={0,1,2}
}

ENTRY main {
  param_0 = f32[128,256]{1,0} parameter(0)
  ROOT triton_fusion = f32[64,2,256,2]{3,2,1,0} fusion(param_0), kind=kCustom,
    calls=triton_computation, backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["4","2","8","2"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      auto xtile_module_and_hlo_module,
      CreateXTileIrAndFileCheck(kHloText, "triton_computation", R"(
CHECK: stablehlo.reshape
)"));

  TF_ASSERT_OK(LowerXTileIrToTritonAndFileCheck(
      xtile_module_and_hlo_module.first.get(), R"(
CHECK: tt.reshape
)",
      GetFusionInstruction(*xtile_module_and_hlo_module.second,
                           "triton_computation")));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_P(TmaParameterizedTritonEmitterTest,
       SimpleBitcastNormalizedLayoutIsLoweredCorrectly) {
  constexpr absl::string_view kHloTextTemplate = R"(
triton_computation {
  p = s16[16,32,64]{2,1,0} parameter(0)
  ROOT bitcast = s16[16,32,64] bitcast(p)
}

ENTRY entry_computation {
  p = s16[16,32,64]{2,1,0} parameter(0)
  ROOT fusion = s16[16,32,64] fusion(p), kind=kCustom, calls=triton_computation,
    backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
      "output_tiles":[{"sizes":["16","16","32"]}],
      "num_warps":"1",
      "num_ctas":"1",
      "num_stages":"1",
      "is_tma_allowed":"$0"}}}
})";

  const bool is_tma_allowed = GetParam();
  const std::string hlo_text =
      absl::Substitute(kHloTextTemplate, is_tma_allowed);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, kExactMatch));
}

// Parameterized the test to make sure that non-canonical layouts are handled
// correctly when TMA is enabled.
TEST_P(TmaParameterizedTritonEmitterTest,
       SimpleBitcastNonNormalizedInputLayoutIsLoweredCorrectly) {
  constexpr absl::string_view kHloTextTemplate = R"(
triton_computation {
  p = s32[64,32,16]{0,1,2} parameter(0)
  ROOT bitcast = s32[16,32,64] bitcast(p)
}

ENTRY entry_computation {
  p = s32[64,32,16]{0,1,2} parameter(0)
  ROOT fusion = s32[16,32,64] fusion(p), kind=kCustom, calls=triton_computation,
    backend_config={
    "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16","16","32"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1",
        "is_tma_allowed":"$0"}}}
})";

  const bool is_tma_allowed = GetParam();
  const std::string hlo_text =
      absl::Substitute(kHloTextTemplate, is_tma_allowed);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, kExactMatch));
}

// Parameterized the test to make sure that non-canonical layouts are handled
// correctly when TMA is enabled.
TEST_P(TmaParameterizedTritonEmitterTest,
       SimpleBitcastNonNormalizedOutputLayoutIsLoweredCorrectly) {
  constexpr absl::string_view kHloTextTemplate = R"(
triton_computation {
p = s32[64,16] parameter(0)
ROOT bitcast = s32[16,64]{0,1} bitcast(p)
}

ENTRY entry_computation {
p = s32[64,16] parameter(0)
ROOT fusion = s32[16,64]{0,1} fusion(p), kind=kCustom, calls=triton_computation,
backend_config={
"fusion_backend_config":{
 "kind":"__triton",
 "block_level_fusion_config":{
   "output_tiles":[{"sizes":["16","32"]}],
   "num_warps":"1",
   "num_ctas":"1",
   "num_stages":"1",
   "is_tma_allowed":"$0"}}}
})";

  const bool is_tma_allowed = GetParam();
  const std::string hlo_text =
      absl::Substitute(kHloTextTemplate, is_tma_allowed);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, kExactMatch));
}

// Parameterized the test to make sure that non-canonical layouts are handled
// correctly when TMA is enabled.
TEST_P(
    TmaParameterizedTritonEmitterTest,
    SimpleBitcastNonNormalizedOutputLayoutAndBitcastConvertIsLoweredCorrectly) {
  constexpr absl::string_view kHloTextTemplate = R"(
triton_computation {
p = f32[64,15] parameter(0)
bitcast = s32[15,64]{0,1} bitcast(p)
ROOT negate = s32[15,64]{0,1} negate(bitcast)
}

ENTRY entry_computation {
p = f32[64,15] parameter(0)
ROOT fusion = s32[15,64]{0,1} fusion(p), kind=kCustom, calls=triton_computation,
backend_config={
"fusion_backend_config":{
 "kind":"__triton",
 "block_level_fusion_config":{
   "output_tiles":[{"sizes":["15","32"]}],
   "num_warps":"1",
   "num_ctas":"1",
   "num_stages":"1",
   "is_tma_allowed":"$0"}}}
})";

  const bool is_tma_allowed = GetParam();
  const std::string hlo_text =
      absl::Substitute(kHloTextTemplate, is_tma_allowed);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, kExactMatch));
}

// When TMA is enabled, it is important to test this in an end-to-end fashion.
// This test covers the logic that adjusts box_dims based on the swizzle mode.
// See tensorflow/compiler/xla/backends/gpu/codegen/triton/tma_utils.cc.
TEST_P(TmaParameterizedTritonEmitterTest,
       ContiguousDimensionExceedsSwizzleLimitIsLoweredCorrectly) {
  constexpr absl::string_view kHloTextTemplate = R"(
triton_computation {
p = s32[16,128] parameter(0)
ROOT bitcast = s32[16,128] bitcast(p)
}

ENTRY entry_computation {
p = s32[16,128] parameter(0)
ROOT fusion = s32[16,128] fusion(p), kind=kCustom, calls=triton_computation,
backend_config={
"fusion_backend_config":{
 "kind":"__triton",
 "block_level_fusion_config":{
 "output_tiles":[{"sizes":["16","64"]}],
 "num_warps":"1",
 "num_ctas":"1",
 "num_stages":"1",
 "is_tma_allowed":"$0"}}}
})";

  const bool is_tma_allowed = GetParam();
  const std::string hlo_text =
      absl::Substitute(kHloTextTemplate, is_tma_allowed);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, kExactMatch));
}

// TODO(b/390559452): Capture the iteration order from the propagated tiling.
// When computing the tiling separately we need to use the same iteration order.
TEST_F(TritonEmitterTest, DISABLED_Transpose3DWithExtraOutput) {
  constexpr absl::string_view kHloText = R"(
HloModule m

fused_computation {
  param_0.1 = f32[15,7,3] parameter(0)
  abs = f32[15,7,3] abs(param_0.1)
  transpose = f32[3,15,7] transpose(abs), dimensions={2,0,1}
  ROOT tuple = (f32[3,15,7], f32[15,7,3]) tuple(transpose, abs)
}

ENTRY entry_computation {
  param_0.2 = f32[15,7,3] parameter(0)
  ROOT fusion = (f32[3,15,7], f32[15,7,3]) fusion(param_0.2), kind=kCustom,
    calls=fused_computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","8","4"]},{"sizes":["4","8","1"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto xtile_module_and_hlo_module,
      CreateXTileIrAndFileCheck(kHloText, "fused_computation", R"(
CHECK:         %[[TILE:.*]] = xtile.extract {{.*}} -> tensor<15x7x3xf32> to tensor<8x4x1xf32>
CHECK-NOT:     xtile.extract
CHECK:         %[[ABS:.*]] = math.absf %[[TILE]]
CHECK:         stablehlo.transpose %[[ABS]], dims = [2, 0, 1] : (tensor<8x4x1xf32>) -> tensor<1x8x4xf32>
CHECK-COUNT-2: xtile.insert
          )"));

  TF_ASSERT_OK(LowerXTileIrToTritonAndFileCheck(
      xtile_module_and_hlo_module.first.get(), R"(
CHECK:         %[[TILE:.*]] = xtile.extract {{.*}} -> tensor<15x7x3xf32> to tensor<8x4x1xf32>
CHECK-NOT:     xtile.extract
CHECK:         %[[ABS:.*]] = math.absf %[[TILE]]
CHECK:         tt.trans %[[ABS]] {order = array<i32: 2, 0, 1>} : tensor<8x4x1xf32> -> tensor<1x8x4xf32>
CHECK-COUNT-2: xtile.insert
  )",
      GetFusionInstruction(*xtile_module_and_hlo_module.second,
                           "fused_computation")));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

class IotaEmitterParametrizedTest
    : public TritonEmitterTest,
      public ::testing::WithParamInterface<PrimitiveType> {};

TEST_P(IotaEmitterParametrizedTest, Iota4DIsCodegeneratedCorrectly) {
  auto data_type = GetParam();
  const std::string kHloText =
      absl::Substitute(R"(
triton_computation {
  ROOT iota = $0[3,4,1000,5] iota(), iota_dimension=2
}

ENTRY main {
  ROOT triton_fusion = $0[3,4,1000,5] fusion(),
    kind=kCustom, calls=triton_computation, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","2","64","8"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})",
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  TF_ASSERT_OK_AND_ASSIGN(
      auto xtile_module_and_hlo_module,
      CreateXTileIrAndFileCheck(kHloText, "triton_computation", R"(
CHECK:      %[[RANGE:.*]] = stablehlo.iota dim = 0 : tensor<64xi32>
CHECK:      %[[MUL:.*]] = arith.muli %[[RANGE]], {{.*}} : tensor<64xi32>
CHECK:      arith.addi{{.*}} %[[MUL]]
            // Omit the data type below, since it depends on a test parameter
            // and is not abbreviated the same as in HLO.
CHECK:      stablehlo.broadcast_in_dim {{.*}}, dims = [2] : {{.*}}
          )"));

  TF_ASSERT_OK(LowerXTileIrToTritonAndFileCheck(
      xtile_module_and_hlo_module.first.get(), R"(
CHECK:      %[[RANGE:.*]] = tt.make_range {{.*}} : tensor<64xi32>
CHECK:      arith.addi{{.*}} %[[RANGE]]
            // Omit the data type below, since it depends on a test parameter
            // and is not abbreviated the same as in HLO.
CHECK:      tt.broadcast {{.*}} -> tensor<1x2x64x8x
  )",
      GetFusionInstruction(*xtile_module_and_hlo_module.second,
                           "triton_computation")));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

std::string TypeTestParamToString(
    const ::testing::TestParamInfo<PrimitiveType>& data) {
  return primitive_util::LowercasePrimitiveTypeName(data.param);
}

INSTANTIATE_TEST_SUITE_P(IotaEmitterParametrizedTestSuite,
                         IotaEmitterParametrizedTest,
                         ::testing::ValuesIn({S8, S16, S32, S64, BF16, F16, F32,
                                              F64}),
                         TypeTestParamToString);

// Reproducer from b/384110192.
TEST_F(TritonEmitterTest,
       FusionWithOutputContainingMoreThanInt32MaxElementsExecutesCorrectly) {
  if (GpuComputeCapability().IsRocm()) {
    GTEST_SKIP() << "Requires more than 4GB GPU memory, exceeds ROCm RBE "
                    "worker limits";
  }
  // The point here is to check the output of the Triton fusion. The `slice` op
  // at the end is inserted to allow the comparison of output to run in a
  // reasonable amount of time, and has been proven to still correctly capture
  // the indexing overflow behaviour of the Triton fusion that we're checking
  // for.
  constexpr absl::string_view kTritonHloText = R"(
computation {
  p0 = s8[256]{0} parameter(0)
  ROOT broadcast = s8[16777217,256]{1,0} broadcast(p0), dimensions={1}
}

ENTRY entry_computation {
  p0 = s8[256]{0} parameter(0)
  fusion = s8[16777217,256]{1,0} fusion(p0), kind=kCustom,
    calls=computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["2","256"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
  ROOT slice = s8[1000,256]{1,0} slice(fusion), slice={[16776217:16777217], [0:256]}
})";

  constexpr absl::string_view kEmittersHloText = R"(
computation {
  p0 = s8[256]{0} parameter(0)
  ROOT broadcast = s8[16777217,256]{1,0} broadcast(p0), dimensions={1}
}

ENTRY entry_computation {
  p0 = s8[256]{0} parameter(0)
  fusion = s8[16777217,256]{1,0} fusion(p0), kind=kCustom,
    calls=computation
  ROOT slice = s8[1000,256]{1,0} slice(fusion), slice={[16776217:16777217], [0:256]}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> triton_module,
                          ParseAndReturnVerifiedModule(kTritonHloText));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> emitters_module,
                          ParseAndReturnVerifiedModule(kEmittersHloText));

  const Shape& triton_fusion_shape = triton_module->entry_computation()
                                         ->root_instruction()
                                         ->operand(0)
                                         ->shape();

  ASSERT_GT(Product(triton_fusion_shape.dimensions()), 1l << 32);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(emitters_module),
                                      std::move(triton_module), kExactMatch,
                                      /*run_hlo_passes=*/false));
}

TEST_F(TritonEmitterTest, ConvertF16ToF8E5M2Exhaustive) {
  // TODO(b/396595945): enable post-Ampere once Triton respects RTNE semantics
  // on H100.
  if (auto cc = GpuComputeCapability().cuda_compute_capability();
      cc && cc->IsAtLeastHopper()) {
    GTEST_SKIP() << "Skipping tests above Ampere, Triton's conversion isn't "
                    "always correct";
  }
  if (GpuComputeCapability().IsRocm()) {
    GTEST_SKIP() << "Triton's F16 to F8E5M2 conversion doesn't preserve "
                    "infinities on ROCm";
  }

  constexpr absl::string_view kHloTextTemplate = R"(
computation {
  p0 = f16[65536]{0} parameter(0)
  ROOT convert = f8e5m2[65536]{0} convert(p0)
}

ENTRY entry_computation {
  p0 = f16[65536]{0} constant({$0})
  ROOT fusion = f8e5m2[65536]{0} fusion(p0), kind=kCustom,
    calls=computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";

  std::vector<Eigen::half> all_f16_values;
  for (int i = 0; i < 65536; i++) {
    all_f16_values.push_back(
        Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(i)));
  }

  std::string hlo_text =
      absl::Substitute(kHloTextTemplate, absl::StrJoin(all_f16_values, ", "));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), kExactMatch));
}

TEST_F(TritonEmitterTest, ConvertS4ToS8Exhaustive) {
  constexpr absl::string_view kHloText = R"(
computation {
  p0 = s4[16]{0:E(4)} parameter(0)
  ROOT convert = s8[16]{0} convert(p0)
}

ENTRY entry_computation {
  p0 = s4[16]{0:E(4)} parameter(0)
  ROOT fusion = s8[16]{0} fusion(p0), kind=kCustom,
    calls=computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  auto values = {s4(-8), s4(-7), s4(-6), s4(-5), s4(-4), s4(-3), s4(-2), s4(-1),
                 s4(0),  s4(1),  s4(2),  s4(3),  s4(4),  s4(5),  s4(6),  s4(7)};
  Literal literal = LiteralUtil::CreateR1<s4>(values);
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), {&literal}, kExactMatch));
}

TEST_P(TmaParameterizedTritonEmitterTest, ConvertS4ToS8For2D) {
  constexpr absl::string_view kHloTextTemplate = R"(
computation {
  p0 = s4[64,64]{1,0:E(4)} parameter(0)
  ROOT convert = s8[64,64]{1,0} convert(p0)
}

ENTRY entry_computation {
  p0 = s4[64,64]{1,0:E(4)} parameter(0)
  ROOT fusion = s8[64,64]{1,0} fusion(p0), kind=kCustom,
    calls=computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32", "32"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1",
          "is_tma_allowed":"$0"}}}
})";

  const bool is_tma_allowed = GetParam();
  std::string hlo_text = absl::Substitute(kHloTextTemplate, is_tma_allowed);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, kExactMatch));
}

// Parameterized as a sanity check to make sure dots work with TMA.
TEST_P(TmaParameterizedTritonEmitterTest, DotIsEmittedCorrectly) {
  const std::string kHloTextTemplate = R"(
fdot {
  fdot.p0 = f32[32,123] parameter(0)
  fdot.p1 = f32[123,512] parameter(1)
  lhs.root = f32[32,123] negate(fdot.p0)
  frhs.root = f32[123,512] abs(fdot.p1)
  ROOT fdot.root = f32[32,512]{1,0} dot(lhs.root, frhs.root),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_f32_f32_f32, backend_config={sizes:[32]}
}

ENTRY entry {
  entry.p0 = f32[32,123] parameter(0)
  entry.p1 = f32[123,512] parameter(1)
  ROOT fusion = f32[32,512] fusion(entry.p0, entry.p1),
    kind=kCustom, calls=fdot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16", "64"]}],
          "num_warps":"1",
          "num_ctas":"1",
          "num_stages":"1",
          "is_tma_allowed":"$0"}}}
})";

  const bool is_tma_allowed = GetParam();
  std::string hlo_text = absl::Substitute(kHloTextTemplate, is_tma_allowed);

  TF_ASSERT_OK_AND_ASSIGN(auto xtile_module_and_hlo_module,
                          CreateXTileIrAndFileCheck(hlo_text, "fdot",
                                                    R"(
CHECK:      xtile.entry_func @xtile_dialect_fn(%[[ARG0:[A-Za-z0-9_]*]]: memref<32x123xf32>
CHECK-SAME:                             %[[ARG1:[A-Za-z0-9_]*]]: memref<123x512xf32>
CHECK-SAME:                             %[[ARG2:[A-Za-z0-9_]*]]: memref<32x512xf32>
CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
CHECK:      {{.*}} = scf.for %{{.*}} = %[[C0]] to %[[C4]] step %[[C1]]
CHECK-SAME: iter_args({{.*}}) -> (tensor<16x64xf32>) {
CHECK-DAG:  xtile.extract %[[ARG0]]
CHECK-DAG:  xtile.extract %[[ARG1]]
CHECK-DAG:  arith.negf {{.*}} : tensor<16x32xf32>
CHECK-DAG:  math.absf {{.*}} : tensor<32x64xf32>
CHECK:      stablehlo.dot_general {{.*}} (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
CHECK:      arith.addf {{.*}}
CHECK:      scf.yield {{.*}} : tensor<16x64xf32>
CHECK-COUNT-1: xtile.insert

          )"));

  TF_ASSERT_OK(LowerXTileIrToTritonAndFileCheck(
      xtile_module_and_hlo_module.first.get(), R"(
CHECK:      xtile.entry_func @xtile_dialect_fn(%[[ARG0:[A-Za-z0-9_]*]]: memref<32x123xf32>
CHECK-SAME:                             %[[ARG1:[A-Za-z0-9_]*]]: memref<123x512xf32>
CHECK-SAME:                             %[[ARG2:[A-Za-z0-9_]*]]: memref<32x512xf32>
CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
CHECK:      {{.*}} = scf.for %{{.*}} = %[[C0]] to %[[C4]] step %[[C1]]
CHECK-SAME: iter_args({{.*}}) -> (tensor<16x64xf32>) {
CHECK-DAG:  xtile.extract %[[ARG0]]
CHECK-DAG:  xtile.extract %[[ARG1]]
CHECK-DAG:  arith.negf {{.*}} : tensor<16x32xf32>
CHECK-DAG:  math.absf {{.*}} : tensor<32x64xf32>
CHECK:      tt.dot {{.*}} tensor<16x32xf32> * tensor<32x64xf32> -> tensor<16x64xf32>
CHECK:      scf.yield {{.*}} : tensor<16x64xf32>
CHECK-COUNT-1: xtile.insert
  )",
      GetFusionInstruction(*xtile_module_and_hlo_module.second, "fdot")));

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      hlo_text, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

TEST_F(WarpSpecializationTritonEmitterTest,
       DotAccumulationLoopUsesWarpSpecialization) {
  if (!GetCudaComputeCapability().IsAtLeastBlackwell()) {
    GTEST_SKIP() << "Currently only supported on Blackwell and newer.";
  }

  const std::string hlo_text = R"(
fdot {
  fdot.p0 = f16[256,256] parameter(0)
  fdot.p1 = f16[256,256] parameter(1)
  ROOT fdot.root = f16[256,256]{1,0} dot(fdot.p0, fdot.p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_f16_f16_f32, backend_config={sizes:[64]}
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

  // Check that the IR attribute is set correctly.
  TF_EXPECT_OK(CreateTritonIrFromHloTextAndFileCheck(hlo_text, "fdot", R"(
  // CHECK:       scf.for
  // CHECK:       scf.yield
  // CHECK-NEXT:  tt.warp_specialize
  // )"));

  // Make sure it runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      hlo_text, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_P(TmaParameterizedTritonEmitterTest, DotFromBroadcastIsEmittedCorrectly) {
  // TODO(b/393299275): add a deviceless test to run the whole pipeline as
  // other passes might change the module but we are starting from a fixed
  // state.
  const std::string kHloTextTemplate = R"(
HloModule module

triton_dot (p0: f32[256], p1: f32[128,32]) -> f32[256,32] {
  p0 = f32[256]{0} parameter(0)
  p1 = f32[128,32]{1,0} parameter(1)
  bcast = f32[256,128]{1,0} broadcast(p0), dimensions={0}
  ROOT result = f32[256,32]{1,0} dot(bcast, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_f32_f32_f32, backend_config={sizes:[16]}
}

ENTRY e (p0.1: f32[11,1,24,1], p1.1: f32[128,32]) -> f32[256,32] {
  p0.1 = f32[11,1,24,1]{3,2,1,0} parameter(0)
  bitcast = f32[256]{0} bitcast(p0.1)
  p1.1 = f32[128,32]{1,0} parameter(1)
  ROOT result.1 = f32[256,32]{1,0} fusion(bitcast, p1.1), kind=kCustom,
    calls=triton_dot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","16"]}],
          "num_warps":"1",
          "num_stages":"1",
          "num_ctas":"1",
          "is_tma_allowed":"$0"}}}
}
)";

  const bool is_tma_allowed = GetParam();
  const std::string hlo_text =
      absl::Substitute(kHloTextTemplate, is_tma_allowed);
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      hlo_text, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

// The template is parametrized by the type of the lhs/rhs, the type of the
// dot output, and the algorithm.
constexpr absl::string_view kHloForDotAlgorithmTestTemplate = R"(
dot {
  p0 = $0[512,512] parameter(0)
  p1 = $0[512,512] parameter(1)
  ROOT dot = $1[512,512] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=$2,
    backend_config={sizes:[32]}
}

ENTRY entry {
  p0 = $0[512,512] parameter(0)
  p1 = $0[512,512] parameter(1)
  ROOT fusion = $1[512,512] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config":{
        "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_warps":"1", "num_ctas":"1", "num_stages":"1"
    }}}
})";

std::string GetDotAlgorithmHlo(PrimitiveType in_ty, PrimitiveType out_ty,
                               PrecisionConfig::Algorithm algorithm) {
  constexpr absl::string_view kAlgorithmPrefix = "ALG_";
  std::string in_ty_str = primitive_util::LowercasePrimitiveTypeName(in_ty);
  std::string out_ty_str = primitive_util::LowercasePrimitiveTypeName(out_ty);
  std::string algorithm_str = PrecisionConfig::Algorithm_Name(algorithm).substr(
      kAlgorithmPrefix.size());
  return absl::Substitute(kHloForDotAlgorithmTestTemplate, in_ty_str,
                          out_ty_str, algorithm_str);
}

// TODO(b/407744579): narrow down the error specs for the various dot
// algorithms.
//
// The non-default values are either taken from the pre-existing
// `dot_algorithms_test` as of 2025-04-01, or approximated. It's not clear
// whether even the pre-existing values were derived to adhere precisely to the
// numerical expectations of the corresponding algorithms. We should narrow this
// down in the future.
ErrorSpec ErrorSpecForDotAlgorithm(PrecisionConfig::Algorithm algorithm) {
  // A default error spec, not particularly tuned to any algorithm.
  ErrorSpec default_error_spec{/*aabs=*/1e-4, /*arel=*/1e-6};
  switch (algorithm) {
    case PrecisionConfig::ALG_UNSET:
      // Give a loose tolerance to ALG_UNSET, as the expected behaviour is
      // not deducible from the algorithm name alone.
      return ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2};
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-3};
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      return default_error_spec;
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/2e-6, /*arel=*/2e-6};
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
      return default_error_spec;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      // Taken from `dot_algorithms_test`.
      return ErrorSpec{/*aabs=*/0, /*arel=*/6e-5};
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
      // Taken from `dot_algorithms_test`.
      return ErrorSpec{/*aabs=*/0, /*arel=*/7e-6};
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/2e-6, /*arel=*/2e-6};
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3};
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/2e-6, /*arel=*/3e-6};
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      // Computed to make the tests pass (and it seems reasonable on the face of
      // it), and not derived from first principles.
      return ErrorSpec{/*aabs=*/2e-6, /*arel=*/2e-6};
    case PrecisionConfig::ALG_DOT_BF16_BF16_BF16:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
      return kExactMatch;
    // Keep in order to make the switch exhaustive.
    case PrecisionConfig_Algorithm_PrecisionConfig_Algorithm_INT_MIN_SENTINEL_DO_NOT_USE_:  // NOLINT(whitespace/line_length)
    case PrecisionConfig_Algorithm_PrecisionConfig_Algorithm_INT_MAX_SENTINEL_DO_NOT_USE_:  // NOLINT(whitespace/line_length)
      LOG(FATAL) << "Unsupported algorithm: " << algorithm;
  }
}

class TritonEmitterTestWithAlgorithmParam
    : public TritonEmitterTest,
      public ::testing::WithParamInterface<PrecisionConfig::Algorithm> {};

// Regroups tests for dot algorithms that have no ambiguous type parameters as
// per `algorithm_util::GetAllowedOperandsTypeForAlgorithm` and
// `algorithm_util::GetDotAccumulatorType`, and do not decompose each tiled step
// into multiple `dot` operations. We call these algorithms "basic" algorithms
// here.
using BasicDotAlgorithmEmitterTest = TritonEmitterTestWithAlgorithmParam;

constexpr std::array kBasicAlgorithms = {
    PrecisionConfig::ALG_DOT_F16_F16_F16,
    PrecisionConfig::ALG_DOT_F32_F32_F32,
    PrecisionConfig::ALG_DOT_F64_F64_F64,
    PrecisionConfig::ALG_DOT_F16_F16_F32,
    PrecisionConfig::ALG_DOT_BF16_BF16_F32,
    PrecisionConfig::ALG_DOT_TF32_TF32_F32,
};

TEST_P(BasicDotAlgorithmEmitterTest, BasicAlgorithmIsEmittedCorrectly) {
  auto algorithm = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<PrimitiveType> allowed_types,
      algorithm_util::GetAllowedOperandsTypeForAlgorithm(algorithm));
  ASSERT_EQ(allowed_types.size(), 1);
  PrimitiveType in_ty = allowed_types.front();
  TF_ASSERT_OK_AND_ASSIGN(PrimitiveType out_ty,
                          algorithm_util::GetDotAccumulatorType(algorithm));
  const std::string kHloText = GetDotAlgorithmHlo(in_ty, out_ty, algorithm);

  TF_ASSERT_OK_AND_ASSIGN(
      auto xtile_module_and_hlo_module,
      CreateXTileIrAndFileCheck(
          kHloText, "dot",
          absl::Substitute(
              R"(
  CHECK:  stablehlo.dot_general{{.*}} : (tensor<16x32x$0>, tensor<32x64x$0>) -> tensor<16x64x$1>
  CHECK:  arith.addf
  )",
              primitive_util::LowercasePrimitiveTypeName(in_ty),
              primitive_util::LowercasePrimitiveTypeName(out_ty))));

  TF_ASSERT_OK(LowerXTileIrToTritonAndFileCheck(
      xtile_module_and_hlo_module.first.get(),
      absl::Substitute(R"(
  CHECK:  tt.dot{{.*}} : tensor<16x32x$0> * tensor<32x64x$0> -> tensor<16x64x$1>
  )",
                       primitive_util::LowercasePrimitiveTypeName(in_ty),
                       primitive_util::LowercasePrimitiveTypeName(out_ty)),
      GetFusionInstruction(*xtile_module_and_hlo_module.second, "dot")));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpecForDotAlgorithm(algorithm)));
}

std::string DotAlgorithmTestToString(
    const ::testing::TestParamInfo<PrecisionConfig::Algorithm>& data) {
  return PrecisionConfig::Algorithm_Name(data.param);
}

INSTANTIATE_TEST_SUITE_P(BasicDotAlgorithmEmitterTestSuite,
                         BasicDotAlgorithmEmitterTest,
                         ::testing::ValuesIn(kBasicAlgorithms),
                         DotAlgorithmTestToString);

// Regroups tests for dot algorithms that issue several dot instructions.
using MultiDotAlgorithmEmitterTest = TritonEmitterTestWithAlgorithmParam;

constexpr std::array kMultiDotAlgorithms = {
    PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3,
    PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6,
    PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3,
    // TODO(basioli): re-enable this algorithm testing once the attribute
    // importer supports the conversion.
    // PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9,
};

TEST_P(MultiDotAlgorithmEmitterTest, MultiDotAlgorithmIsEmittedCorrectly) {
  auto algorithm = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(PrimitiveType out_ty,
                          algorithm_util::GetDotAccumulatorType(algorithm));
  PrimitiveType in_ty =
      algorithm == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3 ? F32 : BF16;
  // Dummy value to ensure that the dot count is explicitly set.
  int dot_count_for_algorithm = 0x1337;
  int stablehlo_dot_count_for_algorithm = 0x1337;
  std::string input_precision_string = "";
  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
      dot_count_for_algorithm = 3;
      stablehlo_dot_count_for_algorithm = 3;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
      dot_count_for_algorithm = 6;
      stablehlo_dot_count_for_algorithm = 6;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      dot_count_for_algorithm = 9;
      stablehlo_dot_count_for_algorithm = 9;
      break;
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      // Triton implements TF32x3 as a specific precision mode.
      input_precision_string = "tf32x3";
      dot_count_for_algorithm = 1;
      stablehlo_dot_count_for_algorithm = 3;
      break;
    default:
      // Unreachable.
      ASSERT_TRUE(false);
  }

  const std::string kHloText = GetDotAlgorithmHlo(in_ty, out_ty, algorithm);

  TF_ASSERT_OK_AND_ASSIGN(
      auto xtile_module_and_hlo_module,
      CreateXTileIrAndFileCheck(kHloText, "dot",
                                absl::Substitute(
                                    R"(
  CHECK:  stablehlo.dot_general{{.*}} num_primitive_operations = $0, {{.*}}
  )",
                                    stablehlo_dot_count_for_algorithm)));

  TF_ASSERT_OK(LowerXTileIrToTritonAndFileCheck(
      xtile_module_and_hlo_module.first.get(),
      absl::Substitute(R"(
  CHECK-COUNT-$2:  tt.dot{{.*}}$3{{.*}} : tensor<16x32x$0> * tensor<32x64x$0> -> tensor<16x64x$1>
  )",
                       primitive_util::LowercasePrimitiveTypeName(in_ty),
                       primitive_util::LowercasePrimitiveTypeName(out_ty),
                       dot_count_for_algorithm, input_precision_string),
      GetFusionInstruction(*xtile_module_and_hlo_module.second, "dot")));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHloText, ErrorSpecForDotAlgorithm(algorithm)));
}

INSTANTIATE_TEST_SUITE_P(MultiDotAlgorithmEmitterTestSuite,
                         MultiDotAlgorithmEmitterTest,
                         ::testing::ValuesIn(kMultiDotAlgorithms),
                         DotAlgorithmTestToString);

// Regroups tests that use TF32 precision by definition.
using TF32DotAlgorithmEmitterTest = TritonEmitterTestWithAlgorithmParam;

constexpr std::array kTF32DotAlgorithms = {
    PrecisionConfig::ALG_DOT_TF32_TF32_F32,
    PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3};

TEST_P(TF32DotAlgorithmEmitterTest, TF32AlgorithmsUseTF32InputPrecision) {
  auto algorithm = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<PrimitiveType> allowed_types,
      algorithm_util::GetAllowedOperandsTypeForAlgorithm(algorithm));
  ASSERT_EQ(allowed_types.size(), 1);
  PrimitiveType in_ty = allowed_types.front();
  TF_ASSERT_OK_AND_ASSIGN(PrimitiveType out_ty,
                          algorithm_util::GetDotAccumulatorType(algorithm));
  const std::string kHloText = GetDotAlgorithmHlo(in_ty, out_ty, algorithm);

  std::string input_precision_string =
      algorithm == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3 ? "tf32x3"
                                                             : "tf32";

  std::string num_primitive_operations_string =
      algorithm == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3 ? "3" : "1";

  // TODO(basioli): maybe algorithm string?
  TF_ASSERT_OK_AND_ASSIGN(
      auto xtile_module_and_hlo_module,
      CreateXTileIrAndFileCheck(
          kHloText, "dot",
          absl::Substitute(
              R"(
  CHECK:  stablehlo.dot_general{{.*}}, contracting_dims = [1] x [0], {{.*}} algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = $2, allow_imprecise_accumulation = false> : (tensor<16x32x$0>, tensor<32x64x$0>) -> tensor<16x64x$1>
  )",
              primitive_util::LowercasePrimitiveTypeName(in_ty),
              primitive_util::LowercasePrimitiveTypeName(out_ty),
              num_primitive_operations_string)));

  TF_ASSERT_OK(LowerXTileIrToTritonAndFileCheck(
      xtile_module_and_hlo_module.first.get(),
      absl::Substitute(R"(
  CHECK:  tt.dot{{.*}} inputPrecision = $2 : tensor<16x32x$0> * tensor<32x64x$0> -> tensor<16x64x$1>
  )",
                       primitive_util::LowercasePrimitiveTypeName(in_ty),
                       primitive_util::LowercasePrimitiveTypeName(out_ty),
                       input_precision_string),
      GetFusionInstruction(*xtile_module_and_hlo_module.second, "dot")));

  // No need to `RunAndCompare` here, these algorithms are already covered by
  // other tests.
}

INSTANTIATE_TEST_SUITE_P(TF32DotAlgorithmEmitterTestSuite,
                         TF32DotAlgorithmEmitterTest,
                         ::testing::ValuesIn(kTF32DotAlgorithms),
                         DotAlgorithmTestToString);

class DotUnsetAlgorithmEmitterTest
    : public TritonEmitterTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType>> {
 public:
  static std::string ParamToString(
      const ::testing::TestParamInfo<DotUnsetAlgorithmEmitterTest::ParamType>&
          data) {
    auto [result_type, input_type] = data.param;
    return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(result_type),
                        "_",
                        primitive_util::LowercasePrimitiveTypeName(input_type));
  };
};

TEST_P(DotUnsetAlgorithmEmitterTest, UnsetAlgorithmIsEmittedCorrectly) {
  auto [result_type, input_type] = GetParam();
  const std::string kHloText =
      GetDotAlgorithmHlo(input_type, result_type, PrecisionConfig::ALG_UNSET);
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  if (!IsTritonSupportedComputation(*module->entry_computation(),
                                    GpuComputeCapability())) {
    GTEST_SKIP() << "Not supported on this platform.";
  }

  ErrorSpec error_spec = ErrorSpecForDotAlgorithm(PrecisionConfig::ALG_UNSET);
  // For 8-bit floating point types, we need to allow large errors.
  if (primitive_util::IsFloatingPointType(result_type) &&
      primitive_util::BitWidth(result_type) == 8) {
    error_spec = ErrorSpec{/*aabs=*/1e0, /*arel=*/1e-1};
  }
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, error_spec));
}

INSTANTIATE_TEST_SUITE_P(
    DotUnsetAlgorithmEmitterTestSuite, DotUnsetAlgorithmEmitterTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllXlaDataTypes())),
    DotUnsetAlgorithmEmitterTest::ParamToString);

TEST_F(TritonEmitterTest, ScaledDotIsSupportedByReferencePlatform) {
  constexpr absl::string_view kHloText = R"(
    HloModule ScaledDotIsSupportedByReferencePlatform

    ENTRY entry {
     lhs = bf16[16,128] parameter(0)
     rhs = bf16[128,16] parameter(1)
     lhs_scale = bf16[1,4] parameter(2)
     rhs_scale = bf16[4,1] parameter(3)
     ROOT dot = bf16[16,16] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
         lhs_contracting_dims={1},
         rhs_contracting_dims={0}
    }
  )";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonEmitterTest, RocmWarpSizeIsSetCorrectly) {
  if (GpuComputeCapability().IsCuda()) {
    GTEST_SKIP() << "Warp size is always 32 on CUDA";
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> verified_module,
                          ParseAndReturnVerifiedModule(GetDotAlgorithmHlo(
                              F16, F16, PrecisionConfig::ALG_UNSET)));

  std::string output_directory;
  if (!tsl::io::GetTestUndeclaredOutputsDir(&output_directory)) {
    output_directory = tsl::testing::TmpDir();
  }
  DebugOptions debug_options = verified_module->config().debug_options();
  debug_options.set_xla_dump_to(output_directory);
  debug_options.set_xla_dump_emitter_re("triton-to-llvm");
  verified_module->mutable_config().set_debug_options(debug_options);

  const HloFusionInstruction* triton_fusion = Cast<HloFusionInstruction>(
      verified_module->entry_computation()->root_instruction());

  llvm::LLVMContext llvm_ctx;
  mlir::MLIRContext mlir_context;
  llvm::Triple target_triple(nvptx::TargetTriple());
  std::string data_layout(nvptx::DataLayout());
  std::vector<std::string> paths;
  std::string triton_passes_log;

  // https://github.com/openxla/xla/commit/e00d5aa8029d228b148bf0ac463bdc5d1b70ea16
  // adds bounds checks in Triton fusion emitter.
  // Consequently valid, non-empty, tile parameters/sizes must be provided.
  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 64}};
  block_level_parameters.num_warps = 1;

  // For MI210 warp_size should be 64
  se::DeviceDescription dev_info = TestGpuDeviceInfo::AMDMI210DeviceInfo();
  // Now, we pass valud tiles, we also need to set non-zero
  // `shared_memory_per_block_optin` to pass this check
  // https://github.com/openxla/xla/blob/c8b710f1b70f890c9ee4b8756bc53f3a599a0ed5/xla/backends/gpu/codegen/triton/fusion_emitter.cc#L1863-L1867
  dev_info.set_shared_memory_per_block_optin(64 * 1024);
  TF_ASSERT_OK(TritonWrapper(
      "test_fn", *triton_fusion,
      se::GpuComputeCapability{se::RocmComputeCapability("gfx942")}, dev_info,
      block_level_parameters, target_triple, data_layout, llvm_ctx,
      mlir_context));
  TF_EXPECT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(output_directory, "*.triton-to-llvm.txt"), &paths));
  EXPECT_EQ(paths.size(), 1);
  TF_ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), paths[0], &triton_passes_log));
  constexpr absl::string_view kPattern = R"(
      // CHECK: "ttg.threads-per-warp" = 64
    )";
  EXPECT_THAT(RunFileCheck(triton_passes_log, kPattern), true);

  // For RX7900 warp_size should be 32
  se::DeviceDescription dev_info_n = TestGpuDeviceInfo::AMDRX7900DeviceInfo();
  // Now, we pass valud tiles, we also need to set non-zero
  // `shared_memory_per_block_optin` to pass this check
  // https://github.com/openxla/xla/blob/c8b710f1b70f890c9ee4b8756bc53f3a599a0ed5/xla/backends/gpu/codegen/triton/fusion_emitter.cc#L1863-L1867
  dev_info_n.set_shared_memory_per_block_optin(64 * 1024);
  TF_ASSERT_OK(TritonWrapper(
      "test_fn", *triton_fusion,
      se::GpuComputeCapability{se::RocmComputeCapability("gfx1100")},
      dev_info_n, block_level_parameters, target_triple, data_layout, llvm_ctx,
      mlir_context));
  TF_EXPECT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(output_directory, "*.triton-to-llvm.txt"), &paths));
  EXPECT_EQ(paths.size(), 1);
  TF_ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), paths[0], &triton_passes_log));
  constexpr absl::string_view kPattern_n = R"(
      // CHECK: "ttg.threads-per-warp" = 32
    )";
  EXPECT_THAT(RunFileCheck(triton_passes_log, kPattern_n), true);
}

TEST_F(TritonEmitterTest, RocmWavesPerEuAttributeIsSet) {
  if (GpuComputeCapability().IsCuda()) {
    GTEST_SKIP() << "waves_per_eu is ROCm-specific";
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> verified_module,
                          ParseAndReturnVerifiedModule(GetDotAlgorithmHlo(
                              F16, F16, PrecisionConfig::ALG_UNSET)));

  const HloFusionInstruction* triton_fusion = Cast<HloFusionInstruction>(
      verified_module->entry_computation()->root_instruction());

  llvm::LLVMContext llvm_ctx;
  mlir::MLIRContext mlir_context;
  llvm::Triple target_triple(amdgpu::TargetTriple());
  std::string data_layout(amdgpu::DataLayout());

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 64}};
  block_level_parameters.num_warps = 1;
  block_level_parameters.waves_per_eu = 4;

  se::DeviceDescription dev_info = TestGpuDeviceInfo::AMDMI210DeviceInfo();

  TF_ASSERT_OK_AND_ASSIGN(
      TritonWrapperResult result,
      TritonWrapper(
          "test_fn", *triton_fusion,
          se::GpuComputeCapability{se::RocmComputeCapability("gfx90a")},
          dev_info, block_level_parameters, target_triple, data_layout,
          llvm_ctx, mlir_context));

  ASSERT_NE(result.llvm_module, nullptr);
  auto* fn = result.llvm_module->getFunction("test_fn");
  ASSERT_NE(fn, nullptr)
      << "Kernel function 'test_fn' not found in LLVM module";
  auto attr = fn->getFnAttribute("amdgpu-waves-per-eu");
  ASSERT_TRUE(attr.isStringAttribute());
  EXPECT_EQ(attr.getValueAsString().str(), "4, 4");
}

TEST_F(TritonEmitterTest, RocmWavesPerEuZeroOmitsAttribute) {
  if (GpuComputeCapability().IsCuda()) {
    GTEST_SKIP() << "waves_per_eu is ROCm-specific";
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> verified_module,
                          ParseAndReturnVerifiedModule(GetDotAlgorithmHlo(
                              F16, F16, PrecisionConfig::ALG_UNSET)));

  const HloFusionInstruction* triton_fusion = Cast<HloFusionInstruction>(
      verified_module->entry_computation()->root_instruction());

  llvm::LLVMContext llvm_ctx;
  mlir::MLIRContext mlir_context;
  llvm::Triple target_triple(amdgpu::TargetTriple());
  std::string data_layout(amdgpu::DataLayout());

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 64}};
  block_level_parameters.num_warps = 1;
  block_level_parameters.waves_per_eu = 0;

  se::DeviceDescription dev_info = TestGpuDeviceInfo::AMDMI210DeviceInfo();

  TF_ASSERT_OK_AND_ASSIGN(
      TritonWrapperResult result,
      TritonWrapper(
          "test_fn", *triton_fusion,
          se::GpuComputeCapability{se::RocmComputeCapability("gfx90a")},
          dev_info, block_level_parameters, target_triple, data_layout,
          llvm_ctx, mlir_context));

  ASSERT_NE(result.llvm_module, nullptr);
  auto* fn = result.llvm_module->getFunction("test_fn");
  ASSERT_NE(fn, nullptr)
      << "Kernel function 'test_fn' not found in LLVM module";
  EXPECT_FALSE(fn->hasFnAttribute("amdgpu-waves-per-eu"))
      << "waves_per_eu=0 should not set amdgpu-waves-per-eu attribute";
}

struct ScaleDotTestParams {
  std::string lhs_type;
  std::string rhs_type;
  std::string lhs_scale_type;
  std::string rhs_scale_type;
  std::string output_type;
  std::string expected_triton_type;

  std::string PrepareHloText(absl::string_view hlo_template) const {
    return absl::StrReplaceAll(hlo_template,
                               {{"$lhs_type", lhs_type},
                                {"$rhs_type", rhs_type},
                                {"$lhs_scale_type", lhs_scale_type},
                                {"$rhs_scale_type", rhs_scale_type},
                                {"$output_type", output_type}});
  }
  static std::string ToString(
      const ::testing::TestParamInfo<ScaleDotTestParams>& info) {
    const ScaleDotTestParams& params = info.param;
    auto name = absl::StrCat(params.lhs_type, "_", params.rhs_type, "_",
                             params.lhs_scale_type, "_", params.rhs_scale_type,
                             "_", params.output_type);
    absl::StrReplaceAll({{"[", "_"}, {"]", "_"}, {",", "x"}}, &name);
    return name;
  }
};

std::ostream& operator<<(std::ostream& stream, const ScaleDotTestParams& tc) {
  return stream << "{\n\tlhs_type:" << tc.lhs_type
                << ",\n\trhs_type:" << tc.rhs_type
                << ",\n\tlhs_scale_type:" << tc.lhs_scale_type
                << ",\n\trhs_scale_type:" << tc.rhs_scale_type
                << ",\n\toutput_type:" << tc.output_type << "\n}";
}

class TritonScaledDotGemmTest
    : public TritonEmitterTest,
      public ::testing::WithParamInterface<ScaleDotTestParams> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = TritonEmitterTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(true);
    debug_options.set_xla_gpu_autotune_level(0);
    debug_options.set_xla_gpu_cublas_fallback(false);
    return debug_options;
  }
};

TEST_P(TritonScaledDotGemmTest,
       FP8ScaledDotCompilesToPtxIntrinsicsWhenAvailable) {
  const ScaleDotTestParams& params = GetParam();
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule m

triton_dot {
  lhs = $lhs_type parameter(0)
  rhs = $rhs_type parameter(1)
  lhs_scale = $lhs_scale_type parameter(2)
  rhs_scale = $rhs_scale_type parameter(3)
  ROOT _ = $output_type{1,0} scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    backend_config={sizes:[128]}
}

ENTRY e {
  lhs = $lhs_type{1,0} parameter(0)
  rhs = $rhs_type{1,0} parameter(1)
  lhs_scale = $lhs_scale_type{1,0} parameter(2)
  rhs_scale = $rhs_scale_type{1,0} parameter(3)
  ROOT _ = $output_type{1,0} fusion(lhs, rhs, lhs_scale, rhs_scale),
    kind=kCustom,
    calls=triton_dot,
    backend_config={
      "fusion_backend_config": {
        kind: "__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128", "256"]}],
          "num_warps":"4",
          "num_stages":"1",
          "num_ctas":"1"
        }
      }
    }
}
)hlo";

  auto hlo_text = params.PrepareHloText(kHloTextTemplate);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  constexpr absl::string_view kExpectedTritonIrTmpl = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x128x$triton_type>, tensor<128x4xi8>
      CHECK: tensor<128x256x$triton_type>, tensor<256x4xi8>
      CHECK: -> tensor<128x256xf32>
  )";
  auto expected_triton_ir = absl::StrReplaceAll(
      kExpectedTritonIrTmpl, {{"$triton_type", params.expected_triton_type}});
  EXPECT_THAT(
      CreateTritonIrAndFileCheckForDot(
          *module->GetComputationWithName("triton_dot"), expected_triton_ir),
      absl_testing::IsOk());
  if (GetCudaComputeCapability().IsAtLeastBlackwell()) {
    CompileAndOptionallyVerifyPtx(
        std::move(module), R"(CHECK: mxf8f6f4.block_scale.scale_vec::1X)");
  }
}

TEST_P(TritonScaledDotGemmTest, FP8ScaledDotGetsFusedAndExecutesCorrectly) {
  const ScaleDotTestParams& params = GetParam();
  if (!GetCudaComputeCapability().IsAtLeastBlackwell()) {
    GTEST_SKIP() << "Skipping test for pre-Blackwell GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule FP8ScaledDotGetsFusedAndExecutesCorrectly

ENTRY e {
  lhs = $lhs_type parameter(0)
  rhs = $rhs_type parameter(2)
  lhs_scale = $lhs_scale_type parameter(1)
  rhs_scale = $rhs_scale_type parameter(3)
  ROOT _ = $output_type{1,0} scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0}
}
)hlo";

  auto hlo_text = params.PrepareHloText(kHloTextTemplate);

  TF_ASSERT_OK_AND_ASSIGN(auto optimized_module, GetOptimizedModule(hlo_text));
  EXPECT_TRUE(*RunFileCheck(optimized_module->ToString(), R"(
    CHECK: fusion
    CHECK: ROOT {{.*}} scaled-dot
    CHECK: ENTRY
    CHECK: __triton_nested_gemm_fusion
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

INSTANTIATE_TEST_SUITE_P(
    TritonScaledDotGemmTest, TritonScaledDotGemmTest,
    ::testing::Values(ScaleDotTestParams{"f8e4m3fn[128,128]",
                                         "f8e4m3fn[128,256]",
                                         "f8e8m0fnu[128,4]", "f8e8m0fnu[4,256]",
                                         "bf16[128,256]", "f8E4M3FN"},
                      ScaleDotTestParams{"f8e5m2[128,128]", "f8e5m2[128,256]",
                                         "f8e8m0fnu[128,4]", "f8e8m0fnu[4,256]",
                                         "bf16[128,256]", "f8E5M2"}),
    ScaleDotTestParams::ToString);

class TritonScaledDotTest : public TritonEmitterTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = TritonEmitterTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(true);
    debug_options.set_xla_gpu_autotune_level(0);
    debug_options.set_xla_gpu_cublas_fallback(false);
    return debug_options;
  }

  HloComputation* GetFirstComputationWithInstruction(const HloModule& module,
                                                     HloOpcode opcode) const {
    for (const auto& computation : module.computations()) {
      for (const auto& instruction : computation->instructions()) {
        if (instruction->opcode() == opcode) {
          return computation;
        }
      }
    }
    return nullptr;
  }
};

TEST_F(TritonScaledDotTest,
       ScaledDotWithOmmittedLhsScaleGetFusedAndExecutedCorrectly) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP() << "Scaled dot isn't supported by Triton for pre-Hopper GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule ScaledDotWithOmmittedLhsScaleGetFusedAndExecutedCorrectly

ENTRY e {
  lhs = bf16[3,128,128] parameter(0)
  rhs = f8e4m3fn[3,128,128] parameter(1)
  constant = bf16[1,1,1] constant(1.0)
  rhs_scale = f8e8m0fnu[3,128,4] parameter(2)
  ROOT _ = bf16[3,128,128] scaled-dot(lhs, rhs, constant, rhs_scale),
    lhs_batch_dims={0},
    rhs_batch_dims={0},
    lhs_contracting_dims={2},
    rhs_contracting_dims={2}
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(auto optimized_module,
                          GetOptimizedModule(kHloTextTemplate));
  constexpr absl::string_view kExpectedOptimizedHLO = R"(
    CHECK: fusion
    CHECK: ROOT {{.*}} scaled-dot
    CHECK: ENTRY
    CHECK: __triton_nested_gemm_fusion
  )";
  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), kExpectedOptimizedHLO),
              true);
  for (const auto& computation : optimized_module->computations()) {
    for (const auto& instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kScaledDot) {
        LOG(INFO) << "Instruction: " << instruction->name();
      }
    }
  }

  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x128xbf16>
      CHECK: tensor<128x16xf8E4M3FN>, tensor<16x4xi8>
      CHECK: -> tensor<128x16xf32>
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              absl_testing::IsOk());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonScaledDotTest, ScaledDotWithBatchGetFusedAndExecutedCorrectly) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP() << "Scaled dot isn't supported by Triton for pre-Hopper GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule ScaledDotWithBatchGetFusedAndExecutedCorrectly

ENTRY e {
  lhs = f8e4m3fn[3,128,128] parameter(0)
  rhs = f8e4m3fn[3,128,128] parameter(1)
  lhs_scale = f8e8m0fnu[3,128,4] parameter(2)
  rhs_scale = f8e8m0fnu[3,128,4 ] parameter(3)
  ROOT _ = bf16[3,128,128] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_batch_dims={0},
    rhs_batch_dims={0},
    lhs_contracting_dims={2},
    rhs_contracting_dims={2}
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(auto optimized_module,
                          GetOptimizedModule(kHloTextTemplate));
  constexpr absl::string_view kExpectedOptimizedHLO = R"(
    CHECK: fusion
    CHECK: ROOT {{.*}} scaled-dot
    CHECK: ENTRY
    CHECK: __triton_nested_gemm_fusion
  )";
  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), kExpectedOptimizedHLO),
              true);

  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x128xf8E4M3FN>, tensor<128x4xi8>
      CHECK: tensor<128x16xf8E4M3FN>, tensor<16x4xi8>
      CHECK: -> tensor<128x16xf32>
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              absl_testing::IsOk());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonScaledDotTest, BroadcastAndReshapeGetFused) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP() << "Scaled dot isn't supported by Triton for pre-Hopper GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule ScaledDotWithBatchGetFusedAndExecutedCorrectly

ENTRY e {
  lhs = f8e4m3fn[3,128,128] parameter(0)
  rhs = f8e4m3fn[3,128,128] parameter(1)
  lhs_scale = f8e8m0fnu[3,128,1] parameter(2)
  lhs_scale_broadcasted = f8e8m0fnu[3,128,1,4] broadcast(lhs_scale),
      dimensions={0,1,2}
  lhs_scale_reshaped = f8e8m0fnu[3,128,4] reshape(lhs_scale_broadcasted)
  rhs_scale = f8e8m0fnu[3,128,1] parameter(3)
  rhs_scale_broadcasted = f8e8m0fnu[3,128,1,4] broadcast(rhs_scale),
      dimensions={0,1,2}
  rhs_scale_reshaped = f8e8m0fnu[3,128,4] reshape(rhs_scale_broadcasted)
  ROOT _ = bf16[3,128,128] scaled-dot(
      lhs,
      rhs,
      lhs_scale_reshaped,
      rhs_scale_reshaped),
    lhs_batch_dims={0},
    rhs_batch_dims={0},
    lhs_contracting_dims={2},
    rhs_contracting_dims={2}
}
  )hlo";

  TF_ASSERT_OK_AND_ASSIGN(auto optimized_module,
                          GetOptimizedModule(kHloTextTemplate));
  constexpr absl::string_view kExpectedOptimizedHLO = R"(
    CHECK: %fusion
    CHECK: %{{.*}} = f8e8m0fnu[3,128,4]{2,1,0} broadcast(%{{.*}}), dimensions={0,1}
    CHECK: %{{.*}} = f8e8m0fnu[3,128,4]{2,1,0} broadcast(%{{.*}}), dimensions={0,1}
    CHECK: ROOT {{.*}} scaled-dot
    CHECK: ENTRY
    CHECK: __triton_nested_gemm_fusion
  )";
  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), kExpectedOptimizedHLO),
              true);

  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x128xf8E4M3FN>, tensor<128x4xi8>
      CHECK: tensor<128x16xf8E4M3FN>, tensor<16x4xi8>
      CHECK: -> tensor<128x16xf32>
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              absl_testing::IsOk());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonScaledDotTest, Fp4Succeeds) {
  if (!GetCudaComputeCapability().IsAtLeastBlackwell()) {
    GTEST_SKIP() << "Scaled dot with FP4 isn't supported by Triton for "
                    "pre-Blackwell GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
    HloModule jit_scaled_dot_fn

    ENTRY %main.2 {
      %lhs = f4e2m1fn[1,1024,256]{2,1,0} parameter(0)
      %rhs = f4e2m1fn[1,256,256]{2,1,0} parameter(1)
      %lhs_scale = f8e8m0fnu[1,1024,8]{2,1,0} parameter(2)
      %rhs_scale = f8e8m0fnu[1,8,256]{2,1,0} parameter(3)
      ROOT %scaled-dot = bf16[1,1024,256]{2,1,0} scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
          lhs_batch_dims={0},
          lhs_contracting_dims={2},
          rhs_batch_dims={0},
          rhs_contracting_dims={1}
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       GetOptimizedModule(kHloTextTemplate));
  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x64xi8>, tensor<128x4xi8>
      CHECK: tensor<128x16xi8>, tensor<32x4xi8>
      CHECK: -> tensor<128x32xf32>
  )";

  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              absl_testing::IsOk());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonScaledDotTest, GlobalScalerSucceeds) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP() << "Scaled dot isn't supported by Triton for pre-Hopper GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule ScaledDotWithGlobalScaler

ENTRY e {
  lhs = f8e4m3fn[3,128,128] parameter(0)
  rhs = f8e4m3fn[3,128,128] parameter(1)
  lhs_scale = f8e8m0fnu[3,128,4] parameter(2)
  rhs_scale = f8e8m0fnu[3,128,4] parameter(3)
  scaled_dot = bf16[3,128,128] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_batch_dims={0},
    rhs_batch_dims={0},
    lhs_contracting_dims={2},
    rhs_contracting_dims={2}
  global_scaler = bf16[] constant(1.42)
  global_scaler_broadcasted = bf16[3,128,128] broadcast(global_scaler),
      dimensions={}
  ROOT _ = bf16[3,128,128] multiply(scaled_dot, global_scaler_broadcasted)
}
  )hlo";

  TF_ASSERT_OK_AND_ASSIGN(auto optimized_module,
                          GetOptimizedModule(kHloTextTemplate));
  constexpr absl::string_view kExpectedOptimizedHLO = R"(
    CHECK: %[[fusion_name:.*]] (parameter
    CHECK: %[[scaled_dot:.*]] = bf16[3,128,128]{2,1,0} scaled-dot
    CHECK: %[[global_scaler:.*]] = bf16[3,128,128]{2,1,0} broadcast
    CHECK: ROOT %{{.*}} = bf16[3,128,128]{2,1,0} multiply(%[[scaled_dot]], %[[global_scaler]])
    CHECK: ENTRY
    CHECK: ROOT {{.*}} fusion({{.*}}), kind=kCustom, calls=%[[fusion_name]]
  )";
  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), kExpectedOptimizedHLO),
              true);

  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x128xf8E4M3FN>, tensor<128x4xi8>
      CHECK: tensor<128x16xf8E4M3FN>, tensor<16x4xi8>
      CHECK: -> tensor<128x16xf32>
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              absl_testing::IsOk());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
