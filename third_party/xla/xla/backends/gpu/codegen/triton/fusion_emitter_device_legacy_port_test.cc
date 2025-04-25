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

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/test_utils.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/gpu/transforms/nest_gemm_fusion.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;
using tsl::testing::StatusIs;

struct ModuleAndNestedFusionMetadata {
  std::unique_ptr<VerifiedHloModule> module;
  HloComputation* computation;
  BlockLevelParameters block_level_parameters;
};

// Returns the "real" root instruction of a computation, which is either the
// root instruction itself, or the first instruction feeding into the root that
// is not a bitcast.
HloInstruction* GetNonBitcastRoot(const HloComputation* computation) {
  HloInstruction* root = computation->root_instruction();
  while (root->opcode() == HloOpcode::kBitcast) {
    root = root->mutable_operand(0);
  }
  return root;
}

class TritonTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options
        .set_xla_gpu_unsupported_enable_generic_triton_emitter_for_gemms(true);
    // Disable autotuning by default, re-enable it on a per-test basis in order
    // to avoid unnecessary slowness.
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }

  stream_executor::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  const stream_executor::GpuComputeCapability& GpuComputeCapability() {
    return device_desc().gpu_compute_capability();
  }

  stream_executor::GpuComputeCapability CudaAmpereOrRocm() {
    if (std::holds_alternative<stream_executor::RocmComputeCapability>(
            GpuComputeCapability())) {
      return stream_executor::GpuComputeCapability{
          device_desc().rocm_compute_capability()};
    }
    return se::CudaComputeCapability::Ampere();
  }

  // Returns the module, its fusion computation and associated block level
  // parameters from an HLO module text whose entry computation contains a
  // single GEMM fusion.
  absl::StatusOr<ModuleAndNestedFusionMetadata>
  GetModuleAndNestedFusionMetadata(absl::string_view hlo_text) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                        ParseAndReturnVerifiedModule(hlo_text));
    TF_ASSIGN_OR_RETURN(
        bool fusion_was_nested,
        NestGemmFusion(GpuComputeCapability()).Run(module.get()));
    if (!fusion_was_nested) {
      return absl::InternalError("Failed to nest the GEMM fusion.");
    }
    HloFusionInstruction* fusion =
        Cast<HloFusionInstruction>(hlo_query::GetFirstInstructionWithOpcode(
            *module->entry_computation(), HloOpcode::kFusion));
    HloComputation* computation = fusion->fused_instructions_computation();
    BlockLevelParameters block_level_parameters =
        BlockLevelParameters::FromBlockLevelFusionConfig(
            fusion->backend_config<GpuBackendConfig>()
                ->fusion_backend_config()
                .block_level_fusion_config());
    return ModuleAndNestedFusionMetadata{std::move(module), computation,
                                         std::move(block_level_parameters)};
  }

 protected:
  const stream_executor::DeviceDescription& device_desc() {
    return backend().default_stream_executor()->GetDeviceDescription();
  }
};

class TritonGemmTest : public TritonTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = TritonTest::GetDebugOptionsForTest();
    // Disable autotuning by default, re-enable it on a per-test basis in order
    // to avoid unnecessary slowness.
    debug_options.set_xla_gpu_autotune_level(0);
    // Do not fall back to cuBLAS and disable cuDNN; we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);
    debug_options.set_xla_gpu_cudnn_gemm_fusion_level(0);
    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    // Always rewrite Gemms with Triton regardless of size.
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
    return debug_options;
  }

  void MatchHloModule(HloModule& module, absl::string_view pattern) {
    TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                            RunFileCheck(module.ToString(), pattern));
    EXPECT_TRUE(filecheck_result);
  }
};

class TritonGemmTestWithSplitK : public TritonGemmTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = TritonGemmTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_split_k_autotuning(true);
    return debug_options;
  }
};

TEST_F(TritonTest, TestGemmWithTrivialNonContractingDimension) {
  constexpr absl::string_view kHloText = R"(
HloModule t, is_scheduled=true

triton_dot {
  p0 = f32[137,115]{1,0} parameter(0)
  p1 = f32[1,115]{1,0} parameter(1)
  ROOT dot = f32[137,1]{1,0} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f32[137,115]{1,0} parameter(0)
  p1 = f32[1,115]{1,0} parameter(1)
  ROOT custom-call = f32[137,1]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":2,
                         "num_ctas":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(*module_and_metadata.computation,
                                 module_and_metadata.block_level_parameters,
                                 R"(
CHECK: tt.dot {{.*}} : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
)"));
}

TEST_F(TritonTest, PredParametersAreTruncatedToI1) {
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_gemm_computation {
  p = pred[2,2]{1,0} parameter(0)
  a = f32[2,2]{1,0} parameter(1)
  b = f32[2,2]{1,0} parameter(2)
  c = f32[2,2]{1,0} parameter(3)
  compare = pred[2,2]{1,0} compare(a, b), direction=LT
  and = pred[2,2]{1,0} and(p, compare)
  convert = f32[2,2]{1,0} convert(and)
  ROOT r = f32[2,2]{1,0} dot(convert, c),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p = pred[2,2]{1,0} parameter(0)
  a = f32[2,2]{1,0} parameter(1)
  b = f32[2,2]{1,0} parameter(2)
  c = f32[2,2]{1,0} parameter(3)
  ROOT triton_gemm = f32[2,2]{1,0} fusion(p, a, b, c), kind=kCustom,
    calls=triton_gemm_computation,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
        triton_gemm_config: {
          "block_m":16,"block_n":16,"block_k":16,
          "split_k":1,"num_stages":1,"num_warps":1,
          "num_ctas":1
        }
      }
    }
})";
  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(*module_and_metadata.computation,
                                 module_and_metadata.block_level_parameters,
                                 R"(
CHECK: %[[LOAD:.*]] = triton_xla.extract {{.*}} : tensor<2x2xi8> to tensor<16x16xi8>
CHECK: %[[TRUNCI:.*]] = arith.trunci %[[LOAD]] : tensor<16x16xi8> to tensor<16x16xi1>
CHECK: %{{.*}} = arith.andi %[[TRUNCI]], %{{.*}} : tensor<16x16xi1>
)"));
}

TEST_F(TritonTest, CodegenBatchedDotWithConcatenationWithCorrectBatchStride) {
  constexpr absl::string_view kHloText = R"(
HloModule t

triton_gemm {
  parameter_0 = f32[2,3,10]{2,1,0} parameter(0)
  parameter_1 = f32[2,10,128]{2,1,0} parameter(1)
  parameter_2 = f32[2,10,256]{2,1,0} parameter(2)
  concatenate = f32[2,10,384]{2,1,0} concatenate(parameter_1, parameter_2), dimensions={2}
  ROOT dot = f32[2,3,384]{2,1,0} dot(parameter_0, concatenate),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  parameter_0 = f32[2,3,10]{2,1,0} parameter(0)
  parameter_1 = f32[2,10,128]{2,1,0} parameter(1)
  parameter_2 = f32[2,10,256]{2,1,0} parameter(2)
  ROOT dot = f32[2,3,384]{2,1,0} fusion(parameter_0, parameter_1, parameter_2),
    kind=kCustom, calls=triton_gemm,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":64,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":2,
                         "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));
  TF_EXPECT_OK(
      CreateTritonIrAndFileCheck(*module_and_metadata.computation,
                                 module_and_metadata.block_level_parameters, R"(
CHECK: scf.if {{.*}} -> (tensor<1x32x64xf32>)
CHECK: tt.dot {{.*}} : tensor<16x32xf32> * tensor<32x64xf32> -> tensor<16x64xf32>
)"));
}

// TODO(b/393299275): this requires adding support for dynamic-slice in the
// generic Triton emitter.
TEST_F(TritonTest, DISABLED_CodegenDynamicSliceWithCorrectOffsets) {
  // The start index(es) for the non-majormost dimension(s) are constant zero(s)
  // because we don't support dynamic slice on those dimensions.
  constexpr absl::string_view kHloText = R"(
HloModule t

triton_gemm {
  dot_lhs = f32[2,4] parameter(0)
  dynamic_slice_input = f32[4,5,2] parameter(1)
  start_index0 = s32[] parameter(2)
  start_index1 = s32[] parameter(3)
  start_index2 = s32[] parameter(4)
  dynamic_slice = f32[1,5,2] dynamic-slice(dynamic_slice_input, start_index0, start_index1, start_index2), dynamic_slice_sizes={1,5,2}
  bitcast = f32[5,2] bitcast(dynamic_slice)
  ROOT dot = f32[4,5] dot(dot_lhs, bitcast), lhs_contracting_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  dot_lhs = f32[2,4] parameter(0)
  dynamic_slice_input = f32[4,5,2] parameter(1)
  start_index0 = s32[] parameter(2)
  start_index1 = s32[] constant(0)
  start_index2 = s32[] constant(0)
  ROOT fusion = f32[4,5] fusion(dot_lhs, dynamic_slice_input, start_index0, start_index1, start_index2),
       kind=kCustom, calls=triton_gemm,
       backend_config={
         "fusion_backend_config":{
           "kind":"__triton_gemm","triton_gemm_config":{
             "block_m":"32","block_n":"32","block_k":"32","split_k":"1",
             "num_stages":"1","num_warps":"4","num_ctas":"1"}}}
})";

  ASSERT_THAT(
      CreateTritonIrAndFileCheckForDot(this, kHloText, "triton_gemm", R"(
CHECK:     tt.func @triton_fn({{[^,]*}}, %[[DYNAMIC_SLICE_INPUT:[^:]*]]: !tt.ptr<f32> {{[^,]*}}, %[[START_INDEX0_PTR:[^:]*]]: !tt.ptr<i32>
CHECK-DAG:   %[[C0_i32:.*]] = arith.constant 0 : i32
CHECK-DAG:   %[[C1_i64:.*]] = arith.constant 1 : i64
CHECK-DAG:   %[[C2_i64:.*]] = arith.constant 2 : i64
CHECK-DAG:   %[[C3_i32:.*]] = arith.constant 3 : i32
CHECK-DAG:   %[[C5_i32:.*]] = arith.constant 5 : i32
CHECK-DAG:   %[[C5_i64:.*]] = arith.constant 5 : i64
CHECK-DAG:   %[[START_INDEX0:.*]] = tt.load %[[START_INDEX0_PTR]] : !tt.ptr<i32>
CHECK-DAG:   %[[SEMI_CLAMPED_START_INDEX0:.*]] = arith.maxsi %[[START_INDEX0]], %[[C0_i32]] : i32
CHECK-DAG:   %[[CLAMPED_START_INDEX0:.*]] = arith.minsi %[[SEMI_CLAMPED_START_INDEX0]], %[[C3_i32]] : i32
CHECK-DAG:   %[[ROW_OFFSET:.*]] = arith.muli %[[CLAMPED_START_INDEX0]], %[[C5_i32]] : i32
CHECK-DAG:   %[[ROW_OFFSET_i64:.*]] = arith.extsi %[[ROW_OFFSET]] : i32 to i64
CHECK-DAG:   %[[ROW_LIMIT:.*]] = arith.addi %[[ROW_OFFSET_i64]], %[[C5_i64]] : i64
CHECK-DAG:   tt.make_tensor_ptr %[[DYNAMIC_SLICE_INPUT]], [%[[C2_i64]], %[[ROW_LIMIT]]], [%[[C1_i64]], %[[C2_i64]]], [%[[C0_i32]], %[[ROW_OFFSET]]]
)"),
      tsl::testing::IsOk());
}

TEST_F(TritonGemmTest, DoNotUseTensorCoresWithNonDefaultPrecision) {
  constexpr absl::string_view kHloText = R"(
triton_gemm_r {
  parameter_0 = s8[80,15]{1,0} parameter(0)
  convert.3 = f32[80,15]{1,0} convert(parameter_0)
  parameter_1 = f32[16,15]{1,0} parameter(1)
  ROOT r.1 = f32[80,16]{1,0} dot(convert.3, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  p1 = f32[16,15]{1,0} parameter(1)
  p0 = s8[80,15]{1,0} parameter(0)
  ROOT triton_gemm_r = f32[80,16]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm_r,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm", triton_gemm_config:
      {"block_m":32,"block_n":32,"block_k":32,
       "split_k":1,"num_stages":1,"num_warps":2,
       "num_ctas":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));

  CompileAndOptionallyVerifyPtx(std::move(module_and_metadata.module),
                                R"(
CHECK-NOT: mma
)");
}

TEST_F(TritonGemmTest, DebugOptionsArePropagated) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = f16[30,30] parameter(0)
  p1 = s8[30,30] parameter(1)
  cp1 = f16[30,30] convert(p1)
  ROOT _ = f16[30,30] dot(p0, cp1),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> verified_module,
                          ParseAndReturnVerifiedModule(kHloText));
  std::string output_directory;
  if (!tsl::io::GetTestUndeclaredOutputsDir(&output_directory)) {
    output_directory = tsl::testing::TmpDir();
  }
  DebugOptions debug_options = verified_module->config().debug_options();
  debug_options.set_xla_dump_to(output_directory);
  debug_options.set_xla_dump_hlo_pass_re("triton-fusion-emitter");
  verified_module->mutable_config().set_debug_options(debug_options);

  EXPECT_TRUE(RunAndCompare(std::move(verified_module),
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));

  std::vector<std::string> paths;
  TF_EXPECT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(output_directory, "*.triton-passes.log"), &paths));
  EXPECT_EQ(paths.size(), 1);
}

TEST_F(TritonGemmTest, DotWithPredFromCompareProducesCorrectResult) {
  constexpr absl::string_view kHloText = R"(
triton_dot {
  parameter_0 = s32[4,128]{1,0} parameter(0)
  broadcast.255 = s32[4,128,64]{2,1,0} broadcast(parameter_0), dimensions={0,1}
  parameter_1 = s32[4,128,64]{2,1,0} parameter(1)
  compare.39 = pred[4,128,64]{2,1,0} compare(broadcast.255, parameter_1), direction=EQ
  bitcast.1097 = pred[512,64]{1,0} reshape(compare.39)
  convert.229 = bf16[512,64]{1,0} convert(bitcast.1097)
  parameter_2 = bf16[64,256]{0,1} parameter(2)
  ROOT dot.21 = bf16[512,256]{1,0} dot(convert.229, parameter_2),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
ENTRY main {
  p0 = s32[4,128]{1,0} parameter(0)
  p1 = s32[4,128,64]{2,1,0} parameter(1)
  p2 = bf16[64,256]{0,1} parameter(2)
  ROOT gemm_fusion_dot.0 = bf16[512,256]{1,0} fusion(p0, p1, p2), kind=kCustom, calls=triton_dot, backend_config={"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"64","block_n":"128","block_k":"32","split_k":"1","num_stages":"4","num_warps":"4","num_ctas":"1"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(module_and_metadata.module->ToString(),
                               ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

TEST_F(TritonGemmTest, UseTensorCoresForF32OnAmpere) {
  constexpr absl::string_view kHloText = R"(
triton_gemm_r {
  parameter_0 = f16[80,15]{1,0} parameter(0)
  convert.3 = f32[80,15]{1,0} convert(parameter_0)
  parameter_1 = f32[16,15]{1,0} parameter(1)
  ROOT r.1 = f32[80,16]{1,0} dot(convert.3, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p1 = f32[16,15]{1,0} parameter(1)
  p0 = f16[80,15]{1,0} parameter(0)
  ROOT triton_gemm_r = f32[80,16]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm_r,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm", triton_gemm_config:
      {"block_m":32,"block_n":32,"block_k":32,
      "split_k":1,"num_stages":1,"num_warps":2,
      "num_ctas":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));
  CompileAndOptionallyVerifyPtx(std::move(module_and_metadata.module),
                                R"(
CHECK: mma
)");
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonGemmTest, FailIfTooMuchShmem) {
  auto cc = se::CudaComputeCapability::Ampere();
  const se::DeviceDescription device_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  constexpr absl::string_view kHloTextTemplate = R"(
triton_gemm_dot {
  p0 = s8[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  c0 = f32[1024,1024] convert(p0)
  ROOT dot = f32[1024,1024] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = s8[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  ROOT r = f32[1024,1024] fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
      triton_gemm_config: {"block_m":$0,"block_n":$1,"block_k":$2,
      "split_k":1,"num_stages":$3,"num_warps":4,
      "num_ctas":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module1_and_metadata,
                          GetModuleAndNestedFusionMetadata(absl::Substitute(
                              kHloTextTemplate, 16, 32, 512, 8)));

  const HloFusionInstruction* fusion1 = Cast<HloFusionInstruction>(
      module1_and_metadata.computation->FusionInstruction());
  EXPECT_THAT(
      TritonWrapper("test_fn", fusion1, cc, device_info,
                    module1_and_metadata.block_level_parameters, &llvm_module,
                    mlir_context),
      StatusIs(tsl::error::RESOURCE_EXHAUSTED,
               ::testing::HasSubstr("Shared memory size limit exceeded")));

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module2_and_metadata,
                          GetModuleAndNestedFusionMetadata(absl::Substitute(
                              kHloTextTemplate, 64, 128, 128, 1)));

  const HloFusionInstruction* fusion2 = Cast<HloFusionInstruction>(
      module2_and_metadata.computation->FusionInstruction());

  TF_ASSERT_OK_AND_ASSIGN(
      const auto result,
      TritonWrapper("test_fn", fusion2, cc, device_info,
                    module2_and_metadata.block_level_parameters, &llvm_module,
                    mlir_context));
  // Use optin shared memory which is > shared_memory_per_block.
  EXPECT_GT(result.shmem_bytes, device_info.shared_memory_per_block());
}

TEST_F(TritonGemmTest, MultipleDims) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f16[1,16,17,3] parameter(0)
  p1 = s8[16,17,3] parameter(1)
  cp1 = f16[16,17,3] convert(p1)
  ROOT _ = f16[1,16,16] dot(p0, cp1),
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: ENTRY
; CHECK-NOT:  convert
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-SAME: "__triton_nested_gemm_fusion"
  )");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, PredWithBF16DotProducesCorrectResult) {
  constexpr absl::string_view kHloText = R"(
triton_dot {
  p0 = pred[8,640]{1,0} parameter(0)
  cvt = bf16[8,640]{1,0} convert(pred[8,640]{1,0} p0)
  p1 = bf16[4096,640]{1,0} parameter(1)
  ROOT dot.10277 = bf16[8,4096]{1,0} dot(cvt, p1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = pred[8,640]{1,0} parameter(0)
  p1 = bf16[4096,640]{1,0} parameter(1)
  ROOT dot = bf16[8,4096]{1,0} fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm", triton_gemm_config:
      {"block_m":16,"block_n":32,"block_k":64,
      "split_k":1,"num_stages":2,"num_warps":8,
      "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(module_and_metadata.module->ToString(),
                               ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, SplitLhsNoncontractingTransposeRhs) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = pred[3,122,96,12]{3,2,1,0} parameter(0)
  cp0 = f16[3,122,96,12]{3,2,1,0} convert(p0)
  p1 = pred[1,5,122]{2,1,0} parameter(1)
  cp1 = f16[1,5,122]{2,1,0} convert(p1)
  ROOT _ = f16[3,96,12,1,5]{4,3,2,1,0} dot(cp0, cp1),
    lhs_contracting_dims={1}, rhs_contracting_dims={2}
})";

  // Check that the transpose is in the nested fusion but not in the entry.
  MatchOptimizedHlo(kHloText, R"(
; CHECK: transpose
; CHECK: ENTRY
; CHECK-NOT: transpose
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-SAME: __triton_nested_gemm_fusion
)");
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(TritonGemmTest, SplitLhsNoncontracting) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = f32[72,72] parameter(0)
  bc1 = f32[4,3,3,2,4,3,3,2] reshape(p0)
  tr = f32[4,3,3,2,2,4,3,3] transpose(bc1), dimensions={0,1,2,3,7,4,5,6}
  bc2 = f32[144,36] reshape(tr)
  p1 = f16[36,3] parameter(1)
  c7 = f32[36,3] convert(p1)
  ROOT _ = f32[144,3] dot(bc2, c7),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  // Check that the transpose is in the nested fusion but not in the entry.
  MatchOptimizedHlo(kHloText, R"(
; CHECK: f32[72,2,36]{2,1,0} transpose(
; CHECK-NEXT: ROOT
; CHECK: ENTRY
; CHECK-NOT: transpose
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-SAME: __triton_nested_gemm_fusion
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, SplitAndTransposeLhsExecutesCorrectly) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  tmp_0 = s8[5,50,2,128] parameter(1)
  tmp_2 = s8[50,5,2,128] transpose(tmp_0), dimensions={1,0,2,3}
  tmp_3 = s8[50,1280] reshape(tmp_2)
  tmp_4 = f16[50,1280] convert(tmp_3)
  tmp_5 = f16[50,79] parameter(0)
  ROOT tmp_6 = f16[1280,79] dot(tmp_4, tmp_5),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: ENTRY
; CHECK-NOT: transpose
; CHECK-NOT: convert
; CHECK: fusion
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config={{.*}}"kind":"__triton_nested_gemm_fusion"
)");
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// TODO(b/393299275): it's not clear that this test is actually testing what it
// claims to be testing. It should either be rewritten to start from
// post-optimization HLO, or hoisted out to test the fusion logic specifically.
TEST_F(TritonGemmTest, NondefaultOperandLayoutIsSupported) {
  constexpr absl::string_view kHloText = R"(
ENTRY r {
  p1 = f16[3,10,128]{2,1,0} parameter(1)
  cp = f16[3,10,128]{2,0,1} copy(p1)
  cv = f32[3,10,128]{2,0,1} convert(cp)
  p0 = f32[3,10,123]{2,1,0} parameter(0)
  ROOT d = f32[3,128,123]{2,1,0} dot(cv, p0),
    lhs_batch_dims={0}, lhs_contracting_dims={1},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-SAME: "__triton_nested_gemm_fusion"
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// TODO(b/393299275): this is a pure test of fusion logic. It should be moved to
// a separate, fusion-specific, deviceless test.
TEST_F(TritonGemmTest, DoNotFuseSplitRhsContractingTranspose) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f16[5,8] parameter(0)
  p1 = s8[2,3,4] parameter(1)
  c0 = f16[2,3,4] convert(p1)
  t1 = f16[3,2,4] transpose(c0), dimensions={1,0,2}
  r1 = f16[3,8] reshape(t1)
  ROOT _ = f16[5,3] dot(p0, r1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: ENTRY
; CHECK: transpose
; CHECK: fusion
; CHECK-SAME: kind=kCustom
; CHECK-SAME: "__triton_nested_gemm_fusion"
)");
}

// TODO(b/393299275): this is a pure test of fusion logic. It should be moved to
// a separate, fusion-specific, deviceless test.
TEST_F(TritonGemmTest, DoNotFuseSplitLhsContractingTranspose) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f16[3,16,25]{2,1,0} parameter(0)
  p0t = f16[16,3,25]{2,1,0} transpose(p0), dimensions={1,0,2}
  p0tr = f16[16,75]{1,0} reshape(p0t)
  p1 = s8[128,75]{1,0} parameter(1)
  cp1 = f16[128,75]{1,0} convert(p1)
  ROOT dot = f16[16,128]{1,0} dot(p0tr, cp1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: ENTRY
; CHECK: transpose
; CHECK: fusion
; CHECK-SAME: kind=kCustom
; CHECK-SAME: "__triton_nested_gemm_fusion"
)");
}

// TODO(b/393299275): this test should be rewritten to start from
// post-optimization HLO. (Though I'm not entirely sure it's even worth keeping
// it.)
TEST_F(TritonGemmTest, NonMajorMostInputBatchWorksCorrectly) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  x = f32[20,50,30] parameter(0)
  y = f16[30,50,40] parameter(1)
  cy = f32[30,50,40] convert(y)
  ROOT _ = f32[50,20,40] dot(x, cy),
    lhs_contracting_dims={2}, rhs_contracting_dims={0},
    lhs_batch_dims={1}, rhs_batch_dims={1}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: fusion
; CHECK-SAME: kind=kCustom
; CHECK-SAME: "__triton_nested_gemm_fusion"
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// TODO(b/393299275): this is a pure test of fusion logic. It should be moved to
// a separate, fusion-specific, deviceless test.
TEST_F(TritonGemmTest, DoNotFuseArbitraryReshape) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  p0 = f16[5,2,3] parameter(0)
  p0c = f32[5,2,3] convert(p0)
  p1 = f32[20,3] parameter(1)
  p1r = f32[5,3,4] reshape(p1)
  ROOT dot = f32[5,2,4] dot(p0c, p1r),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: ENTRY
; CHECK: f32[5,3,4]{2,1,0} bitcast
; CHECK: fusion
; CHECK-SAME: kind=kCustom
; CHECK-SAME: "__triton_nested_gemm_fusion"
)");
}

TEST_F(TritonGemmTest, MultipleBatchRequireSeparateTranspose) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  Arg_0 = f16[3,4,2,5,4] parameter(0)
  c = f32[3,4,2,5,4] convert(Arg_0)
  Arg_1 = f32[5,3,4,3,2] parameter(1)
  ROOT dot.3 = f32[5,3,4,4,3] dot(c, Arg_1),
    lhs_batch_dims={3,0,1}, lhs_contracting_dims={2},
    rhs_batch_dims={0,1,2}, rhs_contracting_dims={4}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: ROOT
; CHECK: transpose(
; CHECK: bitcast(
; CHECK: kCustom
; CHECK-SAME: backend_config={{.*}}"kind":"__triton_nested_gemm_fusion"
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// This tests the complexity heuristics in TritonWrapper.
// TODO(b/393299275): this is not worth keeping as a codegen test. Really, we
// should not reject tilings that are slow/spill in codegen. If this has use in
// autotuning, then this should be tested/called in the autotuner.
// The generic Triton emitter does not want to deal with this.
TEST_F(TritonGemmTest, DISABLED_FailForTooComplexTiling) {
  auto cc = se::CudaComputeCapability::Ampere();
  const se::DeviceDescription device_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  constexpr absl::string_view kHloTextTemplate = R"(
HloModule module

triton_gemm_dot {
  p0 = s8[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  c0 = f32[1024,1024] convert(p0)
  ROOT dot.0 = f32[1024,1024] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = s8[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  ROOT r = f32[1024,1024] fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
      triton_gemm_config: {"block_m":$0,"block_n":$1,"block_k":$2,
      "split_k":1,"num_stages":1,"num_warps":2,
      "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module1_and_metadata,
                          GetModuleAndNestedFusionMetadata(absl::Substitute(
                              kHloTextTemplate, 512, 512, 32)));

  const HloFusionInstruction* fusion1 = Cast<HloFusionInstruction>(
      module1_and_metadata.computation->FusionInstruction());
  EXPECT_THAT(TritonWrapper("test_fn", fusion1, cc, device_info,
                            module1_and_metadata.block_level_parameters,
                            &llvm_module, mlir_context),
              StatusIs(tsl::error::RESOURCE_EXHAUSTED,
                       "Tiling complexity heuristic exceeded"));

  // Succeeds if the tiling is not too complex.
  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module2_and_metadata,
                          GetModuleAndNestedFusionMetadata(
                              absl::Substitute(kHloTextTemplate, 32, 32, 32)));

  const HloFusionInstruction* fusion2 = Cast<HloFusionInstruction>(
      module1_and_metadata.computation->FusionInstruction());

  TF_EXPECT_OK(TritonWrapper("test_fn", fusion2, cc, device_info,
                             module2_and_metadata.block_level_parameters,
                             &llvm_module, mlir_context)
                   .status());
}

// TODO(b/393299275): this test may have some value while Triton tiling
// propagation is being replaced, but has little worth as a codegen test.
// Consider moving this.
TEST_F(TritonGemmTest,
       BroadcastsOfTriviallySizedNonContractingDimensionsAreSupported) {
  constexpr absl::string_view kHloText = R"(
f {
  p0 = f32[64,6464] parameter(0)
  p1 = f32[16,6464] parameter(1)
  dot = f32[16,64] dot(p1, p0),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bc0 = f32[1,16,64] bitcast(dot)
  p2 = f32[64] parameter(2)
  bc1 = f32[1,64] bitcast(p2)
  br = f32[1,16,64] broadcast(bc1), dimensions={0,2}
  m = f32[1,16,64] multiply(bc0, br)
}

e {
  p0 = f32[64,6464] parameter(0)
  p1 = f32[16,6464] parameter(1)
  p2 = f32[64] parameter(2)
  f = f32[1,16,64] fusion(p0, p1, p2),
    kind=kCustom, calls=f, backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
    "triton_gemm_config": {"block_m":"16","block_n":"16","block_k":"64","split_k":"1",
          "num_stages":"1","num_warps":"4","num_ctas":"1"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module_and_metadata.module),
                               ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// TODO(b/393299275): this test may have some value while Triton tiling
// propagation is being replaced, but has little worth as a codegen test.
// Consider moving this.
// TODO(b/393299275): likely uncovered a bug in `NestGemmFusion`, where after
// transformations and collapse of a dimension, broadcast dimensions are wrong.
TEST_F(TritonGemmTest,
       DISABLED_BroadcastsOfTriviallySizedContractingDimensionsAreSupported) {
  constexpr absl::string_view kHloText = R"(
f {
  a = f16[2] parameter(0)
  bc0 = f16[1,2] bitcast(a)
  br = f16[1,4000,2] broadcast(bc0), dimensions={0,2}
  bc1 = f16[4000,2] bitcast(br)
  b = f16[3,4000] parameter(1)
  d = f16[2,3] dot(bc1, b),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
}

e {
  a = f16[2] parameter(0)
  b = f16[3,4000] parameter(1)
  f = f16[2,3] fusion(a, b),
    kind=kCustom, calls=f, backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
    "triton_gemm_config": {"block_m":"16","block_n":"16","block_k":"16","split_k":"1",
          "num_stages":"1","num_warps":"1","num_ctas":"1"}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module_and_metadata.module),
                               ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// TODO(b/393299275): this requires adding support for dynamic-slice in the
// generic Triton emitter.
TEST_F(TritonGemmTest, DISABLED_DynamicSliceIsSupportedInLhsEndToEnd) {
  // The select is used to restrict the start index to values that make sense.
  // If it was constant, then the dynamic-slice would be optimized to slice. It
  // is not strictly needed, because we also support clamping the indices.
  // The start index(es) for the non-majormost dimension(s) are constant zero(s)
  // because we don't support dynamic slice on those dimensions.
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  dot_lhs = f32[2,4] parameter(0)
  dynamic_slice_input = f32[7,2] parameter(1)
  pred0 = pred[] parameter(2)
  c1 = s32[] constant(1)
  c2 = s32[] constant(2)
  start_index0 = s32[] select(pred0, c1, c2)
  start_index1 = s32[] constant(0)
  dynamic_slice = f32[5,2] dynamic-slice(dynamic_slice_input, start_index0, start_index1),
                  dynamic_slice_sizes={5,2}
  ROOT dot = f32[4,5] dot(dot_lhs, dynamic_slice),
          lhs_contracting_dims={0}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(),
                           m::Fusion(m::Parameter()), m::Constant())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
  // Check that it's not optimized away.
  MatchHloModule(*module, "; CHECK: dynamic-slice(");
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

// TODO(b/393299275): this requires adding support for dynamic-slice in the
// generic Triton emitter.
TEST_F(TritonGemmTest, DISABLED_DynamicSliceIsSupportedInRhs) {
  // The start index(es) for the non-majormost dimension(s) are constant zero(s)
  // because we don't support dynamic slice on those dimensions.
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_gemm {
  dynamic_slice_input = f32[7,2] parameter(0)
  dot_rhs = f32[2,4] parameter(1)
  start_index0 = s32[] parameter(2)
  start_index1 = s32[] parameter(3)
  dynamic_slice = f32[5,2] dynamic-slice(dynamic_slice_input, start_index0, start_index1),
                  dynamic_slice_sizes={5,2}
  ROOT dot = f32[5, 4] dot(dynamic_slice, dot_rhs),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  dynamic_slice_input = f32[7,2] parameter(0)
  dot_rhs = f32[2,4] parameter(1)
  start_index0 = s32[] constant(1)
  start_index1 = s32[] constant(0)
  ROOT fusion = f32[5,4] fusion(dynamic_slice_input, dot_rhs, start_index0, start_index1),
       kind=kCustom, calls=triton_gemm,
       backend_config={
         "fusion_backend_config":{
           "kind":"__triton_gemm","triton_gemm_config":{
             "block_m":"32","block_n":"32","block_k":"32","split_k":"1",
             "num_stages":"1","num_warps":"4","num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

class TritonGemmDynamicSliceClampingTest
    : public TritonTest,
      public ::testing::WithParamInterface<int> {};

// TODO(b/393299275): this requires adding support for dynamic-slice in the
// generic Triton emitter.
TEST_P(TritonGemmDynamicSliceClampingTest,
       DISABLED_DynamicSliceIsSupportedWhenTheStartIndexNeedsClamping) {
  // The start index(es) for the non-majormost dimension(s) are constant zero(s)
  // because we don't support dynamic slice on those dimensions.

  const std::string hlo_text = absl::Substitute(R"(
HloModule m

triton_gemm {
  dynamic_slice_input = f32[7,2] parameter(0)
  dot_rhs = f32[2,4] parameter(1)
  start_index0 = s32[] parameter(2)
  start_index1 = s32[] parameter(3)
  dynamic_slice = f32[5,2] dynamic-slice(dynamic_slice_input, start_index0, start_index1),
                  dynamic_slice_sizes={5,2}
  ROOT dot = f32[5, 4] dot(dynamic_slice, dot_rhs),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  dynamic_slice_input = f32[7,2] parameter(0)
  dot_rhs = f32[2,4] parameter(1)
  start_index0 = s32[] constant($0)
  start_index1 = s32[] constant(0)
  ROOT fusion = f32[5,4] fusion(dynamic_slice_input, dot_rhs, start_index0, start_index1),
       kind=kCustom, calls=triton_gemm,
       backend_config={
         "fusion_backend_config":{
           "kind":"__triton_gemm","triton_gemm_config":{
             "block_m":"32","block_n":"32","block_k":"32","split_k":"1",
             "num_stages":"1","num_warps":"4","num_ctas":"1"}}}
})",
                                                GetParam());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      hlo_text, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

std::string OffsetParamToString(const ::testing::TestParamInfo<int>& data) {
  return absl::StrCat("WithOffsetEq", data.param < 0 ? "Negative" : "",
                      std::abs(data.param));
}

INSTANTIATE_TEST_SUITE_P(All, TritonGemmDynamicSliceClampingTest,
                         ::testing::Values(-100, 3, 999), OffsetParamToString);

// TODO(b/393299275): this requires adding support for dynamic-slice in the
// generic Triton emitter.
TEST_F(TritonGemmTest,
       DISABLED_DynamicSliceOfMajormostContractingDimIsSupported) {
  // Tests that dynamic-slice works on the majormost dimension even if that
  // dimension is contracted.
  // The start index(es) for the non-majormost dimension(s) are constant zero(s)
  // because we don't support dynamic slice on those dimensions.
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_gemm {
  dot_lhs = f32[2,4] parameter(0)
  dynamic_slice_input = f32[5,4] parameter(1)
  start_index0 = s32[] parameter(2)
  start_index1 = s32[] parameter(3)
  dynamic_slice = f32[2,4] dynamic-slice(dynamic_slice_input, start_index0, start_index1),
                  dynamic_slice_sizes={2,4}
  ROOT dot = f32[4,4] dot(dot_lhs, dynamic_slice),
             lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  dot_lhs = f32[2,4] parameter(0)
  dynamic_slice_input = f32[5,4] parameter(1)
  start_index0 = s32[] constant(2)
  start_index1 = s32[] constant(0)
  ROOT fusion = f32[4,4] fusion(dot_lhs, dynamic_slice_input, start_index0, start_index1),
       kind=kCustom, calls=triton_gemm,
       backend_config={
         "fusion_backend_config":{
           "kind":"__triton_gemm","triton_gemm_config":{
             "block_m":"32","block_n":"32","block_k":"32","split_k":"1",
             "num_stages":"1","num_warps":"4","num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

// TODO(b/393299275): this requires adding support for dynamic-slice in the
// generic Triton emitter.
TEST_F(TritonGemmTest, DISABLED_DynamicSliceOfMajormostBatchDimIsSupported) {
  // Tests that dynamic-slice works on the majormost dimension even if that
  // dimension is a batch.
  // The start index(es) for the non-majormost dimension(s) are constant zero(s)
  // because we don't support dynamic slice on those dimensions.
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_gemm {
  dot_lhs = f32[2,2,4] parameter(0)
  dynamic_slice_input = f32[7,2,4] parameter(1)
  start_index0 = s32[] parameter(2)
  start_index1 = s32[] parameter(3)
  start_index2 = s32[] parameter(4)
  dynamic_slice = f32[2,2,4] dynamic-slice(dynamic_slice_input, start_index0, start_index1, start_index2),
                  dynamic_slice_sizes={2,2,4}
  ROOT dot = f32[2,4,4] dot(dot_lhs, dynamic_slice),
             lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  dot_lhs = f32[2,2,4] parameter(0)
  dynamic_slice_input = f32[7,2,4] parameter(1)
  start_index0 = s32[] constant(2)
  start_index1 = s32[] constant(0)
  start_index2 = s32[] constant(0)
  ROOT fusion = f32[2,4,4] fusion(dot_lhs, dynamic_slice_input, start_index0, start_index1, start_index2),
       kind=kCustom, calls=triton_gemm,
       backend_config={
         "fusion_backend_config":{
           "kind":"__triton_gemm","triton_gemm_config":{
             "block_m":"32","block_n":"32","block_k":"32","split_k":"1",
             "num_stages":"1","num_warps":"4","num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

// TODO(b/393299275): this requires adding support for dynamic-slice in the
// generic Triton emitter.
TEST_F(TritonGemmTest,
       DISABLED_DynamicSliceSingleDimensionIntoReshapeIsSupported) {
  // This directly tests the targeted use case (b/307922364) of iterating over
  // layer weights and extracting them with dynamic slice.
  // The start index(es) for the non-majormost dimension(s) are constant zero(s)
  // because we don't support dynamic slice on those dimensions.
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_gemm {
  dot_lhs = f32[2,4] parameter(0)
  dynamic_slice_input = f32[4,5,2] parameter(1)
  start_index0 = s32[] parameter(2)
  start_index1 = s32[] parameter(3)
  start_index2 = s32[] parameter(4)
  dynamic_slice = f32[1,5,2] dynamic-slice(dynamic_slice_input, start_index0, start_index1, start_index2),
                             dynamic_slice_sizes={1,5,2}
  reshape = f32[5,2] reshape(dynamic_slice)
  ROOT d = f32[4,5] dot(dot_lhs, reshape),
           lhs_contracting_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  dot_lhs = f32[2,4] parameter(0)
  dynamic_slice_input = f32[4,5,2] parameter(1)
  start_index0 = s32[] constant(3)
  start_index1 = s32[] constant(0)
  start_index2 = s32[] constant(0)
  ROOT fusion = f32[4,5] fusion(dot_lhs, dynamic_slice_input, start_index0, start_index1, start_index2),
       kind=kCustom, calls=triton_gemm,
       backend_config={
         "fusion_backend_config":{
           "kind":"__triton_gemm","triton_gemm_config":{
             "block_m":"32","block_n":"32","block_k":"32","split_k":"1",
             "num_stages":"1","num_warps":"4","num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-6}));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, DoNotFuseConcatenationOfSplitNonContractingDimension) {
  if (std::holds_alternative<se::RocmComputeCapability>(
          GpuComputeCapability())) {
    GTEST_SKIP() << "Not using autotuner on ROCM yet.";
  }
  if (!SupportsBF16(GpuComputeCapability())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  x = bf16[2,128,10] parameter(0)
  y = bf16[2,256,10] parameter(1)
  concat = bf16[2,384,10] concatenate(x, y), dimensions={1}
  z = bf16[10,20] parameter(2)
  ROOT d = bf16[2,384,20] dot(concat, z), lhs_contracting_dims={2}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK:      ENTRY
; CHECK:      concatenate
; CHECK:      fusion
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   "__triton_nested_gemm_fusion"
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, BinaryOperationWithSmallInputsIsFused) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  p0 = s8[7,3] parameter(0)
  p1 = f32[3,16] parameter(1)
  p2 = f32[3,16] parameter(2)
  e = f32[3,16] exponential(p1)
  a = f32[3,16] add(e, p2)
  c = f32[7,3] convert(p0)
  ROOT d = f32[7,16] dot(c, a),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

TEST_F(TritonGemmTest, BinaryOperationWithLargeInputsIsNotFused) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  p0 = f16[333,1000] parameter(0)
  p1 = f32[1000,333] parameter(1)
  p1n = f32[1000,333] negate(p1)
  p2 = f32[1000,333] parameter(2)
  p2n = f32[1000,333] negate(p2)
  s = f32[1000,333] subtract(p1n, p2n)
  c = f32[333,1000] convert(p0)
  ROOT d = f32[1000,1000] dot(s, c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: fused_subtract
; CHECK: negate
; CHECK: negate
; CHECK: ROOT
; CHECK-SAME: subtract
; CHECK: ENTRY
; CHECK: kLoop
; CHECK: kCustom
; CHECK-SAME: backend_config={{.*}}"kind":"__triton_nested_gemm_fusion"
)");
}

TEST_F(TritonGemmTest, ParametersWithDifferentLayoutsAreSupportedInOneScope) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = s8[5,3] parameter(0)
  p0c = f16[5,3] convert(p0)
  p1 = f16[5,7] parameter(1)
  p2 = f16[7,5] parameter(2)
  t = f16[5,7] transpose(p2), dimensions={1,0}
  a = f16[5,7] add(t, p1)
  ROOT d = f16[3,7] dot(p0c, a),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

TEST_F(TritonGemmTest, BinaryOperationOnLargeParametersIsFused) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  p0 = f16[1000,111] parameter(0)
  p1 = f32[111,10000] parameter(1)
  p2 = f32[111,10000] parameter(2)
  s = f32[111,10000] subtract(p1, p2)
  c = f32[1000,111] convert(p0)
  ROOT d = f32[10000,1000] dot(s, c),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

TEST_F(TritonGemmTest, LinkingLibdeviceTwiceWorks) {
  // TODO(b/393299275): This test looks weird. It's testing the whole
  // optimization pipeline end-to-end to check that linking libdevice twice
  // works? rewrite this to just use post-optimization HLO
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = s8[7,3] parameter(0)
  c0 = f32[7,3] convert(p0)
  p1 = f32[3,16] parameter(1)
  e1 = f32[3,16] exponential(p1)
  d0 = f32[7,16] dot(c0, e1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p2 = s8[7,3] parameter(2)
  c2 = f32[7,3] convert(p2)
  e2 = f32[7,3] exponential(c2)
  p3 = f32[3,16] parameter(3)
  d1 = f32[7,16] dot(e2, p3),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT a = f32[7,16] add(d0, d1)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Add(
                  m::Fusion(m::Parameter(), m::Parameter())
                      .WithFusionKind(HloInstruction::FusionKind::kCustom),
                  m::Fusion(m::Parameter(), m::Parameter())
                      .WithFusionKind(HloInstruction::FusionKind::kCustom))));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, BroadcastOfScalarParameterIsFused) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = f16[64,256] parameter(0)
  p0c = f32[64,256] convert(p0)
  p1 = f32[] parameter(1)
  b = f32[256,128] broadcast(p1), dimensions={}
  ROOT d = f32[64,128] dot(p0c, b),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, BroadcastOfScalarConstantIsFused) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  p0 = f16[70,30] parameter(0)
  p0c = f32[70,30] convert(p0)
  constant_3663 = f32[] constant(4321)
  bc0 = f32[30,5] broadcast(constant_3663)
  ROOT d = f32[70,5] dot(p0c, bc0),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, DoubleBroadcastOfScalarConstantIsFused) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  c = s32[] constant(1)
  bc1 = s32[21]{0} broadcast(c), dimensions={}
  p0 = s32[21]{0} parameter(0)
  cmp = pred[21]{0} compare(bc1, p0), direction=EQ
  convert.6 = bf16[21]{0} convert(cmp)
  bc2 = bf16[3,21]{1,0} broadcast(convert.6), dimensions={1}
  p1 = bf16[21,71]{1,0} parameter(1)
  ROOT d = bf16[3,71]{1,0} dot(bc2, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      GetNonBitcastRoot(module->entry_computation()),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, BroadcastOfVectorConstantIsFused) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  p0 = s8[60,5] parameter(0)
  c0 = f16[60,5] convert(p0)
  cst1 = f16[120] constant({...})
  r1 = f16[5,120] broadcast(cst1), dimensions={1}
  ROOT d = f16[60,120] dot(c0, r1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      GetNonBitcastRoot(module->entry_computation()),
      GmockMatch(m::Fusion(m::Parameter(), m::Constant())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, AlwaysFuseScalarConstantAtBroadcastInput) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = bf16[2,3,3]{2,1,0} parameter(0)
  p1 = bf16[3,2,3]{2,1,0} parameter(1)
  d = bf16[2,3,3]{2,1,0} dot(p0, p1),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={1}, rhs_contracting_dims={0}
  t = bf16[3,2,3]{2,0,1} transpose(d), dimensions={1,0,2}
  c = bf16[] constant(0.123)
  b = bf16[3,2,3]{2,1,0} broadcast(c), dimensions={}
  m = bf16[3,2,3]{2,0,1} multiply(t, b)
  ROOT tu = (bf16[3,2,3]{2,0,1}, bf16[3,2,3]{2,1,0}) tuple(m, b)
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: gemm_fusion_dot
; CHECK: dot(
; CHECK: bf16[] constant(0.123)
; CHECK: ROOT
; CHECK: ENTRY
; CHECK: kCustom
; CHECK-SAME: backend_config={{.*}}"kind":"__triton_nested_gemm_fusion"
)");
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, FuseConcatenation) {
  if (!SupportsBF16(GpuComputeCapability())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  constexpr absl::string_view kHloText = R"(
e {
  p0 = s8[153,1536] parameter(0)
  p1 = s8[153,128] parameter(1)
  p2 = s8[153,128] parameter(2)
  cat = s8[153,1792] concatenate(p0, p1, p2), dimensions={1}
  cvt = bf16[153,1792] convert(cat)
  p3 = bf16[16,153] parameter(3)
  ROOT d = bf16[16,1792] dot(p3, cvt),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      GetNonBitcastRoot(module->entry_computation()),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter(),
                           m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-2,
                                                /*arel=*/1e-2}));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, SineOutputIsNotFused) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  p0 = s8[7,101] parameter(0)
  p1 = f32[101,16] parameter(1)
  c = f32[7,101] convert(p0)
  d = f32[7,16] dot(c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT r = f32[7,16] sine(d)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Sin(
                  m::Fusion(m::Parameter(), m::Parameter())
                      .WithFusionKind(HloInstruction::FusionKind::kCustom))));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, SliceInputIsFused) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = f16[97,121] parameter(0)
  s0 = f16[7,101] slice(p0), slice={[3:10], [10:111]}
  p1 = f32[101,16] parameter(1)
  c = f32[7,101] convert(s0)
  ROOT d = f32[16,7] dot(p1, c),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, SliceInputWithReshapeIsFused) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = f32[363,1536] parameter(0)
  p1 = f32[4,1536,611] parameter(1)
  s = f32[1,1536,611] slice(p1),
    slice={[1:2], [0:1536], [0:611]}
  r = f32[1536,611] reshape(s)
  ROOT d = f32[363,611] dot(p0, r),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      GetNonBitcastRoot(module->entry_computation()),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory. From the point of view of codegen, this is purely
// about compositionality.
TEST_F(TritonGemmTest, NestedSlicingWorks) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p1 = f32[6,24] parameter(1)
  slice1 = f32[5,20] slice(p1), slice={[1:6], [3:23]}
  n1 = f32[5,20] negate(slice1)
  slice2 = f32[3,7] slice(n1), slice={[1:4], [13:20]}
  p0 = f32[7,37] parameter(0)
  ROOT d = f32[3,37] dot(slice2, p0),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, SlicedBatchDimensionIsSupported) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = f16[3,3,256] parameter(0)
  s0 = f16[3,3,128] slice(p0), slice={[0:3], [0:3], [123:251]}
  r0 = f16[3,3,128] reshape(s0)
  p1 = f16[3,3,256] parameter(1)
  svar1 = f16[3,3,128] slice(p1), slice={[0:3], [0:3], [30:158]}
  r1 = f16[3,3,128] reshape(svar1)
  ROOT d = f16[128,3,3]{2,1,0} dot(r0, r1),
    lhs_batch_dims={2}, lhs_contracting_dims={1},
    rhs_batch_dims={2}, rhs_contracting_dims={1}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

// TODO(b/393299275): symbolic tile analysis fails to derive a tile for one
// outer parameter here. However, we shouldn't be deriving this tile anyway,
// and the underlying indexing map is incorrect. This requires a fix in
// symbolic tile derivation.
// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTestWithSplitK,
       DISABLED_SplitKDoesNotBreakSlicedFragmentedContractingDimension) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = f16[16,8,128]{2,1,0} parameter(0)
  s0 = f16[16,4,128]{2,1,0} slice(p0),
    slice={[0:16], [0:4], [0:128]}
  r0 = f16[16,512]{1,0} reshape(s0)
  p1 = s8[4096,4,128]{2,1,0} parameter(1)
  r1 = s8[512,4096]{0,1} reshape(p1)
  c1 = f16[512,4096]{0,1} convert(r1)
  ROOT d = f16[16,4096]{1,0} dot(r0, c1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

// TODO(b/393299275): this should be rewritten to work on post-optimization HLO,
// and potentially have an associated fusion test.
TEST_F(TritonGemmTestWithSplitK, SplitKWithTrivialDimension) {
  constexpr absl::string_view kHloText = R"(
ENTRY entry_computation {
  p0 = f16[1001,1]{1,0} parameter(0)
  convert = f32[1001,1]{1,0} convert(p0)
  p1 = f32[1001,2048]{1,0} parameter(1)
  ROOT dot = f32[1,2048]{1,0} dot(convert, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, NarrowingConvertOutputIsFused) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  p0 = s8[22,80] parameter(0)
  p1 = f32[80,54] parameter(1)
  c = f32[22,80] convert(p0)
  d = f32[54,22] dot(p1, c),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
  ROOT r = f16[54,22] convert(d)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, ParameterAfterDotIsFused) {
  if (!SupportsBF16(GpuComputeCapability())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  p0 = bf16[350,1280]{1,0} parameter(0)
  p1 = s16[1280,690]{0,1} parameter(1)
  p1c = bf16[1280,690]{0,1} convert(p1)
  dot.21 = bf16[350,690]{1,0} dot(p0, p1c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p2 = bf16[350,690]{1,0} parameter(2)
  ROOT r = bf16[350,690]{1,0} multiply(p2, dot.21)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  const HloInstruction* instr = module->entry_computation()->root_instruction();
  if (!instr->IsCustomFusion()) {
    instr = instr->operand(0);
    ASSERT_TRUE(instr->IsCustomFusion());
  }
  EXPECT_THAT(
      instr,
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/2e-2, /*arel=*/2e-2}));
}

TEST_F(TritonGemmTest, OutputFusionExecutesCorrectly) {
  // TODO(b/393299275): it this test useful?
  if (!SupportsBF16(GpuComputeCapability())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  p0 = f16[350,1280]{1,0} parameter(0)
  p0c = bf16[350,1280]{1,0} convert(p0)
  p1 = bf16[1280,690]{0,1} parameter(1)
  d = bf16[350,690]{1,0} dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p3 = bf16[350,690]{1,0} parameter(3)
  multiply.8811 = bf16[350,690]{1,0} multiply(d, p3)
  neg.484 = bf16[350,690]{1,0} negate(multiply.8811)
  p2 = bf16[350,690]{1,0} parameter(2)
  ROOT multiply.8808 = bf16[350,690]{1,0} multiply(neg.484, p2)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  const HloInstruction* instr = module->entry_computation()->root_instruction();
  if (!instr->IsCustomFusion()) {
    instr = instr->operand(0);
    ASSERT_TRUE(instr->IsCustomFusion());
  }
  EXPECT_THAT(
      instr,
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter(),
                           m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/2e-2, /*arel=*/2e-2}));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
TEST_F(TritonGemmTest, SplitLHSOutputTransposeAloneIsNotFused) {
  if (!SupportsBF16(GpuComputeCapability())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  p0 = s8[18,15000] parameter(0)
  p0c = bf16[18,15000] convert(p0)
  p1 = bf16[42,18] parameter(1)
  d = bf16[15000,42] dot(p0c, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
  r1 = bf16[5,200,15,42] reshape(d)
  ROOT t1 = bf16[5,42,200,15] transpose(r1), dimensions={0,3,1,2}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Bitcast(
          m::Fusion(m::Fusion(m::Parameter(), m::Parameter())
                        .WithFusionKind(HloInstruction::FusionKind::kCustom))
              .WithFusionKind(HloInstruction::FusionKind::kInput))));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// TODO(b/393299275): this should just be a fusion test and does not need to be
// in the codegen directory.
// TODO(b/393299275): symbolic tile analysis fails to derive a tile for one
// outer parameter here. However, we shouldn't be deriving this tile anyway,
// and the underlying indexing map is incorrect. This requires a fix in
// symbolic tile derivation.
TEST_F(TritonGemmTest, DISABLED_SplitLHSInputOutputIsFused) {
  if (!SupportsBF16(GpuComputeCapability())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  if (std::holds_alternative<se::RocmComputeCapability>(
          GpuComputeCapability())) {
    GTEST_SKIP() << "Skipped until corresponding issue on ROCm is fixed.";
  }

  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = s8[5,18,20,150] parameter(0)
  p0c = bf16[5,18,20,150] convert(p0)
  t0 = bf16[18,5,20,150] transpose(p0c), dimensions={1,0,2,3}
  r0 = bf16[18,15000] reshape(t0)
  p1 = bf16[42,18] parameter(1)
  d = bf16[15000,42] dot(r0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
  r1 = bf16[5,20,150,42] reshape(d)
  ROOT t1 = bf16[5,42,20,150] transpose(r1), dimensions={0,3,1,2}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, SupportPredParametersUsedInExpressions) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p = pred[2,2]{1,0} parameter(0)
  a = f32[2,2]{1,0} parameter(1)
  b = f32[2,2]{1,0} parameter(2)
  c = f32[2,2]{1,0} parameter(3)
  compare = pred[2,2]{1,0} compare(a, b), direction=LT
  and = pred[2,2]{1,0} and(p, compare)
  convert = f32[2,2]{1,0} convert(and)
  ROOT r = f32[2,2]{1,0} dot(convert, c),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter(),
                           m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

// TODO(b/393299275): This test name might be a bit misleading, since the dot is
// given a non-contracting dimension by the time it gets passed down to the
// Triton emitter. This should probably be a fusion test.
// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonGemmTest, LowerDotWithLhsWithoutNonContractingDimThroughTriton) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  parameter_0 = f32[1,40] parameter(0)
  parameter_1 = f32[1,40,250000] parameter(1)
  ROOT dot = f32[1,250000] dot(parameter_0, parameter_1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  EXPECT_THAT(
      GetNonBitcastRoot(module->entry_computation()),
      GmockMatch(
          m::Fusion(m::Bitcast(m::Parameter()), m::Bitcast(m::Parameter()))
              .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

// TODO(b/393299275): This test name might be a bit misleading, since the dot is
// given a non-contracting dimension by the time it gets passed down to the
// Triton emitter. This should probably be a fusion test.
// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonGemmTest, LowerDotWithRhsWithoutNonContractingDimThroughTriton) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  parameter_0 = f32[1,40,250000] parameter(0)
  parameter_1 = f32[1,40] parameter(1)
  ROOT dot = f32[1,250000] dot(parameter_0, parameter_1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  EXPECT_THAT(
      GetNonBitcastRoot(module->entry_computation()),
      GmockMatch(
          m::Fusion(m::Bitcast(m::Parameter()), m::Bitcast(m::Parameter()))
              .WithFusionKind(HloInstruction::FusionKind::kCustom)));
}

// This group of tests compares GPU results of dots already rewritten
// into Triton fusions.
using CompareTest = TritonGemmTest;

TEST_F(CompareTest, F32WithTrivialNonContractingDimension) {
  constexpr absl::string_view kHloTextRef = R"(
HloModule r

ENTRY e {
  arg0 = f32[5,7] parameter(0)
  arg1 = f32[1,7] parameter(1)
  gemm = (f32[5,1], s8[0]{0}) custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[1],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f32[5,1]{1,0} get-tuple-element((f32[5,1]{1,0}, s8[0]{0}) gemm), index=0
}
)";

  constexpr absl::string_view kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[5,7] parameter(0)
  p1 = f32[1,7] parameter(1)
  ROOT dot = f32[5,1] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f32[5,7]{1,0} parameter(0)
  p1 = f32[1,7]{1,0} parameter(1)
  ROOT _ = f32[5,1] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":32,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":1,
                         "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      ModuleAndNestedFusionMetadata test_module_and_metadata,
      GetModuleAndNestedFusionMetadata(kHloText));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> ref_module,
                          ParseAndReturnVerifiedModule(kHloTextRef));

  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(ref_module), std::move(test_module_and_metadata.module),
      ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3},
      /*run_hlo_passes=*/false));
}

// TODO(b/353484968, b/393299275): the e2e test path was never really testing
// anything useful until now---it used to not actually set opt-in shared memory
// on properly. Additionally, it claimed to be testing Ampere specifically but
// runs across every possible chip that is supported by Triton. The test should
// probably be made deviceless and repurposed to test that opt-in shared memory
// is used only.
TEST_F(CompareTest, UsingOptinSharedMemoryProducesSameResult) {
  if (std::holds_alternative<se::RocmComputeCapability>(
          GpuComputeCapability())) {
    GTEST_SKIP() << "No Optin Shared Memory on AMD.";
  }
  const se::DeviceDescription dev_info =
      backend().default_stream_executor()->GetDeviceDescription();
  // TODO(b/353484968): pin this test to a specific device type to ensure
  // correct expectations.
  //
  // On Hopper, the RHS has to be provided through shared memory, so a minima,
  // the kernel will get away with using
  //   num_stages * block_k * block_n * sizeof(rhs_element_type)
  // = 2          * 128     * 128     * 2
  // = 65536 bytes.
  //
  // This should hold on Blackwell as well.
  constexpr int kBytesOfSharedMemoryTested = 2 * 128 * 128 * 2;
  EXPECT_GE(dev_info.shared_memory_per_block_optin(),
            kBytesOfSharedMemoryTested);

  const std::string kHloTextOptinShmem = R"(
triton_dot {
  p0 = s8[332,441]{1,0} parameter(0)
  convert = bf16[332,441]{1,0} convert(p0)
  p1 = bf16[441,39]{1,0} parameter(1)
  // Fix an algorithm on the dot in order to explicitly control the size of the
  // operands in shared memory, as well as the precision.
  ROOT dot = bf16[332,39]{1,0} dot(convert, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32
}

ENTRY e {
  p0 = s8[332,441]{1,0} parameter(0)
  p1 = bf16[441,39]{1,0} parameter(1)
  ROOT _ = bf16[332,39]{1,0} fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":128,"block_n":128,"block_k":128,
                         "split_k":1,"num_stages":2,"num_warps":32,
                         "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      ModuleAndNestedFusionMetadata optin_shmem_module_and_metadata,
      GetModuleAndNestedFusionMetadata(kHloTextOptinShmem));
  const HloFusionInstruction* triton_dot_fusion = Cast<HloFusionInstruction>(
      optin_shmem_module_and_metadata.computation->FusionInstruction());
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  TF_ASSERT_OK_AND_ASSIGN(
      const auto result,
      TritonWrapper("test_fn", triton_dot_fusion, GpuComputeCapability(),
                    dev_info,
                    optin_shmem_module_and_metadata.block_level_parameters,
                    &llvm_module, mlir_context));
  // The config is chosen so that the used memory size is slightly above the
  // 48 kB boundary of standard / opt-in shared memory so that any GPU that
  // has the opt-in one should be able to execute the test.
  EXPECT_EQ(result.shmem_bytes, kBytesOfSharedMemoryTested);
  // Make sure the written config indeed has to use optin shared memory.
  EXPECT_GT(result.shmem_bytes, dev_info.shared_memory_per_block());

  const std::string kHloTextLowShmem = R"(
HloModule t

triton_dot {
  p0 = s8[332,441]{1,0} parameter(0)
  convert = bf16[332,441]{1,0} convert(p0)
  p1 = bf16[441,39]{1,0} parameter(1)
  // Fix an algorithm on the dot in order to explicitly control the size of the
  // operands in shared memory, as well as the precision.
  ROOT dot = bf16[332,39]{1,0} dot(convert, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32
}

ENTRY e {
  p0 = s8[332,441]{1,0} parameter(0)
  p1 = bf16[441,39]{1,0} parameter(1)
  ROOT _ = bf16[332,39]{1,0} fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":32,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":4,
                         "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      ModuleAndNestedFusionMetadata low_shmem_module_and_metadata,
      GetModuleAndNestedFusionMetadata(kHloTextLowShmem));

  EXPECT_TRUE(
      RunAndCompareTwoModules(std::move(low_shmem_module_and_metadata.module),
                              std::move(optin_shmem_module_and_metadata.module),
                              ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6},
                              /*run_hlo_passes=*/false));
}

// TODO(b/393299275): there seems to be a (not yet diagnosed) miscompile here.
// We have to investigate.
TEST_F(CompareTest, DISABLED_SplitK) {
  if (!SupportsBF16(GpuComputeCapability())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  constexpr absl::string_view hlo_text_ref = R"(
HloModule t

triton_gemm_r {
  parameter_0 = s8[480,120]{1,0} parameter(0)
  convert.3 = bf16[480,120]{1,0} convert(parameter_0)
  parameter_1 = bf16[16,120]{1,0} parameter(1)
  ROOT r.1 = bf16[480,16]{1,0} dot(convert.3, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0_pred = s8[480,120]{1,0} parameter(0)
  p0 = s8[480,120]{1,0} convert(p0_pred)
  p1_pred = pred[16,120]{1,0} parameter(1)
  p1 = bf16[16,120]{1,0} convert(p1_pred)
  ROOT triton_gemm_r = bf16[480,16]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm_r,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":64,"block_n":16,"block_k":16,
                         "split_k":1,"num_stages":4,"num_warps":4,
                         "num_ctas":1}}}
})";

  constexpr absl::string_view hlo_text_splitk = R"(
HloModule t

triton_gemm_r {
  parameter_0 = s8[480,120]{1,0} parameter(0)
  convert.3 = bf16[480,120]{1,0} convert(parameter_0)
  bitcast.11 = bf16[480,4,30]{2,1,0} bitcast(convert.3)
  parameter_1 = bf16[16,120]{1,0} parameter(1)
  bitcast.12 = bf16[16,4,30]{2,1,0} bitcast(parameter_1)
  ROOT dot.1 = bf16[4,480,16]{2,1,0} dot(bitcast.11, bitcast.12),
    lhs_batch_dims={1}, lhs_contracting_dims={2},
    rhs_batch_dims={1}, rhs_contracting_dims={2}
}

add {
  rhs.1 = f32[] parameter(1)
  lhs.1 = f32[] parameter(0)
  ROOT add.1 = f32[] add(lhs.1, rhs.1)
}

fused_computation {
  param_0.2 = bf16[4,480,16]{2,1,0} parameter(0)
  convert.18 = f32[4,480,16]{2,1,0} convert(param_0.2)
  constant_1 = bf16[] constant(0)
  convert.17 = f32[] convert(constant_1)
  reduce.1 = f32[480,16]{1,0} reduce(convert.18, convert.17), dimensions={0},
    to_apply=add
  ROOT convert.16 = bf16[480,16]{1,0} convert(reduce.1)
}

ENTRY e {
  p0_pred = s8[480,120]{1,0} parameter(0)
  p0 = s8[480,120]{1,0} convert(p0_pred)
  p1_pred = pred[16,120]{1,0} parameter(1)
  p1 = bf16[16,120]{1,0} convert(p1_pred)
  triton_gemm_r = bf16[4,480,16]{2,1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm_r,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":64,"block_n":16,"block_k":16,
                         "split_k":4,"num_stages":1,"num_warps":4,
                         "num_ctas":1}}}
  ROOT fusion.1 = bf16[480,16]{1,0} fusion(triton_gemm_r), kind=kLoop,
    calls=fused_computation
})";

  TF_ASSERT_OK_AND_ASSIGN(
      ModuleAndNestedFusionMetadata test_module_and_metadata,
      GetModuleAndNestedFusionMetadata(hlo_text_splitk));

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata ref_module_and_metadata,
                          GetModuleAndNestedFusionMetadata(hlo_text_ref));

  EXPECT_TRUE(
      RunAndCompareTwoModules(std::move(ref_module_and_metadata.module),
                              std::move(test_module_and_metadata.module),
                              ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6},
                              /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, SplitKBatch) {
  if (!SupportsBF16(GpuComputeCapability())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloTextRef = R"(
triton_gemm_dot.24 {
  parameter_1 = bf16[1,1,800,5,128]{4,3,2,1,0} parameter(1)
  bitcast.3 = bf16[800,5,128]{2,1,0} bitcast(parameter_1)
  convert.3 = f32[800,5,128]{2,1,0} convert(bitcast.3)
  parameter_0 = f32[1,5,700,800]{3,2,1,0} parameter(0)
  bitcast.2 = f32[5,700,800]{2,1,0} bitcast(parameter_0)
  ROOT dot.26 = f32[5,128,700]{2,1,0} dot(convert.3, bitcast.2), lhs_batch_dims={1}, lhs_contracting_dims={0}, rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY e {
  tmp_3 = f32[1,5,700,800]{3,2,1,0} parameter(0)
  tmp_0 = bf16[1,1,800,5,128]{4,3,2,1,0} parameter(1)
  ROOT triton_gemm_dot.24 = f32[5,128,700]{2,1,0} fusion(tmp_3, tmp_0),
    kind=kCustom, calls=triton_gemm_dot.24,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":64,"block_n":32,"block_k":64,
                         "split_k":1,"num_stages":2,"num_warps":8,
                         "num_ctas":1}}}
})";

  const std::string kHloTextTest = R"(
triton_gemm_dot {
  parameter_1 = bf16[1,1,800,5,128]{4,3,2,1,0} parameter(1)
  bitcast.3 = bf16[800,5,128]{2,1,0} bitcast(parameter_1)
  convert.3 = f32[800,5,128]{2,1,0} convert(bitcast.3)
  bitcast = f32[8,100,5,128]{3,2,1,0} bitcast(convert.3)
  parameter_0 = f32[1,5,700,800]{3,2,1,0} parameter(0)
  bitcast.2 = f32[5,700,800]{2,1,0} bitcast(parameter_0)
  bitcast.1 = f32[5,700,8,100]{3,2,1,0} bitcast(bitcast.2)
  ROOT dot = f32[8,5,128,700]{3,2,1,0} dot(bitcast, bitcast.1), lhs_batch_dims={0,2}, lhs_contracting_dims={1}, rhs_batch_dims={2,0}, rhs_contracting_dims={3}
}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY e {
  tmp_3 = f32[1,5,700,800]{3,2,1,0} parameter(0)
  tmp_0 = bf16[1,1,800,5,128]{4,3,2,1,0} parameter(1)
  triton_gemm_dot.24 = f32[8,5,128,700]{3,2,1,0} fusion(tmp_3, tmp_0),
    kind=kCustom, calls=triton_gemm_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":64,"block_n":32,"block_k":64,
                         "split_k":8,"num_stages":1,"num_warps":4,
                         "num_ctas":1}}}
  constant = f32[] constant(0)
  ROOT reduce = f32[5,128,700]{2,1,0} reduce(triton_gemm_dot.24, constant), dimensions={0}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(
      ModuleAndNestedFusionMetadata test_module_and_metadata,
      GetModuleAndNestedFusionMetadata(kHloTextTest));

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata ref_module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloTextRef));

  EXPECT_TRUE(
      RunAndCompareTwoModules(std::move(ref_module_and_metadata.module),
                              std::move(test_module_and_metadata.module),
                              ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3},
                              /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, SplitKNontrivialBitcast) {
  if (!SupportsBF16(GpuComputeCapability())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloTextRef = R"(
HloModule module

dot {
  p0 = s8[512,96]{1,0} parameter(0)
  convert = bf16[512,96]{1,0} convert(p0)
  p1 = bf16[16,4,128]{2,1,0} parameter(1)
  bitcast = bf16[16,512]{1,0} bitcast(p1)
  ROOT dot = bf16[16,96]{1,0} dot(bitcast, convert),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = s8[512,96]{1,0} parameter(0)
  p1 = bf16[16,4,128]{2,1,0} parameter(1)
  ROOT dot = bf16[16,96]{1,0} fusion(p0, p1),
    kind=kCustom, calls=dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":32,"block_k":256,
                         "split_k":1,"num_stages":1,"num_warps":4,
                         "num_ctas":1}}}
})";

  const std::string kHloTextTest = R"(
HloModule module

dot {
  p0 = s8[512,96]{1,0} parameter(0)
  convert = bf16[512,96]{1,0} convert(p0)
  p1 = bf16[16,4,128]{2,1,0} parameter(1)
  bitcast_p1 = bf16[16,16,32]{2,1,0} bitcast(p1)
  bitcast_convert = bf16[16,32,96]{2,1,0} bitcast(convert)
  ROOT dot = bf16[16,16,96]{2,1,0} dot(bitcast_p1, bitcast_convert),
    lhs_batch_dims={1}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
}

reducer {
  rhs = f32[] parameter(1)
  lhs = f32[] parameter(0)
  ROOT add = f32[] add(lhs, rhs)
}

split_k_reducer {
  p0 = bf16[16,16,96]{2,1,0} parameter(0)
  convert = f32[16,16,96]{2,1,0} convert(p0)
  c0 = f32[] constant(0)
  reduce = f32[16,96]{1,0} reduce(convert, c0),
    dimensions={0}, to_apply=reducer
  ROOT output = bf16[16,96]{1,0} convert(reduce)
}

ENTRY entry {
  p0 = s8[512,96]{1,0} parameter(0)
  p1 = bf16[16,4,128]{2,1,0} parameter(1)
  dot = bf16[16,16,96]{2,1,0} fusion(p0, p1),
    kind=kCustom, calls=dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":64,"block_n":32,"block_k":32,
                         "split_k":16,"num_stages":1,"num_warps":4,
                         "num_ctas":1}}}
  ROOT output = bf16[16,96]{1,0} fusion(dot), kind=kLoop, calls=split_k_reducer
})";
  TF_ASSERT_OK_AND_ASSIGN(
      ModuleAndNestedFusionMetadata test_module_and_metadata,
      GetModuleAndNestedFusionMetadata(kHloTextTest));

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata ref_module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloTextRef));

  EXPECT_TRUE(
      RunAndCompareTwoModules(std::move(ref_module_and_metadata.module),
                              std::move(test_module_and_metadata.module),
                              ErrorSpec{/*aabs=*/2, /*arel=*/1e-2},
                              /*run_hlo_passes=*/false));
}

// This is based on gemm_fusion_test.cc/SplitKTest.SupportsIndivisible.
//
// There were relatively large numeric errors with an f16 temporary buffer, so I
// ended up using --xla_gpu_triton_gemm_disable_reduced_precision_reduction=true
// when generating this test case.
//
// TODO(b/393299275): transform this test once padding derivation if fixed.
TEST_F(CompareTest, DISABLED_SupportsSplitKWithIndivisibleKComplexExample) {
  constexpr absl::string_view kHloTextRef = R"(
dot {
  p0 = s8[3,129,5,32]{3,2,1,0} parameter(0)
  bitcast = s8[3,5,32,129]{2,1,3,0} bitcast(p0)
  copy = s8[3,5,32,129]{3,2,1,0} copy(bitcast)
  reshape = s8[480,129]{1,0} reshape(copy)
  convert = f16[480,129]{1,0} convert(reshape)
  p1 = f16[16,129]{1,0} parameter(1)
  ROOT dot = f16[480,16]{1,0} dot(convert, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY entry_computation {
  p0 = s8[3,129,5,32]{3,2,1,0} parameter(0)
  p1 = f16[16,129]{1,0} parameter(1)
  ROOT fusion = f16[480,16]{1,0} fusion(p0, p1), kind=kCustom, calls=dot,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
    "triton_gemm_config":{"block_m":"32","block_n":"32","block_k":"256",
                          "split_k":"1","num_stages":"1","num_warps":"4",
                          "num_ctas":"1"}}}
}
)";

  constexpr absl::string_view kHloTextTest = R"(
dot {
  p0 = s8[3,129,5,32]{3,2,1,0} parameter(0)
  bitcast_p0 = s8[3,5,32,129]{2,1,3,0} bitcast(p0)
  copy_p0 = s8[3,5,32,129]{3,2,1,0} copy(bitcast_p0)
  reshape_p0 = s8[480,129]{1,0} reshape(copy_p0)
  convert.8 = f16[480,129]{1,0} convert(reshape_p0)
  c0 = f16[] constant(0)
  pad_p0 = f16[480,130]{1,0} pad(convert.8, c0), padding=0_0x0_1
  bitcast_pad_p0 = f16[480,2,65]{2,1,0} bitcast(pad_p0)
  dot_lhs = f32[480,2,65]{2,1,0} convert(bitcast_pad_p0)
  p1 = f16[16,129]{1,0} parameter(1)
  pad_p1 = f16[16,130]{1,0} pad(p1, c0), padding=0_0x0_1
  bitcast_pad_p1 = f16[16,2,65]{2,1,0} bitcast(pad_p1)
  dot_rhs = f32[16,2,65]{2,1,0} convert(bitcast_pad_p1)
  ROOT dot.2 = f32[2,480,16]{2,1,0} dot(dot_lhs, dot_rhs),
    lhs_batch_dims={1}, lhs_contracting_dims={2},
    rhs_batch_dims={1}, rhs_contracting_dims={2}
}

reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

split_k_reducer {
  p0 = f32[2,480,16]{2,1,0} parameter(0)
  c0 = f32[] constant(0)
  reduce = f32[480,16]{1,0} reduce(p0, c0), dimensions={0}, to_apply=reducer
  ROOT convert = f16[480,16]{1,0} convert(reduce)
}

ENTRY entry_computation {
  p0 = s8[3,129,5,32]{3,2,1,0} parameter(0)
  p1 = f16[16,129]{1,0} parameter(1)
  fusion = f32[2,480,16]{2,1,0} fusion(p0, p1), kind=kCustom, calls=dot,
  backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
  "triton_gemm_config":{"block_m":"128","block_n":"128","block_k":"64",
                        "split_k":"2","num_stages":"1","num_warps":"8",
                        "num_ctas":"1"}}}
  ROOT output = f16[480,16]{1,0} fusion(fusion), kind=kLoop,
    calls=split_k_reducer
})";
  TF_ASSERT_OK_AND_ASSIGN(
      ModuleAndNestedFusionMetadata test_module_and_metadata,
      GetModuleAndNestedFusionMetadata(kHloTextTest));

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata ref_module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloTextRef));

  EXPECT_TRUE(
      RunAndCompareTwoModules(std::move(ref_module_and_metadata.module),
                              std::move(test_module_and_metadata.module),
                              ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2},
                              /*run_hlo_passes=*/false));
}

// TODO(b/393299275): transform this test once padding derivation if fixed.
TEST_F(CompareTest, DISABLED_SupportsSplitKWithIndivisibleKUsingPaddingEqual1) {
  constexpr absl::string_view kHloTextRef = R"(
HloModule extracted, entry_computation_layout={(f16[1,8,4,1023]{3,2,1,0}, f16[1,1023,128]{2,1,0})->f16[1,8,4,128]{3,2,1,0}}

triton_gemm_dot.7103_computation.clone {
  parameter_0.499 = f16[1,8,4,1023]{3,2,1,0} parameter(0)
  bitcast.7923 = f16[32,1023]{1,0} bitcast(parameter_0.499)
  parameter_1.499 = f16[1,1023,128]{2,1,0} parameter(1)
  bitcast.7924 = f16[1023,128]{1,0} bitcast(parameter_1.499)
  dot.9350 = f16[32,128]{1,0} dot(bitcast.7923, bitcast.7924), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT bitcast.7925 = f16[1,8,4,128]{3,2,1,0} bitcast(dot.9350)
}

ENTRY entry_computation {
  p0 = f16[1,8,4,1023]{3,2,1,0} parameter(0)
  p1 = f16[1,1023,128]{2,1,0} parameter(1)
  ROOT triton_gemm_dot.7103 = f16[1,8,4,128]{3,2,1,0} fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot.7103_computation.clone,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
    "triton_gemm_config":{"block_m":"128","block_n":"128","block_k":"32",
                          "split_k":"1","num_stages":"4","num_warps":"4",
                          "num_ctas":"1"}}}
})";

  constexpr absl::string_view kHloTextSplitK = R"(
HloModule extracted, entry_computation_layout={(f16[1,8,4,1023]{3,2,1,0}, f16[1,1023,128]{2,1,0})->f16[1,8,4,128]{3,2,1,0}}

triton_gemm_dot.7103_computation.clone {
  parameter_0.499 = f16[1,8,4,1023]{3,2,1,0} parameter(0)
  bitcast.7923 = f16[32,1023]{1,0} bitcast(parameter_0.499)
  constant = f16[] constant(0)
  pad = f16[32,1024]{1,0} pad(bitcast.7923, constant), padding=0_0x0_1
  bitcast = f16[32,8,128]{2,1,0} bitcast(pad)
  parameter_1.499 = f16[1,1023,128]{2,1,0} parameter(1)
  bitcast.7924 = f16[1023,128]{1,0} bitcast(parameter_1.499)
  constant.1 = f16[] constant(0)
  pad.1 = f16[1024,128]{1,0} pad(bitcast.7924, constant.1), padding=0_1x0_0
  bitcast.1 = f16[8,128,128]{2,1,0} bitcast(pad.1)
  dot.1 = f16[8,32,128]{2,1,0} dot(bitcast, bitcast.1), lhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  ROOT bitcast.7925.clone = f16[8,1,8,4,128]{4,3,2,1,0} bitcast(dot.1)
}

triton_gemm_dot.7103.reduce_sub_computation.clone {
  lhs.1 = f32[] parameter(0)
  rhs.1 = f32[] parameter(1)
  add.2 = f32[] add(lhs.1, rhs.1)
  convert.13 = f16[] convert(add.2)
  ROOT convert.12 = f32[] convert(convert.13)
}

fused_computation.1 {
  param_0.5 = f16[8,1,8,4,128]{4,3,2,1,0} parameter(0)
  convert.16 = f32[8,1,8,4,128]{4,3,2,1,0} convert(param_0.5)
  constant.3 = f16[] constant(0)
  convert.15 = f32[] convert(constant.3)
  reduce.1 = f32[1,8,4,128]{3,2,1,0} reduce(convert.16, convert.15), dimensions={0}, to_apply=triton_gemm_dot.7103.reduce_sub_computation.clone
  ROOT convert.14 = f16[1,8,4,128]{3,2,1,0} convert(reduce.1)
}

ENTRY entry_computation {
  p0 = f16[1,8,4,1023]{3,2,1,0} parameter(0)
  p1 = f16[1,1023,128]{2,1,0} parameter(1)
  triton_gemm_dot.7103 = f16[8,1,8,4,128]{4,3,2,1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm_dot.7103_computation.clone,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
    "triton_gemm_config":{"block_m":"16","block_n":"128","block_k":"32",
                          "split_k":"8","num_stages":"1","num_warps":"4",
                          "num_ctas":"1"}}}
  ROOT fusion.1 = f16[1,8,4,128]{3,2,1,0} fusion(triton_gemm_dot.7103), kind=kLoop, calls=fused_computation.1
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextSplitK,
                                      ErrorSpec{/*aabs=*/4e-2, /*arel=*/2e-2},
                                      /*run_hlo_passes=*/false));
}

// TODO(b/393299275): symbolic tile derivation fails for one of the padded
// operands, with indexing map
//  (d0, d1, d2, d3) -> (d1, d0 * 64 + d3)
//    domain: d0 in [0, 15]
//            d1 in [0, 31]
//            d2 in [0, 127]
//            d3 in [0, 63]
//            d0 * 64 + d3 in [0, 1018]
// While the expression should be processed without any issue, padding
// introduces a non-redundant pre-existing constraint d0 * 64 + d3 in [0, 1018],
// which causes the derivation to be rejected. The reason for this is that it's
// not quite clear how to handle these pre-existing constraints in the general
// sense. But wrt HLO specifically and symbolic tile analysis, we could probably
// decide to just drop them from symbolic tile derivation: the reason for that
// is that offset constraints are handled via `tile_offsets_indexing` anyway,
// and it's all that should be relevant afaik. We can probably let the caller
// decide to drop pre-existing constraints.
TEST_F(CompareTest, DISABLED_SupportsSplitKWithIndivisibleKUsingPaddingEqual5) {
  constexpr absl::string_view kHloTextRef = R"(
HloModule extracted

dot {
  p0 = f16[1,8,4,1019]{3,2,1,0} parameter(0)
  bitcast_p0 = f16[32,1019]{1,0} bitcast(p0)
  p1 = f16[1,1019,128]{2,1,0} parameter(1)
  bitcast_p1 = f16[1019,128]{1,0} bitcast(p1)
  dot = f16[32,128]{1,0} dot(bitcast_p0, bitcast_p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT bitcast_dot = f16[1,8,4,128]{3,2,1,0} bitcast(dot)
}

ENTRY entry_computation {
  p0 = f16[1,8,4,1019]{3,2,1,0} parameter(0)
  p1 = f16[1,1019,128]{2,1,0} parameter(1)
  ROOT dot = f16[1,8,4,128]{3,2,1,0} fusion(p0, p1), kind=kCustom,
    calls=dot,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
    "triton_gemm_config":{"block_m":"32","block_n":"32","block_k":"256",
                          "split_k":"1","num_stages":"1","num_warps":"4",
                          "num_ctas":"1"}}}
}
)";

  constexpr absl::string_view kHloTextTest = R"(
HloModule extracted

split_k_dot {
  p0 = f16[1,8,4,1019]{3,2,1,0} parameter(0)
  bitcast_p0 = f16[32,1019]{1,0} bitcast(p0)
  c0 = f16[] constant(0)
  pad_p0 = f16[32,1024]{1,0} pad(bitcast_p0, c0), padding=0_0x0_5
  bitcast_pad_p0 = f16[32,16,64]{2,1,0} bitcast(pad_p0)
  p1 = f16[1,1019,128]{2,1,0} parameter(1)
  bitcast_p1 = f16[1019,128]{1,0} bitcast(p1)
  pad_p1 = f16[1024,128]{1,0} pad(bitcast_p1, c0), padding=0_5x0_0
  bitcast_pad_p1 = f16[16,64,128]{2,1,0} bitcast(pad_p1)
  dot = f16[16,32,128]{2,1,0} dot(bitcast_pad_p0, bitcast_pad_p1),
    lhs_batch_dims={1}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
  ROOT bitcast.7925.clone = f16[16,1,8,4,128]{4,3,2,1,0} bitcast(dot)
}

reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  add = f32[] add(lhs, rhs)
  convert = f16[] convert(add)
  ROOT output = f32[] convert(convert)
}

split_k_reducer {
  p0 = f16[16,1,8,4,128]{4,3,2,1,0} parameter(0)
  convert = f32[16,1,8,4,128]{4,3,2,1,0} convert(p0)
  c0 = f32[] constant(0)
  reduce = f32[1,8,4,128]{3,2,1,0} reduce(convert, c0), dimensions={0}, to_apply=reducer
  ROOT output = f16[1,8,4,128]{3,2,1,0} convert(reduce)
}

ENTRY entry_computation {
  p0 = f16[1,8,4,1019]{3,2,1,0} parameter(0)
  p1 = f16[1,1019,128]{2,1,0} parameter(1)
  dot = f16[16,1,8,4,128]{4,3,2,1,0} fusion(p0, p1), kind=kCustom,
    calls=split_k_dot,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
    "triton_gemm_config":{"block_m":"64","block_n":"32","block_k":"32",
                          "split_k":"16","num_stages":"1","num_warps":"4",
                          "num_ctas":"1"}}}
  ROOT fusion = f16[1,8,4,128]{3,2,1,0} fusion(dot), kind=kLoop,
    calls=split_k_reducer
})";
  TF_ASSERT_OK_AND_ASSIGN(
      ModuleAndNestedFusionMetadata test_module_and_metadata,
      GetModuleAndNestedFusionMetadata(kHloTextTest));

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata ref_module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloTextRef));

  EXPECT_TRUE(
      RunAndCompareTwoModules(std::move(ref_module_and_metadata.module),
                              std::move(test_module_and_metadata.module),
                              ErrorSpec{/*aabs=*/4e-2, /*arel=*/2e-2},
                              /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, NonMajorMostOutputBatchWorksCorrectly) {
  const std::string kHloTextTest = R"(
HloModule m

dot {
  p0 = pred[32,26,104]{2,1,0} parameter(0)
  p1 = f32[32,50,104]{2,1,0} parameter(1)
  convert = f32[32,26,104]{2,1,0} convert(p0)
  ROOT dot = f32[32,50,26]{2,0,1} dot(p1, convert),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY e {
  p0 = pred[32,26,104]{2,1,0} parameter(0)
  p1 = f32[32,50,104]{2,1,0} parameter(1)
  ROOT dot = f32[32,50,26]{2,0,1} fusion(p0, p1),
    kind=kCustom, calls=dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":64,"block_n":16,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":4,
                         "num_ctas":1}}}
})";

  const std::string kHloTextRef = R"(
HloModule m

dot {
  p0 = pred[32,26,104]{2,1,0} parameter(0)
  p1 = f32[32,50,104]{2,1,0} parameter(1)
  convert = f32[32,26,104]{2,1,0} convert(p0)
  ROOT dot = f32[32,50,26]{2,1,0} dot(p1, convert),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
}

loop_fusion {
  p0 = f32[32,50,26]{2,1,0} parameter(0)
  transpose = f32[50,32,26]{2,1,0} transpose(p0), dimensions={1,0,2}
  ROOT bitcast = f32[32,50,26]{2,0,1} bitcast(transpose)
}

ENTRY e {
  p0 = pred[32,26,104]{2,1,0} parameter(0)
  p1 = f32[32,50,104]{2,1,0} parameter(1)
  dot = f32[32,50,26]{2,1,0} fusion(p0, p1),
    kind=kCustom, calls=dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":128,"block_k":64,
                         "split_k":1,"num_stages":2,"num_warps":4,
                         "num_ctas":1}}}
  ROOT fusion = f32[32,50,26]{2,0,1} fusion(dot), kind=kLoop,
    calls=loop_fusion
})";

  TF_ASSERT_OK_AND_ASSIGN(
      ModuleAndNestedFusionMetadata test_module_and_metadata,
      GetModuleAndNestedFusionMetadata(kHloTextTest));

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata ref_module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloTextRef));

  EXPECT_TRUE(
      RunAndCompareTwoModules(std::move(ref_module_and_metadata.module),
                              std::move(test_module_and_metadata.module),
                              ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6},
                              /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, TritonDotFusionCanHaveOnlyRHSParameter) {
  const std::string kHloTextTest = R"(
HloModule m, is_scheduled=true

triton_gemm___computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  c = f16[] constant(321)
  b = f16[11,63] broadcast(c)
  cc = f32[11,63] convert(b)
  ROOT _.1 = f32[63,92]{1,0} dot(cc, parameter_0),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f32[92,11]{1,0} parameter(0)
  ROOT triton_gemm__ = f32[63,92]{1,0} fusion(p0), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
                    "triton_gemm_config":{"block_m":"16","block_n":"64",
                                          "block_k":"16","split_k":"1",
                                          "num_stages":"3","num_warps":"2",
                                          "num_ctas":"1"}}}
})";

  const std::string kHloTextRef = R"(
HloModule m, is_scheduled=true

ENTRY e {
  constant_2 = f32[] constant(321)
  parameter_0 = f32[92,11]{1,0} parameter(0)
  broadcast.2 = f32[11,63]{1,0} broadcast(constant_2), dimensions={}
  gemm = (f32[63,92]{1,0}, s8[0]{0}) custom-call(broadcast.2, parameter_0),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["0"],"rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f32[63,92]{1,0} get-tuple-element((f32[63,92]{1,0}, s8[0]{0}) gemm), index=0
})";
  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloTextTest));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> ref_module,
                          ParseAndReturnVerifiedModule(kHloTextRef));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(ref_module),
                                      std::move(module_and_metadata.module),
                                      ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, TritonDotFusionCanHaveNoParametersAtAll) {
  const std::string kHloTextTest = R"(
HloModule m, is_scheduled=true

triton_gemm___computation {
  c = f32[] constant(7)
  b = f32[11,61] broadcast(c)
  c2 = f32[] constant(5)
  b2 = f32[61,45] broadcast(c2)
  ROOT _.1 = f32[11,45]{1,0} dot(b, b2),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  ROOT triton_gemm__ = f32[11,45]{1,0} fusion(), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
                    "triton_gemm_config":{"block_m":"16","block_n":"64",
                                          "block_k":"16","split_k":"1",
                                          "num_stages":"3","num_warps":"2",
                                          "num_ctas":"1"}}}
})";

  const std::string kHloTextRef = R"(
HloModule m, is_scheduled=true

ENTRY triton_gemm___computation {
  constant_1 = f32[] constant(7)
  constant = f32[] constant(5)
  broadcast = f32[11,61]{1,0} broadcast(constant), dimensions={}
  broadcast.1 = f32[61,45]{1,0} broadcast(constant_1), dimensions={}
  gemm = (f32[11,45]{1,0}, s8[0]{0}) custom-call(broadcast, broadcast.1),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f32[11,45]{1,0} get-tuple-element((f32[11,45]{1,0}, s8[0]{0}) gemm), index=0
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloTextTest));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> ref_module,
                          ParseAndReturnVerifiedModule(kHloTextRef));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(ref_module),
                                      std::move(module_and_metadata.module),
                                      ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, TritonDotFusionCanHaveManyParameters) {
  const std::string kHloTextTest = R"(
HloModule m

triton_gemm_dot_computation {
  tmp_1 = pred[3,32]{1,0} parameter(0)
  tmp_2 = f32[3,32]{1,0} parameter(1)
  tmp_3 = f32[3,32]{1,0} parameter(2)
  tmp_4 = f32[3,32]{1,0} select(tmp_1, tmp_2, tmp_3)
  tmp_5 = f32[3,32]{1,0} parameter(3)
  tmp_6 = f32[3,32]{1,0} multiply(tmp_4, tmp_5)
  tmp_7 = f32[3,32]{1,0} parameter(4)
  tmp_8 = f32[3,32]{1,0} maximum(tmp_6, tmp_7)
  tmp_9 = f32[3,57]{1,0} parameter(9)
  tmp_10 = f32[3,57]{1,0} parameter(10)
  tmp_11 = f32[3,57]{1,0} multiply(tmp_9, tmp_10)
  tmp_12 = f32[3,57]{1,0} parameter(11)
  tmp_13 = f32[3,57]{1,0} add(tmp_11, tmp_12)
  tmp_14 = pred[3,57]{1,0} parameter(5)
  tmp_15 = f32[3,57]{1,0} parameter(6)
  tmp_16 = f32[3,57]{1,0} parameter(7)
  tmp_17 = f32[3,57]{1,0} select(tmp_14, tmp_15, tmp_16)
  tmp_18 = f32[3,57]{1,0} parameter(8)
  tmp_19 = f32[3,57]{1,0} multiply(tmp_17, tmp_18)
  tmp_20 = f32[3,57]{1,0} negate(tmp_19)
  tmp_21 = f32[3,57]{1,0} add(tmp_13, tmp_20)
  const_1 = f32[] constant(-3e-3)
  const_2 = f32[] constant(3e-2)
  broadcast_1 = f32[3,57]{1,0} broadcast(const_1), dimensions={}
  broadcast_2 = f32[3,57]{1,0} broadcast(const_2), dimensions={}
  tmp_22 = f32[3,57]{1,0} clamp(broadcast_1, tmp_21, broadcast_2)
  ROOT tmp_23 = f32[32,57]{0,1} dot(tmp_8, tmp_22), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  tmp_1 = pred[3,32]{1,0} parameter(0)
  tmp_2 = f32[3,32]{1,0} parameter(1)
  tmp_3 = f32[3,32]{1,0} parameter(2)
  tmp_5 = f32[3,32]{1,0} parameter(3)
  tmp_7 = f32[3,32]{1,0} parameter(4)
  tmp_14 = pred[3,57]{1,0} parameter(5)
  tmp_15 = f32[3,57]{1,0} parameter(6)
  tmp_16 = f32[3,57]{1,0} parameter(7)
  tmp_18 = f32[3,57]{1,0} parameter(8)
  tmp_9 = f32[3,57]{1,0} parameter(9)
  tmp_10 = f32[3,57]{1,0} parameter(10)
  tmp_12 = f32[3,57]{1,0} parameter(11)
  ROOT r = f32[32,57]{0,1} fusion(tmp_1, tmp_2, tmp_3, tmp_5, tmp_7, tmp_14, tmp_15, tmp_16, tmp_18, tmp_9, tmp_10, tmp_12), kind=kCustom,
    calls=triton_gemm_dot_computation,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
                    "triton_gemm_config":{"block_m":"64","block_n":"64",
                                          "block_k":"64","split_k":"1",
                                          "num_stages":"1","num_warps":"4",
                                          "num_ctas":"1"}}}
})";

  const std::string kHloTextRef = R"(
HloModule m

fused_computation {
  param_5.1 = f32[3,57]{1,0} parameter(5)
  param_6 = f32[3,57]{1,0} parameter(6)
  multiply.4 = f32[3,57]{1,0} multiply(param_5.1, param_6)
  param_4.2 = f32[3,57]{1,0} parameter(4)
  add.3 = f32[3,57]{1,0} add(multiply.4, param_4.2)
  param_1.4 = pred[3,57]{1,0} parameter(1)
  param_2.2 = f32[3,57]{1,0} parameter(2)
  param_3.1 = f32[3,57]{1,0} parameter(3)
  select.2 = f32[3,57]{1,0} select(param_1.4, param_2.2, param_3.1)
  param_0.1 = f32[3,57]{1,0} parameter(0)
  multiply.3 = f32[3,57]{1,0} multiply(select.2, param_0.1)
  negate.1 = f32[3,57]{1,0} negate(multiply.3)
  add.2 = f32[3,57]{1,0} add(add.3, negate.1)
  const.1 = f32[] constant(-3e-3)
  const.2 = f32[] constant(3e-2)
  broadcast.1 = f32[3,57]{1,0} broadcast(const.1), dimensions={}
  broadcast.2 = f32[3,57]{1,0} broadcast(const.2), dimensions={}
  ROOT clamp = f32[3,57]{1,0} clamp(broadcast.1, add.2, broadcast.2)
}

fused_computation.1 {
  param_2.4 = pred[3,32]{1,0} parameter(2)
  param_3.2 = f32[3,32]{1,0} parameter(3)
  param_4.3 = f32[3,32]{1,0} parameter(4)
  select.3 = f32[3,32]{1,0} select(param_2.4, param_3.2, param_4.3)
  param_1.7 = f32[3,32]{1,0} parameter(1)
  multiply.5 = f32[3,32]{1,0} multiply(select.3, param_1.7)
  param_0.3 = f32[3,32]{1,0} parameter(0)
  ROOT maximum.1 = f32[3,32]{1,0} maximum(multiply.5, param_0.3)
}

ENTRY e {
  tmp_18 = f32[3,57]{1,0} parameter(8)
  tmp_16 = f32[3,57]{1,0} parameter(7)
  tmp_15 = f32[3,57]{1,0} parameter(6)
  tmp_14 = pred[3,57]{1,0} parameter(5)
  tmp_12 = f32[3,57]{1,0} parameter(11)
  tmp_10 = f32[3,57]{1,0} parameter(10)
  tmp_9 = f32[3,57]{1,0} parameter(9)
  tmp_7 = f32[3,32]{1,0} parameter(4)
  tmp_5 = f32[3,32]{1,0} parameter(3)
  tmp_3 = f32[3,32]{1,0} parameter(2)
  tmp_2 = f32[3,32]{1,0} parameter(1)
  tmp_1 = pred[3,32]{1,0} parameter(0)
  fusion.1 = f32[3,32]{1,0} fusion(tmp_7, tmp_5, tmp_1, tmp_2, tmp_3), kind=kLoop, calls=fused_computation.1
  fusion = f32[3,57]{1,0} fusion(tmp_18, tmp_14, tmp_15, tmp_16, tmp_12, /*index=5*/tmp_9, tmp_10), kind=kLoop, calls=fused_computation
  gemm = (f32[32,57]{0,1}, s8[0]{0}) custom-call(fusion.1, fusion),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["0"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f32[32,57]{0,1} get-tuple-element((f32[32,57]{0,1}, s8[0]{0}) gemm), index=0
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloTextTest));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> ref_module,
                          ParseAndReturnVerifiedModule(kHloTextRef));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(ref_module),
                                      std::move(module_and_metadata.module),
                                      ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, PredToBF16ConversionWorks) {
  if (!SupportsBF16(GpuComputeCapability())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloTextTest = R"(
HloModule m, is_scheduled=true

triton_gemm_computation {
  parameter_0 = bf16[92,11]{1,0} parameter(0)
  parameter_1 = s32[11,63]{1,0} parameter(1)
  parameter_2 = s32[11,63]{1,0} parameter(2)
  f1.1 = pred[11,63]{1,0} compare(parameter_1, parameter_2), direction=GE
  c.1 = bf16[11,63]{1,0} convert(f1.1)
  ROOT _.1 = bf16[92,63]{1,0} dot(parameter_0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = bf16[92,11]{1,0} parameter(0)
  p1 = s32[11,63]{1,0} parameter(1)
  p2 = s32[11,63]{1,0} parameter(2)
  ROOT triton_gemm__ = bf16[92,63]{1,0} fusion(p0, p1, p2), kind=kCustom,
    calls=triton_gemm_computation,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
                    "triton_gemm_config":{"block_m":"32","block_n":"16",
                                          "block_k":"32","split_k":"1",
                                          "num_stages":"1","num_warps":"4",
                                          "num_ctas":"1"}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloTextTest));

  const std::string kHloTextRef = R"(
HloModule m, is_scheduled=true

fused_computation {
  p0 = s32[11,63]{1,0} parameter(0)
  p1 = s32[11,63]{1,0} parameter(1)
  f.1 = pred[11,63]{1,0} compare(p0, p1), direction=GE
  ROOT convert.1 = bf16[11,63]{1,0} convert(f.1)
}

ENTRY e {
  p2 = s32[11,63]{1,0} parameter(2)
  p1 = s32[11,63]{1,0} parameter(1)
  p0 = bf16[92,11]{1,0} parameter(0)
  fusion = bf16[11,63]{1,0} fusion(p1, p2), kind=kLoop, calls=fused_computation
  gemm = (bf16[92,63]{1,0}, s8[0]{0}) custom-call(p0, fusion),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":
      {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],
      "lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},
      "alpha_imag":0,"precision_config":
      {"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = bf16[92,63]{1,0} get-tuple-element((bf16[92,63]{1,0}, s8[0]{0}) gemm), index=0
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> ref_module,
                          ParseAndReturnVerifiedModule(kHloTextRef));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(ref_module),
                                      std::move(module_and_metadata.module),
                                      ErrorSpec{/*aabs=*/0, /*arel=*/0},
                                      /*run_hlo_passes=*/false));
}

// TODO(b/393299275): symbolic tile analysis fails to derive a tile for one
// outer parameter here. However, we shouldn't be deriving this tile anyway,
// and the underlying indexing map is incorrect. This requires a fix in
// symbolic tile derivation.
TEST_F(CompareTest, DISABLED_DifferentLayoutsAreSupportedInOneScope) {
  const std::string kHloTextTest = R"(
triton_dot {
  p1 = f16[3,3,2,16]{1,3,2,0} parameter(1)
  cvt1 = f32[3,3,2,16]{1,3,2,0} convert(p1)
  p0 = f16[9,32]{0,1} parameter(0)
  b0 = f16[3,3,2,16]{1,0,3,2} bitcast(p0)
  cp0b0 = f16[2,16,3,3]{3,2,1,0} bitcast(b0)
  cp0t0 = f16[3,2,16,3]{3,2,1,0} transpose(cp0b0), dimensions={2,0,1,3}
  cp0b1 = f16[3,3,2,16]{1,3,2,0} bitcast(cp0t0)
  cvt0 = f32[3,3,2,16]{1,3,2,0} convert(cp0b1)
  m = f32[3,3,2,16]{1,3,2,0} multiply(cvt1, cvt0)
  cvt2 = f16[3,3,2,16]{1,3,2,0} convert(m)
  cp1b0 = f16[3,2,16,3]{3,2,1,0} bitcast(cvt2)
  cp1t0 = f16[3,3,2,16]{3,2,1,0} transpose(cp1b0), dimensions={0,3,1,2}
  b1 = f16[9,32]{1,0} bitcast(cp1t0)
  p2 = f16[32,32]{1,0} parameter(2)
  ROOT r = f16[9,32]{1,0} dot(b1, p2),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[9,32]{0,1} parameter(0)
  p1 = f16[3,3,2,16]{1,3,2,0} parameter(1)
  p2 = f16[32,32]{1,0} parameter(2)
  ROOT r = f16[9,32]{1,0} fusion(p0, p1, p2),
    kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":32,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":2,
                         "num_ctas":"1"}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloTextTest));

  const std::string kHloTextRef = R"(
ENTRY e {
  p1 = f16[3,3,2,16]{1,3,2,0} parameter(1)
  cvt1 = f32[3,3,2,16]{1,3,2,0} convert(p1)
  p0 = f16[9,32]{0,1} parameter(0)
  b0 = f16[3,3,2,16]{1,0,3,2} bitcast(p0)
  cp0b0 = f16[2,16,3,3]{3,2,1,0} bitcast(b0)
  cp0t0 = f16[3,2,16,3]{3,2,1,0} transpose(cp0b0), dimensions={2,0,1,3}
  cp0b1 = f16[3,3,2,16]{1,3,2,0} bitcast(cp0t0)
  cvt0 = f32[3,3,2,16]{1,3,2,0} convert(cp0b1)
  m = f32[3,3,2,16]{1,3,2,0} multiply(cvt1, cvt0)
  cvt2 = f16[3,3,2,16]{1,3,2,0} convert(m)
  cp1b0 = f16[3,2,16,3]{3,2,1,0} bitcast(cvt2)
  cp1t0 = f16[3,3,2,16]{3,2,1,0} transpose(cp1b0), dimensions={0,3,1,2}
  b1 = f16[9,32]{1,0} bitcast(cp1t0)
  p2 = f16[32,32]{1,0} parameter(2)
  ROOT r = f16[9,32]{1,0} dot(b1, p2),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> ref_module,
                          ParseAndReturnVerifiedModule(kHloTextRef));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(ref_module),
                                      std::move(module_and_metadata.module),
                                      ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4},
                                      /*run_hlo_passes=*/false));
}

TEST_F(TritonTest, UseTF32For8BitOrLessWithF32) {
  constexpr absl::string_view kHloText = R"(
HloModule t

triton_dot {
  parameter_0 = s32[11,24]{1,0} parameter(0)
  broadcast = s32[11,24,128]{2,1,0} broadcast(parameter_0),
  dimensions={0,1}
  parameter_1 = s32[11,24,128]{2,1,0} parameter(1)
  compare = pred[11,24,128]{2,1,0} compare(broadcast, parameter_1),
      direction=EQ
  bitcast = pred[264,128]{1,0} bitcast(compare)
  convert = f32[264,128]{1,0} convert(bitcast)
  parameter_2 = f32[128,8]{1,0} parameter(2)
  ROOT dot = f32[264,8]{1,0} dot(convert, parameter_2),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s32[11,24]{1,0} parameter(0)
  p1 = s32[11,24,128]{2,1,0} parameter(1)
  p2 = f32[128,8]{1,0} parameter(2)
  ROOT _ = f32[264,8] fusion(p0, p1, p2), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
      triton_gemm_config:
        {"block_m":32,"block_n":16,"block_k":128,
         "split_k":1,"num_stages":1,"num_warps":4,
         "num_ctas":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(*module_and_metadata.computation,
                                 module_and_metadata.block_level_parameters,
                                 R"(
CHECK:      tt.dot
CHECK:      inputPrecision = tf32
  )"));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module_and_metadata.module),
                               ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, S8ToF16DotWithSmallTileDoesNotCrash) {
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_dot {
  p0 = s8[33,33]{1,0} parameter(0)
  c0 = f16[33,33]{1,0} convert(p0)
  p1 = f16[33,33]{1,0} parameter(1)
  ROOT _ = f16[33,33]{1,0} dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[33,33]{1,0} parameter(0)
  p1 = f16[33,33]{1,0} parameter(1)
  ROOT _ = f16[33,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":16,
                         "split_k":1,"num_stages":2,"num_warps":2,
                         "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));
  EXPECT_TRUE(Run(std::move(module_and_metadata.module),
                  /*run_hlo_passes=*/false));
}

TEST_F(TritonGemmTest, S8ToF32DotWithManyWarpsDoesNotCrash) {
  constexpr absl::string_view kHloText = R"(
HloModule m

triton_dot {
  p0 = s8[16,65]{0,1} parameter(0)
  c0 = f32[16,65]{1,0} convert(p0)
  p1 = f32[65,128]{1,0} parameter(1)
  ROOT _ = f32[16,128]{1,0} dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[16,65]{1,0} parameter(0)
  p1 = f32[65,128]{1,0} parameter(1)
  ROOT _ = f32[16,128] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":128,"block_k":32,
                         "split_k":1,"num_stages":2,"num_warps":16,
                         "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));
  EXPECT_TRUE(Run(std::move(module_and_metadata.module),
                  /*run_hlo_passes=*/false));
}

TEST_F(TritonGemmTest, Fp8DotWithSmallTileDoesNotCrash) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP() << "Doesn't pass on pre-Hopper GPUs.";
  }

  constexpr absl::string_view kHloText = R"(
HloModule m

triton_dot {
  p0 = f8e4m3fn[33,33]{1,0} parameter(0)
  p1 = f8e4m3fn[33,33]{1,0} parameter(1)
  ROOT _ = bf16[33,33]{1,0} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f8e4m3fn[33,33]{1,0} parameter(0)
  p1 = f8e4m3fn[33,33]{1,0} parameter(1)
  ROOT _ = bf16[33,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":16,
                         "split_k":1,"num_stages":2,"num_warps":2,
                         "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));
  EXPECT_TRUE(Run(std::move(module_and_metadata.module),
                  /*run_hlo_passes=*/false));
}

TEST_F(TritonGemmTest, Fp8DotWithManyWarpsDoesNotCrash) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP() << "Doesn't pass on pre-Hopper GPUs.";
  }

  constexpr absl::string_view kHloText = R"(
HloModule m

triton_dot {
  p0 = f8e4m3fn[33,33]{1,0} parameter(0)
  p1 = f8e4m3fn[33,33]{1,0} parameter(1)
  ROOT _ = bf16[33,33]{1,0} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f8e4m3fn[33,33]{1,0} parameter(0)
  p1 = f8e4m3fn[33,33]{1,0} parameter(1)
  ROOT _ = bf16[33,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":16,"block_k":32,
                         "split_k":1,"num_stages":2,"num_warps":16,
                         "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));
  EXPECT_TRUE(Run(std::move(module_and_metadata.module),
                  /*run_hlo_passes=*/false));
}

// Test PreventMmaV3LoopUnrolling pass in order to keep compile time low.
// See b/344841434.
// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonGemmTest, TestPreventMMAV3LoopUnrolling) {
  if (GetCudaComputeCapability().major != se::CudaComputeCapability::kHopper) {
    GTEST_SKIP() << "wgmma instruction is only available on Hopper";
  }
  constexpr absl::string_view kHloText = R"(
gemm_fusion_dot {
  p0 = f16[64,1024]{1,0} parameter(0)
  p1 = f16[1024,32,32]{2,1,0} parameter(1)
  bitcast = f16[1024,1024]{0,1} bitcast(p1)
  ROOT dot = f16[64,1024]{1,0} dot(p0, bitcast),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[64,1024]{1,0} parameter(0)
  p1 = f16[1024,32,32]{2,1,0} parameter(1)
  ROOT triton_gemm_fusion_dot = f16[64,1024]{1,0} fusion(p0, p1), kind=kCustom,
    calls=gemm_fusion_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
      triton_gemm_config:
        {"block_m":64,"block_n":32,"block_k":32,
         "split_k":1,"num_stages":1,"num_warps":4,
         "num_ctas":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));

  CompileAndOptionallyVerifyPtx(std::move(module_and_metadata.module), R"(
                                R"(
CHECK: $L__BB0_1:
CHECK-NEXT: // begin inline asm
CHECK-NEXT: .pragma "nounroll";
CHECK: wgmma
)",
                                /*run_optimization_passes=*/false);
}

// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonGemmTest, WgmmaIsUsedForMemBoundShape) {
  if (GetCudaComputeCapability().major != se::CudaComputeCapability::kHopper) {
    GTEST_SKIP() << "wgmma instruction is only available on Hopper";
  }
  constexpr absl::string_view kHloText = R"(
gemm_fusion_dot {
  p0 = s8[128,128]{1,0} parameter(0)
  p1 = bf16[128,16]{1,0} parameter(1)
  convert = bf16[128,128]{1,0} convert(p0)
  ROOT %dot = bf16[128,16]{1,0} dot(convert, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[128,128]{1,0} parameter(0)
  p1 = bf16[128,16]{1,0} parameter(1)
  ROOT triton_gemm_fusion_dot = bf16[128,16]{1,0} fusion(p0, p1), kind=kCustom,
    calls=gemm_fusion_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
      triton_gemm_config:
        {"block_m":128,"block_n":16,"block_k":16,
         "split_k":1,"num_stages":1,"num_warps":4,
         "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(ModuleAndNestedFusionMetadata module_and_metadata,
                          GetModuleAndNestedFusionMetadata(kHloText));

  CompileAndOptionallyVerifyPtx(std::move(module_and_metadata.module), R"(
CHECK: wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16
)",
                                /*run_optimization_passes=*/false);
}

// Test presence of default matmul config information when the GEMM autotuner is
// not present in the compilation pipeline (which is always the case on ROCM).
//
// TODO(b/353484968): Tests that don't run RunAndCompareNoHloPasses should be
// moved to deviceless test file.
TEST_F(TritonGemmTest, TestNoAutotuner) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = f16[30,30] parameter(0)
  p1 = s8[30,30] parameter(1)
  cp1 = f16[30,30] convert(p1)
  ROOT _ = f16[30,30] dot(p0, cp1),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> verified_module,
                          ParseAndReturnVerifiedModule(kHloText));
  DebugOptions debug_options = verified_module->config().debug_options();
  debug_options.set_xla_gpu_autotune_level(0);
  verified_module->mutable_config().set_debug_options(debug_options);

  MatchOptimizedHlo(kHloText, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-SAME: __triton_nested_gemm_fusion
  )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
