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

#include "xla/service/gpu/ir_emitter_triton.h"

#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "xla/autotuning.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/xla.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

// TODO(b/317016172): Inspect usages of TritonGemmConfig and potentially update
// them to to use newly exposed parameters.

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class TritonTest : public GpuCodegenTest {
  const auto& device_desc() {
    return backend().default_stream_executor()->GetDeviceDescription();
  }

 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  const se::GpuComputeCapability& GpuComputeComp() {
    return device_desc().gpu_compute_capability();
  }

  bool SkipBF16Tests() {
    if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
      auto rcc = device_desc().rocm_compute_capability();
      return !rcc.has_bf16_dtype_support();
    }
    return false;
  }

  se::GpuComputeCapability CudaAmpereOrRocm() {
    if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
      return se::GpuComputeCapability{device_desc().rocm_compute_capability()};
    } else {
      return se::GpuComputeCapability{
          se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
    }
  }
};

class TritonGemmTest : public TritonTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = TritonTest::GetDebugOptionsForTest();
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);
    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    // Always rewrite Gemms with Triton regardless of size.
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
    return debug_options;
  }
};

class TritonGemmTestWithSplitK : public TritonGemmTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = TritonGemmTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_split_k_autotuning(true);
    return debug_options;
  }
};

class TritonGemmTestWithoutTritonGemmAny : public TritonGemmTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = TritonGemmTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_triton_gemm_any(false);
    return debug_options;
  }
};

class TritonFilecheckTest : public TritonTest {
 public:
  absl::Status CreateTritonIrAndFileCheck(absl::string_view hlo_text,
                                          const TritonGemmConfig& config,
                                          TritonIrEmitter emitter,
                                          absl::string_view triton_fusion_name,
                                          absl::string_view filecheck_pattern);
};

absl::Status TritonFilecheckTest::CreateTritonIrAndFileCheck(
    absl::string_view hlo_text, const TritonGemmConfig& config,
    TritonIrEmitter emitter, absl::string_view triton_fusion_name,
    absl::string_view filecheck_pattern) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> verified_module,
                      ParseAndReturnVerifiedModule(hlo_text));

  auto* computation =
      verified_module->GetComputationWithName(triton_fusion_name);
  TF_RET_CHECK(computation != nullptr);
  TF_ASSIGN_OR_RETURN(auto analysis,
                      TritonFusionAnalysis::Execute(*computation));

  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(
      auto module, CreateTritonModule(analysis, "triton_fn", computation,
                                      TestGpuDeviceInfo::RTXA6000DeviceInfo(),
                                      config, emitter, context));

  std::string out;
  llvm::raw_string_ostream os(out);
  module->print(os);
  TF_ASSIGN_OR_RETURN(bool succeeded, RunFileCheck(out, filecheck_pattern));
  if (!succeeded) {
    return absl::InternalError("FileCheck failed.");
  }
  return absl::OkStatus();
}

TEST_F(TritonFilecheckTest, TestGemm) {
  const std::string kHloText = R"(
HloModule t, is_scheduled=true

triton_gemm_r {
  parameter_0 = s8[80,115]{1,0} parameter(0)
  convert.3 = f32[80,115]{1,0} convert(parameter_0)
  parameter_1 = f32[137,115]{1,0} parameter(1)
  ROOT r.1 = f32[80,137]{1,0} dot(convert.3, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p1 = f32[137,115]{1,0} parameter(1)
  p0 = s8[80,115]{1,0} parameter(0)
  ROOT triton_gemm_r = f32[80,137]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm_r,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":64,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":2,
                         "num_ctas":1}}}
})";
  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul,
                                          "triton_gemm_r", R"(
CHECK:    tt.func @triton_fn(%[[LHS:.*]]: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %[[RHS:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[OUT:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK-DAG:  %[[ZERO_KN:.*]] = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
CHECK-DAG:  %[[ZERO_MK:.*]] = arith.constant dense<0.000000e+00> : tensor<16x32xf32>
CHECK-DAG:  %[[ZERO_MN:.*]] = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
CHECK-DAG:  %[[SIZE_K:.*]] = arith.constant 115 : i32
CHECK-DAG:  %[[SIZE_M:.*]] = arith.constant 137 : i64
CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : i64
CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : i32
CHECK-DAG:  %[[C80:.*]] = arith.constant 80 : i64
CHECK-DAG:  %[[TILE_SIZE_K:.*]] = arith.constant 32 : i32
CHECK-DAG:  %[[TILE_SIZE_N:.*]] = arith.constant 64 : i32
CHECK-DAG:  %[[TILE_SIZE_M:.*]] = arith.constant 16 : i32
CHECK-DAG:  %[[NUM_TILES_M:.*]] = arith.constant 5 : i32
CHECK-DAG:  %[[GROUP_M:.*]] = arith.constant 8 : i32
CHECK-DAG:  %[[WIDTH:.*]] = arith.constant 24 : i32
CHECK:      %[[PID_NC:.*]] = tt.get_program_id x
CHECK:      %[[GROUP_ID:.*]] = arith.divsi %[[PID_NC]], %[[WIDTH]]
CHECK:      %[[FIRST_PID_M:.*]] = arith.muli %[[GROUP_ID]], %[[GROUP_M]]
CHECK:      %[[MAX_M:.*]] = arith.subi %[[NUM_TILES_M]], %[[FIRST_PID_M]]
CHECK:      %[[CMP:.*]] = arith.cmpi slt, %[[MAX_M]], %[[GROUP_M]]
CHECK:      %[[GROUP_SIZE:.*]] = arith.select %[[CMP]], %[[MAX_M]], %[[GROUP_M]]
CHECK:      %[[PID_M:.*]] = arith.remsi %[[PID_NC]], %[[GROUP_SIZE]]
CHECK:      %[[TILE_INDEX_M:.*]] = arith.addi %[[FIRST_PID_M]], %[[PID_M]] : i32
CHECK:      %[[TMP:.*]] = arith.remsi %[[PID_NC]], %[[WIDTH]] : i32
CHECK:      %[[TILE_INDEX_N:.*]] = arith.divsi %[[TMP]], %[[GROUP_SIZE]] : i32
CHECK:      %[[TILE_OFFSET_M_LHS:.*]] = arith.muli %[[TILE_INDEX_M]], %[[TILE_SIZE_M]]
CHECK:      %[[LHS_PTR:.*]] = tt.make_tensor_ptr %[[LHS]]
CHECK:      %[[LHS_TILE_PTR:.*]] = tt.advance %[[LHS_PTR]], [%[[TILE_OFFSET_M_LHS]], %[[C0]]]
CHECK:      %[[TILE_OFFSET_N_RHS:.*]] = arith.muli %[[TILE_INDEX_N]], %[[TILE_SIZE_N]]
CHECK:      %[[RHS_PTR:.*]] = tt.make_tensor_ptr %[[RHS]]
CHECK:      %[[RHS_TILE_PTR:.*]] = tt.advance %[[RHS_PTR]], [%[[C0]], %[[TILE_OFFSET_N_RHS]]]
CHECK:        %[[FOR:.*]]:3 = scf.for %[[BLOCK_K:.*]] = %[[C0]] to %[[SIZE_K]] step %[[TILE_SIZE_K]]
CHECK-SAME:       iter_args(%[[LHS_ITER_PTR:.*]] = %[[LHS_TILE_PTR]], %[[RHS_ITER_PTR:.*]] = %[[RHS_TILE_PTR]], %[[ACC:.*]] = %[[ZERO_MN]])
CHECK:        %[[LHS_TILE:.*]] = tt.load %[[LHS_ITER_PTR]] {boundaryCheck = array<i32: 1>
CHECK:        %[[LHS_ITER_PTR_NEXT:.*]] = tt.advance %[[LHS_ITER_PTR]], [%[[C0]], %[[TILE_SIZE_K]]]
CHECK:        %[[RHS_TILE:.*]] = tt.load %[[RHS_ITER_PTR]] {boundaryCheck = array<i32: 0, 1>
CHECK:        %[[RHS_ITER_PTR_NEXT:.*]] = tt.advance %[[RHS_ITER_PTR]], [%[[TILE_SIZE_K]], %[[C0]]]
CHECK:        %[[CONVERTED:.*]] = arith.sitofp %[[LHS_TILE]] : tensor<16x32xi8> to tensor<16x32xf32>
CHECK:        %[[TILE_K_LIMIT:.*]] = arith.subi %[[SIZE_K]], %[[BLOCK_K]] : i32
CHECK:        %[[K_TILE_IOTA:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
CHECK:        %[[K_OFFSETS_1K:.*]] = tt.expand_dims %[[K_TILE_IOTA]] {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
CHECK:        %[[TILE_K_LIMIT_1K:.*]] = tt.splat %[[TILE_K_LIMIT]] : i32 -> tensor<1x32xi32>
CHECK:        %[[LHS_INBOUNDS_1K:.*]] = arith.cmpi slt, %[[K_OFFSETS_1K]], %[[TILE_K_LIMIT_1K]] : tensor<1x32xi32>
CHECK:        %[[LHS_INBOUNDS_MK:.*]] = tt.broadcast %[[LHS_INBOUNDS_1K]] : tensor<1x32xi1> -> tensor<16x32xi1>
CHECK:        %[[LHS_MASKED:.*]] = arith.select %[[LHS_INBOUNDS_MK]], %[[CONVERTED]], %[[ZERO_MK]]
CHECK:        %[[K_OFFSETS_K1:.*]] = tt.expand_dims %[[K_TILE_IOTA]] {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
CHECK:        %[[TILE_K_LIMIT_K1:.*]] = tt.splat %[[TILE_K_LIMIT]] : i32 -> tensor<32x1xi32>
CHECK:        %[[RHS_INBOUNDS_K1:.*]] = arith.cmpi slt, %[[K_OFFSETS_K1]], %[[TILE_K_LIMIT_K1]] : tensor<32x1xi32>
CHECK:        %[[RHS_INBOUNDS_KN:.*]] = tt.broadcast %[[RHS_INBOUNDS_K1]] : tensor<32x1xi1> -> tensor<32x64xi1>
CHECK:        %[[RHS_MASKED:.*]] = arith.select %[[RHS_INBOUNDS_KN]], %[[RHS_TILE]], %[[ZERO_KN]] : tensor<32x64xi1>, tensor<32x64xf32>
CHECK:        %[[ACC_NEXT:.*]] = tt.dot %[[LHS_MASKED]], %[[RHS_MASKED]], %[[ACC]]
CHECK:        scf.yield %[[LHS_ITER_PTR_NEXT]], %[[RHS_ITER_PTR_NEXT]], %[[ACC_NEXT]] : !tt.ptr<tensor<16x32xi8>>, !tt.ptr<tensor<32x64xf32>>, tensor<16x64xf32>
CHECK:      }
CHECK:      %[[OUT_PTR:.*]] = tt.make_tensor_ptr %[[OUT]], [%[[C80]], %[[SIZE_M]]], [%[[SIZE_M]], %[[C1]]], [%[[C0]], %[[C0]]] {order = array<i32: 1, 0>} : <tensor<16x64xf32>>
CHECK:      %[[OUT_OFFSET:.*]] = tt.advance %[[OUT_PTR]], [%[[TILE_OFFSET_M_LHS]], %[[TILE_OFFSET_N_RHS]]] : <tensor<16x64xf32>>
CHECK:      tt.store %[[OUT_OFFSET]], %[[FOR]]#2 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<16x64xf32>>
CHECK:      tt.return
CHECK:    }
)"));
}

TEST_F(TritonFilecheckTest, TestGemmWithTrivialNonContractingDimension) {
  const std::string kHloText = R"(
HloModule t, is_scheduled=true

triton_dot {
  param_0.1 = f32[137,115]{1,0} parameter(0)
  param_1.1 = f32[1,115]{1,0} parameter(1)
  ROOT dot = f32[137,1]{1,0} dot(param_0.1, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f32[137,115]{1,0} parameter(0)
  p1 = f32[1,115]{1,0} parameter(1)
  ROOT custom-call = f32[137,1]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":64,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":2,
                         "num_ctas":1}}}
})";

  TritonGemmConfig config(16, 16, 32, 1, 1, 1);
  EXPECT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK:    tt.func @triton_fn(%[[LHS:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[RHS:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[OUT:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK-DAG:  %[[ZERO_KN:.*]] = arith.constant dense<0.000000e+00> : tensor<32x16xf32>
CHECK-DAG:  %[[ZERO_MK:.*]] = arith.constant dense<0.000000e+00> : tensor<16x32xf32>
CHECK-DAG:  %[[ZERO_MN:.*]] = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
CHECK-DAG:  %[[SIZE_K:.*]] = arith.constant 115 : i32
CHECK-DAG:  %[[SIZE_M:.*]] = arith.constant 137 : i64
CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : i64
CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : i32
CHECK-DAG:  %[[C115:.*]] = arith.constant 115 : i64
CHECK-DAG:  %[[TILE_SIZE_K:.*]] = arith.constant 32 : i32
CHECK-DAG:  %[[TILE_SIZE_M:.*]] = arith.constant 16 : i32
CHECK-DAG:  %[[C8:.*]] = arith.constant 8 : i32
CHECK-DAG:  %[[NUM_TILES_M:.*]] = arith.constant 9 : i32
CHECK:    %[[PID_NC:.*]] = tt.get_program_id x : i32
CHECK:    %[[GROUP_ID:.*]] = arith.divsi %[[PID_NC]], %[[C8]]
CHECK:    %[[FIRST_PID_M:.*]] = arith.muli %[[GROUP_ID]], %[[C8]]
CHECK:    %[[MAX_M:.*]] = arith.subi %[[NUM_TILES_M]], %[[FIRST_PID_M]]
CHECK:    %[[CMP:.*]] = arith.cmpi slt, %[[MAX_M]], %[[C8]]
CHECK:    %[[GROUP_SIZE:.*]] = arith.select %[[CMP]], %[[MAX_M]], %[[C8]]
CHECK:    %[[PID_M:.*]] = arith.remsi %[[PID_NC]], %[[GROUP_SIZE]]
CHECK:    %[[TILE_INDEX_M:.*]] = arith.addi %[[FIRST_PID_M]], %[[PID_M]]
CHECK:    %[[TMP:.*]] = arith.remsi %[[PID_NC]], %[[C8]]
CHECK:    %[[TILE_INDEX_N:.*]] = arith.divsi %[[TMP]], %[[GROUP_SIZE]]
CHECK:    %[[TILE_OFFSET_M_LHS:.*]] = arith.muli %[[TILE_INDEX_M]], %[[TILE_SIZE_M]]
CHECK:    %[[LHS_PTR:.*]] = tt.make_tensor_ptr %[[LHS]]
CHECK:    %[[LHS_TILE_PTR:.*]] = tt.advance %[[LHS_PTR]], [%[[TILE_OFFSET_M_LHS]], %[[C0]]]
CHECK:    %[[TILE_OFFSET_N_RHS:.*]] = arith.muli %[[TILE_INDEX_N]], %[[TILE_SIZE_M]]
CHECK:    %[[RHS_PTR:.*]] = tt.make_tensor_ptr %[[RHS]]
CHECK:    %[[RHS_TILE_PTR:.*]] = tt.advance %[[RHS_PTR]], [%[[C0]], %[[TILE_OFFSET_N_RHS]]]
CHECK:    %[[FOR:.*]]:3 = scf.for %[[BLOCK_K:.*]] = %[[C0]] to %[[SIZE_K]] step %[[TILE_SIZE_K]]
CHECK-SAME:       iter_args(%[[LHS_ITER_PTR:.*]] = %[[LHS_TILE_PTR]], %[[RHS_ITER_PTR:.*]] = %[[RHS_TILE_PTR]], %[[ACC:.*]] = %[[ZERO_MN]])
CHECK:      %[[LHS_TILE:.*]] = tt.load %[[LHS_ITER_PTR]] {boundaryCheck = array<i32: 0, 1>
CHECK:      %[[LHS_ITER_PTR_NEXT:.*]] = tt.advance %[[LHS_ITER_PTR]], [%[[C0]], %[[TILE_SIZE_K]]]
CHECK:      %[[RHS_TILE:.*]] = tt.load %[[RHS_ITER_PTR]] {boundaryCheck = array<i32: 0, 1>
CHECK:      %[[RHS_ITER_PTR_NEXT:.*]] = tt.advance %[[RHS_ITER_PTR]], [%[[TILE_SIZE_K]], %[[C0]]]
CHECK:      %[[TILE_K_LIMIT:.*]] = arith.subi %[[SIZE_K]], %[[BLOCK_K]] : i32
CHECK:      %[[K_TILE_IOTA:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
CHECK:      %[[K_OFFSETS_1K:.*]] = tt.expand_dims %[[K_TILE_IOTA]] {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
CHECK:      %[[TILE_K_LIMIT_1K:.*]] = tt.splat %[[TILE_K_LIMIT]] : i32 -> tensor<1x32xi32>
CHECK:      %[[LHS_INBOUNDS_1K:.*]] = arith.cmpi slt, %[[K_OFFSETS_1K]], %[[TILE_K_LIMIT_1K]] : tensor<1x32xi32>
CHECK:      %[[LHS_INBOUNDS_MK:.*]] = tt.broadcast %[[LHS_INBOUNDS_1K]] : tensor<1x32xi1> -> tensor<16x32xi1>
CHECK:      %[[LHS_MASKED:.*]] = arith.select %[[LHS_INBOUNDS_MK]], %[[LHS_TILE]], %[[ZERO_MK]]
CHECK:      %[[K_OFFSETS_K1:.*]] = tt.expand_dims %[[K_TILE_IOTA]] {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
CHECK:      %[[TILE_K_LIMIT_K1:.*]] = tt.splat %[[TILE_K_LIMIT]] : i32 -> tensor<32x1xi32>
CHECK:      %[[RHS_INBOUNDS_K1:.*]] = arith.cmpi slt, %[[K_OFFSETS_K1]], %[[TILE_K_LIMIT_K1]] : tensor<32x1xi32>
CHECK:      %[[RHS_INBOUNDS_KN:.*]] = tt.broadcast %[[RHS_INBOUNDS_K1]] : tensor<32x1xi1> -> tensor<32x16xi1>
CHECK:      %[[RHS_MASKED:.*]] = arith.select %[[RHS_INBOUNDS_KN]], %[[RHS_TILE]], %[[ZERO_KN]] : tensor<32x16xi1>, tensor<32x16xf32>
CHECK:      %[[ACC_NEXT:.*]] = tt.dot %[[LHS_MASKED]], %[[RHS_MASKED]], %[[ACC]]
CHECK:      scf.yield %[[LHS_ITER_PTR_NEXT]], %[[RHS_ITER_PTR_NEXT]], %[[ACC_NEXT]] : !tt.ptr<tensor<16x32xf32>>, !tt.ptr<tensor<32x16xf32>>, tensor<16x16xf32>
CHECK:    }

CHECK:    %[[OUT_PTR:.*]] = tt.make_tensor_ptr %[[OUT]], [%[[SIZE_M]], %[[C1]]], [%[[C1]], %[[C1]]], [%[[C0]], %[[C0]]] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
CHECK:    %[[OUT_OFFSET:.*]] = tt.advance %[[OUT_PTR]], [%[[TILE_OFFSET_M_LHS]], %[[TILE_OFFSET_N_RHS]]] : <tensor<16x16xf32>>
CHECK:    tt.store %[[OUT_OFFSET]], %[[FOR]]#2 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
CHECK:    tt.return
CHECK:  }
)"));
}

TEST_F(TritonFilecheckTest, TestSoftmaxEmitterWithSingleParameter) {
  const std::string kHloText = R"(
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
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
})";
  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitSoftMax,
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 127)>
CHECK:        tt.func @triton_fn(%[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK:            %[[PID:.*]] = tt.get_program_id x : i32
CHECK:            arith.index_castui %[[PID]] : i32 to index
CHECK:            tt.addptr %[[P0]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG2:[^:]*]]: f32, %[[ARG3:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG2]], %[[ARG3]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<128xf32>) -> f32
CHECK:            tt.splat
CHECK:            arith.mulf
CHECK-SAME:       tensor<128xf32>
CHECK:            tt.addptr %[[P1]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.store
CHECK-SAME:       {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.return
CHECK:        }
)"));
}

TEST_F(TritonFilecheckTest, TestSoftmaxEmitterWithSingleScalarParameter) {
  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  parameter_0 = f32[] parameter(0)
  broadcast_1 = f32[125,127]{1,0} broadcast(parameter_0), dimensions={}
  multiply_0 = f32[125,127]{1,0} multiply(broadcast_1, broadcast_1)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[] constant(42)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
})";
  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitSoftMax,
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 127)>
CHECK:        tt.func @triton_fn(%[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK-DAG:            %[[PID:.*]] = tt.get_program_id x : i32
CHECK-DAG:            arith.index_castui %[[PID]] : i32 to index
CHECK-DAG:            %[[ZERO_OFFSET:.*]] = arith.constant 0 : i64
CHECK-DAG:            %[[ARG_0:.*]] = tt.addptr %[[P0]], %[[ZERO_OFFSET]] : !tt.ptr<f32>, i64
CHECK:            tt.load %[[ARG_0]] : !tt.ptr<f32>
CHECK-NEXT:       tt.splat
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG2:[^:]*]]: f32, %[[ARG3:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG2]], %[[ARG3]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<128xf32>) -> f32
CHECK:            tt.splat
CHECK:            arith.mulf
CHECK-SAME:       tensor<128xf32>
CHECK:            tt.addptr %[[P1]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.store
CHECK-SAME:       {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.return
CHECK:        }
)"));
}

TEST_F(TritonFilecheckTest, TestSoftmaxEmitterWithMultipleParameters) {
  const std::string kHloText = R"(
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
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0, param_1), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
}
)";
  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitSoftMax,
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 127)>
CHECK:        tt.func @triton_fn(%[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P2:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK-DAG:        %[[PID:.*]] = tt.get_program_id x : i32
CHECK-DAG:        %[[PID_INDEX:.*]] = arith.index_castui %[[PID]] : i32 to index
CHECK-DAG:        %[[C127_i64:.*]] = arith.constant 127 : i64
CHECK-DAG:        %[[ZERO_OFFSET:.*]] = arith.constant 0 : i64
CHECK:            %[[ROW_OFFSET_INDEX:.*]] = affine.apply #[[MAP]]()[%[[PID_INDEX]]]
CHECK:            %[[ROW_OFFSET:.*]] = arith.index_castui %[[ROW_OFFSET_INDEX]] : index to i64
CHECK:            %[[ARG0:.*]] = tt.addptr %[[P0]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<128xf32>>
CHECK:            %[[ARG1:.*]] = tt.addptr %[[P1]], %[[ZERO_OFFSET]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG3:[^:]*]]: f32, %[[ARG4:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG3]], %[[ARG4]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<128xf32>) -> f32
CHECK:            tt.addptr %[[P2]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.store
CHECK-SAME:       {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.return
CHECK:        }
)"));
}

TEST_F(TritonFilecheckTest,
       TestSoftmaxEmitterWithMultipleParametersOrderSwapped) {
  // This mirrors the multiple parameter test above, but with the parameter to
  // be batch-broadcasted in the parameter_0 place instead of parameter_1.
  const std::string kHloText = R"(
HloModule t

add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  param_0 = f32[125,127]{1,0} parameter(1)
  param_1 = f32[127]{0} parameter(0)
  broadcast_0 = f32[125,127]{1,0} broadcast(param_1), dimensions={1}
  multiply_0 = f32[125,127]{1,0} multiply(param_0, broadcast_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(1)
  param_1 = f32[127]{0} parameter(0)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_1, param_0), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
}
)";
  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitSoftMax,
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 127)>
CHECK:        tt.func @triton_fn(%[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P2:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK-DAG:        %[[PID:.*]] = tt.get_program_id x : i32
CHECK-DAG:        %[[PID_INDEX:.*]] = arith.index_castui %[[PID]] : i32 to index
CHECK-DAG:        %[[C127_i64:.*]] = arith.constant 127 : i64
CHECK-DAG:        %[[ZERO_OFFSET:.*]] = arith.constant 0 : i64
CHECK:            %[[ROW_OFFSET_INDEX:.*]] = affine.apply #[[MAP]]()[%[[PID_INDEX]]]
CHECK:            %[[ROW_OFFSET:.*]] = arith.index_castui %[[ROW_OFFSET_INDEX]] : index to i64
CHECK:            %[[ARG1:.*]] = tt.addptr %[[P1]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<128xf32>>
CHECK:            %[[ARG0:.*]] = tt.addptr %[[P0]], %[[ZERO_OFFSET]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG3:[^:]*]]: f32, %[[ARG4:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG3]], %[[ARG4]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<128xf32>) -> f32
CHECK:            tt.splat
CHECK:            tt.addptr %[[P2]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.store
CHECK-SAME:       {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.return
CHECK:        }
)"));
}

TEST_F(TritonFilecheckTest,
       TestSoftmaxEmitterWithAdditionalParameterEnteringAfterDiamond) {
  const std::string kHloText = R"(
HloModule t

add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  param_0 = f32[125,127]{1,0} parameter(0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(param_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  param_1 = f32[127]{0} parameter(1)
  broadcast_0 = f32[125,127]{1,0} broadcast(param_1), dimensions={1}
  ROOT multiply_0 = f32[125,127]{1,0} multiply(broadcast_4, broadcast_0)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[127]{0} parameter(1)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0, param_1), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
}
)";
  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitSoftMax,
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 127)>
CHECK:        tt.func @triton_fn(%[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P2:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK-DAG:        %[[PID:.*]] = tt.get_program_id x : i32
CHECK-DAG:        %[[PID_INDEX:.*]] = arith.index_castui %[[PID]] : i32 to index
CHECK-DAG:        %[[C127_i64:.*]] = arith.constant 127 : i64
CHECK-DAG:        %[[ZERO_OFFSET:.*]] = arith.constant 0 : i64
CHECK:            %[[ROW_OFFSET_INDEX:.*]] = affine.apply #[[MAP]]()[%[[PID_INDEX]]]
CHECK:            %[[ROW_OFFSET:.*]] = arith.index_castui %[[ROW_OFFSET_INDEX]] : index to i64
CHECK:            %[[ARG0:.*]] = tt.addptr %[[P0]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG3:[^:]*]]: f32, %[[ARG4:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG3]], %[[ARG4]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<128xf32>) -> f32
CHECK:            %[[ARG1:.*]] = tt.addptr %[[P1]], %[[ZERO_OFFSET]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.addptr %[[P2]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.store
CHECK-SAME:       {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.return
CHECK:        }
)"));
}

TEST_F(TritonFilecheckTest,
       TestSoftmaxEmitterWithMultipleParametersAlongTiledDimension) {
  const std::string kHloText = R"(
HloModule t

add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[127]{0} parameter(1)
  param_2 = f32[125]{0} parameter(2)
  broadcast_0 = f32[125,127]{1,0} broadcast(param_1), dimensions={1}
  multiply_0 = f32[125,127]{1,0} multiply(param_0, broadcast_0)
  broadcast_1 = f32[125,127]{1,0} broadcast(param_2), dimensions={0}
  multiply_1 = f32[125,127]{1,0} multiply(multiply_0, broadcast_1)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_1, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_1, broadcast_4)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(1)
  param_1 = f32[127]{0} parameter(0)
  param_2 = f32[125]{0} parameter(2)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(param_0, param_1, param_2), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
}
)";
  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitSoftMax,
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 127)>
CHECK:        tt.func @triton_fn(%[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P2:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P3:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK-DAG:        %[[C127_i64:.*]] = arith.constant 127 : i64
CHECK-DAG:        %[[ZERO_OFFSET:.*]] = arith.constant 0 : i64
CHECK-DAG:        %[[PID:.*]] = tt.get_program_id x : i32
CHECK-DAG:        %[[PID_INDEX:.*]] = arith.index_castui %[[PID]] : i32 to index
CHECK:            %[[ROW_OFFSET_INDEX:.*]] = affine.apply #[[MAP]]()[%[[PID_INDEX]]]
CHECK:            %[[ROW_OFFSET:.*]] = arith.index_castui %[[ROW_OFFSET_INDEX]] : index to i64
CHECK:            %[[ARG0:.*]] = tt.addptr %[[P0]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<128xf32>>
CHECK:            %[[ARG1:.*]] = tt.addptr %[[P1]], %[[ZERO_OFFSET]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<128xf32>>
CHECK:            %[[PID_i64:.*]] = arith.index_castui %[[PID_INDEX]] : index to i64
CHECK:            %[[ARG2:.*]] = tt.addptr %[[P2]], %[[PID_i64]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.load %[[ARG2]] : !tt.ptr<f32>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG4:[^:]*]]: f32, %[[ARG5:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG4]], %[[ARG5]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<128xf32>) -> f32
CHECK:            tt.splat
CHECK:            arith.mulf
CHECK-SAME:       tensor<128xf32>
CHECK:            tt.addptr %[[P3]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.store
CHECK-SAME:       {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.return
CHECK:        }
)"));
}

TEST_F(TritonFilecheckTest, TestSoftmaxEmitterWithMultipleTiledDimensions) {
  const std::string kHloText = R"(
HloModule t

add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  param_0 = f32[10,125,127]{2,1,0} parameter(0)
  param_1 = f32[127]{0} parameter(1)
  param_2 = f32[10,125]{1,0} parameter(2)
  broadcast_0 = f32[10,125,127]{2,1,0} broadcast(param_1), dimensions={2}
  multiply_0 = f32[10,125,127]{2,1,0} multiply(param_0, broadcast_0)
  broadcast_1 = f32[10,125,127]{2,1,0} broadcast(param_2), dimensions={0,1}
  multiply_1 = f32[10,125,127]{2,1,0} multiply(multiply_0, broadcast_1)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[10,125]{1,0} reduce(multiply_1, constant_0), dimensions={2}, to_apply=add
  broadcast_4 = f32[10,125,127]{2,1,0} broadcast(reduce_0), dimensions={0,1}
  ROOT multiply = f32[10,125,127]{2,1,0} multiply(multiply_1, broadcast_4)
}

ENTRY main {
  param_0 = f32[10,125,127]{2,1,0} parameter(0)
  param_1 = f32[127]{0} parameter(1)
  param_2 = f32[10,125]{1,0} parameter(2)
  ROOT triton_softmax = f32[10,125,127]{2,1,0} fusion(param_0, param_1, param_2), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton_softmax"}}
}
)";
  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitSoftMax,
                                          "triton_softmax_computation", R"(
CHECK:        #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 127)>
CHECK:        tt.func @triton_fn(%[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P2:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[P3:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK-DAG:        %[[PID:.*]] = tt.get_program_id x : i32
CHECK-DAG:        %[[PID_INDEX:.*]] = arith.index_castui %[[PID]] : i32 to index
CHECK-DAG:        %[[C127_i64:.*]] = arith.constant 127 : i64
CHECK-DAG:        %[[ZERO_OFFSET:.*]] = arith.constant 0 : i64
CHECK-DAG:        %[[ROW_OFFSET_INDEX:.*]] = affine.apply #[[MAP]]()[%[[PID_INDEX]]]
CHECK-DAG:        %[[ROW_OFFSET:.*]] = arith.index_castui %[[ROW_OFFSET_INDEX]] : index to i64
CHECK:            %[[ARG0:.*]] = tt.addptr %[[P0]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<128xf32>>
CHECK:            %[[ARG1:.*]] = tt.addptr %[[P1]], %[[ZERO_OFFSET]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.load
CHECK-SAME:       {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<128xf32>>
CHECK:            %[[PID_i64:.*]] = arith.index_castui %[[PID_INDEX]] : index to i64
CHECK:            %[[ARG2:.*]] = tt.addptr %[[P2]], %[[PID_i64]] : !tt.ptr<f32>, i64
CHECK-NEXT:       tt.load %[[ARG2]] : !tt.ptr<f32>
CHECK:            tt.reduce
CHECK-NEXT:       ^bb0(%[[ARG4:[^:]*]]: f32, %[[ARG5:[^:]*]]: f32):
CHECK-NEXT:           %[[ADD:.*]] = arith.addf %[[ARG4]], %[[ARG5]] : f32
CHECK-NEXT:           tt.reduce.return %[[ADD]] : f32
CHECK-NEXT:       }) : (tensor<128xf32>) -> f32
CHECK:            tt.splat
CHECK:            arith.mulf
CHECK-SAME:       tensor<128xf32>
CHECK:            tt.addptr %[[P3]]
CHECK-NEXT:       tt.make_tensor_ptr
CHECK-SAME:       <tensor<128xf32>>
CHECK-NEXT:       tt.store
CHECK-SAME:       {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<128xf32>>
CHECK:            tt.return
CHECK:        }
)"));
}

TEST_F(
    TritonFilecheckTest,
    DiamondWithAdditionalDiamondParameterBroadcastedAlongReductionDimProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  const std::string kHloText = R"(
HloModule h1

max_computation {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT _ = f32[] maximum(x, y)
}

triton_softmax_computation {
  parameter_1 = f32[32]{0} parameter(1)
  broadcast_1 = f32[32,16]{1,0} broadcast(parameter_1), dimensions={0}
  parameter_0 = f32[32,16]{1,0} parameter(0)
  add_0 = f32[32,16]{1,0} add(broadcast_1, parameter_0)
  c = f32[] constant(0)
  reduce_0 = f32[32]{0} reduce(parameter_0, c), dimensions={1}, to_apply=max_computation
  broadcast_0 = f32[32,16]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT _ = f32[32,16]{1,0} add(add_0, broadcast_0)
}

ENTRY main {
  parameter_1 = f32[32]{0} parameter(1)
  parameter_0 = f32[32,16]{1,0} parameter(0)
  ROOT _ = f32[32,16]{1,0} fusion(parameter_0, parameter_1), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config":{"kind":"__triton_softmax"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  TF_ASSERT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitSoftMax,
                                          "triton_softmax_computation", R"(
CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 16)>
CHECK-LABEL:   tt.func @triton_fn(
CHECK-SAME:        %[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
CHECK-SAME:        %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
CHECK-SAME:        %[[P2:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK-DAG:       %[[ZERO_OFFSET:.*]] = arith.constant 0 : i32
CHECK-DAG:       %[[C1_i64:.*]] = arith.constant 1 : i64
CHECK-DAG:       %[[C16_i64:.*]] = arith.constant 16 : i64
CHECK-DAG:       %[[PID:.*]] = tt.get_program_id x : i32
CHECK:           %[[PID_INDEX:.*]] = arith.index_castui %[[PID]] : i32 to index
CHECK:           %[[PID_i64:.*]] = arith.index_castui %[[PID_INDEX]] : index to i64
CHECK:           tt.addptr %[[P1]], %[[PID_i64]] : !tt.ptr<f32>, i64
CHECK:           tt.splat
CHECK:           %[[ROW_OFFSET_INDEX:.*]] = affine.apply #[[MAP]]()[%[[PID_INDEX]]]
CHECK:           %[[ROW_OFFSET:.*]] = arith.index_castui %[[ROW_OFFSET_INDEX]] : index to i64
CHECK:           tt.addptr %[[P0]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
CHECK:           tt.make_tensor_ptr
CHECK-SAME:      <tensor<16xf32>>
CHECK:           tt.load
CHECK:           tt.reduce
CHECK-NEXT:      ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
CHECK:             %[[MAX:.*]] = arith.maximumf %[[ARG3]], %[[ARG4]] : f32
CHECK:             tt.reduce.return %[[MAX]] : f32
CHECK:           }) : (tensor<16xf32>) -> f32
CHECK:           tt.addptr %[[P2]]
CHECK:           tt.make_tensor_ptr
CHECK-SAME:      tensor<16xf32>
CHECK:           tt.store
CHECK-SAME:      !tt.ptr<tensor<16xf32>>
)"));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/0,
                                                /*arel=*/0}));
}

TEST_F(TritonFilecheckTest, NestedReducerFusionGetsCodegenedCorrectly) {
  // TODO(b/327336797): remove filter once V100 codegen in Triton is removed.
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Doesn't pass on pre-Ampere GPUs.";
  }
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }

  const std::string kHloText = R"(
HloModule softmax

fused_convert {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  convert0 = bf16[] convert(p0)
  convert1 = bf16[] convert(p1)
  add = bf16[] add(convert0, convert1)
  ROOT output = f32[] convert(add)
}

add_computation {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT fusion = f32[] fusion(p0, p1), kind=kLoop, calls=fused_convert
}

triton_softmax_computation {
  p0 = pred[10,128]{1,0} parameter(0)
  p0_f32 = f32[10,128]{1,0} convert(p0)
  zero = f32[] constant(0)
  reduce = f32[10]{0} reduce(p0_f32, zero), dimensions={1}, to_apply=add_computation
  broadcast = f32[10,128]{1,0} broadcast(reduce), dimensions={0}
  ROOT add = f32[10,128]{1,0} add(p0_f32, broadcast)
}

ENTRY main {
  p0 = pred[10,128]{1,0} parameter(0)
  ROOT softmax = f32[10,128] fusion(p0), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config":{"kind":"__triton_softmax"}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/0,
                                                           /*arel=*/0}));
}

TEST_F(
    TritonFilecheckTest,
    DiamondWithAdditionalDiamondParameterBroadcastedAlongBatchDimProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  const std::string kHloText = R"(
HloModule h1

max_computation {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT _ = f32[] maximum(x, y)
}

triton_softmax_computation {
  parameter_1 = f32[32]{0} parameter(1)
  broadcast_1 = f32[16,32]{1,0} broadcast(parameter_1), dimensions={1}
  parameter_0 = f32[16,32]{1,0} parameter(0)
  add_0 = f32[16,32]{1,0} add(broadcast_1, parameter_0)
  c = f32[] constant(0)
  reduce_0 = f32[16]{0} reduce(parameter_0, c), dimensions={1}, to_apply=max_computation
  broadcast_0 = f32[16,32]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT _ = f32[16,32]{1,0} add(add_0, broadcast_0)
}

ENTRY main {
  parameter_0 = f32[16,32]{1,0} parameter(0)
  parameter_1 = f32[32]{0} parameter(1)
  ROOT _ = f32[16,32]{1,0} fusion(parameter_0,parameter_1), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config":{"kind":"__triton_softmax"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  TF_ASSERT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitSoftMax,
                                          "triton_softmax_computation", R"(
CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 32)>
CHECK-LABEL:   tt.func @triton_fn(
CHECK-SAME:        %[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
CHECK-SAME:        %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
CHECK-SAME:        %[[P2:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
CHECK-DAG:       %[[ZERO_OFFSET:.*]] = arith.constant 0 : i32
CHECK-DAG:       %[[C0_i64:.*]] = arith.constant 0 : i64
CHECK-DAG:       %[[C1_i64:.*]] = arith.constant 1 : i64
CHECK-DAG:       %[[PID:.*]] = tt.get_program_id x : i32
CHECK:           %[[PID_INDEX:.*]] = arith.index_castui %[[PID]] : i32 to index
CHECK:           tt.addptr %[[P1]], %[[C0_i64]] : !tt.ptr<f32>, i64
CHECK:           tt.make_tensor_ptr
CHECK-SAME:      <tensor<32xf32>>
CHECK:           tt.load
CHECK-SAME:      !tt.ptr<tensor<32xf32>>
CHECK:           %[[ROW_OFFSET_INDEX:.*]] = affine.apply #[[MAP]]()[%[[PID_INDEX]]]
CHECK:           %[[ROW_OFFSET:.*]] = arith.index_castui %[[ROW_OFFSET_INDEX]] : index to i64
CHECK:           tt.addptr %[[P0]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
CHECK-NEXT:      tt.make_tensor_ptr
CHECK-SAME:      <tensor<32xf32>>
CHECK:           tt.load
CHECK-SAME:      !tt.ptr<tensor<32xf32>>
CHECK:           tt.reduce
CHECK-NEXT:      ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
CHECK:             %[[MAX:.*]] = arith.maximumf %[[ARG3]], %[[ARG4]] : f32
CHECK:             tt.reduce.return %[[MAX]] : f32
CHECK:           }) : (tensor<32xf32>) -> f32
CHECK:           tt.addptr %[[P2]]
CHECK:           tt.make_tensor_ptr
CHECK-SAME:      <tensor<32xf32>>
CHECK:           tt.store
CHECK-SAME:      !tt.ptr<tensor<32xf32>>
)"));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(
    TritonFilecheckTest,
    DiamondWithAdditionalSplatDiamondScalarParameterProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  const std::string kHloText = R"(
HloModule h1

max_computation {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT _ = f32[] maximum(x,y)
}

triton_softmax_computation {
  parameter_1 = f32[] parameter(1)
  broadcast_1 = f32[64,32,16]{2,1,0} broadcast(parameter_1), dimensions={}
  parameter_0 = f32[64,32,16]{2,1,0} parameter(0)
  add_0 = f32[64,32,16]{2,1,0} add(broadcast_1, parameter_0)
  c = f32[] constant(0)
  reduce_0 = f32[64,32]{1,0} reduce(parameter_0, c), dimensions={2}, to_apply=max_computation
  broadcast_0 = f32[64,32,16]{2,1,0} broadcast(reduce_0), dimensions={0,1}
  ROOT _ = f32[64,32,16]{2,1,0} add(add_0, broadcast_0)
}

ENTRY main {
  parameter_1 = f32[64,32,16]{2,1,0} parameter(1)
  parameter_0 = f32[] parameter(0)
  ROOT _ = f32[64,32,16]{2,1,0} fusion(parameter_1, parameter_0), kind=kCustom, calls=triton_softmax_computation, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_softmax"}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  TF_ASSERT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitSoftMax,
                                          "triton_softmax_computation", R"(
// CHECK:         #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 16)>
// CHECK-LABEL:   tt.func @triton_fn(
// CHECK-SAME:                       %[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                       %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                       %[[P2:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
// CHECK-DAG:       %[[ZERO_OFFSET_i32:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[C16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG:       %[[ZERO_OFFSET_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[PID:.*]] = tt.get_program_id x : i32
// CHECK:           %[[PID_INDEX:.*]] = arith.index_castui %[[PID]] : i32 to index
// CHECK:           tt.addptr %[[P1]], %[[ZERO_OFFSET_i64]] : !tt.ptr<f32>, i64
// CHECK-NEXT:      tt.load
// CHECK-SAME:      : !tt.ptr<f32>
// CHECK:           %[[ROW_OFFSET_INDEX:.*]] = affine.apply #[[MAP]]()[%[[PID_INDEX]]]
// CHECK:           %[[ROW_OFFSET:.*]] = arith.index_castui %[[ROW_OFFSET_INDEX]] : index to i64
// CHECK:           tt.addptr %[[P0]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
// CHECK:           tt.make_tensor_ptr
// CHECK-SAME:      <tensor<16xf32>>
// CHECK:           tt.load
// CHECK-SAME:      !tt.ptr<tensor<16xf32>>
// CHECK:           tt.reduce
// CHECK:           ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
// CHECK:             %[[MAX:.*]] = arith.maximumf %[[ARG3]], %[[ARG4]] : f32
// CHECK:             tt.reduce.return %[[MAX]] : f32
// CHECK:           }) : (tensor<16xf32>) -> f32
// CHECK:           tt.addptr %[[P2]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
// CHECK-NEXT:      tt.make_tensor_ptr
// CHECK-SAME:      <tensor<16xf32>>
// CHECK:           tt.store
// CHECK-SAME:      !tt.ptr<tensor<16xf32>>
)"));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(
    TritonFilecheckTest,
    DiamondWithAdditionalBroadcastOf1DParameterAlongNonReductionDimensionsProducesAccurateResults) {  // NOLINT(whitespace/line_length)
  const std::string kHloText = R"(
HloModule h1

max_computation {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT _ = f32[] maximum(x,y)
}

triton_softmax_computation {
  parameter_1 = f32[16]{0} parameter(1)
  broadcast_1 = f32[64,32,16]{2,1,0} broadcast(f32[16]{0} parameter_1), dimensions={2}
  parameter_0 = f32[64,32,16]{2,1,0} parameter(0)
  add_0 = f32[64,32,16]{2,1,0} add(f32[64,32,16]{2,1,0} broadcast_1, f32[64,32,16]{2,1,0} parameter_0)
  c = f32[] constant(0)
  reduce_0 = f32[64,32]{1,0} reduce(f32[64,32,16]{2,1,0} parameter_0, f32[] c), dimensions={2}, to_apply=max_computation
  broadcast_0 = f32[64,32,16]{2,1,0} broadcast(f32[64,32]{1,0} reduce_0), dimensions={0,1}
  ROOT _ = f32[64,32,16]{2,1,0} add(f32[64,32,16]{2,1,0} add_0, f32[64,32,16]{2,1,0} broadcast_0)
}

ENTRY main {
  parameter_1 = f32[64,32,16]{2,1,0} parameter(1)
  parameter_0 = f32[16]{0} parameter(0)
  ROOT _ = f32[64,32,16]{2,1,0} fusion(f32[64,32,16]{2,1,0} parameter_1, f32[16]{0} parameter_0), kind=kCustom, calls=%triton_softmax_computation, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_softmax"}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  TritonGemmConfig config(16, 64, 32, 1, 1, 1);
  ASSERT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitSoftMax,
                                       "triton_softmax_computation", R"(
// CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 16)>
// CHECK-LABEL:   tt.func @triton_fn(
// CHECK-SAME:                       %[[P0:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                       %[[P1:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                       %[[P2:[^:]*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
// CHECK-DAG:       %[[ZERO_OFFSET_i32:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[C1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[C16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG:       %[[PID:.*]] = tt.get_program_id x : i32
// CHECK:           %[[PID_INDEX:.*]] = arith.index_castui %[[PID]] : i32 to index
// CHECK:           tt.addptr %[[P1]], %[[C0_i64]] : !tt.ptr<f32>, i64
// CHECK-NEXT:      tt.make_tensor_ptr
// CHECK-SAME:      <tensor<16xf32>>
// CHECK:           tt.load
// CHECK-SAME:      !tt.ptr<tensor<16xf32>>
// CHECK:           %[[ROW_OFFSET_INDEX:.*]] = affine.apply #[[MAP]]()[%[[PID_INDEX]]]
// CHECK:           %[[ROW_OFFSET:.*]] = arith.index_castui %[[ROW_OFFSET_INDEX]] : index to i64
// CHECK:           tt.addptr %[[P0]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
// CHECK-NEXT:      tt.make_tensor_ptr
// CHECK-SAME:      <tensor<16xf32>>
// CHECK-NEXT:      tt.load
// CHECK-SAME:      !tt.ptr<tensor<16xf32>>
// CHECK:           tt.reduce
// CHECK:           ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
// CHECK:             %[[MAX:.*]] = arith.maximumf %[[ARG3]], %[[ARG4]] : f32
// CHECK:             tt.reduce.return %[[MAX]] : f32
// CHECK:           }) : (tensor<16xf32>) -> f32
// CHECK:           tt.addptr %[[P2]], %[[ROW_OFFSET]] : !tt.ptr<f32>, i64
// CHECK-NEXT:      tt.make_tensor_ptr
// CHECK-SAME:      <tensor<16xf32>>
// CHECK:           tt.store
// CHECK-SAME:      !tt.ptr<tensor<16xf32>>
)"));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(TritonFilecheckTest, PredParametersAreTruncatedToI1) {
  const std::string kHloText = R"(
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
}
)";
  TritonGemmConfig config(16, 16, 16, 1, 1, 1);
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul,
                                          "triton_gemm_computation", R"(
CHECK: %[[LOAD:.*]] = tt.load %{{.*}} {{.*}} : !tt.ptr<tensor<16x16xi8>>
CHECK: %[[TRUNCI:.*]] = arith.trunci %[[LOAD]] : tensor<16x16xi8> to tensor<16x16xi1>
CHECK: %{{.*}} = arith.andi %[[TRUNCI]], %{{.*}} : tensor<16x16xi1>
)"));
}

TEST_F(TritonFilecheckTest,
       CodegenBatchedDotWithConcatenationWithCorrectBatchStride) {
  constexpr absl::string_view kHloText = R"(
HloModule t, is_scheduled=true

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

  TritonGemmConfig config(16, 64, 32, 1, 1, 2);
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul,
                                          "triton_gemm", R"(
CHECK:   tt.func @triton_fn(%[[P0:[^:]*]]: !tt.ptr<f32>
CHECK-SAME:                 %[[P1:[^:]*]]: !tt.ptr<f32>
CHECK-SAME:                 %[[P2:[^:]*]]: !tt.ptr<f32>
CHECK-DAG: %[[ARG_PTR:.*]] = arith.select %[[CONCAT_COND:.*]], %[[P1]], %[[P2]]
CHECK-DAG: %[[BATCH_STRIDE_P1:.*]] = arith.constant 1280
CHECK-DAG: %[[BATCH_STRIDE_P2:.*]] = arith.constant 2560
CHECK-DAG: %[[BATCH_STRIDE:.*]] = arith.select %[[CONCAT_COND_2:.*]], %[[BATCH_STRIDE_P1]], %[[BATCH_STRIDE_P2]]
CHECK-DAG: %[[PID_BATCH:.*]] = tt.get_program_id y
CHECK-DAG: %[[OFFSET:.*]] = arith.muli %[[PID_BATCH]], %[[BATCH_STRIDE]]
CHECK:     %[[BLOCK_BASE_PTR:.*]] = tt.addptr %[[ARG_PTR]], %[[OFFSET]]
)"));
}

TEST_F(TritonFilecheckTest, SparseDot) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  lhs = f16[128,160] parameter(0)
  rhs = f16[320,64] parameter(1)
  meta = u16[128,20] parameter(2)
  ROOT dot = f16[128,64] dot(lhs, rhs, meta),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
}

ENTRY e {
  lhs = f16[128,160] parameter(0)
  rhs = f16[320,64] parameter(1)
  meta = u16[128,20] parameter(2)
  ROOT _ = f16[128,64] fusion(lhs, rhs, meta), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK: %[[LHS:[0-9]+]] = tt.load
CHECK: %[[RHS:[0-9]+]] = tt.load
CHECK: %[[META:[0-9]+]] = tt.load
CHECK: triton_gpu.sparse_dot %[[LHS]], %[[RHS]], %{{[^:]+}}, %[[META]] :
    )"));
}

TEST_F(TritonFilecheckTest, SparseDotWithMasking) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  lhs = f16[32,24] parameter(0)
  rhs = f16[48,32] parameter(1)
  meta = u16[32,3] parameter(2)
  ROOT dot = f16[32,32] dot(lhs, rhs, meta),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
}

ENTRY e {
  lhs = f16[32,24] parameter(0)
  rhs = f16[48,32] parameter(1)
  meta = u16[32,3] parameter(2)
  ROOT _ = f16[32,32] fusion(lhs, rhs, meta), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":64,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 64, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK-DAG: %[[C24:.+]] = arith.constant dense<24>
CHECK-DAG: %[[C48:.+]] = arith.constant dense<48>
CHECK: %[[LHS:[0-9]+]] = tt.load %{{.+}} {boundaryCheck = array<i32: 1>
CHECK: %[[RHS:[0-9]+]] = tt.load %{{.+}} {boundaryCheck = array<i32: 0>
CHECK: %[[META:[0-9]+]] = tt.load %{{.+}} {boundaryCheck = array<i32: 1>
CHECK: arith.cmpi slt, %{{.+}}, %[[C24]] :
CHECK: %[[LHS_MASKED:[0-9]+]] = arith.select %{{.+}}, %[[LHS]],
CHECK: arith.cmpi slt, %{{.+}}, %[[C48]] :
CHECK: %[[RHS_MASKED:[0-9]+]] = arith.select %{{.+}}, %[[RHS]],
CHECK: triton_gpu.sparse_dot %[[LHS_MASKED]], %[[RHS_MASKED]], %{{[^:]+}}, %[[META]] :
    )"));
}

TEST_F(TritonFilecheckTest, SparseDotBroadcastMetadata) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  lhs = f16[10,32,64] parameter(0)
  rhs = f16[10,128,256] parameter(1)
  meta_partial = u16[8] parameter(2)
  meta = u16[10,32,8] broadcast(meta_partial), dimensions={2}
  ROOT dot = f16[10,32,256] dot(lhs, rhs, meta),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}, sparsity=L.2@2:4
}

ENTRY e {
  lhs = f16[10,32,64] parameter(0)
  rhs = f16[10,128,256] parameter(1)
  meta_partial = u16[8] parameter(2)
  ROOT _ = f16[10,32,256] fusion(lhs, rhs, meta_partial), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK: %[[TWO:.+]] = arith.constant 2 : i32
CHECK: %[[LHS:[0-9]+]] = tt.load
CHECK: %[[RHS:[0-9]+]] = tt.load
CHECK: %[[T1:[0-9]+]] = tt.load %[[PTR:.+]] :
CHECK: tt.advance %[[PTR]], [%[[TWO]]]
CHECK: %[[T2:[0-9]+]] = tt.expand_dims %[[T1]]
CHECK: %[[META:[0-9]+]] = tt.broadcast %[[T2]]
CHECK: triton_gpu.sparse_dot %[[LHS]], %[[RHS]], %{{[^:]+}}, %[[META]] :
    )"));
}

TEST_F(TritonGemmTest, DoNotUseTensorCoresWithNonDefaultPrecision) {
  const std::string kHloText = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> verified_module,
                          ParseAndReturnVerifiedModule(kHloText));

  CompileAndOptionallyVerifyPtx(std::move(verified_module),
                                R"(
CHECK-NOT: mma
)");
}

TEST_F(TritonGemmTest, DebugOptionsArePropagated) {
  const std::string kHloText = R"(
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
  debug_options.set_xla_gpu_dump_llvmir(true);
  verified_module->mutable_config().set_debug_options(debug_options);

  EXPECT_TRUE(RunAndCompare(std::move(verified_module),
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));

  std::vector<std::string> paths;
  TF_EXPECT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(output_directory, "*.triton-passes.log"), &paths));
  EXPECT_EQ(paths.size(), 1);
}

TEST_F(TritonGemmTest, UseTensorCoresForF32OnAmpere) {
  const std::string kHloText = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> verified_module,
                          ParseAndReturnVerifiedModule(kHloText));

  CompileAndOptionallyVerifyPtx(std::move(verified_module),
                                R"(
CHECK: mma
)");
}

TEST_F(TritonGemmTest, FailIfTooMuchShmem) {
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "GEMM padding requirements for ROCM not included yet.";
  }
  const std::string kHloText = R"(
HloModule module, is_scheduled=true

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
    kind=kCustom, calls=triton_gemm_dot
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloText));
  const HloComputation* triton_dot_computation =
      hlo_module->entry_computation()
          ->root_instruction()
          ->fused_instructions_computation();
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  TritonGemmConfig config(16, 32, 512, 1, 4, 8);
  EXPECT_THAT(
      TritonWrapper(*TritonFusionAnalysis::Execute(*triton_dot_computation),
                    "test_fn", triton_dot_computation, CudaAmpereOrRocm(),
                    dev_info, config, &llvm_module, &EmitMatMul, mlir_context),
      tsl::testing::StatusIs(
          tsl::error::RESOURCE_EXHAUSTED,
          ::testing::HasSubstr("Shared memory size limit exceeded")));

  config.block_m = 64;
  config.block_n = 128;
  config.block_k = 128;
  config.num_stages = 1;
  TF_ASSERT_OK_AND_ASSIGN(
      const auto result,
      TritonWrapper(*TritonFusionAnalysis::Execute(*triton_dot_computation),
                    "test_fn", triton_dot_computation, CudaAmpereOrRocm(),
                    dev_info, config, &llvm_module, &EmitMatMul, mlir_context));
  // Use optin shared memory which is > shared_memory_per_block.
  EXPECT_GT(result.shmem_bytes, dev_info.shared_memory_per_block());
}

TEST_F(TritonGemmTestWithSplitK,
       WorksWhenKIsDivisibleByBlockKButNotByBlockKTimesSplitK) {
  // The condition mentioned in the test name is fulfilled by
  // GemmKey(16, 64, 256, 8, 1, 4), which was part of the default configs for
  // Ampere at the time of the addition of this test case.
  constexpr absl::string_view kHloText = R"(
HloModule extracted

ENTRY e {
  a = f16[16,5120]{1,0} parameter(0)
  b = s8[5120,10240]{1,0} parameter(1)
  converted_b = f16[5120,10240]{1,0} convert(b)
  ROOT r = f16[16,10240]{1,0} dot(a, converted_b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  // This check tests if Triton is used at all plus it runs GemmFusionAutotuner,
  // which verifies if the generated kernels can run without errors such as
  // CUDA_ERROR_ILLEGAL_ADDRESS.
  MatchOptimizedHlo(kHloText, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: "block_m":
  )");

  // Not doing a comparison here, because the input matrices are quite big.
  // If I reduce their size then they can no longer trigger the error, that I
  // want to avoid with this test case.
}

TEST_F(TritonGemmTest, MultipleDims) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  p0 = f16[1,16,17,3] parameter(0)
  p1 = s8[16,17,3] parameter(1)
  cp1 = f16[16,17,3] convert(p1)
  ROOT _ = f16[1,16,16] dot(p0, cp1),
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: "block_m":
  )");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, PredWithBF16DotProducesCorrectResult) {
  const std::string hlo_text = R"(
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

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, NoPadding) {
  const char* hlo_text = R"(
HloModule t

ENTRY e {
  p0 = f16[15,19] parameter(0)
  p1 = s8[19,17] parameter(1)
  cp1 = f16[19,17] convert(p1)
  ROOT _ = f16[15,17] dot(p0, cp1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: ROOT
; CHECK-SAME: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: "block_m":
; CHECK-NOT: pad
; CHECK-NOT: slice
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, SplitLhsNoncontractingTransposeRhs) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  p0 = pred[3,122,96,12]{3,2,1,0} parameter(0)
  cp0 = f16[3,122,96,12]{3,2,1,0} convert(p0)
  p1 = pred[1,5,122]{2,1,0} parameter(1)
  cp1 = f16[1,5,122]{2,1,0} convert(p1)
  ROOT _ = f16[3,96,12,1,5]{4,3,2,1,0} dot(cp0, cp1),
    lhs_contracting_dims={1}, rhs_contracting_dims={2}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: "block_m":
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(TritonGemmTest, SplitLhsNoncontracting) {
  const std::string hlo_text = R"(
HloModule t

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

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: "block_m":
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, SplitAndTransposeLhsExecutesCorrectly) {
  const std::string kHloText = R"(
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
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: ROOT
; CHECK-SAME: fusion
; CHECK-SAME: kind=kCustom
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, NondefaultOperandLayoutIsSupported) {
  // TODO(bchetioui): reenable when b/285866137 is fixed.
#ifndef NDEBUG
  GTEST_SKIP() << "This test times out when -UNDEBUG is set.";
#endif
  const std::string kHloText = R"(
ENTRY r {
  p1 = f16[9,140,128]{2,1,0} parameter(1)
  cp = f16[9,140,128]{2,0,1} copy(p1)
  cv = f32[9,140,128]{2,0,1} convert(cp)
  p0 = f32[9,140,123]{2,1,0} parameter(0)
  ROOT d = f32[9,128,123]{2,1,0} dot(cv, p0),
    lhs_batch_dims={0}, lhs_contracting_dims={1},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, DoNotFuseSplitRhsContractingTranspose) {
  const std::string hlo_text = R"(
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

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: ENTRY
; CHECK: transpose
; CHECK: fusion
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: "block_m":
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, DoNotFuseSplitLhsContractingTranspose) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  p0 = f16[3,16,25]{2,1,0} parameter(0)
  p0t = f16[16,3,25]{2,1,0} transpose(p0), dimensions={1,0,2}
  p0tr = f16[16,75]{1,0} reshape(p0t)
  p1 = s8[128,75]{1,0} parameter(1)
  cp1 = f16[128,75]{1,0} convert(p1)
  ROOT dot.126 = f16[16,128]{1,0} dot(p0tr, cp1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: ENTRY
; CHECK: transpose
; CHECK: fusion
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: "block_m":
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, BatchF32F16) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  x = f32[5,2,3] parameter(0)
  y = f16[5,3,4] parameter(1)
  cy = f32[5,3,4] convert(y)
  ROOT _ = f32[5,2,4] dot(x, cy),
    lhs_contracting_dims={2}, rhs_contracting_dims={1},
    lhs_batch_dims={0}, rhs_batch_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: fusion
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: "block_m":
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-2}));
}

TEST_F(TritonGemmTest, NonMajorMostInputBatchWorksCorrectly) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  x = f32[20,50,30] parameter(0)
  y = f16[30,50,40] parameter(1)
  cy = f32[30,50,40] convert(y)
  ROOT _ = f32[50,20,40] dot(x, cy),
    lhs_contracting_dims={2}, rhs_contracting_dims={0},
    lhs_batch_dims={1}, rhs_batch_dims={1}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: fusion
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: "block_m":
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, BatchTransposeF32F16) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  x = f32[5,3,2] parameter(0)
  y = f16[5,3,4] parameter(1)
  cy = f32[5,3,4] convert(y)
  x_transposed = f32[5,2,3] transpose(x), dimensions={0, 2, 1}
  ROOT _ = f32[5,2,4] dot(x_transposed, cy),
    lhs_contracting_dims={2}, rhs_contracting_dims={1},
    lhs_batch_dims={0}, rhs_batch_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: fusion
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: "block_m":
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-2}));
}

TEST_F(TritonGemmTest, DoNotFuseArbitraryReshape) {
  const std::string hlo_text = R"(
HloModule m

ENTRY e {
  p0 = f16[5,2,3] parameter(0)
  p0c = f32[5,2,3] convert(p0)
  p1 = f32[20,3] parameter(1)
  p1r = f32[5,3,4] reshape(p1)
  ROOT dot.5 = f32[5,2,4] dot(p0c, p1r),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: ENTRY
; CHECK: f32[5,3,4]{2,1,0} bitcast
; CHECK: fusion
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: "block_m":
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(TritonGemmTest, MultipleBatchRequireSeparateTranspose) {
  const std::string kHloText = R"(
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
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(TritonGemmTest, CanCodegenNonBatchedDotWithConcatenationCorrectly) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  parameter_0 = f32[3,10]{1,0} parameter(0)
  parameter_1 = f32[10,128]{1,0} parameter(1)
  parameter_2 = f32[10,256]{1,0} parameter(2)
  concatenate = f32[10,384]{1,0} concatenate(parameter_1, parameter_2), dimensions={1}
  ROOT dot = f32[3,384]{1,0} dot(parameter_0, concatenate),
    lhs_batch_dims={}, lhs_contracting_dims={1},
    rhs_batch_dims={}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK:     ENTRY
; CHECK-NOT:   concatenate
; CHECK:       fusion
; CHECK-SAME:    kind=kCustom
)");

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, CanCodegenBatchedDotWithConcatenationCorrectly) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  parameter_0 = f32[2,3,10]{2,1,0} parameter(0)
  parameter_1 = f32[2,10,128]{2,1,0} parameter(1)
  parameter_2 = f32[2,10,256]{2,1,0} parameter(2)
  concatenate = f32[2,10,384]{2,1,0} concatenate(parameter_1, parameter_2), dimensions={2}
  ROOT dot = f32[2,3,384]{2,1,0} dot(parameter_0, concatenate),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK:     ENTRY
; CHECK-NOT:   concatenate
; CHECK:       fusion
; CHECK-SAME:    kind=kCustom
)");

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTestWithoutTritonGemmAny, SkipU8) {
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "GEMM padding requirements for ROCM not included yet.";
  }
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  p0 = f32[3,3]{1,0} parameter(0)
  p1 = u8[3,3]{1,0} parameter(1)
  c = f32[3,3]{1,0} convert(p1)
  ROOT r = f32[3,3]{1,0} dot(p0, c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: cublas
; CHECK-NOT: triton
)");
}

TEST_F(TritonGemmTestWithoutTritonGemmAny, SkipF32F32) {
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "GEMM padding requirements for ROCM not included yet.";
  }
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  p0 = f32[3,5] parameter(0)
  p1 = f32[5,7] parameter(1)
  ROOT _ = f32[3,7] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: cublas
; CHECK-NOT: triton
)");
}

// This tests the complexity heuristics in TritonWrapper.
TEST_F(TritonGemmTest, FailForTooComplexTiling) {
  const std::string kHloText = R"(
HloModule module, is_scheduled=true

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
    kind=kCustom, calls=triton_gemm_dot
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloText));
  const HloComputation* triton_dot_computation =
      hlo_module->entry_computation()
          ->root_instruction()
          ->fused_instructions_computation();
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  // Fails if the tiling is too complex.
  TritonGemmConfig config(512, 512, 32, 1, 1, 2);
  EXPECT_THAT(
      TritonWrapper(*TritonFusionAnalysis::Execute(*triton_dot_computation),
                    "test_fn", triton_dot_computation, CudaAmpereOrRocm(),
                    dev_info, config, &llvm_module, &EmitMatMul, mlir_context),
      tsl::testing::StatusIs(
          tsl::error::RESOURCE_EXHAUSTED,
          "Tiling complexity heuristic exceeded: 147456 > 9000"));

  // Succeeds if the tiling is not too complex.
  config.block_m = 32;
  config.block_n = 32;
  config.block_k = 32;
  TF_CHECK_OK(
      TritonWrapper(*TritonFusionAnalysis::Execute(*triton_dot_computation),
                    "test_fn", triton_dot_computation, CudaAmpereOrRocm(),
                    dev_info, config, &llvm_module, &EmitMatMul, mlir_context)
          .status());
}

// Triton compiler used to have an issue with reordering constants:
// https://github.com/openai/triton/issues/1864
TEST_F(TritonGemmTest, TritonCompilerDoesNotFailOnConstants) {
  TF_CHECK_OK(GetOptimizedModule(R"(
HloModule m

triton_gemm___computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  c = f32[] constant(0)
  b = f32[11,63] broadcast(c)
  ROOT _.1 = f32[92,63]{1,0} dot(parameter_0, b),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[92,11]{1,0} parameter(0)
  ROOT triton_gemm__ = f32[92,63]{1,0} fusion(p0), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
                    "triton_gemm_config":{"block_m":"16","block_n":"64",
                                          "block_k":"16","split_k":"1",
                                          "num_stages":"3","num_warps":"2",
                                          "num_ctas":"1"}}}
})")
                  .status());
}

// Normally optimized HLO should contain `copy` instead of `transpose` but
// it's also possible to get transposes by modifying the compiler's pipeline.
// The emitter just has to skip through the transpose - it's handled by the
// tiled fusion analysis.
TEST_F(TritonGemmTest, TritonEmitterCanHandleTransposes) {
  MatchOptimizedHlo(R"(
t {
  p0 = f16[55,77,111]{2,1,0} parameter(0)
  p1 = f16[111,77,99]{2,1,0} parameter(1)
  t = f16[77,99,111]{2,1,0} transpose(p1), dimensions={1,2,0}
  ROOT d = f16[77,55,99]{2,1,0} dot(p0, t),
    lhs_batch_dims={1}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY e {
  p0 = f16[55,77,111]{2,1,0} parameter(0)
  p1 = f16[111,77,99]{2,1,0} parameter(1)
  ROOT r = f16[77,55,99]{2,1,0} fusion(p0, p1), kind=kCustom,
    calls=t, backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
})",
                    // This partially optimized HLO will go through the
                    // autotuner which will run the fusion through the emitter
                    // multiple times and assign block sizes on success.
                    R"(
; CHECK: f16[77,99,111]{2,1,0} transpose
; CHECK-PTX: block_m
)");
}

TEST_F(TritonGemmTest, SingleElementTileIsHandled) {
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "Not using autotuner on ROCM yet.";
  }
  MatchOptimizedHlo(R"(
t {
  p0 = f32[2,7,3]{2,1,0} parameter(0)
  p1 = s32[2,1]{1,0} parameter(1)
  c = s32[] constant(1)
  br0 = s32[2,1]{1,0} broadcast(c), dimensions={}
  cmp = pred[2,1]{1,0} compare(p1, br0), direction=LT
  bc0 = pred[2]{0} bitcast(cmp)
  br1 = pred[2,1,3,3]{3,2,0,1} broadcast(bc0), dimensions={0}
  cvt = f32[2,1,3,3]{3,2,0,1} convert(br1)
  bc1 = f32[2,3,3]{2,1,0} bitcast(cvt)
  ROOT d = f32[2,7,3]{2,1,0} dot(p0, bc1),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f32[2,7,3]{2,1,0} parameter(0)
  p1 = s32[2,1]{1,0} parameter(1)
  ROOT r = f32[2,7,3]{2,1,0} fusion(p0, p1), kind=kCustom,
    calls=t, backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
})",
                    // This partially optimized HLO will go through the
                    // autotuner which will run the fusion through the emitter
                    // multiple times and assign block sizes on success.
                    R"(
; CHECK: block_m
)");
}

class TritonGemmTestAny : public TritonGemmTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = TritonGemmTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_triton_gemm_any(true);
    return debug_options;
  }
};

TEST_F(TritonGemmTestAny, DoF32F32) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  p0 = f32[3,5] parameter(0)
  p1 = f32[5,7] parameter(1)
  ROOT _ = f32[3,7] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: block_m
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTestAny, DoAddConstantToScalarAndBroadcastThat) {
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "Not using autotuner on ROCM yet.";
  }
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  p0 = f32[] parameter(0)
  p1 = f32[5,5] parameter(1)
  %constant = f32[] constant(8)
  add = add(p0, constant)
  broadcast = f32[5,5] broadcast(add), dimensions={}
  ROOT _ = f32[5,5] dot(broadcast, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: fusion({{.*}} kind=kCustom, {{.*}}block_m
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, SameInput) {
  const std::string hlo_text = R"(
HloModule m

ENTRY e {
  p0 = pred[5,5]{1,0} parameter(0)
  c = f32[5,5]{1,0} convert(p0)
  ROOT r = f32[5,5]{1,0} dot(c, c),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

  // The fusion has separate parameters for each scope.
  MatchOptimizedHlo(hlo_text, R"(
; CHECK: ENTRY
; CHECK: %[[p0:.*]] = pred[5,5]{1,0} parameter(0)
; CHECK: fusion(%[[p0]], %[[p0]]), kind=kCustom
; CHECK-PTX-SAME: "block_m":
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(TritonGemmTestAny,
       DoNotFuseConcatenationOfSplitNonContractingDimension) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string hlo_text = R"(
HloModule m

ENTRY e {
  x = bf16[2,128,10] parameter(0)
  y = bf16[2,256,10] parameter(1)
  concat = bf16[2,384,10] concatenate(x, y), dimensions={1}
  z = bf16[10,20] parameter(2)
  ROOT d = bf16[2,384,20] dot(concat, z), lhs_contracting_dims={2}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK:      ENTRY
; CHECK:      concatenate
; CHECK:        ROOT
; CHECK-SAME:     fusion
; CHECK-SAME:       kind=kCustom
; CHECK-SAME:       "block_m"
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

class TritonGemmLevel2Test : public TritonGemmTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = TritonGemmTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_triton_fusion_level(2);
    return debug_options;
  }
};

class TritonGemmLevel2TestAny : public TritonGemmLevel2Test {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = TritonGemmLevel2Test::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_triton_gemm_any(true);
    return debug_options;
  }
};

TEST_F(TritonGemmLevel2Test, BinaryOperationWithSmallInputsIsFused) {
  const std::string kHloText = R"(
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

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-1, /*arel=*/1e-2}));
}

TEST_F(TritonGemmLevel2Test, BinaryOperationWithLargeInputsIsNotFused) {
  const std::string kHloText = R"(
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
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-1, /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2Test,
       ParametersWithDifferentLayoutsAreSupportedInOneScope) {
  const std::string kHloText = R"(
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

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(TritonGemmLevel2Test, BinaryOperationOnLargeParametersIsFused) {
  const std::string kHloText = R"(
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

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-1, /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2Test, LinkingLibdeviceTwiceWorks) {
  const std::string kHloText = R"(
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

TEST_F(TritonGemmLevel2Test, BroadcastOfScalarParameterIsFused) {
  const std::string kHloText = R"(
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

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2Test, BroadcastOfScalarConstantIsFused) {
  const std::string kHloText = R"(
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

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/2e-3, /*arel=*/2e-3}));
}

TEST_F(TritonGemmLevel2Test, DoubleBroadcastOfScalarConstantIsHandled) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
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
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(TritonGemmLevel2Test, BroadcastOfVectorConstantIsFused) {
  const std::string kHloText = R"(
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
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Constant())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(TritonGemmLevel2Test, AlwaysFuseScalarConstantAtBroadcastInput) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
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
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2Test, BroadcastOfVectorParameterIsFused) {
  const std::string kHloText = R"(
triton_dot {
  p0 = f16[75] parameter(0)
  bc0 = f16[75,67] broadcast(p0), dimensions={0}
  p1 = f16[92,75] parameter(1)
  ROOT d = f16[92,67] dot(p1, bc0),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[75] parameter(0)
  p1 = f16[92,75] parameter(1)
  ROOT _ = f16[92,67] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm", triton_gemm_config:
      {"block_m":32,"block_n":64,"block_k":32,
      "split_k":1,"num_stages":1,"num_warps":1,
      "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/2e-3, /*arel=*/2e-3}));
}

TEST_F(TritonGemmLevel2Test, FuseConcatenation) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
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
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter(),
                           m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3,
                                                /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2TestAny, MinimumHandlesNaNsOnTheLeft) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f32[5,5] parameter(0)
  neg1 = f32[] constant(-1)
  neg1s = f32[5,5] broadcast(neg1), dimensions={}
  nans = f32[5,5] sqrt(neg1s)
  min = f32[5,5] minimum(nans, neg1s)
  ROOT _ = f32[5,5] dot(p0, min),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: block_m
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2TestAny, MinimumHandlesNaNsOnTheRight) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f32[5,5] parameter(0)
  neg1 = f32[] constant(-1)
  neg1s = f32[5,5] broadcast(neg1), dimensions={}
  nans = f32[5,5] sqrt(neg1s)
  min = f32[5,5] minimum(neg1s, nans)
  ROOT _ = f32[5,5] dot(p0, min),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: block_m
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2TestAny, MaximumHandlesNaNsOnTheLeft) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f32[5,5] parameter(0)
  neg1 = f32[] constant(-1)
  neg1s = f32[5,5] broadcast(neg1), dimensions={}
  nans = f32[5,5] sqrt(neg1s)
  max = f32[5,5] maximum(nans, neg1s)
  ROOT _ = f32[5,5] dot(p0, max),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: block_m
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2TestAny, MaximumHandlesNaNsOnTheRight) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f32[5,5] parameter(0)
  neg1 = f32[] constant(-1)
  neg1s = f32[5,5] broadcast(neg1), dimensions={}
  nans = f32[5,5] sqrt(neg1s)
  max = f32[5,5] maximum(neg1s, nans)
  ROOT _ = f32[5,5] dot(p0, max),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: block_m
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2TestAny, MinimumReturnsLHS) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f32[5,5] parameter(0)
  zero = f32[] constant(0)
  zeros = f32[5,5] broadcast(zero), dimensions={}
  one = f32[] constant(1)
  ones = f32[5,5] broadcast(one), dimensions={}
  min = f32[5,5] minimum(zeros, ones)
  ROOT _ = f32[5,5] dot(p0, min),
  lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: block_m
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3,
                                                /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2TestAny, MinimumReturnsRHS) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f32[5,5] parameter(0)
  zero = f32[] constant(0)
  zeros = f32[5,5] broadcast(zero), dimensions={}
  one = f32[] constant(1)
  ones = f32[5,5] broadcast(one), dimensions={}
  min = f32[5,5] minimum(ones, zeros)
  ROOT _ = f32[5,5] dot(p0, min),
  lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: block_m
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3,
                                                /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2TestAny, MaximumReturnsLHS) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f32[5,5] parameter(0)
  zero = f32[] constant(0)
  zeros = f32[5,5] broadcast(zero), dimensions={}
  one = f32[] constant(1)
  ones = f32[5,5] broadcast(one), dimensions={}
  max = f32[5,5] maximum(ones, zeros)
  ROOT _ = f32[5,5] dot(p0, max),
  lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: block_m
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3,
                                                /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2TestAny, MaximumReturnsRHS) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f32[5,5] parameter(0)
  zero = f32[] constant(0)
  zeros = f32[5,5] broadcast(zero), dimensions={}
  one = f32[] constant(1)
  ones = f32[5,5] broadcast(one), dimensions={}
  max = f32[5,5] maximum(zeros, ones)
  ROOT _ = f32[5,5] dot(p0, max),
  lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: fusion(
; CHECK-SAME: kind=kCustom
; CHECK-PTX-SAME: block_m
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3,
                                                /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, SineOutputIsNotFused) {
  const std::string kHloText = R"(
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

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-1, /*arel=*/1e-2}));
}

TEST_F(TritonGemmTest, SliceInputIsFused) {
  const std::string kHloText = R"(
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

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, SliceInputWithReshapeIsFused) {
  const std::string kHloText = R"(
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
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2Test, NestedSlicingWorks) {
  const std::string kHloText = R"(
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

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, SlicedBatchDimensionIsSupported) {
  const std::string kHloText = R"(
ENTRY e {
  p0 = f16[3,3,256] parameter(0)
  s0 = f16[3,3,128] slice(p0), slice={[0:3], [0:3], [123:251]}
  r0 = f16[3,3,128] reshape(s0)
  p1 = f16[3,3,256] parameter(1)
  s1 = f16[3,3,128] slice(p1), slice={[0:3], [0:3], [30:158]}
  r1 = f16[3,3,128] reshape(s1)
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

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTestWithSplitK,
       SplitKDoesNotBreakSlicedFragmentedContractingDimension) {
  const std::string kHloText = R"(
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

TEST_F(TritonGemmLevel2Test, NarrowingConvertOutputIsFused) {
  const std::string kHloText = R"(
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

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/3e-2, /*arel=*/3e-2}));
}

TEST_F(TritonGemmLevel2Test, ParameterAfterDotIsFused) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
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

TEST_F(TritonGemmLevel2Test, OutputFusionExecutesCorrectly) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
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

TEST_F(TritonGemmLevel2Test, SplitLHSOutputTransposeAloneIsNotFused) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
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
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Transpose(
                  m::Fusion(m::Parameter(), m::Parameter())
                      .WithFusionKind(HloInstruction::FusionKind::kCustom))));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2Test, SplitLHSInputOutputIsFused) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
ENTRY e {
  p0t = (s8[5,18,20,150]) parameter(0)
  p0 = s8[5,18,20,150] get-tuple-element(p0t), index=0
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
      GmockMatch(m::Fusion(m::GetTupleElement(), m::Parameter())
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmLevel2Test, SupportPredParametersUsedInExpressions) {
  const std::string kHloText = R"(
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
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, Naming) {
  const char* hlo_text = R"(
HloModule t

ENTRY e {
  p0 = f16[15,19] parameter(0)
  p1 = s8[19,17] parameter(1)
  cp1 = f16[19,17] convert(p1)
  ROOT r = f16[15,17] dot(p0, cp1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: %gemm_fusion_r_computation (
; CHECK: ROOT %gemm_fusion_r
; CHECK-SAME: kCustom
)");
}

TEST_F(TritonGemmTestAny,
       ShouldNotLowerDotWithLhsWithoutNonContractingDimThroughTriton) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  parameter_0 = f32[32,40] parameter(0)
  parameter_1 = f32[32,40,64] parameter(1)
  ROOT dot = f32[32,64] dot(f32[32,40] parameter_0, f32[32,40,64] parameter_1), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";

  MatchOptimizedHlo(hlo_text, "CHECK-NOT: triton");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(TritonGemmTestAny,
       ShouldNotLowerDotWithRhsWithoutNonContractingDimThroughTriton) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  parameter_0 = f32[32,40,64] parameter(0)
  parameter_1 = f32[32,40] parameter(1)
  ROOT dot = f32[32,64] dot(f32[32,40,64] parameter_0, f32[32,40] parameter_1), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";

  MatchOptimizedHlo(hlo_text, "CHECK-NOT: triton");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

// This group of tests compares GPU results of dots already rewritten
// into Triton fusions.
using CompareTest = TritonGemmTest;

TEST_F(CompareTest, DifferentTilingsProduceSameResult) {
  const char* hlo_text_ref = R"(
HloModule t

triton_dot {
  p0 = s8[101,202] parameter(0)
  p0c = f32[101,202] convert(p0)
  p1 = f32[202,303] parameter(1)
  ROOT dot = f32[101,303] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[101,202]{1,0} parameter(0)
  p1 = f32[202,303]{1,0} parameter(1)
  ROOT _ = f32[101,303] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
      triton_gemm_config:
        {"block_m":16,"block_n":64,"block_k":32,
         "split_k":1,"num_stages":3,"num_warps":8,
         "num_ctas":1}}}
})";

  const char* hlo_text_triton = R"(
HloModule t

triton_dot {
  p0 = s8[101,202] parameter(0)
  p0c = f32[101,202] convert(p0)
  p1 = f32[202,303] parameter(1)
  ROOT dot = f32[101,303] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[101,202]{1,0} parameter(0)
  p1 = f32[202,303]{1,0} parameter(1)
  ROOT _ = f32[101,303] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":128,"block_k":32,
                         "split_k":1,"num_stages":2,"num_warps":4,
                         "num_ctas":1}}}
})";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, F16) {
  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = f16[5,7] parameter(0)
  arg1 = f16[7,33] parameter(1)
  gemm = (f16[5,33], s8[0]{0}) custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[0],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f16[5,33]{1,0} get-tuple-element((f16[5,33]{1,0}, s8[0]{0}) gemm), index=0
}
)";

  const char* hlo_text_triton = R"(
HloModule t

triton_dot {
  p0 = f16[5,7] parameter(0)
  p1 = f16[7,33] parameter(1)
  ROOT dot = f16[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[5,7]{1,0} parameter(0)
  p1 = f16[7,33]{1,0} parameter(1)
  ROOT _ = f16[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":32,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":1,
                         "num_ctas":1}}}
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, F32) {
  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = f32[5,7] parameter(0)
  arg1 = f32[7,33] parameter(1)
  gemm = (f32[5,33], s8[0]{0}) custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[0],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f32[5,33]{1,0} get-tuple-element((f32[5,33]{1,0}, s8[0]{0}) gemm), index=0
}
)";

  const char* hlo_text_triton = R"(
HloModule t

triton_dot {
  p0 = f32[5,7] parameter(0)
  p1 = f32[7,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[5,7]{1,0} parameter(0)
  p1 = f32[7,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":32,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":1,
                         "num_ctas":1}}}
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, F32WithTrivialNonContractingDimension) {
  const char* hlo_text_ref = R"(
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

  const char* hlo_text_triton = R"(
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
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, BF16TransposedLHS) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = bf16[512,16]{1,0} parameter(0)
  arg1 = bf16[512,256]{1,0} parameter(1)
  gemm = (bf16[16,256]{1,0}, s8[0]{0}) custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[0],"rhs_contracting_dimensions":[0],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = bf16[16,256]{1,0} get-tuple-element((bf16[16,256]{1,0}, s8[0]{0}) gemm), index=0
}
)";

  const char* hlo_text_triton = R"(
HloModule t

triton_dot {
  arg0 = bf16[512,16]{1,0} parameter(0)
  arg1 = bf16[512,256]{1,0} parameter(1)
  ROOT dot = bf16[16,256]{1,0} dot(arg0, arg1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  arg0 = bf16[512,16]{1,0} parameter(0)
  arg1 = bf16[512,256]{1,0} parameter(1)
  ROOT _ = bf16[16,256]{1,0} fusion(arg0, arg1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":128,"block_n":32,"block_k":16,
                         "split_k":1,"num_stages":2,"num_warps":4,
                        "num_ctas":1}}}
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, UsingOptinSharedMemoryOnAmpereProducesSameResult) {
  const se::DeviceDescription dev_info =
      backend().default_stream_executor()->GetDeviceDescription();
  constexpr int kBytesOfSharedMemoryTested = 64 * 1024;
  EXPECT_GE(dev_info.shared_memory_per_block_optin(),
            kBytesOfSharedMemoryTested);

  const std::string kHloTextOptinShmem = R"(
HloModule t

triton_dot {
  param_0.1 = s8[332,441]{1,0} parameter(0)
  p0c = f16[332,441]{1,0} convert(param_0.1)
  param_1.1 = f16[441,39]{1,0} parameter(1)
  ROOT dot = f16[332,39]{1,0} dot(p0c, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[332,441]{1,0} parameter(0)
  p1 = f16[441,39]{1,0} parameter(1)
  ROOT _ = f16[332,39]{1,0} fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":128,"block_n":128,"block_k":128,
                         "split_k":1,"num_stages":2,"num_warps":32,
                         "num_ctas":1}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTextOptinShmem));
  const HloComputation* triton_dot_computation =
      hlo_module->entry_computation()
          ->root_instruction()
          ->fused_instructions_computation();
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          hlo_module->entry_computation()
                              ->root_instruction()
                              ->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& config = gpu_config.fusion_backend_config();
  TF_ASSERT_OK_AND_ASSIGN(
      TritonGemmConfig triton_gemm_config,
      TritonGemmConfig::FromProto(config.triton_gemm_config()));
  TF_ASSERT_OK_AND_ASSIGN(
      const auto result,
      TritonWrapper(*TritonFusionAnalysis::Execute(*triton_dot_computation),
                    "test_fn", triton_dot_computation, GpuComputeComp(),
                    dev_info, triton_gemm_config, &llvm_module, &EmitMatMul,
                    mlir_context));
  // The config is chosen so that the used memory size is slightly above the
  // 48 kB boundary of standard / optin shared memory so that any GPU that
  // has the optin one should be able to execute the test.
  EXPECT_EQ(result.shmem_bytes, kBytesOfSharedMemoryTested);
  // Make sure the written config indeed has to use optin shared memory.
  EXPECT_GT(result.shmem_bytes, dev_info.shared_memory_per_block());

  const std::string kHloTextLowShmem = R"(
HloModule t

triton_dot {
  param_0.1 = s8[332,441]{1,0} parameter(0)
  p0c = f16[332,441]{1,0} convert(param_0.1)
  param_1.1 = f16[441,39]{1,0} parameter(1)
  ROOT dot = f16[332,39]{1,0} dot(p0c, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[332,441]{1,0} parameter(0)
  p1 = f16[441,39]{1,0} parameter(1)
  ROOT _ = f16[332,39]{1,0} fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":32,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":4,
                         "num_ctas":1}}}
})";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextLowShmem, kHloTextOptinShmem,
                                      ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, F16TransposedRHS) {
  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = f16[128,32]{1,0} parameter(0)
  arg1 = f16[64,32]{1,0} parameter(1)
  gemm = (f16[128,64]{1,0}, s8[0]{0}) custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[1],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f16[128,64]{1,0} get-tuple-element((f16[128,64]{1,0}, s8[0]{0}) gemm), index=0
}
)";

  const char* hlo_text_triton = R"(
HloModule t

triton_dot {
  arg0 = f16[128,32]{1,0} parameter(0)
  arg1 = f16[64,32]{1,0} parameter(1)
  ROOT dot = f16[128,64]{1,0} dot(arg0, arg1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  arg0 = f16[128,32]{1,0} parameter(0)
  arg1 = f16[64,32]{1,0} parameter(1)
  ROOT _ = f16[128,64]{1,0} fusion(arg0, arg1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":128,"block_n":32,"block_k":64,
                         "split_k":1,"num_stages":2,"num_warps":4,
                         "num_ctas":1}}}
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, F32TransposedBoth) {
  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = f32[64,128]{1,0} parameter(0)
  arg1 = f32[1024,64]{1,0} parameter(1)
  gemm = (f32[128,1024]{1,0}, s8[0]{0}) custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[0],"rhs_contracting_dimensions":[1],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f32[128,1024]{1,0} get-tuple-element((f32[128,1024]{1,0}, s8[0]{0}) gemm), index=0
}
)";

  const char* hlo_text_triton = R"(
HloModule t

triton_dot {
  arg0 = f32[64,128]{1,0} parameter(0)
  arg1 = f32[1024,64]{1,0} parameter(1)
  ROOT dot = f32[128,1024]{1,0} dot(arg0, arg1),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  arg0 = f32[64,128]{1,0} parameter(0)
  arg1 = f32[1024,64]{1,0} parameter(1)
  ROOT _ = f32[128,1024]{1,0} fusion(arg0, arg1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":32,"block_k":64,
                         "split_k":1,"num_stages":2,"num_warps":4,
                         "num_ctas":1}}}
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, S8BF16) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const char* hlo_text_ref = R"(
HloModule r

fused_computation {
  param_0.1 = s8[144,256]{1,0} parameter(0)
  ROOT convert.4 = bf16[144,256]{1,0} convert(param_0.1)
}

ENTRY e {
  p0 = s8[144,256]{1,0} parameter(0)
  fusion = bf16[144,256]{1,0} fusion(p0), kind=kInput, calls=fused_computation
  p1 = bf16[256,122]{1,0} parameter(1)
  gemm = (bf16[144,122]{1,0}, s8[0]{0}) custom-call(fusion, p1),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[0],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = bf16[144,122]{1,0} get-tuple-element((bf16[144,122]{1,0}, s8[0]{0}) gemm), index=0
}
)";

  const char* hlo_text_triton = R"(
HloModule t

triton_dot {
  param_0.1 = s8[144,256]{1,0} parameter(0)
  p0c = bf16[144,256]{1,0} convert(param_0.1)
  param_1.1 = bf16[256,122]{1,0} parameter(1)
  ROOT dot = bf16[144,122]{1,0} dot(p0c, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[144,256]{1,0} parameter(0)
  p1 = bf16[256,122]{1,0} parameter(1)
  ROOT _ = bf16[144,122]{1,0} fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":64,"block_n":64,"block_k":64,
                         "split_k":1,"num_stages":1,"num_warps":2,
                         "num_ctas":1}}}
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, SplitK) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string hlo_text_ref = R"(
HloModule t, is_scheduled=true

triton_gemm_r {
  parameter_0 = s8[480,120]{1,0} parameter(0)
  convert.3 = bf16[480,120]{1,0} convert(parameter_0)
  parameter_1 = bf16[16,120]{1,0} parameter(1)
  ROOT r.1 = bf16[480,16]{1,0} dot(convert.3, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p1 = bf16[16,120]{1,0} parameter(1)
  p0 = s8[3,120,5,32]{3,2,1,0} parameter(0)
  bitcast.4 = s8[480,120]{1,0} bitcast(p0)
  ROOT triton_gemm_r = bf16[480,16]{1,0} fusion(bitcast.4, p1), kind=kCustom,
    calls=triton_gemm_r,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":64,"block_n":32,"block_k":64,
                         "split_k":1,"num_stages":4,"num_warps":4,
                         "num_ctas":1}}}
})";

  const std::string hlo_text_splitk = R"(
HloModule t, is_scheduled=true

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
  p1 = bf16[16,120]{1,0} parameter(1)
  p0 = s8[3,120,5,32]{3,2,1,0} parameter(0)
  bitcast.4 = s8[480,120]{1,0} bitcast(p0)
  triton_gemm_r = bf16[4,480,16]{2,1,0} fusion(bitcast.4, p1), kind=kCustom,
    calls=triton_gemm_r,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":32,"block_k":128,
                         "split_k":4,"num_stages":1,"num_warps":4,
                         "num_ctas":1}}}
  ROOT fusion.1 = bf16[480,16]{1,0} fusion(triton_gemm_r), kind=kLoop,
    calls=fused_computation
})";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_splitk,
                                      ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, SplitKBatch) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloTextRef = R"(
HloModule m, is_scheduled=true

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

  const std::string kHloTextSplitK = R"(
HloModule m, is_scheduled=true

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

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextSplitK,
                                      ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, SplitKNontrivialBitcast) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloTextRef = R"(
HloModule module, is_scheduled=true

triton_gemm_dot.5316 {
  parameter_1 = bf16[16,4,128]{2,1,0} parameter(1)
  bitcast.2 = bf16[16,512]{1,0} bitcast(parameter_1)
  parameter_0 = s8[512,96]{1,0} parameter(0)
  convert.4 = bf16[512,96]{1,0} convert(parameter_0)
  ROOT dot.0 = bf16[16,96]{1,0} dot(bitcast.2, convert.4),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  parameter_0.1 = s8[96,4,128]{2,1,0} parameter(0)
  bitcast.6 = s8[512,96]{1,0} bitcast(parameter_0.1)
  parameter_1.1 = bf16[16,4,128]{2,1,0} parameter(1)
  ROOT triton_gemm_dot.5316 = bf16[16,96]{1,0} fusion(bitcast.6, parameter_1.1),
    kind=kCustom, calls=triton_gemm_dot.5316,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":32,"block_k":256,
                         "split_k":1,"num_stages":1,"num_warps":4,
                         "num_ctas":1}}}
})";

  const std::string kHloTextSplitK = R"(
HloModule module, is_scheduled=true

triton_gemm_dot.5316 {
  parameter_1 = bf16[16,4,128]{2,1,0} parameter(1)
  bitcast.2 = bf16[16,512]{1,0} bitcast(parameter_1)
  bitcast.17 = bf16[16,16,32]{2,1,0} bitcast(bitcast.2)
  parameter_0 = s8[512,96]{1,0} parameter(0)
  convert.4 = bf16[512,96]{1,0} convert(parameter_0)
  bitcast.18 = bf16[16,32,96]{2,1,0} bitcast(convert.4)
  ROOT dot.4 = bf16[16,16,96]{2,1,0} dot(bitcast.17, bitcast.18),
    lhs_batch_dims={1}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
}

triton_gemm_dot.5316.reduce_sub_computation.clone {
  rhs.1 = f32[] parameter(1)
  lhs.1 = f32[] parameter(0)
  ROOT add.1 = f32[] add(lhs.1, rhs.1)
}

fused_computation {
  param_0.2 = bf16[16,16,96]{2,1,0} parameter(0)
  convert.19 = f32[16,16,96]{2,1,0} convert(param_0.2)
  constant_1 = bf16[] constant(0)
  convert.18 = f32[] convert(constant_1)
  reduce.1 = f32[16,96]{1,0} reduce(convert.19, convert.18),
    dimensions={0}, to_apply=triton_gemm_dot.5316.reduce_sub_computation.clone
  ROOT convert.17 = bf16[16,96]{1,0} convert(reduce.1)
}

ENTRY entry {
  parameter_0.1 = s8[96,4,128]{2,1,0} parameter(0)
  bitcast.6 = s8[512,96]{1,0} bitcast(parameter_0.1)
  parameter_1.1 = bf16[16,4,128]{2,1,0} parameter(1)
  triton_gemm_dot.5316 = bf16[16,16,96]{2,1,0} fusion(bitcast.6, parameter_1.1),
    kind=kCustom, calls=triton_gemm_dot.5316,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":64,"block_n":32,"block_k":32,
                         "split_k":16,"num_stages":1,"num_warps":4,
                         "num_ctas":1}}}
  ROOT fusion.1 = bf16[16,96]{1,0} fusion(triton_gemm_dot.5316),
    kind=kLoop, calls=fused_computation
})";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextSplitK,
                                      ErrorSpec{/*aabs=*/2, /*arel=*/1e-2},
                                      /*run_hlo_passes=*/false));
}

// This is based on gemm_fusion_test.cc/SplitKTest.SupportsIndivisible.
//
// There were relatively large numeric errors with an f16 temporary buffer, so I
// ended up using --xla_gpu_triton_gemm_disable_reduced_precision_reduction=true
// when generating this test case.
TEST_F(CompareTest, SupportsSplitKWithIndivisibleKComplexExample) {
  constexpr absl::string_view kHloTextRef = R"(
HloModule extracted, entry_computation_layout={(s8[3,129,5,32]{3,2,1,0}, f16[16,129]{1,0})->f16[480,16]{1,0}}

triton_gemm_dot.clone {
  parameter_0 = s8[3,129,5,32]{3,2,1,0} parameter(0)
  bitcast.1 = s8[3,5,32,129]{2,1,3,0} bitcast(parameter_0)
  copy.1 = s8[3,5,32,129]{3,2,1,0} copy(bitcast.1)
  reshape.5 = s8[480,129]{1,0} reshape(copy.1)
  convert.8 = f16[480,129]{1,0} convert(reshape.5)
  parameter_1 = f16[16,129]{1,0} parameter(1)
  ROOT dot.0 = f16[480,16]{1,0} dot(convert.8, parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY entry_computation {
  p0 = s8[3,129,5,32]{3,2,1,0} parameter(0)
  p1 = f16[16,129]{1,0} parameter(1)
  ROOT fusion = f16[480,16]{1,0} fusion(p0, p1), kind=kCustom, calls=triton_gemm_dot.clone,
  backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
  "triton_gemm_config":{"block_m":"32","block_n":"32","block_k":"256",
                        "split_k":"1","num_stages":"1","num_warps":"4",
                        "num_ctas":"1"}}}
}
)";

  constexpr absl::string_view kHloTextSplitK = R"(
HloModule extracted, entry_computation_layout={(s8[3,129,5,32]{3,2,1,0}, f16[16,129]{1,0})->f16[480,16]{1,0}}

triton_gemm_dot.clone {
  parameter_0 = s8[3,129,5,32]{3,2,1,0} parameter(0)
  bitcast.1 = s8[3,5,32,129]{2,1,3,0} bitcast(parameter_0)
  copy.1 = s8[3,5,32,129]{3,2,1,0} copy(bitcast.1)
  reshape.5 = s8[480,129]{1,0} reshape(copy.1)
  convert.8 = f16[480,129]{1,0} convert(reshape.5)
  constant = f16[] constant(0)
  pad = f16[480,130]{1,0} pad(convert.8, constant), padding=0_0x0_1
  bitcast = f16[480,2,65]{2,1,0} bitcast(pad)
  convert.1 = f32[480,2,65]{2,1,0} convert(bitcast)
  parameter_1 = f16[16,129]{1,0} parameter(1)
  constant.1 = f16[] constant(0)
  pad.1 = f16[16,130]{1,0} pad(parameter_1, constant.1), padding=0_0x0_1
  bitcast.2 = f16[16,2,65]{2,1,0} bitcast(pad.1)
  convert.2 = f32[16,2,65]{2,1,0} convert(bitcast.2)
  ROOT dot.2 = f32[2,480,16]{2,1,0} dot(convert.1, convert.2), lhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}
}

fusion.reduce_sub_computation {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

fused_computation {
  param_0.1 = f32[2,480,16]{2,1,0} parameter(0)
  constant.3 = f32[] constant(0)
  reduce.1 = f32[480,16]{1,0} reduce(param_0.1, constant.3), dimensions={0}, to_apply=fusion.reduce_sub_computation
  ROOT convert.3 = f16[480,16]{1,0} convert(reduce.1)
}

ENTRY entry_computation {
  p0 = s8[3,129,5,32]{3,2,1,0} parameter(0)
  p1 = f16[16,129]{1,0} parameter(1)
  fusion = f32[2,480,16]{2,1,0} fusion(p0, p1), kind=kCustom, calls=triton_gemm_dot.clone,
  backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
  "triton_gemm_config":{"block_m":"128","block_n":"128","block_k":"64",
                        "split_k":"2","num_stages":"1","num_warps":"8",
                        "num_ctas":"1"}}}
  ROOT fusion.1 = f16[480,16]{1,0} fusion(fusion), kind=kLoop, calls=fused_computation
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextSplitK,
                                      ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, SupportsSplitKWithIndivisibleKUsingPaddingEqual1) {
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
}
)";

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

fused_computation {
  param_0.2 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  add.2 = f32[] add(param_0.2, param_1)
  convert.13 = f16[] convert(add.2)
  ROOT convert.12 = f32[] convert(convert.13)
}

triton_gemm_dot.7103.reduce_sub_computation.clone {
  lhs.1 = f32[] parameter(0)
  rhs.1 = f32[] parameter(1)
  ROOT fusion = f32[] fusion(lhs.1, rhs.1), kind=kLoop, calls=fused_computation
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

TEST_F(CompareTest, SupportsSplitKWithIndivisibleKUsingPaddingEqual5) {
  constexpr absl::string_view kHloTextRef = R"(
HloModule extracted, entry_computation_layout={(f16[1,8,4,1019]{3,2,1,0}, f16[1,1019,128]{2,1,0})->f16[1,8,4,128]{3,2,1,0}}

triton_gemm_dot.7103_computation.clone {
  parameter_0.499 = f16[1,8,4,1019]{3,2,1,0} parameter(0)
  bitcast.7923 = f16[32,1019]{1,0} bitcast(parameter_0.499)
  parameter_1.499 = f16[1,1019,128]{2,1,0} parameter(1)
  bitcast.7924 = f16[1019,128]{1,0} bitcast(parameter_1.499)
  dot.9350 = f16[32,128]{1,0} dot(bitcast.7923, bitcast.7924), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT bitcast.7925 = f16[1,8,4,128]{3,2,1,0} bitcast(dot.9350)
}

ENTRY entry_computation {
  p0 = f16[1,8,4,1019]{3,2,1,0} parameter(0)
  p1 = f16[1,1019,128]{2,1,0} parameter(1)
  ROOT triton_gemm_dot.7103 = f16[1,8,4,128]{3,2,1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm_dot.7103_computation.clone,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
    "triton_gemm_config":{"block_m":"32","block_n":"32","block_k":"256",
                          "split_k":"1","num_stages":"1","num_warps":"4",
                          "num_ctas":"1"}}}
}
)";

  constexpr absl::string_view kHloTextSplitK = R"(
HloModule extracted, entry_computation_layout={(f16[1,8,4,1019]{3,2,1,0}, f16[1,1019,128]{2,1,0})->f16[1,8,4,128]{3,2,1,0}}

triton_gemm_dot.7103_computation.clone {
  parameter_0.499 = f16[1,8,4,1019]{3,2,1,0} parameter(0)
  bitcast.7923 = f16[32,1019]{1,0} bitcast(parameter_0.499)
  constant = f16[] constant(0)
  pad = f16[32,1024]{1,0} pad(bitcast.7923, constant), padding=0_0x0_5
  bitcast = f16[32,16,64]{2,1,0} bitcast(pad)
  parameter_1.499 = f16[1,1019,128]{2,1,0} parameter(1)
  bitcast.7924 = f16[1019,128]{1,0} bitcast(parameter_1.499)
  constant.1 = f16[] constant(0)
  pad.1 = f16[1024,128]{1,0} pad(bitcast.7924, constant.1), padding=0_5x0_0
  bitcast.1 = f16[16,64,128]{2,1,0} bitcast(pad.1)
  dot.1 = f16[16,32,128]{2,1,0} dot(bitcast, bitcast.1), lhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  ROOT bitcast.7925.clone = f16[16,1,8,4,128]{4,3,2,1,0} bitcast(dot.1)
}

fused_computation {
  param_0.2 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  add.2 = f32[] add(param_0.2, param_1)
  convert.13 = f16[] convert(add.2)
  ROOT convert.12 = f32[] convert(convert.13)
}

triton_gemm_dot.7103.reduce_sub_computation.clone {
  lhs.1 = f32[] parameter(0)
  rhs.1 = f32[] parameter(1)
  ROOT fusion = f32[] fusion(lhs.1, rhs.1), kind=kLoop, calls=fused_computation
}

fused_computation.1 {
  param_0.5 = f16[16,1,8,4,128]{4,3,2,1,0} parameter(0)
  convert.16 = f32[16,1,8,4,128]{4,3,2,1,0} convert(param_0.5)
  constant.3 = f16[] constant(0)
  convert.15 = f32[] convert(constant.3)
  reduce.1 = f32[1,8,4,128]{3,2,1,0} reduce(convert.16, convert.15), dimensions={0}, to_apply=triton_gemm_dot.7103.reduce_sub_computation.clone
  ROOT convert.14 = f16[1,8,4,128]{3,2,1,0} convert(reduce.1)
}

ENTRY entry_computation {
  p0 = f16[1,8,4,1019]{3,2,1,0} parameter(0)
  p1 = f16[1,1019,128]{2,1,0} parameter(1)
  triton_gemm_dot.7103 = f16[16,1,8,4,128]{4,3,2,1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm_dot.7103_computation.clone,
    backend_config={"fusion_backend_config": {"kind":"__triton_gemm",
    "triton_gemm_config":{"block_m":"64","block_n":"32","block_k":"32",
                          "split_k":"16","num_stages":"1","num_warps":"4",
                          "num_ctas":"1"}}}
  ROOT fusion.1 = f16[1,8,4,128]{3,2,1,0} fusion(triton_gemm_dot.7103), kind=kLoop, calls=fused_computation.1
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextSplitK,
                                      ErrorSpec{/*aabs=*/4e-2, /*arel=*/2e-2},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, NonMajorMostOutputBatchWorksCorrectly) {
  const std::string kHloTextTest = R"(
HloModule m

triton_gemm_dot.6 {
  parameter_1 = f32[32,50,104]{2,1,0} parameter(1)
  parameter_0 = s8[32,26,104]{2,1,0} parameter(0)
  convert.22 = f32[32,26,104]{2,1,0} convert(parameter_0)
  ROOT dot.127 = f32[32,50,26]{2,0,1} dot(parameter_1, convert.22),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY e {
  p0 = s8[32,26,104]{2,1,0} parameter(0)
  p1 = f32[32,50,104]{2,1,0} parameter(1)
  ROOT triton_gemm_dot.6 = f32[32,50,26]{2,0,1} fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot.6,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":64,"block_n":16,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":4,
                         "num_ctas":1}}}
})";

  const std::string kHloTextRef = R"(
HloModule m

%triton_gemm_dot.127 {
  %parameter_1.1 = f32[32,50,104]{2,1,0} parameter(1)
  %parameter_0.1 = s8[32,26,104]{2,1,0} parameter(0)
  %convert.0 = f32[32,26,104]{2,1,0} convert(%parameter_0.1)
  ROOT %dot.0 = f32[32,50,26]{2,1,0} dot(%parameter_1.1, %convert.0),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
}

%fused_computation {
  %param_0.1 = f32[32,50,26]{2,1,0} parameter(0)
  %transpose.1 = f32[50,32,26]{2,1,0} transpose(%param_0.1), dimensions={1,0,2}
  ROOT %bitcast.7 = f32[32,50,26]{2,0,1} bitcast(%transpose.1)
}

ENTRY e {
  %parameter_0 = s8[32,26,104]{2,1,0} parameter(0)
  %parameter_1 = f32[32,50,104]{2,1,0} parameter(1)
  %triton_gemm_dot.127 = f32[32,50,26]{2,1,0} fusion(%parameter_0, %parameter_1),
    kind=kCustom, calls=%triton_gemm_dot.127,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":32,"block_n":128,"block_k":64,
                         "split_k":1,"num_stages":2,"num_warps":4,
                         "num_ctas":1}}}
  ROOT %fusion.1 = f32[32,50,26]{2,0,1} fusion(%triton_gemm_dot.127), kind=kLoop, calls=%fused_computation
})";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextTest,
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

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextTest,
                                      ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, TritonDotFusionCanHaveNoParametersAtAll) {
  const std::string kHloTextTest = R"(
HloModule m, is_scheduled=true

triton_gemm___computation {
  c = f32[] constant(123)
  b = f32[11,63] broadcast(c)
  c2 = f32[] constant(945)
  b2 = f32[63,45] broadcast(c2)
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
  constant_1 = f32[] constant(945)
  constant = f32[] constant(123)
  broadcast = f32[11,63]{1,0} broadcast(constant), dimensions={}
  broadcast.1 = f32[63,45]{1,0} broadcast(constant_1), dimensions={}
  gemm = (f32[11,45]{1,0}, s8[0]{0}) custom-call(broadcast, broadcast.1),
    custom_call_target="__cublas$gemm",
    backend_config={"gemm_backend_config": {"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f32[11,45]{1,0} get-tuple-element((f32[11,45]{1,0}, s8[0]{0}) gemm), index=0
})";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextTest,
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
  ROOT tmp_22 = f32[32,57]{0,1} dot(tmp_8, tmp_21), lhs_contracting_dims={0}, rhs_contracting_dims={0}
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
  ROOT add.2 = f32[3,57]{1,0} add(add.3, negate.1)
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

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextTest,
                                      ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, PredToBF16ConversionWorks) {
  if (SkipBF16Tests()) {
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

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextTest,
                                      ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, DifferentLayoutsAreSupportedInOneScope) {
  const std::string kHloTextTest = R"(
triton_dot {
  p1 = f16[3,3,2,16]{1,3,2,0} parameter(1)
  cvt1 = f32[3,3,2,16]{1,3,2,0} convert(p1)
  p0 = f16[9,32]{0,1} parameter(0)
  b0 = f16[3,3,2,16]{1,0,3,2} bitcast(p0)
  cp0 = f16[3,3,2,16]{1,3,2,0} copy(b0)
  cvt0 = f32[3,3,2,16]{1,3,2,0} convert(cp0)
  m = f32[3,3,2,16]{1,3,2,0} multiply(cvt1, cvt0)
  cvt2 = f16[3,3,2,16]{1,3,2,0} convert(m)
  cp1 = f16[3,3,2,16]{3,2,1,0} copy(cvt2)
  b1 = f16[9,32]{1,0} bitcast(cp1)
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

  const std::string kHloTextRef = R"(
ENTRY e {
  p1 = f16[3,3,2,16]{1,3,2,0} parameter(1)
  cvt1 = f32[3,3,2,16]{1,3,2,0} convert(p1)
  p0 = f16[9,32]{0,1} parameter(0)
  b0 = f16[3,3,2,16]{1,0,3,2} bitcast(p0)
  cp0 = f16[3,3,2,16]{1,3,2,0} copy(b0)
  cvt0 = f32[3,3,2,16]{1,3,2,0} convert(cp0)
  m = f32[3,3,2,16]{1,3,2,0} multiply(cvt1, cvt0)
  cvt2 = f16[3,3,2,16]{1,3,2,0} convert(m)
  cp1 = f16[3,3,2,16]{3,2,1,0} copy(cvt2)
  b1 = f16[9,32]{1,0} bitcast(cp1)
  p2 = f16[32,32]{1,0} parameter(2)
  ROOT r = f16[9,32]{1,0} dot(b1, p2),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextTest,
                                      ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4},
                                      /*run_hlo_passes=*/false));
}

class TritonGemmContractionDims : public TritonGemmTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = TritonGemmTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_ensure_minor_dot_contraction_dims(true);
    debug_options.set_xla_gpu_triton_gemm_any(true);

    return debug_options;
  }
};

TEST_F(TritonGemmContractionDims, TritonDotForceContractionDims_1_0) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
HloModule m

ENTRY e {
  parameter.0 = bf16[16,40]{1,0} parameter(0)
  parameter.1 = bf16[40,32]{1,0} parameter(1)
  ROOT dot.31472 = bf16[16,32]{1,0} dot(parameter.0, parameter.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  EXPECT_THAT(module->entry_computation()
                  ->root_instruction()
                  ->fused_instructions_computation()
                  ->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(BF16, {16, 40}, {1, 0}),
                                m::Op().WithShape(BF16, {40, 32}, {0, 1}))
                             .WithShape(BF16, {16, 32}, {1, 0})));
}

TEST_F(TritonGemmContractionDims, TritonDotForceContractionDims_1_2_1_2) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
HloModule m

ENTRY e {
  parameter_0 = bf16[32,4,36]{2,1,0} parameter(0)
  parameter_1 = bf16[40,4,36]{2,1,0} parameter(1)
  ROOT dot.16450 = bf16[4,32,40]{2,1,0} dot(parameter_0, parameter_1), lhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  EXPECT_THAT(module->entry_computation()
                  ->root_instruction()
                  ->fused_instructions_computation()
                  ->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(BF16, {32, 4, 36}, {2, 0, 1}),
                                m::Op().WithShape(BF16, {40, 4, 36}, {2, 0, 1}))
                             .WithShape(BF16, {4, 32, 40}, {2, 1, 0})));
}

TEST_F(TritonGemmContractionDims, TritonDotForceContractionDims_1_2_0_1) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
HloModule m

ENTRY e {
  parameter_1 = bf16[16,16,48]{2,1,0} parameter(1)
  parameter_2 = bf16[16,48,32]{2,1,0} parameter(0)
  ROOT dot.16125 = bf16[16,16,32]{2,1,0} dot(parameter_1, parameter_2), lhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  EXPECT_THAT(
      module->entry_computation()
          ->root_instruction()
          ->fused_instructions_computation()
          ->root_instruction(),
      GmockMatch(m::Dot(m::Op().WithShape(BF16, {16, 16, 48}, {2, 0, 1}),
                        m::Op().WithShape(BF16, {16, 48, 32}, {1, 2, 0}))
                     .WithShape(BF16, {16, 16, 32}, {2, 1, 0})));
}

TEST_F(TritonGemmContractionDims, TritonDotForceContractionDims_1_1) {
  if (SkipBF16Tests()) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
HloModule m

ENTRY e {
  parameter_0 = bf16[16,32]{1,0} parameter(0)
  parameter_1 = bf16[40,32]{0,1} parameter(1)
  ROOT dot.15148 = bf16[16,40]{1,0} dot(parameter_0, parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_THAT(module->entry_computation()
                  ->root_instruction()
                  ->fused_instructions_computation()
                  ->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(BF16, {16, 32}, {1, 0}),
                                m::Op().WithShape(BF16, {32, 40}, {1, 0}))
                             .WithShape(BF16, {16, 40}, {1, 0})));
}

// In these tests, we depend on "algorithm" annotations for selecting the 6XBF16
// algorithm.
class Triton6xBF16GemmTest : public TritonFilecheckTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = TritonFilecheckTest::GetDebugOptionsForTest();
    // These 2 flags are not strictly necessary now, but we're adding them to be
    // on the safe side against future flakiness.
    //
    // Enable triton fusion for all supported gemms.
    debug_options.set_xla_gpu_triton_gemm_any(true);
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);

    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    return debug_options;
  }

 protected:
  void SetUp() override {
    if (SkipBF16Tests()) {
      GTEST_SKIP() << "BF16 not supported.";
    }
  }
};

// In these tests, we depend on debug option flags for selecting the 6XBF16
// algorithm.
// TODO(b/316147294): Remove this class and the --xla_gpu_enable_bf16_6way_gemm
// flag after we will support the algorithm values through the entire stack.
class Triton6xBF16GemmTestWithFlag : public TritonFilecheckTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = TritonFilecheckTest::GetDebugOptionsForTest();
    // Enable triton fusion for all supported gemms.
    debug_options.set_xla_gpu_triton_gemm_any(true);
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);
    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    // Enable bf16_6way gemm to compute F32 matmul.
    debug_options.set_xla_gpu_enable_bf16_6way_gemm(true);
    return debug_options;
  }
};

TEST_F(Triton6xBF16GemmTest, Emit6xBF16GemmWhenBothInputsAreF32) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[5,7] parameter(0)
  p1 = f32[7,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x6
}

ENTRY e {
  p0 = f32[5,7]{1,0} parameter(0)
  p1 = f32[7,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK:          %[[INFINITY:.*]] = arith.constant dense<0x7F800000> : tensor<32x32xf32>
CHECK:          %[[C_MASK:.*]] = arith.constant dense<-65536> : tensor<32x32xi32>
CHECK:          %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
CHECK:          %[[CAST_I32:.*]] = tt.bitcast %{{.*}} : tensor<32x32xf32> -> tensor<32x32xi32>
CHECK:          %[[EXTRACT_HI:.*]] = arith.andi %[[CAST_I32]], %[[C_MASK]] : tensor<32x32xi32>
CHECK:          %[[CAST_HI:.*]] = tt.bitcast %[[EXTRACT_HI]] : tensor<32x32xi32> -> tensor<32x32xf32>
CHECK:          %[[TRUNC_TO_BF16:.*]] = arith.truncf %[[CAST_HI]] : tensor<32x32xf32> to tensor<32x32xbf16>
CHECK-COUNT-5:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ABS:.*]] = math.absf
CHECK:          %[[CMP:.*]] = arith.cmpf ogt, %[[INFINITY]], %[[ABS]] : tensor<32x32xf32>
CHECK:          %[[SELECT:.*]] = arith.select %[[CMP]], %{{.*}}, %[[C0]] : tensor<32x32xi1>, tensor<32x32xf32>
CHECK:          %[[DOT_LAST:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[SELECT]] : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ACC:.*]] = arith.addf %[[DOT_LAST]], %[[C0]] : tensor<32x32xf32>
    )"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                           /*arel=*/1e-6}));
}

TEST_F(Triton6xBF16GemmTestWithFlag, Emit6xBF16GemmWhenBothInputsAreF32) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[5,7] parameter(0)
  p1 = f32[7,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[5,7]{1,0} parameter(0)
  p1 = f32[7,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK:          %[[INFINITY:.*]] = arith.constant dense<0x7F800000> : tensor<32x32xf32>
CHECK:          %[[C_MASK:.*]] = arith.constant dense<-65536> : tensor<32x32xi32>
CHECK:          %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
CHECK:          %[[CAST_I32:.*]] = tt.bitcast %{{.*}} : tensor<32x32xf32> -> tensor<32x32xi32>
CHECK:          %[[EXTRACT_HI:.*]] = arith.andi %[[CAST_I32]], %[[C_MASK]] : tensor<32x32xi32>
CHECK:          %[[CAST_HI:.*]] = tt.bitcast %[[EXTRACT_HI]] : tensor<32x32xi32> -> tensor<32x32xf32>
CHECK:          %[[TRUNC_TO_BF16:.*]] = arith.truncf %[[CAST_HI]] : tensor<32x32xf32> to tensor<32x32xbf16>
CHECK-COUNT-5:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ABS:.*]] = math.absf
CHECK:          %[[CMP:.*]] = arith.cmpf ogt, %[[INFINITY]], %[[ABS]] : tensor<32x32xf32>
CHECK:          %[[SELECT:.*]] = arith.select %[[CMP]], %{{.*}}, %[[C0]] : tensor<32x32xi1>, tensor<32x32xf32>
CHECK:          %[[DOT_LAST:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[SELECT]] : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ACC:.*]] = arith.addf %[[DOT_LAST]], %[[C0]] : tensor<32x32xf32>
    )"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                           /*arel=*/1e-6}));
}

TEST_F(Triton6xBF16GemmTest, Triton6xBF16GemmWorksForLongContractingDimension) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[5,2048] parameter(0)
  p1 = f32[2048,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x6
}

ENTRY e {
  p0 = f32[5,2048]{1,0} parameter(0)
  p1 = f32[2048,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":64,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":4, "num_ctas":1}}}
}
)";
  TritonGemmConfig config(64, 32, 32, 1, 1, 4);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK-COUNT-6:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x32xbf16> * tensor<32x32xbf16> -> tensor<64x32xf32>
    )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-5,
                                                           /*arel=*/1e-5}));
}

TEST_F(Triton6xBF16GemmTest, Triton6xBF16GemmCanHandleInfinity) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[2,2] parameter(0)
  p1 = f32[2,2] parameter(1)
  ROOT dot = f32[2,2] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x6
}

ENTRY e {
  p0 = f32[2,2]{1, 0} parameter(0)
  p1 = f32[2,2]{1, 0} parameter(1)
  ROOT _ = f32[2,2] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1, "num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK-COUNT-6:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
    )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  std::vector<Literal> arguments(2);
  arguments[0] =
      LiteralUtil::CreateR2<float>({{+std::numeric_limits<float>::infinity(),
                                     +std::numeric_limits<float>::infinity()},
                                    {+std::numeric_limits<float>::infinity(),
                                     +std::numeric_limits<float>::infinity()}});
  arguments[1] = LiteralUtil::CreateR2<float>({{1.0f, 1.0f}, {1.0f, 1.0f}});
  std::vector<Literal*> argument_ptrs;
  absl::c_transform(
      arguments, std::back_inserter(argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), argument_ptrs,
                                       ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(Triton6xBF16GemmTest, Triton6xBF16GemmCanHandleNaN) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[2,2] parameter(0)
  p1 = f32[2,2] parameter(1)
  ROOT dot = f32[2,2] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x6
}

ENTRY e {
  p0 = f32[2,2]{1, 0} parameter(0)
  p1 = f32[2,2]{1, 0} parameter(1)
  ROOT _ = f32[2,2] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1, "num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK-COUNT-6:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
    )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  std::vector<Literal> arguments(2);
  arguments[0] =
      LiteralUtil::CreateR2<float>({{std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::quiet_NaN()},
                                    {std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::quiet_NaN()}});
  arguments[1] = LiteralUtil::CreateR2<float>(
      {{1.0f, +std::numeric_limits<float>::infinity()},
       {1.0f, +std::numeric_limits<float>::infinity()}});
  std::vector<Literal*> argument_ptrs;
  absl::c_transform(
      arguments, std::back_inserter(argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), argument_ptrs,
                                       ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

// Test case shows that why we truncate the middle term instead of rounding.
// If we round the middle term, the splitted terms may disagree in sign. This
// could result in wrong results for extreme values.
// For example, consider:
//   x = -3.40282347e+38
// If we round the middle term, its decomposition would be:
//   x_hi:  -3.38953139e+38
//   x_mid: -1.3240357e+36
//   x_lo:  5.17201445e+33
// The result of x*x would be NaN instead of positive infinity.
TEST_F(Triton6xBF16GemmTest, Triton6xBF16GemmWorksForInputsWithLargeExponent) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[2,2] parameter(0)
  p1 = f32[2,2] parameter(1)
  ROOT dot = f32[2,2] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x6
}

ENTRY e {
  p0 = f32[2,2]{1, 0} parameter(0)
  p1 = f32[2,2]{1, 0} parameter(1)
  ROOT _ = f32[2,2] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1, "num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK-COUNT-6:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
    )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  std::vector<Literal> arguments(2);
  constexpr float kLargeExponentFloat = 0x1.0103p72f;
  arguments[0] = LiteralUtil::CreateR2<float>(
      {{kLargeExponentFloat, 1.0f}, {-kLargeExponentFloat, 1.0f}});
  arguments[1] = LiteralUtil::CreateR2<float>(
      {{kLargeExponentFloat, 1.0f}, {-kLargeExponentFloat, 1.0f}});
  std::vector<Literal*> argument_ptrs;
  absl::c_transform(
      arguments, std::back_inserter(argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), argument_ptrs,
                               ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(Triton6xBF16GemmTest, Emit6xBF16GemmEndToEnd) {
  const char* kHloText = R"(
HloModule t

ENTRY e {
  p0 = f32[5,32] parameter(0)
  p1 = f32[32,7] parameter(1)
  ROOT dot = f32[5,7] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x6
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> verified_module,
                          ParseAndReturnVerifiedModule(kHloText));
  CompileAndOptionallyVerifyPtx(std::move(verified_module),
                                R"(
CHECK: mma.sync.aligned.{{.*}}.row.col.f32.bf16.bf16.f32
CHECK-NOT: mma.sync.aligned.{{.*}}.row.col.f32.tf32.tf32.f32
)");
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                /*arel=*/1e-6}));
}

// In these tests, we depend on "algorithm" annotations for selecting the 3XBF16
// algorithm.
class Triton3xBF16GemmTest : public TritonFilecheckTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = TritonFilecheckTest::GetDebugOptionsForTest();
    // These 2 flags are not strictly necessary now, but we're adding them the
    // to be on the safe side against future flakiness.
    //
    // Enable triton fusion for all supported gemms.
    debug_options.set_xla_gpu_triton_gemm_any(true);
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);

    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    return debug_options;
  }
};

// In these tests, we depend on debug option flags for selecting the 3XBF16
// algorithm.
// TODO(b/316147294): Remove this class and the --xla_gpu_enable_bf16_3way_gemm
// flag after we will support the algorithm values through the entire stack.
class Triton3xBF16GemmTestWithFlag : public TritonFilecheckTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = TritonFilecheckTest::GetDebugOptionsForTest();
    // Enable triton fusion for all supported gemms.
    debug_options.set_xla_gpu_triton_gemm_any(true);
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);
    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    // Enable bf16_3way gemm to compute F32 matmul.
    debug_options.set_xla_gpu_enable_bf16_3way_gemm(true);
    return debug_options;
  }

 protected:
  void SetUp() override {
    if (SkipBF16Tests()) {
      GTEST_SKIP() << "BF16 not supported.";
    }
  }
};

TEST_F(Triton3xBF16GemmTest, Emit3xBF16GemmWhenBothInputsAreF32) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[5,7] parameter(0)
  p1 = f32[7,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x3
}

ENTRY e {
  p0 = f32[5,7]{1,0} parameter(0)
  p1 = f32[7,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK:          %[[INFINITY:.*]] = arith.constant dense<0x7F800000> : tensor<32x32xf32>
CHECK:          %[[C_MASK:.*]] = arith.constant dense<-65536> : tensor<32x32xi32>
CHECK:          %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
CHECK:          %[[CAST_I32:.*]] = tt.bitcast %{{.*}} : tensor<32x32xf32> -> tensor<32x32xi32>
CHECK:          %[[EXTRACT_HI:.*]] = arith.andi %[[CAST_I32]], %[[C_MASK]] : tensor<32x32xi32>
CHECK:          %[[CAST_HI:.*]] = tt.bitcast %[[EXTRACT_HI]] : tensor<32x32xi32> -> tensor<32x32xf32>
CHECK:          %[[TRUNC_TO_BF16:.*]] = arith.truncf %[[CAST_HI]] : tensor<32x32xf32> to tensor<32x32xbf16>
CHECK-COUNT-2:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ABS:.*]] = math.absf
CHECK:          %[[CMP:.*]] = arith.cmpf ogt, %[[INFINITY]], %[[ABS]] : tensor<32x32xf32>
CHECK:          %[[SELECT:.*]] = arith.select %[[CMP]], %{{.*}}, %[[C0]] : tensor<32x32xi1>, tensor<32x32xf32>
CHECK:          %[[DOT_LAST:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[SELECT]] : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ACC:.*]] = arith.addf %[[DOT_LAST]], %[[C0]] : tensor<32x32xf32>
    )"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-5,
                                                           /*arel=*/1e-5}));
}

TEST_F(Triton3xBF16GemmTestWithFlag, Emit3xBF16GemmWhenBothInputsAreF32) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[5,7] parameter(0)
  p1 = f32[7,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[5,7]{1,0} parameter(0)
  p1 = f32[7,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK:          %[[INFINITY:.*]] = arith.constant dense<0x7F800000> : tensor<32x32xf32>
CHECK:          %[[C_MASK:.*]] = arith.constant dense<-65536> : tensor<32x32xi32>
CHECK:          %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
CHECK:          %[[CAST_I32:.*]] = tt.bitcast %{{.*}} : tensor<32x32xf32> -> tensor<32x32xi32>
CHECK:          %[[EXTRACT_HI:.*]] = arith.andi %[[CAST_I32]], %[[C_MASK]] : tensor<32x32xi32>
CHECK:          %[[CAST_HI:.*]] = tt.bitcast %[[EXTRACT_HI]] : tensor<32x32xi32> -> tensor<32x32xf32>
CHECK:          %[[TRUNC_TO_BF16:.*]] = arith.truncf %[[CAST_HI]] : tensor<32x32xf32> to tensor<32x32xbf16>
CHECK-COUNT-2:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ABS:.*]] = math.absf
CHECK:          %[[CMP:.*]] = arith.cmpf ogt, %[[INFINITY]], %[[ABS]] : tensor<32x32xf32>
CHECK:          %[[SELECT:.*]] = arith.select %[[CMP]], %{{.*}}, %[[C0]] : tensor<32x32xi1>, tensor<32x32xf32>
CHECK:          %[[DOT_LAST:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[SELECT]] : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ACC:.*]] = arith.addf %[[DOT_LAST]], %[[C0]] : tensor<32x32xf32>
    )"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-5,
                                                           /*arel=*/1e-5}));
}

TEST_F(Triton3xBF16GemmTestWithFlag, NoEmit3xBF16GemmWhenBothInputsAreNotF32) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f16[5,7] parameter(0)
  p1 = f16[7,33] parameter(1)
  ROOT dot = f16[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[5,7]{1,0} parameter(0)
  p1 = f16[7,33]{1,0} parameter(1)
  ROOT _ = f16[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK:      tt.dot
CHECK-SAME: tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
CHECK-NOT:  tt.dot
    )"));
}

TEST_F(Triton3xBF16GemmTest, Triton3xBF16GemmWorksForLongContractingDimension) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[5,2048] parameter(0)
  p1 = f32[2048,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x3
}

ENTRY e {
  p0 = f32[5,2048]{1,0} parameter(0)
  p1 = f32[2048,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":64,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":4, "num_ctas":1}}}
}
)";
  TritonGemmConfig config(64, 32, 32, 1, 1, 4);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK-COUNT-3:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x32xbf16> * tensor<32x32xbf16> -> tensor<64x32xf32>
    )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-4,
                                                           /*arel=*/1e-4}));
}

TEST_F(Triton3xBF16GemmTest, Triton3xBF16GemmCanHandleInfinity) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[2,2] parameter(0)
  p1 = f32[2,2] parameter(1)
  ROOT dot = f32[2,2] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x3
}

ENTRY e {
  p0 = f32[2,2]{1, 0} parameter(0)
  p1 = f32[2,2]{1, 0} parameter(1)
  ROOT _ = f32[2,2] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1, "num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK-COUNT-3:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
    )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  std::vector<Literal> arguments(2);
  arguments[0] =
      LiteralUtil::CreateR2<float>({{+std::numeric_limits<float>::infinity(),
                                     +std::numeric_limits<float>::infinity()},
                                    {+std::numeric_limits<float>::infinity(),
                                     +std::numeric_limits<float>::infinity()}});
  arguments[1] = LiteralUtil::CreateR2<float>({{1.0f, 1.0f}, {1.0f, 1.0f}});
  std::vector<Literal*> argument_ptrs;
  absl::c_transform(
      arguments, std::back_inserter(argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), argument_ptrs,
                                       ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(Triton3xBF16GemmTest, Triton3xBF16GemmCanHandleNaN) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[2,2] parameter(0)
  p1 = f32[2,2] parameter(1)
  ROOT dot = f32[2,2] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x3
}

ENTRY e {
  p0 = f32[2,2]{1, 0} parameter(0)
  p1 = f32[2,2]{1, 0} parameter(1)
  ROOT _ = f32[2,2] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1, "num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK-COUNT-3:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
    )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  std::vector<Literal> arguments(2);
  arguments[0] =
      LiteralUtil::CreateR2<float>({{std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::quiet_NaN()},
                                    {std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::quiet_NaN()}});
  arguments[1] = LiteralUtil::CreateR2<float>(
      {{1.0f, +std::numeric_limits<float>::infinity()},
       {1.0f, +std::numeric_limits<float>::infinity()}});
  std::vector<Literal*> argument_ptrs;
  absl::c_transform(
      arguments, std::back_inserter(argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), argument_ptrs,
                                       ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(Triton3xBF16GemmTest, Triton3xBF16GemmWorksForInputsWithLargeExponent) {
  const char* kHloText = R"(
HloModule t

triton_dot {
  p0 = f32[2,2] parameter(0)
  p1 = f32[2,2] parameter(1)
  ROOT dot = f32[2,2] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x3
}

ENTRY e {
  p0 = f32[2,2]{1, 0} parameter(0)
  p1 = f32[2,2]{1, 0} parameter(1)
  ROOT _ = f32[2,2] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1, "num_ctas":1}}}
}
)";
  TritonGemmConfig config(32, 32, 32, 1, 1, 1);
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheck(kHloText, config, EmitMatMul, "triton_dot", R"(
CHECK-COUNT-3:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
    )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  std::vector<Literal> arguments(2);
  constexpr float kLargeExponentFloat = 0x1.0103p72f;
  arguments[0] = LiteralUtil::CreateR2<float>(
      {{kLargeExponentFloat, 1.0f}, {-kLargeExponentFloat, 1.0f}});
  arguments[1] = LiteralUtil::CreateR2<float>(
      {{kLargeExponentFloat, 1.0f}, {-kLargeExponentFloat, 1.0f}});
  std::vector<Literal*> argument_ptrs;
  absl::c_transform(
      arguments, std::back_inserter(argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), argument_ptrs,
                               ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(Triton3xBF16GemmTest, Emit3xBF16GemmEndToEnd) {
  const char* kHloText = R"(
HloModule t

ENTRY e {
  p0 = f32[5,32] parameter(0)
  p1 = f32[32,7] parameter(1)
  ROOT dot = f32[5,7] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x3
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> verified_module,
                          ParseAndReturnVerifiedModule(kHloText));
  CompileAndOptionallyVerifyPtx(std::move(verified_module),
                                R"(
CHECK: mma.sync.aligned.{{.*}}.row.col.f32.bf16.bf16.f32
CHECK-NOT: mma.sync.aligned.{{.*}}.row.col.f32.tf32.tf32.f32
)");
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-5,
                                                /*arel=*/1e-5}));
}

using TritonEmitterTest = TritonGemmTest;

TEST_F(TritonEmitterTest, EmitterFailsIfComputeCapabilityIsBelowAmpere) {
  const std::string kHloText = R"(
HloModule module, is_scheduled=true

triton_gemm_dot {
  p0 = f32[10,20] parameter(0)
  p1 = f32[20,30] parameter(1)
  ROOT dot = f32[10,30] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = f32[10,20] parameter(0)
  p1 = f32[20,30] parameter(1)
  ROOT r = f32[10,30] fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloText));
  const HloComputation* triton_dot_computation =
      hlo_module->entry_computation()
          ->root_instruction()
          ->fused_instructions_computation();
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(
      TritonWrapper(*TritonFusionAnalysis::Execute(*triton_dot_computation),
                    "test_fn", triton_dot_computation,
                    se::CudaComputeCapability{se::CudaComputeCapability::VOLTA,
                                              /*minor=*/0},
                    dev_info, TritonGemmConfig{}, &llvm_module, &EmitMatMul,
                    mlir_context),
      tsl::testing::StatusIs(
          absl::StatusCode::kFailedPrecondition,
          ::testing::StrEq(
              "Triton support is only enabled for Ampere GPUs and up.")));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
