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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_triton.h"

#include <memory>
#include <string>

#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info_for_tests.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/protobuf/autotuning.pb.h"

namespace xla {
namespace gpu {
namespace {

class TritonGemmTest : public GpuCodegenTest {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

TEST_F(TritonGemmTest, FailIfTooMuchShmem) {
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
  const GpuDeviceInfo dev_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  llvm::LLVMContext llvm_ctx;
  llvm::Module llvm_module("module", llvm_ctx);
  mlir::MLIRContext mlir_context;

  tensorflow::AutotuneResult::TritonGemmKey config;
  config.set_block_m(512);
  config.set_block_n(512);
  config.set_block_k(512);
  config.set_split_k(1);
  config.set_num_stages(1);
  config.set_num_warps(2);
  EXPECT_THAT(
      TritonWrapper("test_fn", triton_dot_computation,
                    se::CudaComputeCapability{se::CudaComputeCapability::AMPERE,
                                              /*minor=*/0},
                    dev_info, config, &llvm_module, &MatMul, mlir_context),
      tsl::testing::StatusIs(
          tsl::error::RESOURCE_EXHAUSTED,
          absl::StrFormat("Requires too much shared memory: 1310720 > %d",
                          dev_info.shared_memory_per_block_optin)));

  config.set_block_m(64);
  config.set_block_n(128);
  config.set_block_k(128);
  TF_ASSERT_OK_AND_ASSIGN(
      const LaunchDimensions launch_dimensions,
      TritonWrapper("test_fn", triton_dot_computation,
                    se::CudaComputeCapability{se::CudaComputeCapability::AMPERE,
                                              /*minor=*/0},
                    dev_info, config, &llvm_module, &MatMul, mlir_context));
  // Use optin shared memory which is > shared_memory_per_block.
  EXPECT_GT(launch_dimensions.SharedMemBytes(),
            dev_info.shared_memory_per_block);
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
; CHECK: fusion(%p1, %p0)
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
  )");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
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
; CHECK: fusion(%p1, %p0)
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
; CHECK-NOT: pad(
; CHECK-NOT: slice(
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(TritonGemmTest, SplitLhsNoncontractingTransposeRhs) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  p0 = s8[3,122,96,12]{3,2,1,0} parameter(0)
  cp0 = f16[3,122,96,12]{3,2,1,0} convert(p0)
  p1 = f16[1,5,122]{2,1,0} parameter(1)
  ROOT _ = f16[3,96,12,1,5]{4,3,2,1,0} dot(cp0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={2}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: fusion(%p1, %p0)
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
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
; CHECK: fusion(%p1, %p0)
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
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
; CHECK: fusion(%transpose.2, %p0)
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
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
; CHECK: fusion(%p1, %transpose)
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
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
; CHECK: fusion(%y, %x)
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-2}));
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
; CHECK: fusion(%y, %x)
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
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
; CHECK: fusion(%y, %x)
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-2}));
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
; CHECK: f32[5,3,4]{2,1,0} bitcast(%p1)
; CHECK: fusion(%p0, %bitcast
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(TritonGemmTest, SkipMultipleBatch) {
  const std::string hlo_text = R"(
HloModule m

ENTRY e {
  Arg_0 = f16[3,4,2,5,4] parameter(0)
  c = f32[3,4,2,5,4] convert(Arg_0)
  Arg_1 = f32[5,3,4,3,2] parameter(1)
  ROOT dot.3 = f32[5,3,4,4,3] dot(c, Arg_1),
    lhs_batch_dims={3,0,1}, lhs_contracting_dims={2},
    rhs_batch_dims={0,1,2}, rhs_contracting_dims={4}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK-NOT: __triton
)");
}

TEST_F(TritonGemmTest, SkipU8) {
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
; CHECK-NOT: __triton
)");
}

TEST_F(TritonGemmTest, SkipF32F32) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  p0 = f32[3,5] parameter(0)
  p1 = f32[5,7] parameter(1)
  ROOT _ = f32[3,7] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK-NOT: __triton
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
; CHECK: fusion(%p0, %p1)
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
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

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: fusion(%p0), kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-6, 1e-6}));
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
; CHECK: %triton_gemm_r (
; CHECK: %triton_gemm_r =
; CHECK-SAME: fusion
)");
}


// This group of tests compares GPU results of dots already rewritten
// into Triton fusions.
using CompareTest = TritonGemmTest;

TEST_F(CompareTest, DifferentTilingsProduceSameResult) {
  const char* hlo_text_ref = R"(
HloModule t

triton_dot {
  p0 = s8[101,202] parameter(0)
  p1 = f32[202,303] parameter(1)
  ROOT dot = f32[101,303] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[101,202]{1,0} parameter(0)
  p1 = f32[202,303]{1,0} parameter(1)
  ROOT _ = f32[101,303] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config="{\"block_m\":\"16\",\"block_n\":\"64\",\"block_k\":\"32\",\"split_k\":\"1\",\"num_stages\":\"3\",\"num_warps\":\"8\"}"
})";

  const char* hlo_text_triton = R"(
HloModule t

triton_dot {
  p0 = s8[101,202] parameter(0)
  p1 = f32[202,303] parameter(1)
  ROOT dot = f32[101,303] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[101,202]{1,0} parameter(0)
  p1 = f32[202,303]{1,0} parameter(1)
  ROOT _ = f32[101,303] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config="{\"block_m\":\"32\",\"block_n\":\"128\",\"block_k\":\"32\",\"split_k\":\"1\",\"num_stages\":\"2\",\"num_warps\":\"4\"}"
})";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{1e-6, 1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, F16) {
  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = f16[5,7] parameter(0)
  arg1 = f16[7,33] parameter(1)
  ROOT custom-call = f16[5,33] custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config="{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
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
    backend_config="{\"block_m\":\"32\",\"block_n\":\"32\",\"block_k\":\"32\",\"split_k\":\"1\",\"num_stages\":\"1\",\"num_warps\":\"1\"}"
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{1e-6, 1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, F32) {
  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = f32[5,7] parameter(0)
  arg1 = f32[7,33] parameter(1)
  ROOT custom-call = f32[5,33] custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config="{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
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
    backend_config="{\"block_m\":\"32\",\"block_n\":\"32\",\"block_k\":\"32\",\"split_k\":\"1\",\"num_stages\":\"1\",\"num_warps\":\"1\"}"
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{1e-3, 1e-3},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, BF16TransposedLHS) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = bf16[512,16]{1,0} parameter(0)
  arg1 = bf16[512,256]{1,0} parameter(1)
  ROOT custom-call = bf16[16,256]{1,0} custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config="{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"0\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
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
    backend_config="{\"block_m\":\"128\",\"block_n\":\"32\",\"block_k\":\"16\",\"split_k\":\"1\",\"num_stages\":\"2\",\"num_warps\":\"4\"}"
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{1e-2, 1e-2},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, UsingOptinSharedMemoryOnAmpereProducesSameResult) {
  // On pre-Ampere GPUs the test would use a different amount of shared memory.
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "This test is for Ampere+ GPUs.";
  }
  const GpuDeviceInfo dev_info =
      GetGpuDeviceInfo(backend().default_stream_executor());
  constexpr int kBytesOfSharedMemoryTested = 64 * 1024;
  EXPECT_GE(dev_info.shared_memory_per_block_optin, kBytesOfSharedMemoryTested);

  const std::string kHloTextOptinShmem = R"(
HloModule t

triton_dot {
  param_0.1 = s8[332,441]{1,0} parameter(0)
  param_1.1 = f16[441,39]{1,0} parameter(1)
  ROOT dot = f16[332,39]{1,0} dot(param_0.1, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[332,441]{1,0} parameter(0)
  p1 = f16[441,39]{1,0} parameter(1)
  ROOT _ = f16[332,39]{1,0} fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config="{\"block_m\":\"128\",\"block_n\":\"128\",\"block_k\":\"128\",\"split_k\":\"1\",\"num_stages\":\"2\",\"num_warps\":\"32\"}"
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

  TF_ASSERT_OK_AND_ASSIGN(
      const tensorflow::AutotuneResult::TritonGemmKey config,
      hlo_module->entry_computation()
          ->root_instruction()
          ->backend_config<tensorflow::AutotuneResult::TritonGemmKey>());
  TF_ASSERT_OK_AND_ASSIGN(
      const LaunchDimensions launch_dimensions,
      TritonWrapper("test_fn", triton_dot_computation,
                    GetCudaComputeCapability(), dev_info, config, &llvm_module,
                    &MatMul, mlir_context));
  // The config is chosen so that the used memory size is slightly above the
  // 48 kB boundary of standard / optin shared memory so that any GPU that
  // has the optin one should be able to execute the test.
  EXPECT_EQ(launch_dimensions.SharedMemBytes(), kBytesOfSharedMemoryTested);
  // Make sure the written config indeed has to use optin shared memory.
  EXPECT_GT(launch_dimensions.SharedMemBytes(),
            dev_info.shared_memory_per_block);

  const std::string kHloTextLowShmem = R"(
HloModule t

triton_dot {
  param_0.1 = s8[332,441]{1,0} parameter(0)
  param_1.1 = f16[441,39]{1,0} parameter(1)
  ROOT dot = f16[332,39]{1,0} dot(param_0.1, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[332,441]{1,0} parameter(0)
  p1 = f16[441,39]{1,0} parameter(1)
  ROOT _ = f16[332,39]{1,0} fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config="{\"block_m\":\"32\",\"block_n\":\"32\",\"block_k\":\"32\",\"split_k\":\"1\",\"num_stages\":\"1\",\"num_warps\":\"4\"}"
})";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextLowShmem, kHloTextOptinShmem,
                                      ErrorSpec{1e-6, 1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, F16TransposedRHS) {
  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = f16[128,32]{1,0} parameter(0)
  arg1 = f16[64,32]{1,0} parameter(1)
  ROOT custom-call = f16[128,64]{1,0} custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config="{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
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
    backend_config="{\"block_m\":\"128\",\"block_n\":\"32\",\"block_k\":\"64\",\"split_k\":\"1\",\"num_stages\":\"2\",\"num_warps\":\"4\"}"
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{1e-2, 1e-2},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, F32TransposedBoth) {
  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = f32[64,128]{1,0} parameter(0)
  arg1 = f32[1024,64]{1,0} parameter(1)
  ROOT custom-call = f32[128,1024]{1,0} custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config="{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"0\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
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
    backend_config="{\"block_m\":\"32\",\"block_n\":\"32\",\"block_k\":\"64\",\"split_k\":\"1\",\"num_stages\":\"2\",\"num_warps\":\"4\"}"
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{1e-3, 1e-3},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, S8BF16) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "No BF16 before Ampere.";
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
  ROOT custom-call = bf16[144,122]{1,0} custom-call(fusion, p1),
    custom_call_target="__cublas$gemm",
    backend_config="{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
}
)";

  const char* hlo_text_triton = R"(
HloModule t

triton_dot {
  param_0.1 = s8[144,256]{1,0} parameter(0)
  param_1.1 = bf16[256,122]{1,0} parameter(1)
  ROOT dot = bf16[144,122]{1,0} dot(param_0.1, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[144,256]{1,0} parameter(0)
  p1 = bf16[256,122]{1,0} parameter(1)
  ROOT _ = bf16[144,122]{1,0} fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config="{\"block_m\":\"64\",\"block_n\":\"64\",\"block_k\":\"64\",\"split_k\":\"1\",\"num_stages\":\"1\",\"num_warps\":\"2\"}"
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{1e-6, 1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, SplitK) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "No BF16 before Ampere.";
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
    backend_config="{\"block_m\":\"64\",\"block_n\":\"32\",\"block_k\":\"64\",\"split_k\":\"1\",\"num_stages\":\"4\",\"num_warps\":\"4\"}"
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
    backend_config="{\"block_m\":\"32\",\"block_n\":\"32\",\"block_k\":\"128\",\"split_k\":\"4\",\"num_stages\":\"1\",\"num_warps\":\"4\"}"
  ROOT fusion.1 = bf16[480,16]{1,0} fusion(triton_gemm_r), kind=kLoop,
    calls=fused_computation
})";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_splitk,
                                      ErrorSpec{1e-6, 1e-6},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, SplitKBatch) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }
  const std::string hlo_text_ref = R"(
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
    backend_config="{\"block_m\":\"64\",\"block_n\":\"32\",\"block_k\":\"64\",\"split_k\":\"1\",\"num_stages\":\"2\",\"num_warps\":\"8\"}"
})";

  const std::string hlo_text_splitk = R"(
HloModule m, is_scheduled=true

triton_gemm_dot {
  parameter_1 = bf16[1,1,800,5,128]{4,3,2,1,0} parameter(1)
  bitcast.3 = bf16[800,5,128]{2,1,0} bitcast(parameter_1)
  convert.3 = f32[800,5,128]{2,1,0} convert(bitcast.3)
  bitcast = f32[8,100,5,128]{3,2,1,0} bitcast(convert.3)
  parameter_0 = f32[1,5,700,800]{3,2,1,0} parameter(0)
  bitcast.2 = f32[5,700,800]{2,1,0} bitcast(parameter_0)
  bitcast.1 = f32[5,700,8,100]{3,2,1,0} bitcast(bitcast.2)
  ROOT dot = f32[5,8,128,700]{3,2,1,0} dot(bitcast, bitcast.1), lhs_batch_dims={2,0}, lhs_contracting_dims={1}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}
}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY e {
  tmp_3 = f32[1,5,700,800]{3,2,1,0} parameter(0)
  tmp_0 = bf16[1,1,800,5,128]{4,3,2,1,0} parameter(1)
  triton_gemm_dot.24 = f32[5,8,128,700]{3,2,1,0} fusion(tmp_3, tmp_0),
    kind=kCustom, calls=triton_gemm_dot,
    backend_config="{\"block_m\":\"64\",\"block_n\":\"32\",\"block_k\":\"64\",\"split_k\":\"8\",\"num_stages\":\"1\",\"num_warps\":\"4\"}"
  constant = f32[] constant(0)
  ROOT reduce = f32[5,128,700]{2,1,0} reduce(triton_gemm_dot.24, constant), dimensions={1}, to_apply=add
})";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_splitk,
                                      ErrorSpec{1e-3, 1e-3},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, SplitKNontrivialBitcast) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "No BF16 before Ampere.";
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
    backend_config="{\"block_m\":\"32\",\"block_n\":\"32\",\"block_k\":\"256\",\"split_k\":\"1\",\"num_stages\":\"1\",\"num_warps\":\"4\"}"
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
    backend_config="{\"block_m\":\"64\",\"block_n\":\"32\",\"block_k\":\"32\",\"split_k\":\"16\",\"num_stages\":\"1\",\"num_warps\":\"4\"}"
  ROOT fusion.1 = bf16[16,96]{1,0} fusion(triton_gemm_dot.5316),
    kind=kLoop, calls=fused_computation
})";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextSplitK,
                                      ErrorSpec{1e-2, 1},
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
    backend_config="{\"block_m\":\"64\",\"block_n\":\"16\",\"block_k\":\"32\",\"split_k\":\"1\",\"num_stages\":\"1\",\"num_warps\":\"4\"}"
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
    backend_config="{\"block_m\":\"32\",\"block_n\":\"128\",\"block_k\":\"64\",\"split_k\":\"1\",\"num_stages\":\"2\",\"num_warps\":\"4\"}"
  ROOT %fusion.1 = f32[32,50,26]{2,0,1} fusion(%triton_gemm_dot.127), kind=kLoop, calls=%fused_computation
})";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextTest,
                                      ErrorSpec{1e-6, 1e-6},
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
