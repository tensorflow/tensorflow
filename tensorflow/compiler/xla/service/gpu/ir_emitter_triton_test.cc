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

#include <cstdint>
#include <string>

#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class TritonGemmTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(true);
    return debug_options;
  }
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

TEST_F(TritonGemmTest, MultipleDims) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  p0 = f16[1,16,96,32]{3,2,1,0} parameter(0)
  p1 = s8[16,96,32]{2,1,0} parameter(1)
  cp1 = f16[16,96,32]{2,1,0} convert(p1)
  ROOT _ = f16[1,16,16]{2,1,0} dot(p0, cp1),
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: custom_call_target="__triton",
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
; CHECK: custom_call_target="__triton",
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
; CHECK: custom-call(%p1, %p0), custom_call_target="__triton",
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
; CHECK: custom-call(%p1, %p0), custom_call_target="__triton",
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
; CHECK: custom-call(%transpose.2, %p0), custom_call_target="__triton"
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
; CHECK: custom-call(%p1, %transpose), custom_call_target="__triton"
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
  ROOT dot_a = f32[5,2,4] dot(x, cy),
    lhs_contracting_dims={2}, rhs_contracting_dims={1},
    lhs_batch_dims={0}, rhs_batch_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: custom_call_target="__triton",
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-2}));
}

TEST_F(TritonGemmTest, BatchTransposeF32F16) {
  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  x = f32[5,3,2] parameter(0)
  y = f16[5,3,4] parameter(1)
  cy = f32[5,3,4] convert(y)
  x_transposed = f32[5,2,3] transpose(x), dimensions={0, 2, 1}
  ROOT dot_a = f32[5,2,4] dot(x_transposed, cy),
    lhs_contracting_dims={2}, rhs_contracting_dims={1},
    lhs_batch_dims={0}, rhs_batch_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: custom_call_target="__triton",
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-2}));
}

TEST_F(TritonGemmTest, DoNotFuseArbitraryReshape) {
  const std::string hlo_text = R"(
HloModule m

ENTRY e {
  Arg_0.1 = f16[2,2,3]{2,1,0} parameter(0)
  c = f32[2,2,3]{2,1,0} convert(Arg_0.1)
  Arg_1.2 = f32[4,3]{1,0} parameter(1)
  reshape.4 = f32[3,2,2]{2,1,0} reshape(Arg_1.2)
  ROOT dot.5 = f32[2,2,2]{2,1,0} dot(c, reshape.4),
    lhs_batch_dims={1}, lhs_contracting_dims={2},
    rhs_batch_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: f32[3,2,2]{2,1,0} bitcast(%Arg_1.2)
; CHECK: custom-call(%Arg_0.1, %bitcast
; CHECK-SAME: custom_call_target="__triton"
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
; CHECK: custom_call_target="__triton",
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

struct GemmTestParams {
  PrimitiveType lhs_ty;
  PrimitiveType rhs_ty;
  int m;
  int k;
  int n;
  float aabs = 1e-6;
  float arel = 1e-6;
};

class ParametrizedRewriteTest
    : public TritonGemmTest,
      public ::testing::WithParamInterface<GemmTestParams> {};

TEST_P(ParametrizedRewriteTest, Main) {
  GemmTestParams params = GetParam();
  if ((params.lhs_ty == BF16 || params.rhs_ty == BF16) &&
      !GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }
  const std::string hlo_string_template = R"(
HloModule m

ENTRY e {
  p0 = $0[$2,$3] parameter(0)
  p0c = $1[$2,$3] convert(p0)
  p1 = $1[$3,$4] parameter(1)
  ROOT _ = $1[$2,$4] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  std::string hlo_string = absl::Substitute(
      hlo_string_template,
      primitive_util::LowercasePrimitiveTypeName(params.lhs_ty),
      primitive_util::LowercasePrimitiveTypeName(params.rhs_ty), params.m,
      params.k, params.n);
  MatchOptimizedHlo(hlo_string, R"(
; CHECK: custom_call_target="__triton",
)");

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{params.aabs, params.arel}));
}

std::string GemmTestParamsParamsToString(
    const ::testing::TestParamInfo<GemmTestParams>& data) {
  return absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(data.param.lhs_ty), "_",
      primitive_util::LowercasePrimitiveTypeName(data.param.rhs_ty), "_",
      data.param.m, "_", data.param.k, "_", data.param.n);
}

INSTANTIATE_TEST_SUITE_P(RewriteTestSuite, ParametrizedRewriteTest,
                         ::testing::ValuesIn({
                             GemmTestParams{PRED, F16, 16, 32, 8},
                             GemmTestParams{PRED, BF16, 16, 32, 8},
                             GemmTestParams{PRED, F32, 16, 32, 8, 1e-4, 1e-3},
                             GemmTestParams{S8, F16, 16, 32, 8},
                             GemmTestParams{S8, BF16, 16, 32, 8},
                             GemmTestParams{S8, F32, 16, 32, 8, 1e-2, 1e-2},
                             GemmTestParams{S8, F32, 101, 7, 303, 0.1, 0.1},
                             GemmTestParams{S8, F32, 101, 32, 303, 0.1, 0.1},
                             GemmTestParams{S8, F32, 101, 2048, 303, 0.5, 0.1},
                             GemmTestParams{S8, F32, 101, 2555, 303, 0.5, 0.1},
                             // Is supported but overflows.
                             //  GemmTestParams{S32, F16},
                             GemmTestParams{S32, F32, 4, 4, 4, 1, 1e-2},
                             GemmTestParams{F16, BF16, 16, 32, 8},
                             GemmTestParams{F16, F32, 16, 32, 8, 1e-3, 1e-6},
                             GemmTestParams{BF16, F16, 16, 32, 8, 1e-3, 1e-6},
                             GemmTestParams{BF16, F32, 16, 32, 8, 1e-3, 1e-6},
                             // Supported but disabled because narrowing
                             // converts should rather belong to producers.
                             // TODO(b/266862493): Move these to CompareTest.
                             // TritonRewriteTest2Params{S32, BF16},
                             //  TritonRewriteTest2Params{F32, F16},
                             //  TritonRewriteTest2Params{F32, BF16},
                             GemmTestParams{S8, BF16, 24, 40, 8},
                             GemmTestParams{S8, F16, 80, 16, 32},
                             GemmTestParams{F16, F32, 127, 3, 300, 1e-2, 1e-2},
                             GemmTestParams{F16, BF16, 544, 96, 16, 1e-3, 1e-3},
                             GemmTestParams{BF16, F32, 77, 500, 333, 3e-3,
                                            3e-3},
                         }),
                         GemmTestParamsParamsToString);

// This group of tests compares GPU results of dots already rewritten
// into Triton custom calls.
using CompareTest = TritonGemmTest;

TEST_F(CompareTest, DifferentTilings) {
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
  ROOT custom-call = f32[101,303] custom-call(p0, p1),
    custom_call_target="__triton", called_computations={triton_dot},
    backend_config="{\"block_m\":\"128\",\"block_n\":\"64\",\"block_k\":\"32\",\"split_k\":\"1\",\"num_stages\":\"3\",\"num_warps\":\"8\"}"
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
  ROOT custom-call = f32[101,303] custom-call(p0, p1),
    custom_call_target="__triton", called_computations={triton_dot},
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
  ROOT custom-call = f16[5,33] custom-call(p0, p1),
    custom_call_target="__triton", called_computations={triton_dot},
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
  ROOT custom-call = f32[5,33] custom-call(p0, p1),
    custom_call_target="__triton", called_computations={triton_dot},
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
    custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"0\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
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
  ROOT _ = bf16[16,256]{1,0} custom-call(arg0, arg1),
    custom_call_target="__triton", called_computations={triton_dot}, backend_config="{\"block_m\":\"128\",\"block_n\":\"32\",\"block_k\":\"64\",\"split_k\":\"1\",\"num_stages\":\"2\",\"num_warps\":\"4\"}"
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{1e-2, 1e-2},
                                      /*run_hlo_passes=*/false));
}

TEST_F(CompareTest, F16TransposedRHS) {
  const char* hlo_text_ref = R"(
HloModule r

ENTRY e {
  arg0 = f16[128,32]{1,0} parameter(0)
  arg1 = f16[64,32]{1,0} parameter(1)
  ROOT custom-call = f16[128,64]{1,0} custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
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
  ROOT _ = f16[128,64]{1,0} custom-call(arg0, arg1),
    custom_call_target="__triton", called_computations={triton_dot}, backend_config="{\"block_m\":\"128\",\"block_n\":\"32\",\"block_k\":\"64\",\"split_k\":\"1\",\"num_stages\":\"2\",\"num_warps\":\"4\"}"
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
    custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"0\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
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
  ROOT _ = f32[128,1024]{1,0} custom-call(arg0, arg1),
    custom_call_target="__triton", called_computations={triton_dot},
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
    custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
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
  ROOT custom-call = bf16[144,122]{1,0} custom-call(p0, p1),
    custom_call_target="__triton", called_computations={triton_dot}, backend_config="{\"block_m\":\"64\",\"block_n\":\"64\",\"block_k\":\"64\",\"split_k\":\"1\",\"num_stages\":\"1\",\"num_warps\":\"2\"}"
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_triton,
                                      ErrorSpec{1e-6, 1e-6},
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
