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

#include "tensorflow/compiler/xla/service/gpu/cudnn_fused_mha_rewriter.h"

#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

namespace m = xla::match;

class CudnnFusedMhaRewriterTestHloTest : public HloTestBase {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    // Fake a supported compute capability to run tests,
    // we don't run any kernels in these tests so they should be safe
    // to run anywhere.
    return se::CudaComputeCapability(8, 0);
  }

  se::dnn::VersionInfo GetCudnnVersion() {
    // Fake a supported compute capability to run tests,
    // we don't run any kernels in these tests so they should be safe
    // to run anywhere.
    return se::dnn::VersionInfo(8, 8, 0);
  }
  CudnnFusedMhaRewriterTestHloTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/false,
                    /*instruction_can_change_layout_func=*/{}) {}

 protected:
  DebugOptions GetDebugOptionsForTest() override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cudnn_fmha(true);
    return debug_options;
  }

  HloModuleConfig GetModuleConfig() {
    DebugOptions debug_options = GetDebugOptionsForTest();
    HloModuleConfig config_with_fmha;
    config_with_fmha.set_debug_options(debug_options);
    return config_with_fmha;
  }
};

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1Bmm2Pattern) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}
ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}


)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1Bmm2UncanonicalizedPattern) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,64,256]{3,2,1,0}}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
}


)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Transpose(
                  m::GetTupleElement(
                      m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                      .WithShape(BF16, {16, 16, 256, 64}))));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm1_rhs_contracting_dim_not_most_minor) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{2,3,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_TRUE(result);
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm1_lhs_contracting_dim_not_most_minor) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{2,3,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{2,3,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_TRUE(result);
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().lhs_contracting_dimensions()[0],
            2);
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm2_non_contracting_dim_not_most_minor) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{2,3,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{2,3,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{2,3,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_TRUE(result);
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm2_dot_dimension_numbers().lhs_contracting_dimensions()[0],
            3);
  EXPECT_EQ(config.bmm2_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            3);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, F16Bmm1Bmm2Pattern) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0})->f16[16,16,256,64]{3,2,1,0}}
ENTRY main.6 {
  Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.0 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = f16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}


)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(F16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1ScaleMaskSoftmaxBmm2Pattern) {
  const char* module_str = R"(
HloModule jit_bmm_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

region_0.14.clone {
  Arg_0.0 = f32[] parameter(0)
  Arg_1.0 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(Arg_0.0, Arg_1.0)
}

region_1.26 {
  Arg_0.27 = f32[] parameter(0)
  Arg_1.28 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.27, Arg_1.28)
}

ENTRY main.38 {
  constant.10 = pred[16,16,256,256]{3,2,1,0} constant({...})
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.11 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  convert.33 = f32[16,16,256,256]{3,2,1,0} convert(dot.11)
  constant.6 = f32[] constant(2.1)
  broadcast.7 = f32[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
  multiply.12 = f32[16,16,256,256]{3,2,1,0} multiply(convert.33, broadcast.7)
  convert.34 = bf16[16,16,256,256]{3,2,1,0} convert(multiply.12)
  constant.4 = bf16[] constant(0)
  broadcast.5 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
  select.13 = bf16[16,16,256,256]{3,2,1,0} select(constant.10, convert.34, broadcast.5)
  convert.36 = f32[16,16,256,256]{3,2,1,0} convert(select.13)
  constant.9 = f32[] constant(-inf)
  reduce.18 = f32[16,16,256]{2,1,0} reduce(convert.36, constant.9), dimensions={3}, to_apply=region_0.14.clone
  broadcast.22 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.18), dimensions={0,1,2}
  subtract.23 = f32[16,16,256,256]{3,2,1,0} subtract(convert.36, broadcast.22)
  exponential.24 = f32[16,16,256,256]{3,2,1,0} exponential(subtract.23)
  constant.8 = f32[] constant(0)
  reduce.30 = f32[16,16,256]{2,1,0} reduce(exponential.24, constant.8), dimensions={3}, to_apply=region_1.26
  broadcast.35 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.30), dimensions={0,1,2}
  divide.36 = f32[16,16,256,256]{3,2,1,0} divide(exponential.24, broadcast.35)
  convert.49 = bf16[16,16,256,256]{3,2,1,0} convert(divide.36)
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  ROOT dot.37 = bf16[16,16,256,64]{3,2,1,0} dot(convert.49, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleMaskSoftmaxCallTarget}), 0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 2.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 4);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1ScaleBiasMaskSoftmaxBmm2Pattern) {
  const char* module_str = R"(
HloModule jit_bmm_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

region_0.17.clone {
  Arg_0.0 = f32[] parameter(0)
  Arg_1.0 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(Arg_0.0, Arg_1.0)
}

region_1.29 {
  Arg_0.30 = f32[] parameter(0)
  Arg_1.31 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.30, Arg_1.31)
}

ENTRY main.41 {
  constant.10 = pred[16,16,256,256]{3,2,1,0} constant({...})
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.11 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  convert.33 = f32[16,16,256,256]{3,2,1,0} convert(dot.11)
  constant.6 = f32[] constant(3.1)
  constant.11 = f32[] constant(1)
  broadcast.7 = f32[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
  multiply.12 = f32[16,16,256,256]{3,2,1,0} multiply(convert.33, broadcast.7)
  broadcast.11 = f32[16,16,256,256]{3,2,1,0} broadcast(constant.11), dimensions={}
  add.15 = f32[16,16,256,256]{3,2,1,0} add(multiply.12, broadcast.11)
  convert.40 = bf16[16,16,256,256]{3,2,1,0} convert(add.15)
  constant.4 = bf16[] constant(0)
  broadcast.5 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
  select.13 = bf16[16,16,256,256]{3,2,1,0} select(constant.10, convert.40, broadcast.5)
  convert.36 = f32[16,16,256,256]{3,2,1,0} convert(select.13)
  constant.9 = f32[] constant(-inf)
  reduce.18 = f32[16,16,256]{2,1,0} reduce(convert.36, constant.9), dimensions={3}, to_apply=region_0.17.clone
  broadcast.22 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.18), dimensions={0,1,2}
  subtract.23 = f32[16,16,256,256]{3,2,1,0} subtract(convert.36, broadcast.22)
  exponential.24 = f32[16,16,256,256]{3,2,1,0} exponential(subtract.23)
  constant.8 = f32[] constant(0)
  reduce.30 = f32[16,16,256]{2,1,0} reduce(exponential.24, constant.8), dimensions={3}, to_apply=region_1.29
  broadcast.35 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.30), dimensions={0,1,2}
  divide.36 = f32[16,16,256,256]{3,2,1,0} divide(exponential.24, broadcast.35)
  convert.49 = bf16[16,16,256,256]{3,2,1,0} convert(divide.36)
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  ROOT dot.37 = bf16[16,16,256,64]{3,2,1,0} dot(convert.49, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 3.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 5);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1ScaleBiasNonConstantMaskSoftmaxBmm2Pattern) {
  const char* module_str = R"(
HloModule jit_bmm_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

region_0.17.clone {
  Arg_0.0 = f32[] parameter(0)
  Arg_1.0 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(Arg_0.0, Arg_1.0)
}

region_1.29 {
  Arg_0.30 = f32[] parameter(0)
  Arg_1.31 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.30, Arg_1.31)
}

ENTRY main.41 {
  constant.10 = pred[16,16,256,256]{3,2,1,0} constant({...})
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.11 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  convert.33 = f32[16,16,256,256]{3,2,1,0} convert(dot.11)
  constant.6 = f32[] constant(3.1)
  constant.11 = f32[] constant(1)
  broadcast.7 = f32[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
  multiply.12 = f32[16,16,256,256]{3,2,1,0} multiply(convert.33, broadcast.7)
  broadcast.11 = f32[16,16,256,256]{3,2,1,0} broadcast(constant.11), dimensions={}
  add.15 = f32[16,16,256,256]{3,2,1,0} add(multiply.12, broadcast.11)
  convert.40 = bf16[16,16,256,256]{3,2,1,0} convert(add.15)
  constant.4 = bf16[] constant(0)
  broadcast.5 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
  compare = pred[16,16,256,256]{3,2,1,0} compare(convert.40, broadcast.5), direction=GT 
  select.13 = bf16[16,16,256,256]{3,2,1,0} select(compare, convert.40, broadcast.5)
  convert.36 = f32[16,16,256,256]{3,2,1,0} convert(select.13)
  constant.9 = f32[] constant(-inf)
  reduce.18 = f32[16,16,256]{2,1,0} reduce(convert.36, constant.9), dimensions={3}, to_apply=region_0.17.clone
  broadcast.22 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.18), dimensions={0,1,2}
  subtract.23 = f32[16,16,256,256]{3,2,1,0} subtract(convert.36, broadcast.22)
  exponential.24 = f32[16,16,256,256]{3,2,1,0} exponential(subtract.23)
  constant.8 = f32[] constant(0)
  reduce.30 = f32[16,16,256]{2,1,0} reduce(exponential.24, constant.8), dimensions={3}, to_apply=region_1.29
  broadcast.35 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.30), dimensions={0,1,2}
  divide.36 = f32[16,16,256,256]{3,2,1,0} divide(exponential.24, broadcast.35)
  convert.49 = bf16[16,16,256,256]{3,2,1,0} convert(divide.36)
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  ROOT dot.37 = bf16[16,16,256,64]{3,2,1,0} dot(convert.49, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 3.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 5);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16Bmm1ScaleBiasMaskSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,40,64]{3,2,1,0},f16[2,6,64,40]{3,2,1,0},f16[2,6,40,64]{3,2,1,0})->f16[2,6,40,64]{3,2,1,0}}, allow_spmd_sharding_propagation_to_output={true}

region_0.34 {
  Arg_0.35 = f16[] parameter(0)
  Arg_1.36 = f16[] parameter(1)
  ROOT maximum.1 = f16[] maximum(Arg_0.35, Arg_1.36)
}

region_1.46 {
  Arg_0.47 = f32[] parameter(0)
  Arg_1.48 = f32[] parameter(1)
  ROOT add.2 = f32[] add(Arg_0.47, Arg_1.48)
}

ENTRY main.83 {
  constant.5 = u32[1]{0} constant({2718843009})
  constant.7 = u32[1]{0} constant({1272950319})
  constant.9 = u32[1]{0} constant({0})
  constant.11 = u32[1]{0} constant({2711844646})
  custom-call.59 = (u32[1]{0}, u32[1]{0}) custom-call(constant.5, constant.7, constant.9, constant.11), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.60 = u32[1]{0} get-tuple-element(custom-call.59), index=0
  bitcast.112 = u32[] bitcast(get-tuple-element.60)
  broadcast.14 = u32[9600]{0} broadcast(bitcast.112), dimensions={}
  get-tuple-element.61 = u32[1]{0} get-tuple-element(custom-call.59), index=1
  bitcast.113 = u32[] bitcast(get-tuple-element.61)
  broadcast.16 = u32[9600]{0} broadcast(bitcast.113), dimensions={}
  iota.62 = u32[19200]{0} iota(), iota_dimension=0
  slice = u32[9600]{0} slice(iota.62), slice={[0:9600]}
  slice.1 = u32[9600]{0} slice(iota.62), slice={[9600:19200]}
  custom-call.69 = (u32[9600]{0}, u32[9600]{0}) custom-call(broadcast.14, broadcast.16, slice, slice.1), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[9600]{0}, u32[9600]{0}, u32[9600]{0}, u32[9600]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\200%\000\000\000\000\000\000"
  get-tuple-element.70 = u32[9600]{0} get-tuple-element(custom-call.69), index=0
  get-tuple-element.71 = u32[9600]{0} get-tuple-element(custom-call.69), index=1
  concatenate = u32[19200]{0} concatenate(get-tuple-element.70, get-tuple-element.71), dimensions={0}
  constant.13 = u32[] constant(9)
  broadcast.18 = u32[19200]{0} broadcast(constant.13), dimensions={}
  shift-right-logical.1 = u32[19200]{0} shift-right-logical(concatenate, broadcast.18)
  constant.15 = u32[] constant(1065353216)
  broadcast.19 = u32[19200]{0} broadcast(constant.15), dimensions={}
  or.1 = u32[19200]{0} or(shift-right-logical.1, broadcast.19)
  bitcast-convert.1 = f32[19200]{0} bitcast-convert(or.1)
  constant.17 = f32[] constant(-1)
  broadcast.20 = f32[19200]{0} broadcast(constant.17), dimensions={}
  add.3 = f32[19200]{0} add(bitcast-convert.1, broadcast.20)
  constant.39 = f32[] constant(0)
  broadcast.21 = f32[19200]{0} broadcast(constant.39), dimensions={}
  maximum.2 = f32[19200]{0} maximum(add.3, broadcast.21)
  constant.28 = f32[] constant(0.8)
  broadcast.23 = f32[19200]{0} broadcast(constant.28), dimensions={}
  compare.1 = pred[19200]{0} compare(maximum.2, broadcast.23), direction=LT
  bitcast.114 = pred[2,6,40,40]{3,2,1,0} bitcast(compare.1)
  constant.34 = pred[2,6,40,40]{3,2,1,0} constant({...})
  Arg_0.1 = f16[2,6,40,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[2,6,64,40]{3,2,1,0} parameter(1), sharding={replicated}
  dot.30 = f16[2,6,40,40]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.35 = f16[] constant(2)
  broadcast.27 = f16[2,6,40,40]{3,2,1,0} broadcast(constant.35), dimensions={}
  multiply.2 = f16[2,6,40,40]{3,2,1,0} multiply(dot.30, broadcast.27)
  constant.36 = f16[] constant(1)
  broadcast.29 = f16[2,6,40,40]{3,2,1,0} broadcast(constant.36), dimensions={}
  add.5 = f16[2,6,40,40]{3,2,1,0} add(multiply.2, broadcast.29)
  constant.37 = f16[] constant(0)
  broadcast.30 = f16[2,6,40,40]{3,2,1,0} broadcast(constant.37), dimensions={}
  select.1 = f16[2,6,40,40]{3,2,1,0} select(constant.34, add.5, broadcast.30)
  constant.38 = f16[] constant(-inf)
  reduce.38 = f16[2,6,40]{2,1,0} reduce(select.1, constant.38), dimensions={3}, to_apply=region_0.34
  broadcast.32 = f16[2,6,40,40]{3,2,1,0} broadcast(reduce.38), dimensions={0,1,2}
  subtract.1 = f16[2,6,40,40]{3,2,1,0} subtract(select.1, broadcast.32)
  exponential.1 = f16[2,6,40,40]{3,2,1,0} exponential(subtract.1)
  convert.1 = f32[2,6,40,40]{3,2,1,0} convert(exponential.1)
  reduce.50 = f32[2,6,40]{2,1,0} reduce(convert.1, constant.39), dimensions={3}, to_apply=region_1.46
  convert.2 = f16[2,6,40]{2,1,0} convert(reduce.50)
  broadcast.33 = f16[2,6,40,40]{3,2,1,0} broadcast(convert.2), dimensions={0,1,2}
  divide = f16[2,6,40,40]{3,2,1,0} divide(exponential.1, broadcast.33)
  constant.40 = f16[] constant(1.25)
  broadcast.34 = f16[2,6,40,40]{3,2,1,0} broadcast(constant.40), dimensions={}
  multiply.3 = f16[2,6,40,40]{3,2,1,0} multiply(divide, broadcast.34)
  select.2 = f16[2,6,40,40]{3,2,1,0} select(bitcast.114, multiply.3, broadcast.30)
  Arg_2.3 = f16[2,6,40,64]{3,2,1,0} parameter(2), sharding={replicated}
  ROOT dot.82 = f16[2,6,40,64]{3,2,1,0} dot(select.2, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha,
                            {kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget}),
              0)
              .WithShape(F16, {2, 6, 40, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 2);
  EXPECT_NEAR(config.dropout_rate(), 0.2, 1e-2);
  EXPECT_EQ(fmha->operands().size(), 5);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, F16Bmm1UnfusedSoftmaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,40,64]{3,2,1,0},f16[2,6,64,40]{3,2,1,0},f16[2,6,40,64]{3,2,1,0})->f16[2,6,40,64]{3,2,1,0}}

region_0.7 {
  Arg_0.8 = f16[] parameter(0)
  Arg_1.9 = f16[] parameter(1)
  ROOT maximum = f16[] maximum(Arg_0.8, Arg_1.9)
}

region_1.19 {
  Arg_0.20 = f32[] parameter(0)
  Arg_1.21 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.20, Arg_1.21)
}

ENTRY main.31 {
  Arg_0.1 = f16[2,6,40,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[2,6,64,40]{3,2,1,0} parameter(1), sharding={replicated}
  dot = f16[2,6,40,40]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  constant = f16[] constant(-inf)
  reduce.11 = f16[2,6,40]{2,1,0} reduce(dot, constant), dimensions={3}, to_apply=region_0.7
  broadcast.3 = f16[2,6,40,40]{3,2,1,0} broadcast(reduce.11), dimensions={0,1,2}
  subtract.1 = f16[2,6,40,40]{3,2,1,0} subtract(dot, broadcast.3)
  exponential.1 = f16[2,6,40,40]{3,2,1,0} exponential(subtract.1)
  convert.1 = f32[2,6,40,40]{3,2,1,0} convert(exponential.1)
  constant.1 = f32[] constant(0)
  reduce.23 = f32[2,6,40]{2,1,0} reduce(convert.1, constant.1), dimensions={3}, to_apply=region_1.19
  convert.2 = f16[2,6,40]{2,1,0} convert(reduce.23)
  broadcast.4 = f16[2,6,40,40]{3,2,1,0} broadcast(convert.2), dimensions={0,1,2}
  divide = f16[2,6,40,40]{3,2,1,0} divide(exponential.1, broadcast.4)
  Arg_2.3 = f16[2,6,40,64]{3,2,1,0} parameter(2), sharding={replicated}
  ROOT dot.1 = f16[2,6,40,64]{3,2,1,0} dot(divide, Arg_2.3), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHASoftmaxCallTarget}), 0)
                     .WithShape(F16, {2, 6, 40, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 1.0);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 3);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16Bmm1UnfusedSoftmaxWithConvertF32ToReduceMaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[128,6,400,64]{3,2,1,0},f16[128,6,64,400]{3,2,1,0},f16[128,6,400,64]{3,2,1,0})->f16[128,6,400,64]{3,2,1,0}}

region_0.18 {
  Arg_0.19 = f32[] parameter(0)
  Arg_1.20 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(Arg_0.19, Arg_1.20)
}

region_1.29 {
  Arg_0.30 = f32[] parameter(0)
  Arg_1.31 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.30, Arg_1.31)
}

ENTRY main.41 {
  constant.3 = pred[128,6,400,400]{3,2,1,0} constant({...})
  Arg_0.1 = f16[128,6,400,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[128,6,64,400]{3,2,1,0} parameter(1), sharding={replicated}
  constant.1 = f16[] constant(1)
  broadcast.2 = f16[128,6,400,400]{3,2,1,0} broadcast(constant.1), dimensions={}
  constant.50 = f16[] constant(2)
  broadcast.100 = f16[128,6,400,400]{3,2,1,0} broadcast(constant.50), dimensions={}
  dot = f16[128,6,400,400]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  multiply.100 = f16[128,6,400,400]{3,2,1,0} multiply(dot, broadcast.100)
  add.1 = f16[128,6,400,400]{3,2,1,0} add(multiply.100, broadcast.2)
  constant.5 = f16[] constant(0)
  broadcast.4 = f16[128,6,400,400]{3,2,1,0} broadcast(constant.5), dimensions={}
  select.1 = f16[128,6,400,400]{3,2,1,0} select(constant.3, add.1, broadcast.4)
  convert.1 = f32[128,6,400,400]{3,2,1,0} convert(select.1)
  constant.7 = f32[] constant(-inf)
  reduce.22 = f32[128,6,400]{2,1,0} reduce(convert.1, constant.7), dimensions={3}, to_apply=region_0.18
  broadcast.8 = f32[128,6,400,400]{3,2,1,0} broadcast(reduce.22), dimensions={0,1,2}
  subtract.1 = f32[128,6,400,400]{3,2,1,0} subtract(convert.1, broadcast.8)
  exponential.1 = f32[128,6,400,400]{3,2,1,0} exponential(subtract.1)
  constant.11 = f32[] constant(0)
  reduce.33 = f32[128,6,400]{2,1,0} reduce(exponential.1, constant.11), dimensions={3}, to_apply=region_1.29
  broadcast.9 = f32[128,6,400,400]{3,2,1,0} broadcast(reduce.33), dimensions={0,1,2}
  divide = f32[128,6,400,400]{3,2,1,0} divide(exponential.1, broadcast.9)
  convert.2 = f16[128,6,400,400]{3,2,1,0} convert(divide)
  Arg_2.3 = f16[128,6,400,64]{3,2,1,0} parameter(2), sharding={replicated}
  ROOT dot.1 = f16[128,6,400,64]{3,2,1,0} dot(convert.2, Arg_2.3), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(F16, {128, 6, 400, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 2.0);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 5);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1UnfusedScaleMaskBiasSoftmaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[1,16,256,256]{3,2,1,0},pred[16,1,256,256]{3,2,1,0})->bf16[16,256,16,64]{3,2,1,0}}

region_0.32.clone {
  Arg_0.0 = f32[] parameter(0)
  Arg_1.0 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(Arg_0.0, Arg_1.0)
}

region_1.44 {
  Arg_0.45 = f32[] parameter(0)
  Arg_1.46 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.45, Arg_1.46)
}

ENTRY main.61 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  transpose.5 = bf16[16,16,64,256]{3,2,1,0} transpose(Arg_2.3), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  transpose.6 = bf16[16,16,256,64]{3,2,1,0} transpose(Arg_0.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  transpose.7 = bf16[16,16,64,256]{3,2,1,0} transpose(Arg_1.2), dimensions={0,2,3,1}
  Arg_4.5 = pred[16,1,256,256]{3,2,1,0} parameter(4), sharding={replicated}
  bitcast.35 = pred[16,256,256]{2,1,0} bitcast(Arg_4.5)
  convert.49 = s32[16,256,256]{2,1,0} convert(bitcast.35)
  constant.5 = s32[] constant(0)
  broadcast.10 = s32[16,256,256]{2,1,0} broadcast(constant.5), dimensions={}
  constant.50 = bf16[] constant(2)
  broadcast.100 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.50), dimensions={}
  compare = pred[16,256,256]{2,1,0} compare(convert.49, broadcast.10), direction=GT
  constant.7 = bf16[] constant(0)
  broadcast.12 = bf16[16,256,256]{2,1,0} broadcast(constant.7), dimensions={}
  constant.9 = bf16[] constant(-9.999e+09)
  broadcast.13 = bf16[16,256,256]{2,1,0} broadcast(constant.9), dimensions={}
  select = bf16[16,256,256]{2,1,0} select(compare, broadcast.12, broadcast.13)
  convert.51 = f32[16,256,256]{2,1,0} convert(select)
  broadcast.14 = f32[16,16,256,256]{3,2,1,0} broadcast(convert.51), dimensions={0,2,3}
  Arg_3.4 = bf16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  bitcast.52 = bf16[16,256,256]{2,1,0} bitcast(Arg_3.4)
  convert.52 = f32[16,256,256]{2,1,0} convert(bitcast.52)
  broadcast.15 = f32[16,16,256,256]{3,2,1,0} broadcast(convert.52), dimensions={1,2,3}
  add.1 = f32[16,16,256,256]{3,2,1,0} add(broadcast.14, broadcast.15)
  dot = bf16[16,16,256,256]{3,2,1,0} dot(transpose.6, transpose.7), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  multiply.100 = bf16[16,16,256,256]{3,2,1,0} multiply(dot, broadcast.100)
  convert.55 = f32[16,16,256,256]{3,2,1,0} convert(multiply.100)
  add.10 = f32[16,16,256,256]{3,2,1,0} add(convert.55, add.1)
  constant.11 = f32[] constant(-inf)
  reduce.36 = f32[16,16,256]{2,1,0} reduce(add.10, constant.11), dimensions={3}, to_apply=region_0.32.clone
  broadcast.17 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.36), dimensions={0,1,2}
  subtract.1 = f32[16,16,256,256]{3,2,1,0} subtract(add.10, broadcast.17)
  exponential.1 = f32[16,16,256,256]{3,2,1,0} exponential(subtract.1)
  constant.14 = f32[] constant(0)
  reduce.48 = f32[16,16,256]{2,1,0} reduce(exponential.1, constant.14), dimensions={3}, to_apply=region_1.44
  broadcast.18 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.48), dimensions={0,1,2}
  divide = f32[16,16,256,256]{3,2,1,0} divide(exponential.1, broadcast.18)
  convert.68 = bf16[16,16,256,256]{3,2,1,0} convert(divide)
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.5, convert.68), lhs_contracting_dims={3}, rhs_contracting_dims={3}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  ROOT transpose.8 = bf16[16,256,16,64]{3,2,1,0} transpose(dot.1), dimensions={0,3,1,2}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Transpose(
              m::Transpose(m::GetTupleElement(
                  m::CustomCall(&fmha, {kCudnnfMHAScaleBiasSoftmaxCallTarget}),
                  0)))
              .WithShape(BF16, {16, 256, 16, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 4);
  EXPECT_FLOAT_EQ(config.fmha_scale(), 2.0);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1ConvertedMaskAddedAfterFirstGemmSoftmaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},pred[16,1,256,256]{3,2,1,0})->bf16[16,256,16,64]{3,2,1,0}}

region_0.27.clone {
  Arg_0.0 = f32[] parameter(0)
  Arg_1.0 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(Arg_0.0, Arg_1.0)
}

region_1.39 {
  Arg_0.40 = f32[] parameter(0)
  Arg_1.41 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.40, Arg_1.41)
}

ENTRY main.56 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  transpose.5 = bf16[16,16,64,256]{3,2,1,0} transpose(Arg_2.3), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  transpose.6 = bf16[16,16,256,64]{3,2,1,0} transpose(Arg_0.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  transpose.7 = bf16[16,16,64,256]{3,2,1,0} transpose(Arg_1.2), dimensions={0,2,3,1}
  dot = bf16[16,16,256,256]{3,2,1,0} dot(transpose.6, transpose.7), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  convert.47 = f32[16,16,256,256]{3,2,1,0} convert(dot)
  Arg_3.4 = pred[16,1,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  bitcast.37 = pred[16,256,256]{2,1,0} bitcast(Arg_3.4)
  convert.42 = s32[16,256,256]{2,1,0} convert(bitcast.37)
  constant.6 = s32[] constant(0)
  broadcast.9 = s32[16,256,256]{2,1,0} broadcast(constant.6), dimensions={}
  compare = pred[16,256,256]{2,1,0} compare(convert.42, broadcast.9), direction=GT
  constant.8 = bf16[] constant(0)
  broadcast.11 = bf16[16,256,256]{2,1,0} broadcast(constant.8), dimensions={}
  constant.10 = bf16[] constant(-9.999e+09)
  broadcast.12 = bf16[16,256,256]{2,1,0} broadcast(constant.10), dimensions={}
  select = bf16[16,256,256]{2,1,0} select(compare, broadcast.11, broadcast.12)
  convert.48 = f32[16,256,256]{2,1,0} convert(select)
  broadcast.14 = f32[16,16,256,256]{3,2,1,0} broadcast(convert.48), dimensions={0,2,3}
  add.2 = f32[16,16,256,256]{3,2,1,0} add(convert.47, broadcast.14)
  constant.13 = f32[] constant(-inf)
  reduce.31 = f32[16,16,256]{2,1,0} reduce(add.2, constant.13), dimensions={3}, to_apply=region_0.27.clone
  broadcast.16 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.31), dimensions={0,1,2}
  subtract.1 = f32[16,16,256,256]{3,2,1,0} subtract(add.2, broadcast.16)
  exponential.1 = f32[16,16,256,256]{3,2,1,0} exponential(subtract.1)
  constant.14 = f32[] constant(0)
  reduce.43 = f32[16,16,256]{2,1,0} reduce(exponential.1, constant.14), dimensions={3}, to_apply=region_1.39
  broadcast.17 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.43), dimensions={0,1,2}
  divide = f32[16,16,256,256]{3,2,1,0} divide(exponential.1, broadcast.17)
  convert.63 = bf16[16,16,256,256]{3,2,1,0} convert(divide)
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.5, convert.63), lhs_contracting_dims={3}, rhs_contracting_dims={3}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  ROOT transpose.8 = bf16[16,256,16,64]{3,2,1,0} transpose(dot.1), dimensions={0,3,1,2}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Transpose(
              m::Transpose(m::GetTupleElement(
                  m::CustomCall(&fmha, {kCudnnfMHAScaleBiasSoftmaxCallTarget}),
                  0)))
              .WithShape(BF16, {16, 256, 16, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 4);
}

// negative test
TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm1_contracting_dim_not_equal_64) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,32]{3,2,1,0},bf16[16,16,256,32]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}
ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,32]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,32]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(&fmha, m::Dot(m::Parameter(0), m::Parameter(1)),
                                m::Parameter(2))
                             .WithShape(BF16, {16, 16, 256, 64})));
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm1_non_contracting_dim_larger_than_512) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,1024,64]{3,2,1,0},bf16[16,16,1024,64]{3,2,1,0},bf16[16,16,1024,64]{3,2,1,0})->bf16[16,16,1024,64]{3,2,1,0}}
ENTRY main.6 {
  Arg_2.3 = bf16[16,16,1024,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,1024,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,1024,64]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,1024,1024]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,1024,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* dot;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(&dot, m::Op(), m::Parameter(2))
                             .WithShape(BF16, {16, 16, 1024, 64})));
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm2_rhs_non_contracting_dim_not_equal_64) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,32]{3,2,1,0})->bf16[16,16,256,32]{3,2,1,0}}
ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,32]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,32]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(&fmha, m::Op(), m::Parameter(2))
                             .WithShape(BF16, {16, 16, 256, 32})));
}

// // check if MHA is unsupported, canonicalization will not kick in
TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2PatternUncanonicalized_bmm1_contracting_dim_not_equal_64) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,32]{3,2,1,0},bf16[16,16,256,32]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,64,256]{3,2,1,0}}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,32]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,32]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};

  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(&fmha, m::Parameter(2), m::Op())
                             .WithShape(BF16, {16, 16, 64, 256})));
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
