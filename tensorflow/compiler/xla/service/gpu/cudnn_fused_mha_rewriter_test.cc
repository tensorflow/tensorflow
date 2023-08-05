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

#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_fused_mha_transpose_fusion.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/layout_normalization.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/service/reshape_decomposer.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/statusor.h"

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

  se::dnn::VersionInfo GetCudnnVersionWithDbiasAndMaskBwdInputSupport() {
    // Fake a supported compute capability to run tests for training with dbias
    // and mask bwd input support, we don't run any kernels in these tests so
    // they should be safe to run anywhere.
    return se::dnn::VersionInfo(8, 9, 1);
  }

  CudnnFusedMhaRewriterTestHloTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/false,
                    /*instruction_can_change_layout_func=*/{}) {}

 protected:
  size_t CountFusedAttentionCall(HloModule* module, bool is_backward = false) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            [&](const HloInstruction* instr) {
                              if (is_backward) {
                                return IsBwdCustomCallTofMHA(*instr);
                              } else {
                                return IsFwdCustomCallTofMHA(*instr);
                              }
                            });
  }

  DebugOptions GetDebugOptionsForTest() override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_xla_runtime_executable(false);
    debug_options.set_xla_gpu_enable_cudnn_fmha(true);
    debug_options.set_xla_gpu_fused_attention_use_cudnn_rng(true);
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
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));
  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
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
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(optimized_module->entry_computation()->root_instruction(),
              GmockMatch(m::Transpose(
                  m::GetTupleElement(
                      m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                      .WithShape(BF16, {16, 16, 256, 64}))));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
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
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            2);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
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
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().lhs_contracting_dimensions()[0],
            2);
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            2);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
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
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm2_dot_dimension_numbers().lhs_contracting_dimensions()[0],
            3);
  EXPECT_EQ(config.bmm2_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            3);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
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
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(F16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 1.0);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
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

#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleMaskSoftmaxCallTarget}), 0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 2.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 4);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
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

#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 3.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 5);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
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
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());
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

#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 3.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 5);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1CombinedMaskBiasSoftmaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_,
entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[1,16,256,256]{3,2,1,0},pred[16,1,256,256]{3,2,1,0})->bf16[16,256,16,64]{3,2,1,0}}

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
  dot.2 = bf16[16,16,256,256]{3,2,1,0} dot(transpose.6, transpose.7), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  convert.55 = f32[16,16,256,256]{3,2,1,0} convert(dot.2)
  add.18 = f32[16,16,256,256]{3,2,1,0} add(convert.55, add.1)
  constant.11 = f32[] constant(-inf)
  reduce.36 = f32[16,16,256]{2,1,0} reduce(add.18, constant.11), dimensions={3}, to_apply=region_0.32.clone
  broadcast.17 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.36), dimensions={0,1,2}
  subtract.1 = f32[16,16,256,256]{3,2,1,0} subtract(add.18, broadcast.17)
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

// check if MHA is unsupported, canonicalization will not kick in
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

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1BiasSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[1,16,256,256]{3,2,1,0})->bf16[16,256,16,64]{3,2,1,0}}

region_0.34 {
  Arg_0.35 = bf16[] parameter(0)
  Arg_1.36 = bf16[] parameter(1)
  ROOT maximum.37 = bf16[] maximum(Arg_0.35, Arg_1.36)
}

region_1.46 {
  Arg_0.47 = f32[] parameter(0)
  Arg_1.48 = f32[] parameter(1)
  ROOT add.49 = f32[] add(Arg_0.47, Arg_1.48)
}

ENTRY main.82 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = bf16[16,256,16,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.2 = bf16[16,16,64,256]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = bf16[16,16,256,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = bf16[16,16,256,256]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_3.4 = bf16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  reshape.31 = bf16[16,256,256]{2,1,0} reshape(Arg_3.4)
  broadcast.32 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.31), dimensions={1,2,3}
  add.33 = bf16[16,16,256,256]{3,2,1,0} add(dot, broadcast.32)
  constant.21 = bf16[] constant(-inf)
  reduce.38 = bf16[16,16,256]{2,1,0} reduce(add.33, constant.21), dimensions={3}, to_apply=region_0.34
  broadcast.42 = bf16[16,16,256,256]{3,2,1,0} broadcast(reduce.38), dimensions={0,1,2}
  subtract.43 = bf16[16,16,256,256]{3,2,1,0} subtract(add.33, broadcast.42)
  exponential.44 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.43)
  convert.45 = f32[16,16,256,256]{3,2,1,0} convert(exponential.44)
  constant.9 = f32[] constant(0)
  reduce.50 = f32[16,16,256]{2,1,0} reduce(convert.45, constant.9), dimensions={3}, to_apply=region_1.46
  convert.1 = bf16[16,16,256]{2,1,0} convert(reduce.50)
  broadcast.55 = bf16[16,16,256,256]{3,2,1,0} broadcast(convert.1), dimensions={0,1,2}
  divide.56 = bf16[16,16,256,256]{3,2,1,0} divide(exponential.44, broadcast.55)
  constant.18 = u32[1]{0} constant({255383827})
  constant.17 = u32[1]{0} constant({267815257})
  constant.2 = u32[1]{0} constant({0})
  constant.19 = u32[1]{0} constant({3213575472})
  custom-call.26 = (u32[1]{0}, u32[1]{0}) custom-call(constant.18, constant.17, constant.2, constant.19), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.27 = u32[1]{0} get-tuple-element(custom-call.26), index=0
  reshape.58 = u32[] reshape(get-tuple-element.27)
  broadcast.62 = u32[32768]{0} broadcast(reshape.58), dimensions={}
  get-tuple-element.28 = u32[1]{0} get-tuple-element(custom-call.26), index=1
  reshape.59 = u32[] reshape(get-tuple-element.28)
  broadcast.63 = u32[32768]{0} broadcast(reshape.59), dimensions={}
  iota.57 = u32[65536]{0} iota(), iota_dimension=0
  slice.60 = u32[32768]{0} slice(iota.57), slice={[0:32768]}
  slice.61 = u32[32768]{0} slice(iota.57), slice={[32768:65536]}
  custom-call.64 = (u32[32768]{0}, u32[32768]{0}) custom-call(broadcast.62, broadcast.63, slice.60, slice.61), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[32768]{0}, u32[32768]{0}, u32[32768]{0}, u32[32768]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\000\000\000\000\000\000"
  get-tuple-element.65 = u32[32768]{0} get-tuple-element(custom-call.64), index=0
  get-tuple-element.66 = u32[32768]{0} get-tuple-element(custom-call.64), index=1
  concatenate.67 = u32[65536]{0} concatenate(get-tuple-element.65, get-tuple-element.66), dimensions={0}
  constant.15 = u32[] constant(9)
  broadcast.3 = u32[65536]{0} broadcast(constant.15), dimensions={}
  shift-right-logical.0 = u32[65536]{0} shift-right-logical(concatenate.67, broadcast.3)
  constant.13 = u32[] constant(1065353216)
  broadcast.11 = u32[65536]{0} broadcast(constant.13), dimensions={}
  or.0 = u32[65536]{0} or(shift-right-logical.0, broadcast.11)
  bitcast-convert.0 = f32[65536]{0} bitcast-convert(or.0)
  constant.3 = f32[] constant(-1)
  broadcast.17 = f32[65536]{0} broadcast(constant.3), dimensions={}
  add.1 = f32[65536]{0} add(bitcast-convert.0, broadcast.17)
  broadcast.18 = f32[65536]{0} broadcast(constant.9), dimensions={}
  maximum.0 = f32[65536]{0} maximum(add.1, broadcast.18)
  constant.7 = f32[] constant(0.9)
  broadcast.19 = f32[65536]{0} broadcast(constant.7), dimensions={}
  compare.0 = pred[65536]{0} compare(maximum.0, broadcast.19), direction=LT
  constant = bf16[] constant(1.109)
  broadcast.20 = bf16[65536]{0} broadcast(constant), dimensions={}
  constant.4 = bf16[] constant(0)
  broadcast.21 = bf16[65536]{0} broadcast(constant.4), dimensions={}
  select.1 = bf16[65536]{0} select(compare.0, broadcast.20, broadcast.21)
  reshape.19 = bf16[16,16,256]{2,1,0} reshape(select.1)
  broadcast.9 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.19), dimensions={0,1,3}
  multiply.79 = bf16[16,16,256,256]{3,2,1,0} multiply(divide.56, broadcast.9)
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.2, multiply.79), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.81 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  ROOT copy.3 = bf16[16,256,16,64]{3,2,1,0} copy(transpose.81)
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
          m::Copy(m::Transpose(m::Transpose(m::GetTupleElement(
                      m::CustomCall(
                          &fmha, {kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget}),
                      0))))
              .WithShape(BF16, {16, 256, 16, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 4);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1ScaleBiasSoftmaxDropoutForm2Bmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[32,40,60,64]{3,2,1,0},bf16[32,40,60,64]{3,2,1,0},bf16[32,40,60,64]{3,2,1,0})->bf16[32,40,60,64]{3,2,1,0}}, allow_spmd_sharding_propagation_to_output={true}

region_0.29 {
  Arg_0.30 = bf16[] parameter(0)
  Arg_1.31 = bf16[] parameter(1)
  ROOT maximum.32 = bf16[] maximum(Arg_0.30, Arg_1.31)
}

region_1.41 {
  Arg_0.42 = f32[] parameter(0)
  Arg_1.43 = f32[] parameter(1)
  ROOT add.44 = f32[] add(Arg_0.42, Arg_1.43)
}

ENTRY main.79 {
  Arg_2.3 = bf16[32,40,60,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = bf16[32,40,60,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.2 = bf16[32,60,64,40]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  constant.19 = u32[1]{0} constant({2718843009})
  constant.18 = u32[1]{0} constant({1272950319})
  constant.2 = u32[1]{0} constant({0})
  constant.20 = u32[1]{0} constant({2711844646})
  custom-call.54 = (u32[1]{0}, u32[1]{0}) custom-call(constant.19, constant.18, constant.2, constant.20), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.55 = u32[1]{0} get-tuple-element(custom-call.54), index=0
  reshape.58 = u32[] reshape(get-tuple-element.55)
  broadcast.62 = u32[1536000]{0} broadcast(reshape.58), dimensions={}
  get-tuple-element.56 = u32[1]{0} get-tuple-element(custom-call.54), index=1
  reshape.59 = u32[] reshape(get-tuple-element.56)
  broadcast.63 = u32[1536000]{0} broadcast(reshape.59), dimensions={}
  iota.57 = u32[3072000]{0} iota(), iota_dimension=0
  slice.60 = u32[1536000]{0} slice(iota.57), slice={[0:1536000]}
  slice.61 = u32[1536000]{0} slice(iota.57), slice={[1536000:3072000]}
  custom-call.64 = (u32[1536000]{0}, u32[1536000]{0}) custom-call(broadcast.62, broadcast.63, slice.60, slice.61), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1536000]{0}, u32[1536000]{0}, u32[1536000]{0}, u32[1536000]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000p\027\000\000\000\000\000"
  get-tuple-element.65 = u32[1536000]{0} get-tuple-element(custom-call.64), index=0
  get-tuple-element.66 = u32[1536000]{0} get-tuple-element(custom-call.64), index=1
  concatenate.67 = u32[3072000]{0} concatenate(get-tuple-element.65, get-tuple-element.66), dimensions={0}
  constant.16 = u32[] constant(9)
  broadcast.2 = u32[3072000]{0} broadcast(constant.16), dimensions={}
  shift-right-logical.0 = u32[3072000]{0} shift-right-logical(concatenate.67, broadcast.2)
  constant.14 = u32[] constant(1065353216)
  broadcast.6 = u32[3072000]{0} broadcast(constant.14), dimensions={}
  or.0 = u32[3072000]{0} or(shift-right-logical.0, broadcast.6)
  bitcast-convert.0 = f32[3072000]{0} bitcast-convert(or.0)
  constant.3 = f32[] constant(-1)
  broadcast.8 = f32[3072000]{0} broadcast(constant.3), dimensions={}
  add.1 = f32[3072000]{0} add(bitcast-convert.0, broadcast.8)
  constant.10 = f32[] constant(0)
  broadcast.10 = f32[3072000]{0} broadcast(constant.10), dimensions={}
  maximum.0 = f32[3072000]{0} maximum(add.1, broadcast.10)
  constant.8 = f32[] constant(0.9)
  broadcast.12 = f32[3072000]{0} broadcast(constant.8), dimensions={}
  compare.0 = pred[3072000]{0} compare(maximum.0, broadcast.12), direction=LT
  reshape.18 = pred[32,60,40,40]{3,2,1,0} reshape(compare.0)
  Arg_0.1 = bf16[32,40,60,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = bf16[32,40,60,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = bf16[32,60,40,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[32,40,60,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = bf16[32,40,60,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = bf16[32,60,64,40]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = bf16[32,60,40,40]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.25 = bf16[] constant(1)
  broadcast.26 = bf16[32,60,40,40]{3,2,1,0} broadcast(constant.25), dimensions={}
  add.28 = bf16[32,60,40,40]{3,2,1,0} add(dot, broadcast.26)
  constant.24 = bf16[] constant(-inf)
  reduce.33 = bf16[32,60,40]{2,1,0} reduce(add.28, constant.24), dimensions={3}, to_apply=region_0.29
  broadcast.37 = bf16[32,60,40,40]{3,2,1,0} broadcast(reduce.33), dimensions={0,1,2}
  subtract.38 = bf16[32,60,40,40]{3,2,1,0} subtract(add.28, broadcast.37)
  exponential.39 = bf16[32,60,40,40]{3,2,1,0} exponential(subtract.38)
  convert.40 = f32[32,60,40,40]{3,2,1,0} convert(exponential.39)
  reduce.45 = f32[32,60,40]{2,1,0} reduce(convert.40, constant.10), dimensions={3}, to_apply=region_1.41
  convert.0 = bf16[32,60,40]{2,1,0} convert(reduce.45)
  broadcast.50 = bf16[32,60,40,40]{3,2,1,0} broadcast(convert.0), dimensions={0,1,2}
  divide.51 = bf16[32,60,40,40]{3,2,1,0} divide(exponential.39, broadcast.50)
  constant = bf16[] constant(1.109)
  broadcast.1 = bf16[32,60,40,40]{3,2,1,0} broadcast(constant), dimensions={}
  multiply = bf16[32,60,40,40]{3,2,1,0} multiply(divide.51, broadcast.1)
  constant.4 = bf16[] constant(0)
  broadcast.5 = bf16[32,60,40,40]{3,2,1,0} broadcast(constant.4), dimensions={}
  select.76 = bf16[32,60,40,40]{3,2,1,0} select(reshape.18, multiply, broadcast.5)
  dot.1 = bf16[32,60,64,40]{3,2,1,0} dot(transpose.2, select.76), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.78 = bf16[32,40,60,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  ROOT copy.3 = bf16[32,40,60,64]{3,2,1,0} copy(transpose.78)
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
          m::Copy(m::Transpose(m::Transpose(m::GetTupleElement(
                      m::CustomCall(
                          &fmha, {kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget}),
                      0))))
              .WithShape(BF16, {32, 40, 60, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
  EXPECT_EQ(fmha->operands().size(), 4);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16TrainingBmm1Bmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0})->(bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0})}

ENTRY main.17 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = bf16[16,256,16,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.2 = bf16[16,16,64,256]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = bf16[16,16,256,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = bf16[16,16,256,256]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.2, dot), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.7 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  Arg_3.4 = bf16[16,256,16,64]{3,2,1,0} parameter(3), sharding={replicated}
  copy.3 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_3.4), sharding={replicated}
  transpose.4 = bf16[16,16,256,64]{3,2,1,0} transpose(copy.3), dimensions={0,2,1,3}
  dot.2 = bf16[16,16,256,256]{3,2,1,0} dot(transpose.4, transpose.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  copy.4 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.12 = bf16[16,16,256,64]{3,2,1,0} transpose(copy.4), dimensions={0,2,1,3}
  dot.4 = bf16[16,16,256,64]{3,2,1,0} dot(dot.2, transpose.12), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.15 = bf16[16,256,16,64]{3,1,2,0} transpose(dot.4), dimensions={0,2,1,3}
  dot.3 = bf16[16,16,256,64]{3,2,1,0} dot(dot.2, transpose), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.13 = bf16[16,256,16,64]{3,1,2,0} transpose(dot.3), dimensions={0,2,1,3}
  copy.5 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_3.4), sharding={replicated}
  transpose.8 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.5), dimensions={0,2,3,1}
  dot.10 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.8, dot), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.11 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.10), dimensions={0,3,1,2}
  tuple.16 = (bf16[16,256,16,64]{1,3,2,0}, bf16[16,256,16,64]{3,1,2,0}, bf16[16,256,16,64]{3,1,2,0}, bf16[16,256,16,64]{1,3,2,0}) tuple(transpose.7, transpose.15, transpose.13, transpose.11)
  get-tuple-element = bf16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.16), index=0
  copy.6 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element)
  get-tuple-element.1 = bf16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.16), index=1
  copy.7 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.1)
  get-tuple-element.2 = bf16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.16), index=2
  copy.8 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.2)
  get-tuple-element.3 = bf16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.16), index=3
  copy.9 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.3)
  ROOT tuple = (bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}) tuple(copy.6, copy.7, copy.8, copy.9)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  const auto status = RunHloPass(&fusedMhaRewriter, m.get());
  const bool changed = status.value();
  EXPECT_EQ(changed, false);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16TrainingBmm1ScaleBiasSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[1,16,256,256]{3,2,1,0},pred[16,1,256,256]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0})->(bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[1,16,256,256]{3,2,1,0})}

region_0.54 {
  Arg_0.55 = bf16[] parameter(0)
  Arg_1.56 = bf16[] parameter(1)
  ROOT maximum.57 = bf16[] maximum(Arg_0.55, Arg_1.56)
}

region_1.66 {
  Arg_0.67 = f32[] parameter(0)
  Arg_1.68 = f32[] parameter(1)
  ROOT add.69 = f32[] add(Arg_0.67, Arg_1.68)
}

region_2.114 {
  Arg_0.115 = bf16[] parameter(0)
  Arg_1.116 = bf16[] parameter(1)
  ROOT add.117 = bf16[] add(Arg_0.115, Arg_1.116)
}

ENTRY main.146 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = bf16[16,256,16,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.5 = bf16[16,16,64,256]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = bf16[16,16,256,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = bf16[16,16,256,256]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_4.5 = pred[16,1,256,256]{3,2,1,0} parameter(4), sharding={replicated}
  convert.35 = s32[16,1,256,256]{3,2,1,0} convert(Arg_4.5)
  constant.28 = s32[] constant(0)
  broadcast.29 = s32[16,1,256,256]{3,2,1,0} broadcast(constant.28), dimensions={}
  compare.36 = pred[16,1,256,256]{3,2,1,0} compare(convert.35, broadcast.29), direction=GT
  constant.30 = bf16[] constant(0)
  broadcast.1 = bf16[16,1,256,256]{3,2,1,0} broadcast(constant.30), dimensions={}
  constant.10 = bf16[] constant(-9.999e+09)
  broadcast.3 = bf16[16,1,256,256]{3,2,1,0} broadcast(constant.10), dimensions={}
  select.39 = bf16[16,1,256,256]{3,2,1,0} select(compare.36, broadcast.1, broadcast.3)
  reshape.41 = bf16[16,256,256]{2,1,0} reshape(select.39)
  broadcast.42 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.41), dimensions={0,2,3}
  Arg_3.4 = bf16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  reshape.44 = bf16[16,256,256]{2,1,0} reshape(Arg_3.4)
  broadcast.45 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.44), dimensions={1,2,3}
  add.46 = bf16[16,16,256,256]{3,2,1,0} add(broadcast.42, broadcast.45)
  add.53 = bf16[16,16,256,256]{3,2,1,0} add(dot, add.46)
  constant.31 = bf16[] constant(-inf)
  reduce.58 = bf16[16,16,256]{2,1,0} reduce(add.53, constant.31), dimensions={3}, to_apply=region_0.54
  broadcast.62 = bf16[16,16,256,256]{3,2,1,0} broadcast(reduce.58), dimensions={0,1,2}
  subtract.63 = bf16[16,16,256,256]{3,2,1,0} subtract(add.53, broadcast.62)
  exponential.64 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.63)
  convert.65 = f32[16,16,256,256]{3,2,1,0} convert(exponential.64)
  constant.11 = f32[] constant(0)
  reduce.70 = f32[16,16,256]{2,1,0} reduce(convert.65, constant.11), dimensions={3}, to_apply=region_1.66
  convert.4 = bf16[16,16,256]{2,1,0} convert(reduce.70)
  broadcast.75 = bf16[16,16,256,256]{3,2,1,0} broadcast(convert.4), dimensions={0,1,2}
  divide.76 = bf16[16,16,256,256]{3,2,1,0} divide(exponential.64, broadcast.75)
  constant.22 = u32[1]{0} constant({255383827})
  constant.21 = u32[1]{0} constant({267815257})
  constant.2 = u32[1]{0} constant({0})
  constant.23 = u32[1]{0} constant({3213575472})
  custom-call.49 = (u32[1]{0}, u32[1]{0}) custom-call(constant.22, constant.21, constant.2, constant.23), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.50 = u32[1]{0} get-tuple-element(custom-call.49), index=0
  reshape.80 = u32[] reshape(get-tuple-element.50)
  broadcast.84 = u32[32768]{0} broadcast(reshape.80), dimensions={}
  get-tuple-element.51 = u32[1]{0} get-tuple-element(custom-call.49), index=1
  reshape.81 = u32[] reshape(get-tuple-element.51)
  broadcast.85 = u32[32768]{0} broadcast(reshape.81), dimensions={}
  iota.79 = u32[65536]{0} iota(), iota_dimension=0
  slice.82 = u32[32768]{0} slice(iota.79), slice={[0:32768]}
  slice.83 = u32[32768]{0} slice(iota.79), slice={[32768:65536]}
  custom-call.86 = (u32[32768]{0}, u32[32768]{0}) custom-call(broadcast.84, broadcast.85, slice.82, slice.83), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[32768]{0}, u32[32768]{0}, u32[32768]{0}, u32[32768]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\000\000\000\000\000\000"
  get-tuple-element.87 = u32[32768]{0} get-tuple-element(custom-call.86), index=0
  get-tuple-element.88 = u32[32768]{0} get-tuple-element(custom-call.86), index=1
  concatenate.89 = u32[65536]{0} concatenate(get-tuple-element.87, get-tuple-element.88), dimensions={0}
  constant.17 = u32[] constant(9)
  broadcast.13 = u32[65536]{0} broadcast(constant.17), dimensions={}
  shift-right-logical.0 = u32[65536]{0} shift-right-logical(concatenate.89, broadcast.13)
  constant.15 = u32[] constant(1065353216)
  broadcast.21 = u32[65536]{0} broadcast(constant.15), dimensions={}
  or.0 = u32[65536]{0} or(shift-right-logical.0, broadcast.21)
  bitcast-convert.0 = f32[65536]{0} bitcast-convert(or.0)
  constant.3 = f32[] constant(-1)
  broadcast.30 = f32[65536]{0} broadcast(constant.3), dimensions={}
  add.1 = f32[65536]{0} add(bitcast-convert.0, broadcast.30)
  broadcast.31 = f32[65536]{0} broadcast(constant.11), dimensions={}
  maximum.0 = f32[65536]{0} maximum(add.1, broadcast.31)
  constant.9 = f32[] constant(0.9)
  broadcast.32 = f32[65536]{0} broadcast(constant.9), dimensions={}
  compare.0 = pred[65536]{0} compare(maximum.0, broadcast.32), direction=LT
  constant = bf16[] constant(1.109)
  broadcast.33 = bf16[65536]{0} broadcast(constant), dimensions={}
  broadcast.34 = bf16[65536]{0} broadcast(constant.30), dimensions={}
  select.2 = bf16[65536]{0} select(compare.0, broadcast.33, broadcast.34)
  reshape.39 = bf16[16,16,256]{2,1,0} reshape(select.2)
  broadcast.9 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.39), dimensions={0,1,3}
  multiply.101 = bf16[16,16,256,256]{3,2,1,0} multiply(divide.76, broadcast.9)
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.5, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.103 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  Arg_5.6 = bf16[16,256,16,64]{3,2,1,0} parameter(5), sharding={replicated}
  copy.3 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.4 = bf16[16,16,256,64]{3,2,1,0} transpose(copy.3), dimensions={0,2,1,3}
  dot.2 = bf16[16,16,256,256]{3,2,1,0} dot(transpose.4, transpose.5), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  multiply.108 = bf16[16,16,256,256]{3,2,1,0} multiply(dot.2, broadcast.9)
  divide.124 = bf16[16,16,256,256]{3,2,1,0} divide(multiply.108, broadcast.75)
  constant.19 = bf16[] constant(1)
  broadcast.24 = bf16[16,16,256]{2,1,0} broadcast(constant.19), dimensions={}
  multiply.2 = bf16[16,16,256]{2,1,0} multiply(convert.4, convert.4)
  divide.0 = bf16[16,16,256]{2,1,0} divide(broadcast.24, multiply.2)
  broadcast.111 = bf16[16,16,256,256]{3,2,1,0} broadcast(divide.0), dimensions={0,1,2}
  multiply.112 = bf16[16,16,256,256]{3,2,1,0} multiply(multiply.108, broadcast.111)
  multiply.113 = bf16[16,16,256,256]{3,2,1,0} multiply(multiply.112, exponential.64)
  reduce.118 = bf16[16,16,256]{2,1,0} reduce(multiply.113, constant.30), dimensions={3}, to_apply=region_2.114
  negate.1 = bf16[16,16,256]{2,1,0} negate(reduce.118)
  broadcast.11 = bf16[16,16,256,256]{3,2,1,0} broadcast(negate.1), dimensions={0,1,2}
  add.133 = bf16[16,16,256,256]{3,2,1,0} add(divide.124, broadcast.11)
  multiply.134 = bf16[16,16,256,256]{3,2,1,0} multiply(add.133, exponential.64)
  copy.4 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.9 = bf16[16,16,256,64]{3,2,1,0} transpose(copy.4), dimensions={0,2,1,3}
  dot.4 = bf16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose.9), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.144 = bf16[16,256,16,64]{3,1,2,0} transpose(dot.4), dimensions={0,2,1,3}
  dot.3 = bf16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.142 = bf16[16,256,16,64]{3,1,2,0} transpose(dot.3), dimensions={0,2,1,3}
  copy.5 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.104 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.5), dimensions={0,2,3,1}
  dot.106 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.104, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.107 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.106), dimensions={0,3,1,2}
  reduce.139 = bf16[16,256,256]{2,1,0} reduce(multiply.134, constant.30), dimensions={0}, to_apply=region_2.114
  reshape.140 = bf16[1,16,256,256]{3,2,1,0} reshape(reduce.139)
  tuple.145 = (bf16[16,256,16,64]{1,3,2,0}, bf16[16,256,16,64]{3,1,2,0}, bf16[16,256,16,64]{3,1,2,0}, bf16[16,256,16,64]{1,3,2,0}, bf16[1,16,256,256]{3,2,1,0}) tuple(transpose.103, transpose.144, transpose.142, transpose.107, reshape.140)
  get-tuple-element = bf16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=0
  copy.6 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element)
  get-tuple-element.1 = bf16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=1
  copy.7 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.1)
  get-tuple-element.2 = bf16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=2
  copy.8 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.2)
  get-tuple-element.3 = bf16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=3
  copy.9 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.3)
  get-tuple-element.4 = bf16[1,16,256,256]{3,2,1,0} get-tuple-element(tuple.145), index=4
  ROOT tuple = (bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[1,16,256,256]{3,2,1,0}) tuple(copy.6, copy.7, copy.8, copy.9, get-tuple-element.4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  const absl::string_view backward_target =
      kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget;
  auto dbias_index = 5;
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Copy(m::GetTupleElement(
              m::Tuple(
                  m::Transpose().WithShape(BF16, {16, 256, 16, 64}),
                  m::Transpose(m::GetTupleElement(
                                   m::CustomCall(&fmha, {backward_target}), 0))
                      .WithShape(BF16, {16, 256, 16, 64}),
                  m::Transpose(
                      m::GetTupleElement(m::CustomCall({backward_target}), 1))
                      .WithShape(BF16, {16, 256, 16, 64}),
                  m::Transpose(m::Transpose(m::GetTupleElement(
                                   m::CustomCall({backward_target}), 2)))
                      .WithShape(BF16, {16, 256, 16, 64}),
                  m::GetTupleElement(  // dbias
                      m::CustomCall({backward_target}), dbias_index)),
              0)),
          m::Op(), m::Op(), m::Op(), m::Op())));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 5);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16TrainingBmm1ScaleBiasSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0},f16[1,16,256,256]{3,2,1,0},pred[16,1,256,256]{3,2,1,0},f16[16,256,16,64]{3,2,1,0})->(f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[1,16,256,256]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true,true}

region_0.54 {
  Arg_0.55 = f16[] parameter(0)
  Arg_1.56 = f16[] parameter(1)
  ROOT maximum.57 = f16[] maximum(Arg_0.55, Arg_1.56)
}

region_1.66 {
  Arg_0.67 = f32[] parameter(0)
  Arg_1.68 = f32[] parameter(1)
  ROOT add.69 = f32[] add(Arg_0.67, Arg_1.68)
}

region_2.114 {
  Arg_0.115 = f16[] parameter(0)
  Arg_1.116 = f16[] parameter(1)
  ROOT add.117 = f16[] add(Arg_0.115, Arg_1.116)
}

ENTRY main.146 {
  Arg_2.3 = f16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = f16[16,256,16,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.5 = f16[16,16,64,256]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  Arg_0.1 = f16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = f16[16,256,16,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = f16[16,16,256,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = f16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = f16[16,256,16,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = f16[16,16,64,256]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = f16[16,16,256,256]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_4.5 = pred[16,1,256,256]{3,2,1,0} parameter(4), sharding={replicated}
  convert.35 = s32[16,1,256,256]{3,2,1,0} convert(Arg_4.5)
  constant.28 = s32[] constant(0)
  broadcast.29 = s32[16,1,256,256]{3,2,1,0} broadcast(constant.28), dimensions={}
  compare.36 = pred[16,1,256,256]{3,2,1,0} compare(convert.35, broadcast.29), direction=GT
  constant.30 = f16[] constant(0)
  broadcast.1 = f16[16,1,256,256]{3,2,1,0} broadcast(constant.30), dimensions={}
  constant.31 = f16[] constant(-inf)
  broadcast.3 = f16[16,1,256,256]{3,2,1,0} broadcast(constant.31), dimensions={}
  select.39 = f16[16,1,256,256]{3,2,1,0} select(compare.36, broadcast.1, broadcast.3)
  reshape.41 = f16[16,256,256]{2,1,0} reshape(select.39)
  broadcast.42 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.41), dimensions={0,2,3}
  Arg_3.4 = f16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  reshape.44 = f16[16,256,256]{2,1,0} reshape(Arg_3.4)
  broadcast.45 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.44), dimensions={1,2,3}
  add.46 = f16[16,16,256,256]{3,2,1,0} add(broadcast.42, broadcast.45)
  add.53 = f16[16,16,256,256]{3,2,1,0} add(dot, add.46)
  reduce.58 = f16[16,16,256]{2,1,0} reduce(add.53, constant.31), dimensions={3}, to_apply=region_0.54
  broadcast.62 = f16[16,16,256,256]{3,2,1,0} broadcast(reduce.58), dimensions={0,1,2}
  subtract.63 = f16[16,16,256,256]{3,2,1,0} subtract(add.53, broadcast.62)
  exponential.64 = f16[16,16,256,256]{3,2,1,0} exponential(subtract.63)
  convert.65 = f32[16,16,256,256]{3,2,1,0} convert(exponential.64)
  constant.11 = f32[] constant(0)
  reduce.70 = f32[16,16,256]{2,1,0} reduce(convert.65, constant.11), dimensions={3}, to_apply=region_1.66
  convert.4 = f16[16,16,256]{2,1,0} convert(reduce.70)
  broadcast.75 = f16[16,16,256,256]{3,2,1,0} broadcast(convert.4), dimensions={0,1,2}
  divide.76 = f16[16,16,256,256]{3,2,1,0} divide(exponential.64, broadcast.75)
  constant.22 = u32[1]{0} constant({255383827})
  constant.21 = u32[1]{0} constant({267815257})
  constant.2 = u32[1]{0} constant({0})
  constant.23 = u32[1]{0} constant({3213575472})
  custom-call.49 = (u32[1]{0}, u32[1]{0}) custom-call(constant.22, constant.21, constant.2, constant.23), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.50 = u32[1]{0} get-tuple-element(custom-call.49), index=0
  reshape.80 = u32[] reshape(get-tuple-element.50)
  broadcast.84 = u32[32768]{0} broadcast(reshape.80), dimensions={}
  get-tuple-element.51 = u32[1]{0} get-tuple-element(custom-call.49), index=1
  reshape.81 = u32[] reshape(get-tuple-element.51)
  broadcast.85 = u32[32768]{0} broadcast(reshape.81), dimensions={}
  iota.79 = u32[65536]{0} iota(), iota_dimension=0
  slice.82 = u32[32768]{0} slice(iota.79), slice={[0:32768]}
  slice.83 = u32[32768]{0} slice(iota.79), slice={[32768:65536]}
  custom-call.86 = (u32[32768]{0}, u32[32768]{0}) custom-call(broadcast.84, broadcast.85, slice.82, slice.83), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[32768]{0}, u32[32768]{0}, u32[32768]{0}, u32[32768]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\000\000\000\000\000\000"
  get-tuple-element.87 = u32[32768]{0} get-tuple-element(custom-call.86), index=0
  get-tuple-element.88 = u32[32768]{0} get-tuple-element(custom-call.86), index=1
  concatenate.89 = u32[65536]{0} concatenate(get-tuple-element.87, get-tuple-element.88), dimensions={0}
  constant.17 = u32[] constant(9)
  broadcast.13 = u32[65536]{0} broadcast(constant.17), dimensions={}
  shift-right-logical.0 = u32[65536]{0} shift-right-logical(concatenate.89, broadcast.13)
  constant.15 = u32[] constant(1065353216)
  broadcast.21 = u32[65536]{0} broadcast(constant.15), dimensions={}
  or.0 = u32[65536]{0} or(shift-right-logical.0, broadcast.21)
  bitcast-convert.0 = f32[65536]{0} bitcast-convert(or.0)
  constant.3 = f32[] constant(-1)
  broadcast.30 = f32[65536]{0} broadcast(constant.3), dimensions={}
  add.1 = f32[65536]{0} add(bitcast-convert.0, broadcast.30)
  broadcast.31 = f32[65536]{0} broadcast(constant.11), dimensions={}
  maximum.0 = f32[65536]{0} maximum(add.1, broadcast.31)
  constant.9 = f32[] constant(0.9)
  broadcast.32 = f32[65536]{0} broadcast(constant.9), dimensions={}
  compare.0 = pred[65536]{0} compare(maximum.0, broadcast.32), direction=LT
  constant = f16[] constant(1.1113)
  broadcast.33 = f16[65536]{0} broadcast(constant), dimensions={}
  broadcast.34 = f16[65536]{0} broadcast(constant.30), dimensions={}
  select.2 = f16[65536]{0} select(compare.0, broadcast.33, broadcast.34)
  reshape.39 = f16[16,16,256]{2,1,0} reshape(select.2)
  broadcast.9 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.39), dimensions={0,1,3}
  multiply.101 = f16[16,16,256,256]{3,2,1,0} multiply(divide.76, broadcast.9)
  dot.1 = f16[16,16,64,256]{3,2,1,0} dot(transpose.5, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.103 = f16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  Arg_5.6 = f16[16,256,16,64]{3,2,1,0} parameter(5), sharding={replicated}
  copy.3 = f16[16,256,16,64]{3,1,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.4 = f16[16,16,256,64]{3,2,1,0} transpose(copy.3), dimensions={0,2,1,3}
  dot.2 = f16[16,16,256,256]{3,2,1,0} dot(transpose.4, transpose.5), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  multiply.108 = f16[16,16,256,256]{3,2,1,0} multiply(dot.2, broadcast.9)
  divide.124 = f16[16,16,256,256]{3,2,1,0} divide(multiply.108, broadcast.75)
  constant.19 = f16[] constant(1)
  broadcast.24 = f16[16,16,256]{2,1,0} broadcast(constant.19), dimensions={}
  multiply.2 = f16[16,16,256]{2,1,0} multiply(convert.4, convert.4)
  divide.0 = f16[16,16,256]{2,1,0} divide(broadcast.24, multiply.2)
  broadcast.111 = f16[16,16,256,256]{3,2,1,0} broadcast(divide.0), dimensions={0,1,2}
  multiply.112 = f16[16,16,256,256]{3,2,1,0} multiply(multiply.108, broadcast.111)
  multiply.113 = f16[16,16,256,256]{3,2,1,0} multiply(multiply.112, exponential.64)
  reduce.118 = f16[16,16,256]{2,1,0} reduce(multiply.113, constant.30), dimensions={3}, to_apply=region_2.114
  negate.1 = f16[16,16,256]{2,1,0} negate(reduce.118)
  broadcast.11 = f16[16,16,256,256]{3,2,1,0} broadcast(negate.1), dimensions={0,1,2}
  add.133 = f16[16,16,256,256]{3,2,1,0} add(divide.124, broadcast.11)
  multiply.134 = f16[16,16,256,256]{3,2,1,0} multiply(add.133, exponential.64)
  copy.4 = f16[16,256,16,64]{3,1,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.9 = f16[16,16,256,64]{3,2,1,0} transpose(copy.4), dimensions={0,2,1,3}
  dot.4 = f16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose.9), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.144 = f16[16,256,16,64]{3,1,2,0} transpose(dot.4), dimensions={0,2,1,3}
  dot.3 = f16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.142 = f16[16,256,16,64]{3,1,2,0} transpose(dot.3), dimensions={0,2,1,3}
  copy.5 = f16[16,256,16,64]{1,3,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.104 = f16[16,16,64,256]{3,2,1,0} transpose(copy.5), dimensions={0,2,3,1}
  dot.106 = f16[16,16,64,256]{3,2,1,0} dot(transpose.104, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.107 = f16[16,256,16,64]{1,3,2,0} transpose(dot.106), dimensions={0,3,1,2}
  reduce.139 = f16[16,256,256]{2,1,0} reduce(multiply.134, constant.30), dimensions={0}, to_apply=region_2.114
  reshape.140 = f16[1,16,256,256]{3,2,1,0} reshape(reduce.139)
  tuple.145 = (f16[16,256,16,64]{1,3,2,0}, f16[16,256,16,64]{3,1,2,0}, f16[16,256,16,64]{3,1,2,0}, f16[16,256,16,64]{1,3,2,0}, f16[1,16,256,256]{3,2,1,0}) tuple(transpose.103, transpose.144, transpose.142, transpose.107, reshape.140)
  get-tuple-element = f16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=0
  copy.6 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element)
  get-tuple-element.1 = f16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=1
  copy.7 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.1)
  get-tuple-element.2 = f16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=2
  copy.8 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.2)
  get-tuple-element.3 = f16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=3
  copy.9 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.3)
  get-tuple-element.4 = f16[1,16,256,256]{3,2,1,0} get-tuple-element(tuple.145), index=4
  ROOT tuple = (f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[1,16,256,256]{3,2,1,0}) tuple(copy.6, copy.7, copy.8, copy.9, get-tuple-element.4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  const absl::string_view backward_target =
      kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget;
  auto dbias_index = 5;
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Copy(m::GetTupleElement(
              m::Tuple(
                  m::Transpose().WithShape(F16, {16, 256, 16, 64}),
                  m::Transpose(m::GetTupleElement(
                                   m::CustomCall(&fmha, {backward_target}), 0))
                      .WithShape(F16, {16, 256, 16, 64}),
                  m::Transpose(
                      m::GetTupleElement(m::CustomCall({backward_target}), 1))
                      .WithShape(F16, {16, 256, 16, 64}),
                  m::Transpose(m::Transpose(m::GetTupleElement(
                                   m::CustomCall({backward_target}), 2)))
                      .WithShape(F16, {16, 256, 16, 64}),
                  m::GetTupleElement(  // dbias
                      m::CustomCall({backward_target}), dbias_index)),
              0)),
          m::Op(), m::Op(), m::Op(), m::Op())));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 5);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16TrainingBmm1ScaleBiasSoftmaxDropoutBmm2WithTransposeFusion) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0},f16[1,16,256,256]{3,2,1,0},pred[16,1,256,256]{3,2,1,0},f16[16,256,16,64]{3,2,1,0})->(f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[1,16,256,256]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true,true}

region_0.54 {
  Arg_0.55 = f16[] parameter(0)
  Arg_1.56 = f16[] parameter(1)
  ROOT maximum.57 = f16[] maximum(Arg_0.55, Arg_1.56)
}

region_1.66 {
  Arg_0.67 = f32[] parameter(0)
  Arg_1.68 = f32[] parameter(1)
  ROOT add.69 = f32[] add(Arg_0.67, Arg_1.68)
}

region_2.114 {
  Arg_0.115 = f16[] parameter(0)
  Arg_1.116 = f16[] parameter(1)
  ROOT add.117 = f16[] add(Arg_0.115, Arg_1.116)
}

ENTRY main.146 {
  Arg_2.3 = f16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = f16[16,256,16,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.5 = f16[16,16,64,256]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  Arg_0.1 = f16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = f16[16,256,16,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = f16[16,16,256,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = f16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = f16[16,256,16,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = f16[16,16,64,256]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = f16[16,16,256,256]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_4.5 = pred[16,1,256,256]{3,2,1,0} parameter(4), sharding={replicated}
  convert.35 = s32[16,1,256,256]{3,2,1,0} convert(Arg_4.5)
  constant.28 = s32[] constant(0)
  broadcast.29 = s32[16,1,256,256]{3,2,1,0} broadcast(constant.28), dimensions={}
  compare.36 = pred[16,1,256,256]{3,2,1,0} compare(convert.35, broadcast.29), direction=GT
  constant.30 = f16[] constant(0)
  broadcast.1 = f16[16,1,256,256]{3,2,1,0} broadcast(constant.30), dimensions={}
  constant.31 = f16[] constant(-inf)
  broadcast.3 = f16[16,1,256,256]{3,2,1,0} broadcast(constant.31), dimensions={}
  select.39 = f16[16,1,256,256]{3,2,1,0} select(compare.36, broadcast.1, broadcast.3)
  reshape.41 = f16[16,256,256]{2,1,0} reshape(select.39)
  broadcast.42 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.41), dimensions={0,2,3}
  Arg_3.4 = f16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  reshape.44 = f16[16,256,256]{2,1,0} reshape(Arg_3.4)
  broadcast.45 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.44), dimensions={1,2,3}
  add.46 = f16[16,16,256,256]{3,2,1,0} add(broadcast.42, broadcast.45)
  add.53 = f16[16,16,256,256]{3,2,1,0} add(dot, add.46)
  reduce.58 = f16[16,16,256]{2,1,0} reduce(add.53, constant.31), dimensions={3}, to_apply=region_0.54
  broadcast.62 = f16[16,16,256,256]{3,2,1,0} broadcast(reduce.58), dimensions={0,1,2}
  subtract.63 = f16[16,16,256,256]{3,2,1,0} subtract(add.53, broadcast.62)
  exponential.64 = f16[16,16,256,256]{3,2,1,0} exponential(subtract.63)
  convert.65 = f32[16,16,256,256]{3,2,1,0} convert(exponential.64)
  constant.11 = f32[] constant(0)
  reduce.70 = f32[16,16,256]{2,1,0} reduce(convert.65, constant.11), dimensions={3}, to_apply=region_1.66
  convert.4 = f16[16,16,256]{2,1,0} convert(reduce.70)
  broadcast.75 = f16[16,16,256,256]{3,2,1,0} broadcast(convert.4), dimensions={0,1,2}
  divide.76 = f16[16,16,256,256]{3,2,1,0} divide(exponential.64, broadcast.75)
  constant.22 = u32[1]{0} constant({255383827})
  constant.21 = u32[1]{0} constant({267815257})
  constant.2 = u32[1]{0} constant({0})
  constant.23 = u32[1]{0} constant({3213575472})
  custom-call.49 = (u32[1]{0}, u32[1]{0}) custom-call(constant.22, constant.21, constant.2, constant.23), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.50 = u32[1]{0} get-tuple-element(custom-call.49), index=0
  reshape.80 = u32[] reshape(get-tuple-element.50)
  broadcast.84 = u32[32768]{0} broadcast(reshape.80), dimensions={}
  get-tuple-element.51 = u32[1]{0} get-tuple-element(custom-call.49), index=1
  reshape.81 = u32[] reshape(get-tuple-element.51)
  broadcast.85 = u32[32768]{0} broadcast(reshape.81), dimensions={}
  iota.79 = u32[65536]{0} iota(), iota_dimension=0
  slice.82 = u32[32768]{0} slice(iota.79), slice={[0:32768]}
  slice.83 = u32[32768]{0} slice(iota.79), slice={[32768:65536]}
  custom-call.86 = (u32[32768]{0}, u32[32768]{0}) custom-call(broadcast.84, broadcast.85, slice.82, slice.83), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[32768]{0}, u32[32768]{0}, u32[32768]{0}, u32[32768]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\000\000\000\000\000\000"
  get-tuple-element.87 = u32[32768]{0} get-tuple-element(custom-call.86), index=0
  get-tuple-element.88 = u32[32768]{0} get-tuple-element(custom-call.86), index=1
  concatenate.89 = u32[65536]{0} concatenate(get-tuple-element.87, get-tuple-element.88), dimensions={0}
  constant.17 = u32[] constant(9)
  broadcast.13 = u32[65536]{0} broadcast(constant.17), dimensions={}
  shift-right-logical.0 = u32[65536]{0} shift-right-logical(concatenate.89, broadcast.13)
  constant.15 = u32[] constant(1065353216)
  broadcast.21 = u32[65536]{0} broadcast(constant.15), dimensions={}
  or.0 = u32[65536]{0} or(shift-right-logical.0, broadcast.21)
  bitcast-convert.0 = f32[65536]{0} bitcast-convert(or.0)
  constant.3 = f32[] constant(-1)
  broadcast.30 = f32[65536]{0} broadcast(constant.3), dimensions={}
  add.1 = f32[65536]{0} add(bitcast-convert.0, broadcast.30)
  broadcast.31 = f32[65536]{0} broadcast(constant.11), dimensions={}
  maximum.0 = f32[65536]{0} maximum(add.1, broadcast.31)
  constant.9 = f32[] constant(0.9)
  broadcast.32 = f32[65536]{0} broadcast(constant.9), dimensions={}
  compare.0 = pred[65536]{0} compare(maximum.0, broadcast.32), direction=LT
  constant = f16[] constant(1.1113)
  broadcast.33 = f16[65536]{0} broadcast(constant), dimensions={}
  broadcast.34 = f16[65536]{0} broadcast(constant.30), dimensions={}
  select.2 = f16[65536]{0} select(compare.0, broadcast.33, broadcast.34)
  reshape.39 = f16[16,16,256]{2,1,0} reshape(select.2)
  broadcast.9 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.39), dimensions={0,1,3}
  multiply.101 = f16[16,16,256,256]{3,2,1,0} multiply(divide.76, broadcast.9)
  dot.1 = f16[16,16,64,256]{3,2,1,0} dot(transpose.5, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.103 = f16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  Arg_5.6 = f16[16,256,16,64]{3,2,1,0} parameter(5), sharding={replicated}
  copy.3 = f16[16,256,16,64]{3,1,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.4 = f16[16,16,256,64]{3,2,1,0} transpose(copy.3), dimensions={0,2,1,3}
  dot.2 = f16[16,16,256,256]{3,2,1,0} dot(transpose.4, transpose.5), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  multiply.108 = f16[16,16,256,256]{3,2,1,0} multiply(dot.2, broadcast.9)
  divide.124 = f16[16,16,256,256]{3,2,1,0} divide(multiply.108, broadcast.75)
  constant.19 = f16[] constant(1)
  broadcast.24 = f16[16,16,256]{2,1,0} broadcast(constant.19), dimensions={}
  multiply.2 = f16[16,16,256]{2,1,0} multiply(convert.4, convert.4)
  divide.0 = f16[16,16,256]{2,1,0} divide(broadcast.24, multiply.2)
  broadcast.111 = f16[16,16,256,256]{3,2,1,0} broadcast(divide.0), dimensions={0,1,2}
  multiply.112 = f16[16,16,256,256]{3,2,1,0} multiply(multiply.108, broadcast.111)
  multiply.113 = f16[16,16,256,256]{3,2,1,0} multiply(multiply.112, exponential.64)
  reduce.118 = f16[16,16,256]{2,1,0} reduce(multiply.113, constant.30), dimensions={3}, to_apply=region_2.114
  negate.1 = f16[16,16,256]{2,1,0} negate(reduce.118)
  broadcast.11 = f16[16,16,256,256]{3,2,1,0} broadcast(negate.1), dimensions={0,1,2}
  add.133 = f16[16,16,256,256]{3,2,1,0} add(divide.124, broadcast.11)
  multiply.134 = f16[16,16,256,256]{3,2,1,0} multiply(add.133, exponential.64)
  copy.4 = f16[16,256,16,64]{3,1,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.9 = f16[16,16,256,64]{3,2,1,0} transpose(copy.4), dimensions={0,2,1,3}
  dot.4 = f16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose.9), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.144 = f16[16,256,16,64]{3,1,2,0} transpose(dot.4), dimensions={0,2,1,3}
  dot.3 = f16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.142 = f16[16,256,16,64]{3,1,2,0} transpose(dot.3), dimensions={0,2,1,3}
  copy.5 = f16[16,256,16,64]{1,3,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.104 = f16[16,16,64,256]{3,2,1,0} transpose(copy.5), dimensions={0,2,3,1}
  dot.106 = f16[16,16,64,256]{3,2,1,0} dot(transpose.104, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.107 = f16[16,256,16,64]{1,3,2,0} transpose(dot.106), dimensions={0,3,1,2}
  reduce.139 = f16[16,256,256]{2,1,0} reduce(multiply.134, constant.30), dimensions={0}, to_apply=region_2.114
  reshape.140 = f16[1,16,256,256]{3,2,1,0} reshape(reduce.139)
  tuple.145 = (f16[16,256,16,64]{1,3,2,0}, f16[16,256,16,64]{3,1,2,0}, f16[16,256,16,64]{3,1,2,0}, f16[16,256,16,64]{1,3,2,0}, f16[1,16,256,256]{3,2,1,0}) tuple(transpose.103, transpose.144, transpose.142, transpose.107, reshape.140)
  get-tuple-element = f16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=0
  copy.6 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element)
  get-tuple-element.1 = f16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=1
  copy.7 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.1)
  get-tuple-element.2 = f16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=2
  copy.8 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.2)
  get-tuple-element.3 = f16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=3
  copy.9 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.3)
  get-tuple-element.4 = f16[1,16,256,256]{3,2,1,0} get-tuple-element(tuple.145), index=4
  ROOT tuple = (f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[1,16,256,256]{3,2,1,0}) tuple(copy.6, copy.7, copy.8, copy.9, get-tuple-element.4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  AlgebraicSimplifierOptions alg_sim_options;
  alg_sim_options.set_supports_non_canonical_dots(false);
  alg_sim_options.set_is_layout_sensitive(true);
  alg_sim_options.set_enable_conv_operand_swap(false);
  AlgebraicSimplifier alge_simp{alg_sim_options};

  LayoutNormalization layout_normalizer;
  HloCSE cse{/*is_layout_sensitive=*/true};
  TF_ASSERT_OK(RunHloPass(&layout_normalizer, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&cse, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&alge_simp, m.get()).status());

  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  CudnnFusedMHATransposeFusion fmha_transpose_fusion;

  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&alge_simp, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&fmha_transpose_fusion, m.get()).status());

  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  auto dbias_index = 5;
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Bitcast().WithShape(F16, {16, 256, 16, 64}),
          m::Bitcast(
              m::GetTupleElement(
                  m::CustomCall(
                      &fmha,
                      {kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget}),
                  0))
              .WithShape(F16, {16, 256, 16, 64}),
          m::Bitcast(
              m::GetTupleElement(
                  m::CustomCall(
                      {kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget}),
                  1))
              .WithShape(F16, {16, 256, 16, 64}),
          m::Bitcast(
              m::GetTupleElement(
                  m::CustomCall(
                      {kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget}),
                  2))
              .WithShape(F16, {16, 256, 16, 64}),
          m::GetTupleElement(  // dbias
              m::CustomCall(
                  {kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget}),
              dbias_index))));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 5);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16MiniT5xTest) {
  const char* module_str = R"(
HloModule jit__lambda_, entry_computation_layout={(bf16[12,512,32,64]{3,2,1,0},bf16[12,512,2,32,64]{4,3,2,1,0},f32[12,512]{1,0},f32[12,512]{1,0})->(bf16[], bf16[12,512,32,64]{3,2,1,0}, bf16[12,512,2,32,64]{4,3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true}

region_0.51 {
  Arg_0.52 = bf16[] parameter(0)
  Arg_1.53 = bf16[] parameter(1)
  ROOT maximum.54 = bf16[] maximum(Arg_0.52, Arg_1.53)
}

region_1.63 {
  Arg_0.64 = f32[] parameter(0)
  Arg_1.65 = f32[] parameter(1)
  ROOT add.66 = f32[] add(Arg_0.64, Arg_1.65)
}

region_3.99 {
  Arg_0.100 = bf16[] parameter(0)
  Arg_1.101 = bf16[] parameter(1)
  ROOT add.102 = bf16[] add(Arg_0.100, Arg_1.101)
}

ENTRY main.129 {
  Arg_1.2 = bf16[12,512,2,32,64]{4,3,2,1,0} parameter(1), sharding={replicated}
  copy = bf16[12,512,2,32,64]{1,4,3,0,2} copy(Arg_1.2), sharding={replicated}
  slice.42 = bf16[12,512,1,32,64]{1,4,3,0,2} slice(copy), slice={[0:12], [0:512], [1:2], [0:32], [0:64]}
  reshape.44 = bf16[12,512,32,64]{1,3,2,0} reshape(slice.42)
  transpose.5 = bf16[12,32,64,512]{3,2,1,0} transpose(reshape.44), dimensions={0,2,3,1}
  Arg_0.1 = bf16[12,512,32,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = bf16[12,512,32,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  constant.5 = bf16[] constant(0.125)
  broadcast.6 = bf16[12,512,32,64]{3,1,2,0} broadcast(constant.5), dimensions={}
  multiply.45 = bf16[12,512,32,64]{3,1,2,0} multiply(copy.1, broadcast.6)
  transpose = bf16[12,32,512,64]{3,2,1,0} transpose(multiply.45), dimensions={0,2,1,3}
  copy.2 = bf16[12,512,2,32,64]{1,4,3,0,2} copy(Arg_1.2), sharding={replicated}
  slice.41 = bf16[12,512,1,32,64]{1,4,3,0,2} slice(copy.2), slice={[0:12], [0:512], [0:1], [0:32], [0:64]}
  reshape.43 = bf16[12,512,32,64]{1,3,2,0} reshape(slice.41)
  transpose.1 = bf16[12,32,64,512]{3,2,1,0} transpose(reshape.43), dimensions={0,2,3,1}
  dot = bf16[12,32,512,512]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_2.3 = f32[12,512]{1,0} parameter(2), sharding={replicated}
  constant.14 = f32[] constant(0)
  broadcast.19 = f32[12,512]{1,0} broadcast(constant.14), dimensions={}
  compare.24 = pred[12,512]{1,0} compare(Arg_2.3, broadcast.19), direction=GT
  broadcast.30 = pred[12,512,512]{2,1,0} broadcast(compare.24), dimensions={0,1}
  Arg_3.4 = f32[12,512]{1,0} parameter(3), sharding={replicated}
  compare.25 = pred[12,512]{1,0} compare(Arg_3.4, broadcast.19), direction=GT
  broadcast.33 = pred[12,512,512]{2,1,0} broadcast(compare.25), dimensions={0,2}
  and.34 = pred[12,512,512]{2,1,0} and(broadcast.30, broadcast.33)
  convert.4 = s32[12,512,512]{2,1,0} convert(and.34)
  constant.16 = s32[] constant(0)
  broadcast.21 = s32[12,512,512]{2,1,0} broadcast(constant.16), dimensions={}
  compare.0 = pred[12,512,512]{2,1,0} compare(convert.4, broadcast.21), direction=GT
  constant.20 = bf16[] constant(0)
  broadcast.22 = bf16[12,512,512]{2,1,0} broadcast(constant.20), dimensions={}
  constant.11 = bf16[] constant(-9.999e+09)
  broadcast.23 = bf16[12,512,512]{2,1,0} broadcast(constant.11), dimensions={}
  select.0 = bf16[12,512,512]{2,1,0} select(compare.0, broadcast.22, broadcast.23)
  broadcast.49 = bf16[12,32,512,512]{3,2,1,0} broadcast(select.0), dimensions={0,2,3}
  add.50 = bf16[12,32,512,512]{3,2,1,0} add(dot, broadcast.49)
  constant.22 = bf16[] constant(-inf)
  reduce.55 = bf16[12,32,512]{2,1,0} reduce(add.50, constant.22), dimensions={3}, to_apply=region_0.51
  broadcast.59 = bf16[12,32,512,512]{3,2,1,0} broadcast(reduce.55), dimensions={0,1,2}
  subtract.60 = bf16[12,32,512,512]{3,2,1,0} subtract(add.50, broadcast.59)
  exponential.61 = bf16[12,32,512,512]{3,2,1,0} exponential(subtract.60)
  convert.62 = f32[12,32,512,512]{3,2,1,0} convert(exponential.61)
  reduce.67 = f32[12,32,512]{2,1,0} reduce(convert.62, constant.14), dimensions={3}, to_apply=region_1.63
  convert.5 = bf16[12,32,512]{2,1,0} convert(reduce.67)
  broadcast.72 = bf16[12,32,512,512]{3,2,1,0} broadcast(convert.5), dimensions={0,1,2}
  divide.73 = bf16[12,32,512,512]{3,2,1,0} divide(exponential.61, broadcast.72)
  dot.1 = bf16[12,32,64,512]{3,2,1,0} dot(transpose.5, divide.73), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  convert.6 = f32[12,32,64,512]{3,2,1,0} convert(dot.1)
  reduce.83 = f32[] reduce(convert.6, constant.14), dimensions={0,3,1,2}, to_apply=region_1.63
  convert.84 = bf16[] convert(reduce.83)
  constant.2 = bf16[] constant(0.0007935)
  multiply.86 = bf16[] multiply(convert.84, constant.2)
  broadcast.9 = bf16[12,32,512,64]{3,2,1,0} broadcast(constant.2), dimensions={}
  dot.2 = bf16[12,32,512,512]{3,2,1,0} dot(broadcast.9, transpose.5), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  divide.109 = bf16[12,32,512,512]{3,2,1,0} divide(dot.2, broadcast.72)
  constant.10 = bf16[] constant(1)
  broadcast.24 = bf16[12,32,512]{2,1,0} broadcast(constant.10), dimensions={}
  multiply.4 = bf16[12,32,512]{2,1,0} multiply(convert.5, convert.5)
  divide.0 = bf16[12,32,512]{2,1,0} divide(broadcast.24, multiply.4)
  broadcast.96 = bf16[12,32,512,512]{3,2,1,0} broadcast(divide.0), dimensions={0,1,2}
  multiply.97 = bf16[12,32,512,512]{3,2,1,0} multiply(dot.2, broadcast.96)
  multiply.98 = bf16[12,32,512,512]{3,2,1,0} multiply(multiply.97, exponential.61)
  reduce.103 = bf16[12,32,512]{2,1,0} reduce(multiply.98, constant.20), dimensions={3}, to_apply=region_3.99
  negate.0 = bf16[12,32,512]{2,1,0} negate(reduce.103)
  broadcast.10 = bf16[12,32,512,512]{3,2,1,0} broadcast(negate.0), dimensions={0,1,2}
  add.118 = bf16[12,32,512,512]{3,2,1,0} add(divide.109, broadcast.10)
  multiply.119 = bf16[12,32,512,512]{3,2,1,0} multiply(add.118, exponential.61)
  transpose.9 = bf16[12,32,512,64]{2,3,1,0} transpose(reshape.43), dimensions={0,2,1,3}
  copy.3 = bf16[12,32,512,64]{3,2,1,0} copy(transpose.9)
  dot.4 = bf16[12,32,512,64]{3,2,1,0} dot(multiply.119, copy.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  broadcast.12 = bf16[12,32,512,64]{3,2,1,0} broadcast(constant.5), dimensions={}
  multiply.3 = bf16[12,32,512,64]{3,2,1,0} multiply(dot.4, broadcast.12)
  transpose.11 = bf16[12,512,32,64]{3,1,2,0} transpose(multiply.3), dimensions={0,2,1,3}
  broadcast.7 = bf16[12,32,64,512]{3,2,1,0} broadcast(constant.2), dimensions={}
  dot.90 = bf16[12,32,64,512]{3,2,1,0} dot(broadcast.7, divide.73), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.91 = bf16[12,512,32,64]{1,3,2,0} transpose(dot.90), dimensions={0,3,1,2}
  reshape.92 = bf16[12,512,1,32,64]{1,4,3,0,2} reshape(transpose.91)
  pad.93 = bf16[12,512,2,32,64]{1,4,3,0,2} pad(reshape.92, constant.20), padding=0_0x0_0x1_0x0_0x0_0
  dot.3 = bf16[12,32,512,64]{3,2,1,0} dot(multiply.119, transpose), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  copy.4 = bf16[12,32,512,64]{2,3,1,0} copy(dot.3)
  transpose.121 = bf16[12,512,32,64]{1,3,2,0} transpose(copy.4), dimensions={0,2,1,3}
  reshape.124 = bf16[12,512,1,32,64]{1,4,3,0,2} reshape(transpose.121)
  pad.125 = bf16[12,512,2,32,64]{1,4,3,0,2} pad(reshape.124, constant.20), padding=0_0x0_0x0_1x0_0x0_0
  add.126 = bf16[12,512,2,32,64]{1,4,3,0,2} add(pad.93, pad.125)
  tuple.128 = (bf16[], bf16[12,512,32,64]{3,1,2,0}, bf16[12,512,2,32,64]{1,4,3,0,2}) tuple(multiply.86, transpose.11, add.126)
  get-tuple-element = bf16[] get-tuple-element(tuple.128), index=0
  get-tuple-element.1 = bf16[12,512,32,64]{3,1,2,0} get-tuple-element(tuple.128), index=1
  copy.5 = bf16[12,512,32,64]{3,2,1,0} copy(get-tuple-element.1)
  get-tuple-element.2 = bf16[12,512,2,32,64]{1,4,3,0,2} get-tuple-element(tuple.128), index=2
  copy.6 = bf16[12,512,2,32,64]{4,3,2,1,0} copy(get-tuple-element.2)
  ROOT tuple = (bf16[], bf16[12,512,32,64]{3,2,1,0}, bf16[12,512,2,32,64]{4,3,2,1,0}) tuple(get-tuple-element, copy.5, copy.6)
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  AlgebraicSimplifierOptions alg_sim_options;
  alg_sim_options.set_supports_non_canonical_dots(false);
  alg_sim_options.set_is_layout_sensitive(true);
  alg_sim_options.set_enable_conv_operand_swap(false);
  AlgebraicSimplifier alge_simp{alg_sim_options};
  ReshapeDecomposer reshape_decomposer;
  LayoutNormalization layout_normalizer;
  HloCSE cse{/*is_layout_sensitive=*/true};
  TF_ASSERT_OK(RunHloPass(&reshape_decomposer, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&layout_normalizer, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&cse, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&alge_simp, m.get()).status());

  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  CudnnFusedMHATransposeFusion fmha_transpose_fusion;

  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&alge_simp, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&fmha_transpose_fusion, m.get()).status());

  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  EXPECT_EQ(CountFusedAttentionCall(m.get()), 1);
  EXPECT_EQ(CountFusedAttentionCall(m.get(), /*is_backward*/ true), 1);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16TrainingBmm1ScaleBiasMaskSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,128,64]{3,2,1,0},bf16[2,6,64,128]{3,2,1,0},bf16[2,6,128,64]{3,2,1,0},bf16[2,6,128,64]{3,2,1,0})->(bf16[2,6,128,64]{3,2,1,0}, bf16[2,6,128,64]{3,2,1,0}, bf16[2,6,64,128]{3,2,1,0}, bf16[2,6,128,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

region_0.38 {
  Arg_0.39 = bf16[] parameter(0)
  Arg_1.40 = bf16[] parameter(1)
  ROOT maximum.1 = bf16[] maximum(Arg_0.39, Arg_1.40)
}

region_1.50 {
  Arg_0.51 = f32[] parameter(0)
  Arg_1.52 = f32[] parameter(1)
  ROOT add.2 = f32[] add(Arg_0.51, Arg_1.52)
}

region_2.99 {
  Arg_0.100 = bf16[] parameter(0)
  Arg_1.101 = bf16[] parameter(1)
  ROOT add.3 = bf16[] add(Arg_0.100, Arg_1.101)
}

ENTRY main.126 {
  constant.6 = u32[1]{0} constant({2718843009})
  constant.8 = u32[1]{0} constant({1272950319})
  constant.10 = u32[1]{0} constant({0})
  constant.12 = u32[1]{0} constant({2711844646})
  custom-call.65 = (u32[1]{0}, u32[1]{0}) custom-call(constant.6, constant.8, constant.10, constant.12), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.66 = u32[1]{0} get-tuple-element(custom-call.65), index=0
  bitcast.343 = u32[] bitcast(get-tuple-element.66)
  broadcast.27 = u32[98304]{0} broadcast(bitcast.343), dimensions={}
  get-tuple-element.67 = u32[1]{0} get-tuple-element(custom-call.65), index=1
  bitcast.344 = u32[] bitcast(get-tuple-element.67)
  broadcast.28 = u32[98304]{0} broadcast(bitcast.344), dimensions={}
  iota.68 = u32[196608]{0} iota(), iota_dimension=0
  slice = u32[98304]{0} slice(iota.68), slice={[0:98304]}
  slice.1 = u32[98304]{0} slice(iota.68), slice={[98304:196608]}
  custom-call.75 = (u32[98304]{0}, u32[98304]{0}) custom-call(broadcast.27, broadcast.28, slice, slice.1), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[98304]{0}, u32[98304]{0}, u32[98304]{0}, u32[98304]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\001\000\000\000\000\000"
  get-tuple-element.76 = u32[98304]{0} get-tuple-element(custom-call.75), index=0
  get-tuple-element.77 = u32[98304]{0} get-tuple-element(custom-call.75), index=1
  concatenate.2 = u32[196608]{0} concatenate(get-tuple-element.76, get-tuple-element.77), dimensions={0}
  constant.56 = u32[] constant(9)
  broadcast.63 = u32[196608]{0} broadcast(constant.56), dimensions={}
  shift-right-logical.3 = u32[196608]{0} shift-right-logical(concatenate.2, broadcast.63)
  constant.57 = u32[] constant(1065353216)
  broadcast.64 = u32[196608]{0} broadcast(constant.57), dimensions={}
  or.3 = u32[196608]{0} or(shift-right-logical.3, broadcast.64)
  bitcast-convert.3 = f32[196608]{0} bitcast-convert(or.3)
  constant.58 = f32[] constant(-1)
  broadcast.65 = f32[196608]{0} broadcast(constant.58), dimensions={}
  add.10 = f32[196608]{0} add(bitcast-convert.3, broadcast.65)
  constant.48 = f32[] constant(0)
  broadcast.66 = f32[196608]{0} broadcast(constant.48), dimensions={}
  maximum.4 = f32[196608]{0} maximum(add.10, broadcast.66)
  constant.59 = f32[] constant(0.9)
  broadcast.67 = f32[196608]{0} broadcast(constant.59), dimensions={}
  compare.3 = pred[196608]{0} compare(maximum.4, broadcast.67), direction=LT
  bitcast.308 = pred[2,6,128,128]{3,2,1,0} bitcast(compare.3)
  constant.44 = pred[2,6,128,128]{3,2,1,0} constant({...})
  Arg_0.1 = bf16[2,6,128,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = bf16[2,6,64,128]{3,2,1,0} parameter(1), sharding={replicated}
  dot.34 = bf16[2,6,128,128]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.55 = bf16[] constant(2)
  broadcast.61 = bf16[2,6,128,128]{3,2,1,0} broadcast(constant.55), dimensions={}
  multiply.8 = bf16[2,6,128,128]{3,2,1,0} multiply(dot.34, broadcast.61)
  constant.52 = bf16[] constant(1)
  broadcast.39 = bf16[2,6,128,128]{3,2,1,0} broadcast(constant.52), dimensions={}
  add.6 = bf16[2,6,128,128]{3,2,1,0} add(multiply.8, broadcast.39)
  constant.54 = bf16[] constant(0)
  broadcast.52 = bf16[2,6,128,128]{3,2,1,0} broadcast(constant.54), dimensions={}
  select.1 = bf16[2,6,128,128]{3,2,1,0} select(constant.44, add.6, broadcast.52)
  constant.41 = bf16[] constant(-inf)
  reduce.42 = bf16[2,6,128]{2,1,0} reduce(select.1, constant.41), dimensions={3}, to_apply=region_0.38
  broadcast.42 = bf16[2,6,128,128]{3,2,1,0} broadcast(reduce.42), dimensions={0,1,2}
  subtract.1 = bf16[2,6,128,128]{3,2,1,0} subtract(select.1, broadcast.42)
  exponential.1 = bf16[2,6,128,128]{3,2,1,0} exponential(subtract.1)
  convert.5 = f32[2,6,128,128]{3,2,1,0} convert(exponential.1)
  reduce.54 = f32[2,6,128]{2,1,0} reduce(convert.5, constant.48), dimensions={3}, to_apply=region_1.50
  convert.9 = bf16[2,6,128]{2,1,0} convert(reduce.54)
  broadcast.68 = bf16[2,6,128,128]{3,2,1,0} broadcast(convert.9), dimensions={0,1,2}
  divide.5 = bf16[2,6,128,128]{3,2,1,0} divide(exponential.1, broadcast.68)
  constant.60 = bf16[] constant(1.109)
  broadcast.69 = bf16[2,6,128,128]{3,2,1,0} broadcast(constant.60), dimensions={}
  multiply.20 = bf16[2,6,128,128]{3,2,1,0} multiply(divide.5, broadcast.69)
  select.8 = bf16[2,6,128,128]{3,2,1,0} select(bitcast.308, multiply.20, broadcast.52)
  Arg_2.3 = bf16[2,6,128,64]{3,2,1,0} parameter(2), sharding={replicated}
  dot.88 = bf16[2,6,128,64]{3,2,1,0} dot(select.8, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  bitcast.248 = pred[2,6,128,128]{3,2,1,0} bitcast(compare.3)
  Arg_3.4 = bf16[2,6,128,64]{3,2,1,0} parameter(3), sharding={replicated}
  dot.91 = bf16[2,6,128,128]{3,2,1,0} dot(Arg_3.4, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  select.6 = bf16[2,6,128,128]{3,2,1,0} select(bitcast.248, dot.91, broadcast.52)
  multiply.17 = bf16[2,6,128,128]{3,2,1,0} multiply(select.6, broadcast.69)
  divide.4 = bf16[2,6,128,128]{3,2,1,0} divide(multiply.17, broadcast.68)
  broadcast.55 = bf16[2,6,128]{2,1,0} broadcast(constant.52), dimensions={}
  multiply.11 = bf16[2,6,128]{2,1,0} multiply(convert.9, convert.9)
  divide.3 = bf16[2,6,128]{2,1,0} divide(broadcast.55, multiply.11)
  broadcast.56 = bf16[2,6,128]{2,1,0} broadcast(constant.60), dimensions={}
  multiply.12 = bf16[2,6,128]{2,1,0} multiply(divide.3, broadcast.56)
  broadcast.58 = bf16[2,6,128,128]{3,2,1,0} broadcast(multiply.12), dimensions={0,1,2}
  multiply.13 = bf16[2,6,128,128]{3,2,1,0} multiply(select.6, broadcast.58)
  multiply.14 = bf16[2,6,128,128]{3,2,1,0} multiply(multiply.13, exponential.1)
  reduce.103 = bf16[2,6,128]{2,1,0} reduce(multiply.14, constant.54), dimensions={3}, to_apply=region_2.99
  negate.3 = bf16[2,6,128]{2,1,0} negate(reduce.103)
  broadcast.62 = bf16[2,6,128,128]{3,2,1,0} broadcast(negate.3), dimensions={0,1,2}
  add.9 = bf16[2,6,128,128]{3,2,1,0} add(divide.4, broadcast.62)
  multiply.18 = bf16[2,6,128,128]{3,2,1,0} multiply(add.9, exponential.1)
  select.7 = bf16[2,6,128,128]{3,2,1,0} select(constant.44, multiply.18, broadcast.52)
  multiply.19 = bf16[2,6,128,128]{3,2,1,0} multiply(select.7, broadcast.61)
  dot.124 = bf16[2,6,128,64]{3,2,1,0} dot(multiply.19, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = bf16[2,6,64,128]{3,2,1,0} dot(Arg_0.1, multiply.19), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = bf16[2,6,128,64]{3,2,1,0} dot(select.8, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.125 = (bf16[2,6,128,64]{3,2,1,0}, bf16[2,6,128,64]{3,2,1,0}, bf16[2,6,64,128]{3,2,1,0}, bf16[2,6,128,64]{3,2,1,0}) tuple(dot.88, dot.124, dot, dot.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  const absl::string_view target =
      kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget;
  const absl::string_view backward_target =
      kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::CustomCall(&fmha, {target}), 0)
              .WithShape(BF16, {2, 6, 128, 64}),
          m::GetTupleElement(m::CustomCall(&fmha, {backward_target}), 0)
              .WithShape(BF16, {2, 6, 128, 64}),
          m::Transpose(m::GetTupleElement(m::CustomCall({backward_target}), 1))
              .WithShape(BF16, {2, 6, 64, 128}),
          m::GetTupleElement(m::CustomCall({backward_target}), 2)
              .WithShape(BF16, {2, 6, 128, 64}))));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 6);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16TrainingBmm1ScaleBiasMaskSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,128,64]{3,2,1,0},f16[2,6,64,128]{3,2,1,0},f16[2,6,128,64]{3,2,1,0},f16[2,6,128,64]{3,2,1,0})->(f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

region_0.38 {
  Arg_0.39 = f16[] parameter(0)
  Arg_1.40 = f16[] parameter(1)
  ROOT maximum.1 = f16[] maximum(Arg_0.39, Arg_1.40)
}

region_1.50 {
  Arg_0.51 = f32[] parameter(0)
  Arg_1.52 = f32[] parameter(1)
  ROOT add.2 = f32[] add(Arg_0.51, Arg_1.52)
}

region_2.99 {
  Arg_0.100 = f16[] parameter(0)
  Arg_1.101 = f16[] parameter(1)
  ROOT add.3 = f16[] add(Arg_0.100, Arg_1.101)
}

ENTRY main.126 {
  constant.6 = u32[1]{0} constant({2718843009})
  constant.8 = u32[1]{0} constant({1272950319})
  constant.10 = u32[1]{0} constant({0})
  constant.12 = u32[1]{0} constant({2711844646})
  custom-call.65 = (u32[1]{0}, u32[1]{0}) custom-call(constant.6, constant.8, constant.10, constant.12), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.66 = u32[1]{0} get-tuple-element(custom-call.65), index=0
  bitcast.343 = u32[] bitcast(get-tuple-element.66)
  broadcast.27 = u32[98304]{0} broadcast(bitcast.343), dimensions={}
  get-tuple-element.67 = u32[1]{0} get-tuple-element(custom-call.65), index=1
  bitcast.344 = u32[] bitcast(get-tuple-element.67)
  broadcast.28 = u32[98304]{0} broadcast(bitcast.344), dimensions={}
  iota.68 = u32[196608]{0} iota(), iota_dimension=0
  slice = u32[98304]{0} slice(iota.68), slice={[0:98304]}
  slice.1 = u32[98304]{0} slice(iota.68), slice={[98304:196608]}
  custom-call.75 = (u32[98304]{0}, u32[98304]{0}) custom-call(broadcast.27, broadcast.28, slice, slice.1), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[98304]{0}, u32[98304]{0}, u32[98304]{0}, u32[98304]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\001\000\000\000\000\000"
  get-tuple-element.76 = u32[98304]{0} get-tuple-element(custom-call.75), index=0
  get-tuple-element.77 = u32[98304]{0} get-tuple-element(custom-call.75), index=1
  concatenate.2 = u32[196608]{0} concatenate(get-tuple-element.76, get-tuple-element.77), dimensions={0}
  constant.56 = u32[] constant(9)
  broadcast.63 = u32[196608]{0} broadcast(constant.56), dimensions={}
  shift-right-logical.3 = u32[196608]{0} shift-right-logical(concatenate.2, broadcast.63)
  constant.57 = u32[] constant(1065353216)
  broadcast.64 = u32[196608]{0} broadcast(constant.57), dimensions={}
  or.3 = u32[196608]{0} or(shift-right-logical.3, broadcast.64)
  bitcast-convert.3 = f32[196608]{0} bitcast-convert(or.3)
  constant.58 = f32[] constant(-1)
  broadcast.65 = f32[196608]{0} broadcast(constant.58), dimensions={}
  add.10 = f32[196608]{0} add(bitcast-convert.3, broadcast.65)
  constant.48 = f32[] constant(0)
  broadcast.66 = f32[196608]{0} broadcast(constant.48), dimensions={}
  maximum.4 = f32[196608]{0} maximum(add.10, broadcast.66)
  constant.59 = f32[] constant(0.9)
  broadcast.67 = f32[196608]{0} broadcast(constant.59), dimensions={}
  compare.3 = pred[196608]{0} compare(maximum.4, broadcast.67), direction=LT
  bitcast.308 = pred[2,6,128,128]{3,2,1,0} bitcast(compare.3)
  constant.44 = pred[2,6,128,128]{3,2,1,0} constant({...})
  Arg_0.1 = f16[2,6,128,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[2,6,64,128]{3,2,1,0} parameter(1), sharding={replicated}
  dot.34 = f16[2,6,128,128]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.55 = f16[] constant(2)
  broadcast.61 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.55), dimensions={}
  multiply.8 = f16[2,6,128,128]{3,2,1,0} multiply(dot.34, broadcast.61)
  constant.52 = f16[] constant(1)
  broadcast.39 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.52), dimensions={}
  add.6 = f16[2,6,128,128]{3,2,1,0} add(multiply.8, broadcast.39)
  constant.54 = f16[] constant(0)
  broadcast.52 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.54), dimensions={}
  select.1 = f16[2,6,128,128]{3,2,1,0} select(constant.44, add.6, broadcast.52)
  constant.41 = f16[] constant(-inf)
  reduce.42 = f16[2,6,128]{2,1,0} reduce(select.1, constant.41), dimensions={3}, to_apply=region_0.38
  broadcast.42 = f16[2,6,128,128]{3,2,1,0} broadcast(reduce.42), dimensions={0,1,2}
  subtract.1 = f16[2,6,128,128]{3,2,1,0} subtract(select.1, broadcast.42)
  exponential.1 = f16[2,6,128,128]{3,2,1,0} exponential(subtract.1)
  convert.5 = f32[2,6,128,128]{3,2,1,0} convert(exponential.1)
  reduce.54 = f32[2,6,128]{2,1,0} reduce(convert.5, constant.48), dimensions={3}, to_apply=region_1.50
  convert.9 = f16[2,6,128]{2,1,0} convert(reduce.54)
  broadcast.68 = f16[2,6,128,128]{3,2,1,0} broadcast(convert.9), dimensions={0,1,2}
  divide.5 = f16[2,6,128,128]{3,2,1,0} divide(exponential.1, broadcast.68)
  constant.60 = f16[] constant(1.1113)
  broadcast.69 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.60), dimensions={}
  multiply.20 = f16[2,6,128,128]{3,2,1,0} multiply(divide.5, broadcast.69)
  select.8 = f16[2,6,128,128]{3,2,1,0} select(bitcast.308, multiply.20, broadcast.52)
  Arg_2.3 = f16[2,6,128,64]{3,2,1,0} parameter(2), sharding={replicated}
  dot.88 = f16[2,6,128,64]{3,2,1,0} dot(select.8, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  bitcast.248 = pred[2,6,128,128]{3,2,1,0} bitcast(compare.3)
  Arg_3.4 = f16[2,6,128,64]{3,2,1,0} parameter(3), sharding={replicated}
  dot.91 = f16[2,6,128,128]{3,2,1,0} dot(Arg_3.4, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  select.6 = f16[2,6,128,128]{3,2,1,0} select(bitcast.248, dot.91, broadcast.52)
  multiply.17 = f16[2,6,128,128]{3,2,1,0} multiply(select.6, broadcast.69)
  divide.4 = f16[2,6,128,128]{3,2,1,0} divide(multiply.17, broadcast.68)
  broadcast.55 = f16[2,6,128]{2,1,0} broadcast(constant.52), dimensions={}
  multiply.11 = f16[2,6,128]{2,1,0} multiply(convert.9, convert.9)
  divide.3 = f16[2,6,128]{2,1,0} divide(broadcast.55, multiply.11)
  broadcast.56 = f16[2,6,128]{2,1,0} broadcast(constant.60), dimensions={}
  multiply.12 = f16[2,6,128]{2,1,0} multiply(divide.3, broadcast.56)
  broadcast.58 = f16[2,6,128,128]{3,2,1,0} broadcast(multiply.12), dimensions={0,1,2}
  multiply.13 = f16[2,6,128,128]{3,2,1,0} multiply(select.6, broadcast.58)
  multiply.14 = f16[2,6,128,128]{3,2,1,0} multiply(multiply.13, exponential.1)
  reduce.103 = f16[2,6,128]{2,1,0} reduce(multiply.14, constant.54), dimensions={3}, to_apply=region_2.99
  negate.3 = f16[2,6,128]{2,1,0} negate(reduce.103)
  broadcast.62 = f16[2,6,128,128]{3,2,1,0} broadcast(negate.3), dimensions={0,1,2}
  add.9 = f16[2,6,128,128]{3,2,1,0} add(divide.4, broadcast.62)
  multiply.18 = f16[2,6,128,128]{3,2,1,0} multiply(add.9, exponential.1)
  select.7 = f16[2,6,128,128]{3,2,1,0} select(constant.44, multiply.18, broadcast.52)
  multiply.19 = f16[2,6,128,128]{3,2,1,0} multiply(select.7, broadcast.61)
  dot.124 = f16[2,6,128,64]{3,2,1,0} dot(multiply.19, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = f16[2,6,64,128]{3,2,1,0} dot(Arg_0.1, multiply.19), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = f16[2,6,128,64]{3,2,1,0} dot(select.8, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.125 = (f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}) tuple(dot.88, dot.124, dot, dot.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  const absl::string_view target =
      kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget;
  const absl::string_view backward_target =
      kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::CustomCall(&fmha, {target}), 0)
              .WithShape(F16, {2, 6, 128, 64}),
          m::GetTupleElement(m::CustomCall(&fmha, {backward_target}), 0)
              .WithShape(F16, {2, 6, 128, 64}),
          m::Transpose(m::GetTupleElement(m::CustomCall({backward_target}), 1))
              .WithShape(F16, {2, 6, 64, 128}),
          m::GetTupleElement(m::CustomCall({backward_target}), 2)
              .WithShape(F16, {2, 6, 128, 64}))));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 6);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16TrainingBmm1ScaleBiasMaskSoftmaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,128,64]{3,2,1,0},f16[2,6,64,128]{3,2,1,0},f16[2,6,128,64]{3,2,1,0},f16[2,6,128,64]{3,2,1,0})->(f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

region_0.21 {
  Arg_0.22 = f16[] parameter(0)
  Arg_1.23 = f16[] parameter(1)
  ROOT maximum = f16[] maximum(Arg_0.22, Arg_1.23)
}

region_1.33 {
  Arg_0.34 = f32[] parameter(0)
  Arg_1.35 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.34, Arg_1.35)
}

region_2.55 {
  Arg_0.56 = f16[] parameter(0)
  Arg_1.57 = f16[] parameter(1)
  ROOT add.1 = f16[] add(Arg_0.56, Arg_1.57)
}

ENTRY main.82 {
  constant.18 = pred[2,6,128,128]{3,2,1,0} constant({...})
  Arg_0.1 = f16[2,6,128,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[2,6,64,128]{3,2,1,0} parameter(1), sharding={replicated}
  dot.17 = f16[2,6,128,128]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.22 = f16[] constant(2)
  broadcast.24 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.22), dimensions={}
  multiply.2 = f16[2,6,128,128]{3,2,1,0} multiply(dot.17, broadcast.24)
  constant.19 = f16[] constant(1)
  broadcast.13 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.19), dimensions={}
  add.3 = f16[2,6,128,128]{3,2,1,0} add(multiply.2, broadcast.13)
  constant.21 = f16[] constant(0)
  broadcast.23 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.21), dimensions={}
  select.1 = f16[2,6,128,128]{3,2,1,0} select(constant.18, add.3, broadcast.23)
  constant.15 = f16[] constant(-inf)
  reduce.25 = f16[2,6,128]{2,1,0} reduce(select.1, constant.15), dimensions={3}, to_apply=region_0.21
  broadcast.17 = f16[2,6,128,128]{3,2,1,0} broadcast(reduce.25), dimensions={0,1,2}
  subtract.1 = f16[2,6,128,128]{3,2,1,0} subtract(select.1, broadcast.17)
  exponential.1 = f16[2,6,128,128]{3,2,1,0} exponential(subtract.1)
  convert.5 = f32[2,6,128,128]{3,2,1,0} convert(exponential.1)
  constant.17 = f32[] constant(0)
  reduce.37 = f32[2,6,128]{2,1,0} reduce(convert.5, constant.17), dimensions={3}, to_apply=region_1.33
  convert.9 = f16[2,6,128]{2,1,0} convert(reduce.37)
  broadcast.26 = f16[2,6,128,128]{3,2,1,0} broadcast(convert.9), dimensions={0,1,2}
  divide.5 = f16[2,6,128,128]{3,2,1,0} divide(exponential.1, broadcast.26)
  Arg_2.3 = f16[2,6,128,64]{3,2,1,0} parameter(2), sharding={replicated}
  dot.46 = f16[2,6,128,64]{3,2,1,0} dot(divide.5, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_3.4 = f16[2,6,128,64]{3,2,1,0} parameter(3), sharding={replicated}
  dot.49 = f16[2,6,128,128]{3,2,1,0} dot(Arg_3.4, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  divide.4 = f16[2,6,128,128]{3,2,1,0} divide(dot.49, broadcast.26)
  broadcast.20 = f16[2,6,128]{2,1,0} broadcast(constant.19), dimensions={}
  multiply.3 = f16[2,6,128]{2,1,0} multiply(convert.9, convert.9)
  divide.3 = f16[2,6,128]{2,1,0} divide(broadcast.20, multiply.3)
  broadcast.21 = f16[2,6,128,128]{3,2,1,0} broadcast(divide.3), dimensions={0,1,2}
  multiply.4 = f16[2,6,128,128]{3,2,1,0} multiply(dot.49, broadcast.21)
  multiply.5 = f16[2,6,128,128]{3,2,1,0} multiply(multiply.4, exponential.1)
  reduce.59 = f16[2,6,128]{2,1,0} reduce(multiply.5, constant.21), dimensions={3}, to_apply=region_2.55
  negate.2 = f16[2,6,128]{2,1,0} negate(reduce.59)
  broadcast.25 = f16[2,6,128,128]{3,2,1,0} broadcast(negate.2), dimensions={0,1,2}
  add.5 = f16[2,6,128,128]{3,2,1,0} add(divide.4, broadcast.25)
  multiply.8 = f16[2,6,128,128]{3,2,1,0} multiply(add.5, exponential.1)
  select.3 = f16[2,6,128,128]{3,2,1,0} select(constant.18, multiply.8, broadcast.23)
  multiply.9 = f16[2,6,128,128]{3,2,1,0} multiply(select.3, broadcast.24)
  dot.80 = f16[2,6,128,64]{3,2,1,0} dot(multiply.9, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = f16[2,6,64,128]{3,2,1,0} dot(Arg_0.1, multiply.9), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = f16[2,6,128,64]{3,2,1,0} dot(divide.5, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.81 = (f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}) tuple(dot.46, dot.80, dot, dot.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(F16, {2, 6, 128, 64}),
          m::GetTupleElement(
              m::CustomCall(&fmha,
                            {kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget}),
              0)
              .WithShape(F16, {2, 6, 128, 64}),
          m::Transpose(
              m::GetTupleElement(
                  m::CustomCall(
                      {kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget}),
                  1))
              .WithShape(F16, {2, 6, 64, 128}),
          m::GetTupleElement(
              m::CustomCall({kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget}),
              2)
              .WithShape(F16, {2, 6, 128, 64}))));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 6);
  EXPECT_NEAR(config.dropout_rate(), 0, 1e-2);
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
