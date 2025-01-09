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

#include "xla/service/gpu/transforms/cudnn_fused_mha_rewriter.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/transforms/expanders/reshape_decomposer.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/computation_layout.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/transforms/cudnn_fused_mha_transpose_fusion.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_normalization.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cudnn/cudnn.h"  // IWYU pragma: keep
#endif

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
    return se::dnn::VersionInfo(9, 0, 0);
  }

  CudnnFusedMhaRewriterTestHloTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/false,
                    /*instruction_can_change_layout_func=*/{}) {
#if !defined(GOOGLE_CUDA) || CUDA_VERSION < 12000
    skip_reason_ = "cuDNN fused MHA requires CUDA 12 or later.";
    return;
#endif
  }

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

  DebugOptions GetDebugOptionsForTest() const override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
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

  // Centralize skip checks in the constructor. Unfortunately we cannot call
  // GTEST_SKIP from the constructor. Instead, we set (if needed) `skip_reason`,
  // and then check it from all test fixtures.
  // An alternative would be to use the SetUp() override, but for this to be
  // correct we'd have to ensure that all the parents' SetUp() methods are
  // called, which is error prone.
  std::optional<absl::string_view> skip_reason_;
};

constexpr absl::string_view hlo_base_pattern = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,HEAD_DIM]{3,2,1,0},bf16[16,16,256,HEAD_DIM]{3,2,1,0},bf16[16,16,256,HEAD_DIM]{3,2,1,0})->bf16[16,16,256,HEAD_DIM]{3,2,1,0}}

region_0.7 {
  Arg_0.8 = bf16[] parameter(0)
  Arg_1.9 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(Arg_0.8, Arg_1.9)
}

region_1.19 {
  Arg_0.20 = f32[] parameter(0)
  Arg_1.21 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.20, Arg_1.21)
}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,HEAD_DIM]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,HEAD_DIM]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,HEAD_DIM]{2,3,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  constant = bf16[] constant(-inf)
  reduce.11 = bf16[16,16,256]{2,1,0} reduce(dot.0, constant), dimensions={3}, to_apply=region_0.7
  broadcast.3 = bf16[16,16,256,256]{3,2,1,0} broadcast(reduce.11), dimensions={0,1,2}
  subtract.1 = bf16[16,16,256,256]{3,2,1,0} subtract(dot.0, broadcast.3)
  exponential.1 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.1)
  convert.1 = f32[16,16,256,256]{3,2,1,0} convert(exponential.1)
  constant.1 = f32[] constant(0)
  reduce.23 = f32[16,16,256]{2,1,0} reduce(convert.1, constant.1), dimensions={3}, to_apply=region_1.19
  convert.2 = bf16[16,16,256]{2,1,0} convert(reduce.23)
  broadcast.4 = bf16[16,16,256,256]{3,2,1,0} broadcast(convert.2), dimensions={0,1,2}
  divide = bf16[16,16,256,256]{3,2,1,0} divide(exponential.1, broadcast.4)
  ROOT dot.1 = bf16[16,16,256,HEAD_DIM]{3,2,1,0} dot(divide, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
})";

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1SoftmaxBmm2Pattern_bmm1_rhs_contracting_dim_not_most_minor) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const std::string hlo =
      absl::StrReplaceAll(hlo_base_pattern, {{"HEAD_DIM", std::to_string(64)}});
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_TRUE(result);
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHASoftmaxCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1SoftmaxBmm2Pattern_large_head_dim) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  // Large head dim of 256 is supported by Cudnn 9.5+ on Hopper+ GPUs.
  int head_dim = 256;
  const std::string hlo = absl::StrReplaceAll(
      hlo_base_pattern, {{"HEAD_DIM", std::to_string(head_dim)}});
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  CudnnFusedMHARewriter fusedMhaRewriter{se::CudaComputeCapability(9, 0),
                                         se::dnn::VersionInfo(9, 5, 0)};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_TRUE(result);
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHASoftmaxCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, head_dim})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            2);
}

constexpr absl::string_view
    hlo_BF16Bmm1SoftmaxBmm2Pattern_q_hidden_not_most_minor = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

region_0.7 {
  Arg_0.8 = bf16[] parameter(0)
  Arg_1.9 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(Arg_0.8, Arg_1.9)
}

region_1.19 {
  Arg_0.20 = f32[] parameter(0)
  Arg_1.21 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.20, Arg_1.21)
}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{2,3,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{2,3,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  constant = bf16[] constant(-inf)
  reduce.11 = bf16[16,16,256]{2,1,0} reduce(dot.0, constant), dimensions={3}, to_apply=region_0.7
  broadcast.3 = bf16[16,16,256,256]{3,2,1,0} broadcast(reduce.11), dimensions={0,1,2}
  subtract.1 = bf16[16,16,256,256]{3,2,1,0} subtract(dot.0, broadcast.3)
  exponential.1 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.1)
  convert.1 = f32[16,16,256,256]{3,2,1,0} convert(exponential.1)
  constant.1 = f32[] constant(0)
  reduce.23 = f32[16,16,256]{2,1,0} reduce(convert.1, constant.1), dimensions={3}, to_apply=region_1.19
  convert.2 = bf16[16,16,256]{2,1,0} convert(reduce.23)
  broadcast.4 = bf16[16,16,256,256]{3,2,1,0} broadcast(convert.2), dimensions={0,1,2}
  divide = bf16[16,16,256,256]{3,2,1,0} divide(exponential.1, broadcast.4)
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(divide, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
})";

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1SoftmaxBmm2Pattern_bmm1_lhs_contracting_dim_not_most_minor) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  TF_ASSERT_OK_AND_ASSIGN(
      auto m, ParseAndReturnVerifiedModule(
                  hlo_BF16Bmm1SoftmaxBmm2Pattern_q_hidden_not_most_minor));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_TRUE(result);
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHASoftmaxCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().lhs_contracting_dimensions()[0],
            2);
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            2);
}

constexpr absl::string_view
    hlo_BF16Bmm1SoftmaxBmm2Pattern_v_hidden_dim_not_most_minor = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

region_0.7 {
  Arg_0.8 = bf16[] parameter(0)
  Arg_1.9 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(Arg_0.8, Arg_1.9)
}

region_1.19 {
  Arg_0.20 = f32[] parameter(0)
  Arg_1.21 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.20, Arg_1.21)
}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{2,3,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{2,3,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{2,3,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  constant = bf16[] constant(-inf)
  reduce.11 = bf16[16,16,256]{2,1,0} reduce(dot.0, constant), dimensions={3}, to_apply=region_0.7
  broadcast.3 = bf16[16,16,256,256]{3,2,1,0} broadcast(reduce.11), dimensions={0,1,2}
  subtract.1 = bf16[16,16,256,256]{3,2,1,0} subtract(dot.0, broadcast.3)
  exponential.1 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.1)
  convert.1 = f32[16,16,256,256]{3,2,1,0} convert(exponential.1)
  constant.1 = f32[] constant(0)
  reduce.23 = f32[16,16,256]{2,1,0} reduce(convert.1, constant.1), dimensions={3}, to_apply=region_1.19
  convert.2 = bf16[16,16,256]{2,1,0} convert(reduce.23)
  broadcast.4 = bf16[16,16,256,256]{3,2,1,0} broadcast(convert.2), dimensions={0,1,2}
  divide = bf16[16,16,256,256]{3,2,1,0} divide(exponential.1, broadcast.4)
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(divide, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
})";

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1SoftmaxBmm2Pattern_bmm2_non_contracting_dim_not_most_minor) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  TF_ASSERT_OK_AND_ASSIGN(
      auto m, ParseAndReturnVerifiedModule(
                  hlo_BF16Bmm1SoftmaxBmm2Pattern_v_hidden_dim_not_most_minor));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_TRUE(result);
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHASoftmaxCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_EQ(config.bmm2_dot_dimension_numbers().lhs_contracting_dimensions()[0],
            3);
  EXPECT_EQ(config.bmm2_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            3);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1CombinedMaskBiasSoftmaxBmm2) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 4);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, F16Bmm1UnfusedSoftmaxBmm2) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_FLOAT_EQ(config.fmha_scale(), 1.0);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 3);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1ConvertedMaskAddedAfterFirstGemmSoftmaxBmm2) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 4);
}

// negative test
TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm1_contracting_dim_not_equal_64) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
       BF16Bmm1Bmm2Pattern_bmm2_rhs_non_contracting_dim_not_equal_64) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_EQ(fmha->operands().size(), 4);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1ScaleBiasSoftmaxDropoutForm2Bmm2) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
  EXPECT_EQ(fmha->operands().size(), 4);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16TrainingBmm1Bmm2) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16MiniT5xTest) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
       ActivationHasMoreThan1UserShouldNotLower) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule test

%region_50.2457 (Arg_0.2458: bf16[], Arg_1.2459: bf16[]) -> bf16[] {
  %Arg_0.2458 = bf16[] parameter(0)
  %Arg_1.2459 = bf16[] parameter(1)
  ROOT %maximum.2 = bf16[] maximum(bf16[] %Arg_0.2458, bf16[] %Arg_1.2459)
}

%region_36.2316 (Arg_0.2317: f32[], Arg_1.2318: f32[]) -> f32[] {
  %Arg_0.2317 = f32[] parameter(0)
  %Arg_1.2318 = f32[] parameter(1)
  ROOT %add.342 = f32[] add(f32[] %Arg_0.2317, f32[] %Arg_1.2318)
}

ENTRY main {
  %transpose.482 = bf16[4,5,64]{2,1,0} parameter(0)
  %transpose.484 = bf16[4,64,5]{2,1,0} parameter(1)
  %dot.20 = bf16[4,5,5]{2,1,0} dot(bf16[4,5,64]{2,1,0} %transpose.482, bf16[4,64,5]{2,1,0} %transpose.484), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  %constant.2515 = bf16[] constant(0.125)
  %broadcast.789 = bf16[4,5,5]{2,1,0} broadcast(bf16[] %constant.2515), dimensions={}
  %multiply.267 = bf16[4,5,5]{2,1,0} multiply(bf16[4,5,5]{2,1,0} %dot.20, bf16[4,5,5]{2,1,0} %broadcast.789)
  %constant.287 = f32[] constant(-1)
  %broadcast.792 = bf16[4,5,5]{2,1,0} parameter(3)
  %add.348 = bf16[4,5,5]{2,1,0} add(bf16[4,5,5]{2,1,0} %multiply.267, bf16[4,5,5]{2,1,0} %broadcast.792)
  %constant.2510 = bf16[] constant(-inf)
  %reduce.2550 = bf16[4,5]{1,0} reduce(bf16[4,5,5]{2,1,0} %add.348, bf16[] %constant.2510), dimensions={2}, to_apply=%region_50.2457
  %broadcast.793 = bf16[4,5,5]{2,1,0} broadcast(bf16[4,5]{1,0} %reduce.2550), dimensions={0,1}
  %subtract.81 = bf16[4,5,5]{2,1,0} subtract(bf16[4,5,5]{2,1,0} %add.348, bf16[4,5,5]{2,1,0} %broadcast.793)
  %exponential.21 = bf16[4,5,5]{2,1,0} exponential(bf16[4,5,5]{2,1,0} %subtract.81)
  %convert.180 = f32[4,5,5]{2,1,0} convert(bf16[4,5,5]{2,1,0} %exponential.21)
  %constant.2509 = f32[] constant(0)
  %reduce.2558 = f32[4,5]{1,0} reduce(f32[4,5,5]{2,1,0} %convert.180, f32[] %constant.2509), dimensions={2}, to_apply=%region_36.2316
  %convert.182 = bf16[4,5]{1,0} convert(f32[4,5]{1,0} %reduce.2558)
  %broadcast.794 = bf16[4,5,5]{2,1,0} broadcast(bf16[4,5]{1,0} %convert.182), dimensions={0,1}
  %divide.25 = bf16[4,5,5]{2,1,0} divide(bf16[4,5,5]{2,1,0} %exponential.21, bf16[4,5,5]{2,1,0} %broadcast.794)
  %transpose.481 = bf16[4,64,5]{2,1,0} parameter(2)
  %dot.21 = bf16[4,64,5]{2,1,0} dot(bf16[4,64,5]{2,1,0} %transpose.481, bf16[4,5,5]{2,1,0} %divide.25), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={2}
  ROOT %tuple.2668 = (bf16[4,5,5]{2,1,0}, bf16[4,64,5]{2,1,0}) tuple(bf16[4,5,5]{2,1,0} %divide.25, bf16[4,64,5]{2,1,0} %dot.21)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision*/ true);
  ASSERT_IS_OK(verifier.Run(m.get()).status());

  EXPECT_EQ(CountFusedAttentionCall(m.get()), 0);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16InvalidTrainingBmm1ScaleBiasMaskSoftmaxBmm2ShouldNotBeLowered) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
  broadcast.25 = f16[2,6,128,128]{3,2,1,0} broadcast(reduce.59), dimensions={0,1,2}
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
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision*/ true);
  ASSERT_IS_OK(verifier.Run(m.get()).status());

  // The backward pattern in the graph is not a valid fmha pattern,
  // we expect no rewrite happening.
  EXPECT_EQ(CountFusedAttentionCall(m.get()), 0);
  EXPECT_EQ(CountFusedAttentionCall(m.get(), /*is_backward*/ true), 0);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16InvalidTrainingBmm1ScaleBiasMaskSoftmaxDropoutBmm2ShouldNotLower) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
  broadcast.62 = f16[2,6,128,128]{3,2,1,0} broadcast(reduce.103), dimensions={0,1,2}
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
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision*/ true);
  ASSERT_IS_OK(verifier.Run(m.get()).status());

  // The backward pattern in the graph is not a valid fmha pattern,
  // we expect no rewrite happening.
  EXPECT_EQ(CountFusedAttentionCall(m.get()), 0);
  EXPECT_EQ(CountFusedAttentionCall(m.get(), /*is_backward*/ true), 0);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16TrainingBmm1ScaleBiasSoftmaxBmm2QTranspose) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,64,128]{3,2,1,0},f16[2,6,64,128]{3,2,1,0},f16[2,6,128,64]{3,2,1,0},f16[2,6,128,64]{3,2,1,0})->(f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

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
  Arg_0.1 = f16[2,6,64,128]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[2,6,64,128]{3,2,1,0} parameter(1), sharding={replicated}
  dot.17 = f16[2,6,128,128]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.22 = f16[] constant(2)
  broadcast.24 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.22), dimensions={}
  multiply.2 = f16[2,6,128,128]{3,2,1,0} multiply(dot.17, broadcast.24)
  constant.19 = f16[] constant(1)
  broadcast.13 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.19), dimensions={}
  add.3 = f16[2,6,128,128]{3,2,1,0} add(multiply.2, broadcast.13)
  constant.21 = f16[] constant(0)
  constant.15 = f16[] constant(-inf)
  reduce.25 = f16[2,6,128]{2,1,0} reduce(add.3, constant.15), dimensions={3}, to_apply=region_0.21
  broadcast.17 = f16[2,6,128,128]{3,2,1,0} broadcast(reduce.25), dimensions={0,1,2}
  subtract.1 = f16[2,6,128,128]{3,2,1,0} subtract(add.3, broadcast.17)
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
  multiply.9 = f16[2,6,128,128]{3,2,1,0} multiply(multiply.8, broadcast.24)
  dot.80 = f16[2,6,128,64]{3,2,1,0} dot(multiply.9, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = f16[2,6,64,128]{3,2,1,0} dot(Arg_0.1, multiply.9), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = f16[2,6,128,64]{3,2,1,0} dot(divide.5, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.81 = (f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}) tuple(dot.46, dot.80, dot, dot.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
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
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasSoftmaxCallTarget}), 0)
              .WithShape(F16, {2, 6, 128, 64}),
          m::GetTupleElement(
              m::CustomCall(&fmha,
                            {kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget}),
              0)
              .WithShape(F16, {2, 6, 128, 64}),
          m::Transpose(
              m::GetTupleElement(
                  m::CustomCall({kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget}),
                  1))
              .WithShape(F16, {2, 6, 64, 128}),
          m::GetTupleElement(
              m::CustomCall({kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget}), 2)
              .WithShape(F16, {2, 6, 128, 64}))));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_EQ(fmha->operands().size(), 7);
  EXPECT_NEAR(config.dropout_rate(), 0, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16Bmm1UnfusedSoftmaxBmm2IncorrectBmm1NumUsers) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,40,64]{3,2,1,0},f16[2,6,64,40]{3,2,1,0},f16[2,6,40,64]{3,2,1,0})->(f16[2,6,40,64]{3,2,1,0}, f16[2,6,40,40]{3,2,1,0})}

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
  // extra user of bmm1
  neg.1 = f16[2,6,40,40]{3,2,1,0} negate(dot)
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
  dot.1 = f16[2,6,40,64]{3,2,1,0} dot(divide, Arg_2.3), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  ROOT tuple.81 = (f16[2,6,40,64]{3,2,1,0}, f16[2,6,40,40]{3,2,1,0}) tuple(dot.1, neg.1)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Dot(), m::Negate())));
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16Bmm1UnfusedSoftmaxBmm2IncorrectSoftmaxNumUsers) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,40,64]{3,2,1,0},f16[2,6,64,40]{3,2,1,0},f16[2,6,40,64]{3,2,1,0})->(f16[2,6,40,64]{3,2,1,0}, f16[2,6,40,40]{3,2,1,0})}

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
  // extra user of softmax sub node
  neg.1 = f16[2,6,40,40]{3,2,1,0} negate(subtract.1)
  exponential.1 = f16[2,6,40,40]{3,2,1,0} exponential(subtract.1)
  convert.1 = f32[2,6,40,40]{3,2,1,0} convert(exponential.1)
  constant.1 = f32[] constant(0)
  reduce.23 = f32[2,6,40]{2,1,0} reduce(convert.1, constant.1), dimensions={3}, to_apply=region_1.19
  convert.2 = f16[2,6,40]{2,1,0} convert(reduce.23)
  broadcast.4 = f16[2,6,40,40]{3,2,1,0} broadcast(convert.2), dimensions={0,1,2}
  divide = f16[2,6,40,40]{3,2,1,0} divide(exponential.1, broadcast.4)
  Arg_2.3 = f16[2,6,40,64]{3,2,1,0} parameter(2), sharding={replicated}
  dot.1 = f16[2,6,40,64]{3,2,1,0} dot(divide, Arg_2.3), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  ROOT tuple.81 = (f16[2,6,40,64]{3,2,1,0}, f16[2,6,40,40]{3,2,1,0}) tuple(dot.1, neg.1)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Dot(), m::Negate())));
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16TrainingBmm1ScaleBiasSoftmaxBmm2IncorrectSoftmaxBwdNumUsers) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,64,128]{3,2,1,0},f16[2,6,64,128]{3,2,1,0},f16[2,6,128,64]{3,2,1,0},f16[2,6,128,64]{3,2,1,0})->(f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,128]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

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
  Arg_0.1 = f16[2,6,64,128]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[2,6,64,128]{3,2,1,0} parameter(1), sharding={replicated}
  dot.17 = f16[2,6,128,128]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.22 = f16[] constant(2)
  broadcast.24 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.22), dimensions={}
  multiply.2 = f16[2,6,128,128]{3,2,1,0} multiply(dot.17, broadcast.24)
  constant.19 = f16[] constant(1)
  broadcast.13 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.19), dimensions={}
  add.3 = f16[2,6,128,128]{3,2,1,0} add(multiply.2, broadcast.13)
  constant.21 = f16[] constant(0)
  constant.15 = f16[] constant(-inf)
  reduce.25 = f16[2,6,128]{2,1,0} reduce(add.3, constant.15), dimensions={3}, to_apply=region_0.21
  broadcast.17 = f16[2,6,128,128]{3,2,1,0} broadcast(reduce.25), dimensions={0,1,2}
  subtract.1 = f16[2,6,128,128]{3,2,1,0} subtract(add.3, broadcast.17)
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
  // extra user of softmax bwd divide node
  neg.1 = f16[2,6,128,128]{3,2,1,0} negate(divide.4)
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
  multiply.9 = f16[2,6,128,128]{3,2,1,0} multiply(multiply.8, broadcast.24)
  dot.80 = f16[2,6,128,64]{3,2,1,0} dot(multiply.9, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = f16[2,6,64,128]{3,2,1,0} dot(Arg_0.1, multiply.9), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = f16[2,6,128,64]{3,2,1,0} dot(divide.5, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.81 = (f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,128]{3,2,1,0}) tuple(dot.46, dot.80, dot, dot.1, neg.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Dot(), m::Dot(), m::Dot(), m::Dot(),
                                  m::Negate())));
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, F16Bmm1SoftmaxBmm2IncorrectRank) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule reproducer, entry_computation_layout={(f16[1,8,16,5,128]{4,3,2,1,0}, f16[1,8,16,5,128]{4,3,2,1,0}, f16[1,8,16,5,128]{4,3,2,1,0}, f32[128,2,64]{2,1,0}, f32[2,64]{1,0}, /*index=5*/f32[128,2,64]{2,1,0}, f32[2,64]{1,0}, f32[128,2,64]{2,1,0}, f32[2,64]{1,0})->f16[8,16,2,5,64]{4,3,2,1,0}}

region_0.36 {
  Arg_0.37 = f16[] parameter(0)
  Arg_1.38 = f16[] parameter(1)
  ROOT maximum = f16[] maximum(Arg_0.37, Arg_1.38)
}

region_1.48 {
  Arg_0.49 = f32[] parameter(0)
  Arg_1.50 = f32[] parameter(1)
  ROOT add.1 = f32[] add(Arg_0.49, Arg_1.50)
}

ENTRY main {
  arg2.3 = f16[1,8,16,5,128]{4,3,2,1,0} parameter(2), parameter_replication={false}
  bitcast.31 = f16[640,128]{1,0} bitcast(arg2.3)
  arg5.6 = f32[128,2,64]{2,1,0} parameter(5), parameter_replication={false}
  convert.3 = f16[128,2,64]{2,1,0} convert(arg5.6)
  bitcast.36 = f16[128,128]{1,0} bitcast(convert.3)
  dot = f16[640,128]{1,0} dot(bitcast.31, bitcast.36), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={grad_x="false",grad_y="false"}
  bitcast.39 = f16[1,8,16,5,2,64]{5,4,3,2,1,0} bitcast(dot)
  transpose.27 = f16[1,8,16,2,5,64]{5,4,3,2,1,0} transpose(bitcast.39), dimensions={0,1,2,4,3,5}, frontend_attributes={grad_x="false",grad_y="false"}
  arg6.7 = f32[2,64]{1,0} parameter(6), parameter_replication={false}
  convert.4 = f16[2,64]{1,0} convert(arg6.7)
  broadcast.9 = f16[1,8,16,2,5,64]{5,4,3,2,1,0} broadcast(convert.4), dimensions={3,5}
  add.2 = f16[1,8,16,2,5,64]{5,4,3,2,1,0} add(transpose.27, broadcast.9)
  bitcast.49 = f16[8,16,2,5,64]{4,3,2,1,0} bitcast(add.2)
  arg0.1 = f16[1,8,16,5,128]{4,3,2,1,0} parameter(0), parameter_replication={false}
  bitcast.53 = f16[640,128]{1,0} bitcast(arg0.1)
  arg3.4 = f32[128,2,64]{2,1,0} parameter(3), parameter_replication={false}
  convert.5 = f16[128,2,64]{2,1,0} convert(arg3.4)
  bitcast.58 = f16[128,128]{1,0} bitcast(convert.5)
  dot.1 = f16[640,128]{1,0} dot(bitcast.53, bitcast.58), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={grad_x="false",grad_y="false"}
  bitcast.61 = f16[1,8,16,5,2,64]{5,4,3,2,1,0} bitcast(dot.1)
  transpose.28 = f16[1,8,16,2,64,5]{5,4,3,2,1,0} transpose(bitcast.61), dimensions={0,1,2,4,5,3}, frontend_attributes={grad_x="false",grad_y="false"}
  arg4.5 = f32[2,64]{1,0} parameter(4), parameter_replication={false}
  convert.6 = f16[2,64]{1,0} convert(arg4.5)
  broadcast.10 = f16[1,8,16,2,64,5]{5,4,3,2,1,0} broadcast(convert.6), dimensions={3,4}
  add.3 = f16[1,8,16,2,64,5]{5,4,3,2,1,0} add(transpose.28, broadcast.10)
  constant.29 = f16[] constant(0.125)
  broadcast.11 = f16[1,8,16,2,64,5]{5,4,3,2,1,0} broadcast(constant.29), dimensions={}
  multiply = f16[1,8,16,2,64,5]{5,4,3,2,1,0} multiply(add.3, broadcast.11)
  bitcast.74 = f16[8,16,2,64,5]{4,3,2,1,0} bitcast(multiply)
  dot.6 = f16[8,16,2,5,5]{4,3,2,1,0} dot(bitcast.49, bitcast.74), lhs_batch_dims={0,1,2}, lhs_contracting_dims={4}, rhs_batch_dims={0,1,2}, rhs_contracting_dims={3}, frontend_attributes={grad_x="false",grad_y="false"}
  constant.35 = f16[] constant(-inf)
  reduce.1 = f16[8,16,2,5]{3,2,1,0} reduce(dot.6, constant.35), dimensions={3}, to_apply=region_0.36
  broadcast.12 = f16[8,16,2,5,5]{4,3,2,1,0} broadcast(reduce.1), dimensions={0,1,2,4}
  subtract.2 = f16[8,16,2,5,5]{4,3,2,1,0} subtract(dot.6, broadcast.12)
  exponential.2 = f16[8,16,2,5,5]{4,3,2,1,0} exponential(subtract.2)
  convert.7 = f32[8,16,2,5,5]{4,3,2,1,0} convert(exponential.2)
  constant.34 = f32[] constant(0)
  reduce.3 = f32[8,16,2,5]{3,2,1,0} reduce(convert.7, constant.34), dimensions={3}, to_apply=region_1.48
  convert.8 = f16[8,16,2,5]{3,2,1,0} convert(reduce.3)
  broadcast.13 = f16[8,16,2,5,5]{4,3,2,1,0} broadcast(convert.8), dimensions={0,1,2,4}
  divide.2 = f16[8,16,2,5,5]{4,3,2,1,0} divide(exponential.2, broadcast.13)
  bitcast.98 = f16[8,16,2,5,5]{3,4,2,1,0} bitcast(divide.2)
  arg1.2 = f16[1,8,16,5,128]{4,3,2,1,0} parameter(1), parameter_replication={false}
  bitcast.102 = f16[640,128]{1,0} bitcast(arg1.2)
  arg7.8 = f32[128,2,64]{2,1,0} parameter(7), parameter_replication={false}
  convert.9 = f16[128,2,64]{2,1,0} convert(arg7.8)
  bitcast.107 = f16[128,128]{1,0} bitcast(convert.9)
  dot.3 = f16[640,128]{1,0} dot(bitcast.102, bitcast.107), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={grad_x="false",grad_y="false"}
  bitcast.110 = f16[1,8,16,5,2,64]{5,4,3,2,1,0} bitcast(dot.3)
  transpose.30 = f16[1,8,16,2,5,64]{5,4,3,2,1,0} transpose(bitcast.110), dimensions={0,1,2,4,3,5}, frontend_attributes={grad_x="false",grad_y="false"}
  arg8.9 = f32[2,64]{1,0} parameter(8), parameter_replication={false}
  convert.10 = f16[2,64]{1,0} convert(arg8.9)
  broadcast.14 = f16[1,8,16,2,5,64]{5,4,3,2,1,0} broadcast(convert.10), dimensions={3,5}
  add.4 = f16[1,8,16,2,5,64]{5,4,3,2,1,0} add(transpose.30, broadcast.14)
  bitcast.120 = f16[8,16,2,5,64]{4,3,2,1,0} bitcast(add.4)
  ROOT dot.7 = f16[8,16,2,5,64]{4,3,2,1,0} dot(bitcast.98, bitcast.120), lhs_batch_dims={0,1,2}, lhs_contracting_dims={4}, rhs_batch_dims={0,1,2}, rhs_contracting_dims={3}, frontend_attributes={grad_x="false",grad_y="false"}
} // main
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  const auto status_or = RunHloPass(&fusedMhaRewriter, m.get());
  TF_ASSERT_OK(status_or.status());
  EXPECT_FALSE(status_or.value());

  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(), GmockMatch(m::Dot()));
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, F16TrainingBmm2Grad1IncorrectPattern) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,64,128]{3,2,1,0},f16[2,6,64,128]{3,2,1,0},f16[2,6,128,64]{3,2,1,0},f16[2,6,128,64]{3,2,1,0})->(f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,128]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

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
  Arg_0.1 = f16[2,6,64,128]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[2,6,64,128]{3,2,1,0} parameter(1), sharding={replicated}
  dot.17 = f16[2,6,128,128]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.22 = f16[] constant(2)
  broadcast.24 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.22), dimensions={}
  multiply.2 = f16[2,6,128,128]{3,2,1,0} multiply(dot.17, broadcast.24)
  constant.19 = f16[] constant(1)
  broadcast.13 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.19), dimensions={}
  add.3 = f16[2,6,128,128]{3,2,1,0} add(multiply.2, broadcast.13)
  constant.21 = f16[] constant(0)
  constant.15 = f16[] constant(-inf)
  reduce.25 = f16[2,6,128]{2,1,0} reduce(add.3, constant.15), dimensions={3}, to_apply=region_0.21
  broadcast.17 = f16[2,6,128,128]{3,2,1,0} broadcast(reduce.25), dimensions={0,1,2}
  subtract.1 = f16[2,6,128,128]{3,2,1,0} subtract(add.3, broadcast.17)
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
  multiply.9 = f16[2,6,128,128]{3,2,1,0} multiply(multiply.8, broadcast.24)
  dot.80 = f16[2,6,128,64]{3,2,1,0} dot(multiply.9, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = f16[2,6,64,128]{3,2,1,0} dot(Arg_0.1, multiply.9), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  // add another user of ds multiply.9 here, neg.1 should not be pattern matched as bmm2grad1
  neg.1 = f16[2,6,128,128]{3,2,1,0} negate(multiply.9)
  dot.1 = f16[2,6,128,64]{3,2,1,0} dot(divide.5, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.81 = (f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,128]{3,2,1,0}) tuple(dot.46, dot.80, dot, dot.1, neg.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Dot(), m::Dot(), m::Dot(), m::Dot(),
                                  m::Negate())));
}

// flash attention
TEST_F(CudnnFusedMhaRewriterTestHloTest,
       FlashAttentionBF16TrainingBmm1CausalMaskSoftmaxBmm2Pattern) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,128,2048]{3,2,1,0},bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,2048,128]{3,2,1,0})->(bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,128,2048]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}
region_0.32 {
  Arg_0.33 = bf16[] parameter(0)
  Arg_1.34 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(Arg_0.33, Arg_1.34)
}
region_1.44 {
  Arg_0.45 = f32[] parameter(0)
  Arg_1.46 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.45, Arg_1.46)
}
region_2.66 {
  Arg_0.67 = bf16[] parameter(0)
  Arg_1.68 = bf16[] parameter(1)
  ROOT add.1 = bf16[] add(Arg_0.67, Arg_1.68)
}
ENTRY main.92 {
  Arg_0.1 = bf16[2,6,2048,128]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = bf16[2,6,128,2048]{3,2,1,0} parameter(1), sharding={replicated}
  dot.14 = bf16[2,6,2048,2048]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.17 = bf16[] constant(2)
  broadcast.29 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(constant.17), dimensions={}
  multiply.2 = bf16[2,6,2048,2048]{3,2,1,0} multiply(dot.14, broadcast.29)
  iota.2 = s32[2048,2048]{1,0} iota(), iota_dimension=0
  iota.5 = s32[2048,2048]{1,0} iota(), iota_dimension=1
  compare.1 = pred[2048,2048]{1,0} compare(iota.2, iota.5), direction=LT
  constant.6 = bf16[] constant(-2.366e+38)
  broadcast.16 = bf16[2048,2048]{1,0} broadcast(constant.6), dimensions={}
  constant.16 = bf16[] constant(0)
  broadcast.17 = bf16[2048,2048]{1,0} broadcast(constant.16), dimensions={}
  select.2 = bf16[2048,2048]{1,0} select(compare.1, broadcast.16, broadcast.17)
  broadcast.19 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(select.2), dimensions={2,3}
  add.3 = bf16[2,6,2048,2048]{3,2,1,0} add(multiply.2, broadcast.19)
  constant.10 = bf16[] constant(-inf)
  reduce.36 = bf16[2,6,2048]{2,1,0} reduce(add.3, constant.10), dimensions={3}, to_apply=region_0.32
  broadcast.21 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(reduce.36), dimensions={0,1,2}
  subtract.1 = bf16[2,6,2048,2048]{3,2,1,0} subtract(add.3, broadcast.21)
  exponential.1 = bf16[2,6,2048,2048]{3,2,1,0} exponential(subtract.1)
  convert.5 = f32[2,6,2048,2048]{3,2,1,0} convert(exponential.1)
  constant.14 = f32[] constant(0)
  reduce.48 = f32[2,6,2048]{2,1,0} reduce(convert.5, constant.14), dimensions={3}, to_apply=region_1.44
  convert.9 = bf16[2,6,2048]{2,1,0} convert(reduce.48)
  broadcast.32 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(convert.9), dimensions={0,1,2}
  divide.5 = bf16[2,6,2048,2048]{3,2,1,0} divide(exponential.1, broadcast.32)
  Arg_2.3 = bf16[2,6,2048,128]{3,2,1,0} parameter(2), sharding={replicated}
  dot.57 = bf16[2,6,2048,128]{3,2,1,0} dot(divide.5, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_3.4 = bf16[2,6,2048,128]{3,2,1,0} parameter(3), sharding={replicated}
  dot.60 = bf16[2,6,2048,2048]{3,2,1,0} dot(Arg_3.4, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  divide.4 = bf16[2,6,2048,2048]{3,2,1,0} divide(dot.60, broadcast.32)
  constant.15 = bf16[] constant(1)
  broadcast.25 = bf16[2,6,2048]{2,1,0} broadcast(constant.15), dimensions={}
  multiply.3 = bf16[2,6,2048]{2,1,0} multiply(convert.9, convert.9)
  divide.3 = bf16[2,6,2048]{2,1,0} divide(broadcast.25, multiply.3)
  broadcast.26 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(divide.3), dimensions={0,1,2}
  multiply.4 = bf16[2,6,2048,2048]{3,2,1,0} multiply(dot.60, broadcast.26)
  multiply.5 = bf16[2,6,2048,2048]{3,2,1,0} multiply(multiply.4, exponential.1)
  reduce.70 = bf16[2,6,2048]{2,1,0} reduce(multiply.5, constant.16), dimensions={3}, to_apply=region_2.66
  negate.2 = bf16[2,6,2048]{2,1,0} negate(reduce.70)
  broadcast.31 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(negate.2), dimensions={0,1,2}
  add.5 = bf16[2,6,2048,2048]{3,2,1,0} add(divide.4, broadcast.31)
  multiply.8 = bf16[2,6,2048,2048]{3,2,1,0} multiply(add.5, exponential.1)
  multiply.9 = bf16[2,6,2048,2048]{3,2,1,0} multiply(multiply.8, broadcast.29)
  dot.90 = bf16[2,6,2048,128]{3,2,1,0} dot(multiply.9, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = bf16[2,6,128,2048]{3,2,1,0} dot(Arg_0.1, multiply.9), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = bf16[2,6,2048,128]{3,2,1,0} dot(divide.5, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.91 = (bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,128,2048]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}) tuple(dot.57, dot.90, dot, dot.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fwd_fmha;
  const HloInstruction* bwd_fmha;
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(
              m::CustomCall(&fwd_fmha, {kCudnnfMHASoftmaxCallTarget}), 0)
              .WithShape(BF16, {2, 6, 2048, 128}),
          m::GetTupleElement(
              m::CustomCall(&bwd_fmha, {kCudnnfMHASoftmaxBackwardCallTarget}),
              0)
              .WithShape(BF16, {2, 6, 2048, 128}),
          m::Transpose(
              m::GetTupleElement(
                  m::CustomCall({kCudnnfMHASoftmaxBackwardCallTarget}), 1))
              .WithShape(BF16, {2, 6, 128, 2048}),
          m::GetTupleElement(
              m::CustomCall({kCudnnfMHASoftmaxBackwardCallTarget}), 2)
              .WithShape(BF16, {2, 6, 2048, 128}))));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fwd_fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_EQ(fwd_fmha->operands().size(), 3);
  EXPECT_EQ(bwd_fmha->operands().size(), 6);
  EXPECT_NEAR(config.dropout_rate(), 0, 1e-2);
  EXPECT_EQ(config.mask_type(), CudnnfMHABackendConfig::CAUSAL);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       FlashAttentionBF16TrainingBmm1BiasSoftmaxBmm2Pattern) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,128,2048]{3,2,1,0},bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,2048,2048]{3,2,1,0})->(bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,128,2048]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}
region_0.32 {
  Arg_0.33 = bf16[] parameter(0)
  Arg_1.34 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(Arg_0.33, Arg_1.34)
}
region_1.44 {
  Arg_0.45 = f32[] parameter(0)
  Arg_1.46 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.45, Arg_1.46)
}
region_2.66 {
  Arg_0.67 = bf16[] parameter(0)
  Arg_1.68 = bf16[] parameter(1)
  ROOT add.1 = bf16[] add(Arg_0.67, Arg_1.68)
}
ENTRY main.92 {
  Arg_0.1 = bf16[2,6,2048,128]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = bf16[2,6,128,2048]{3,2,1,0} parameter(1), sharding={replicated}
  dot.14 = bf16[2,6,2048,2048]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.17 = bf16[] constant(2)
  broadcast.29 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(constant.17), dimensions={}
  multiply.2 = bf16[2,6,2048,2048]{3,2,1,0} multiply(dot.14, broadcast.29)
  // bias
  Arg_4.5 = bf16[2,6,2048,2048]{3,2,1,0} parameter(4), sharding={replicated}
  add.3 = bf16[2,6,2048,2048]{3,2,1,0} add(multiply.2, Arg_4.5)
  constant.10 = bf16[] constant(-inf)
  constant.16 = bf16[] constant(0)
  reduce.36 = bf16[2,6,2048]{2,1,0} reduce(add.3, constant.10), dimensions={3}, to_apply=region_0.32
  broadcast.21 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(reduce.36), dimensions={0,1,2}
  subtract.1 = bf16[2,6,2048,2048]{3,2,1,0} subtract(add.3, broadcast.21)
  exponential.1 = bf16[2,6,2048,2048]{3,2,1,0} exponential(subtract.1)
  convert.5 = f32[2,6,2048,2048]{3,2,1,0} convert(exponential.1)
  constant.14 = f32[] constant(0)
  reduce.48 = f32[2,6,2048]{2,1,0} reduce(convert.5, constant.14), dimensions={3}, to_apply=region_1.44
  convert.9 = bf16[2,6,2048]{2,1,0} convert(reduce.48)
  broadcast.32 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(convert.9), dimensions={0,1,2}
  divide.5 = bf16[2,6,2048,2048]{3,2,1,0} divide(exponential.1, broadcast.32)
  Arg_2.3 = bf16[2,6,2048,128]{3,2,1,0} parameter(2), sharding={replicated}
  dot.57 = bf16[2,6,2048,128]{3,2,1,0} dot(divide.5, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_3.4 = bf16[2,6,2048,128]{3,2,1,0} parameter(3), sharding={replicated}
  dot.60 = bf16[2,6,2048,2048]{3,2,1,0} dot(Arg_3.4, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  divide.4 = bf16[2,6,2048,2048]{3,2,1,0} divide(dot.60, broadcast.32)
  constant.15 = bf16[] constant(1)
  broadcast.25 = bf16[2,6,2048]{2,1,0} broadcast(constant.15), dimensions={}
  multiply.3 = bf16[2,6,2048]{2,1,0} multiply(convert.9, convert.9)
  divide.3 = bf16[2,6,2048]{2,1,0} divide(broadcast.25, multiply.3)
  broadcast.26 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(divide.3), dimensions={0,1,2}
  multiply.4 = bf16[2,6,2048,2048]{3,2,1,0} multiply(dot.60, broadcast.26)
  multiply.5 = bf16[2,6,2048,2048]{3,2,1,0} multiply(multiply.4, exponential.1)
  reduce.70 = bf16[2,6,2048]{2,1,0} reduce(multiply.5, constant.16), dimensions={3}, to_apply=region_2.66
  negate.2 = bf16[2,6,2048]{2,1,0} negate(reduce.70)
  broadcast.31 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(negate.2), dimensions={0,1,2}
  add.5 = bf16[2,6,2048,2048]{3,2,1,0} add(divide.4, broadcast.31)
  multiply.8 = bf16[2,6,2048,2048]{3,2,1,0} multiply(add.5, exponential.1)
  multiply.9 = bf16[2,6,2048,2048]{3,2,1,0} multiply(multiply.8, broadcast.29)
  dot.90 = bf16[2,6,2048,128]{3,2,1,0} dot(multiply.9, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = bf16[2,6,128,2048]{3,2,1,0} dot(Arg_0.1, multiply.9), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = bf16[2,6,2048,128]{3,2,1,0} dot(divide.5, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.91 = (bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,128,2048]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}) tuple(dot.57, dot.90, dot, dot.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
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
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasSoftmaxCallTarget}), 0)
              .WithShape(BF16, {2, 6, 2048, 128}),
          m::GetTupleElement(
              m::CustomCall(&fmha,
                            {kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget}),
              0)
              .WithShape(BF16, {2, 6, 2048, 128}),
          m::Transpose(
              m::GetTupleElement(
                  m::CustomCall({kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget}),
                  1))
              .WithShape(BF16, {2, 6, 128, 2048}),
          m::GetTupleElement(
              m::CustomCall({kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget}), 2)
              .WithShape(BF16, {2, 6, 2048, 128}))));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_EQ(fmha->operands().size(), 7);
  EXPECT_NEAR(config.dropout_rate(), 0, 1e-2);
  EXPECT_EQ(config.mask_type(), CudnnfMHABackendConfig::NO_MASK);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       FlashAttentionBF16TrainingBmm1SoftmaxBmm2Pattern) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,128,2048]{3,2,1,0},bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,2048,128]{3,2,1,0})->(bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,128,2048]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}
region_0.32 {
  Arg_0.33 = bf16[] parameter(0)
  Arg_1.34 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(Arg_0.33, Arg_1.34)
}
region_1.44 {
  Arg_0.45 = f32[] parameter(0)
  Arg_1.46 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.45, Arg_1.46)
}
region_2.66 {
  Arg_0.67 = bf16[] parameter(0)
  Arg_1.68 = bf16[] parameter(1)
  ROOT add.1 = bf16[] add(Arg_0.67, Arg_1.68)
}
ENTRY main.92 {
  Arg_0.1 = bf16[2,6,2048,128]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = bf16[2,6,128,2048]{3,2,1,0} parameter(1), sharding={replicated}
  dot.14 = bf16[2,6,2048,2048]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.17 = bf16[] constant(2)
  broadcast.29 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(constant.17), dimensions={}
  multiply.2 = bf16[2,6,2048,2048]{3,2,1,0} multiply(dot.14, broadcast.29)
  constant.10 = bf16[] constant(-inf)
  constant.16 = bf16[] constant(0)
  reduce.36 = bf16[2,6,2048]{2,1,0} reduce(multiply.2, constant.10), dimensions={3}, to_apply=region_0.32
  broadcast.21 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(reduce.36), dimensions={0,1,2}
  subtract.1 = bf16[2,6,2048,2048]{3,2,1,0} subtract(multiply.2, broadcast.21)
  exponential.1 = bf16[2,6,2048,2048]{3,2,1,0} exponential(subtract.1)
  convert.5 = f32[2,6,2048,2048]{3,2,1,0} convert(exponential.1)
  constant.14 = f32[] constant(0)
  reduce.48 = f32[2,6,2048]{2,1,0} reduce(convert.5, constant.14), dimensions={3}, to_apply=region_1.44
  convert.9 = bf16[2,6,2048]{2,1,0} convert(reduce.48)
  broadcast.32 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(convert.9), dimensions={0,1,2}
  divide.5 = bf16[2,6,2048,2048]{3,2,1,0} divide(exponential.1, broadcast.32)
  Arg_2.3 = bf16[2,6,2048,128]{3,2,1,0} parameter(2), sharding={replicated}
  dot.57 = bf16[2,6,2048,128]{3,2,1,0} dot(divide.5, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_3.4 = bf16[2,6,2048,128]{3,2,1,0} parameter(3), sharding={replicated}
  dot.60 = bf16[2,6,2048,2048]{3,2,1,0} dot(Arg_3.4, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  divide.4 = bf16[2,6,2048,2048]{3,2,1,0} divide(dot.60, broadcast.32)
  constant.15 = bf16[] constant(1)
  broadcast.25 = bf16[2,6,2048]{2,1,0} broadcast(constant.15), dimensions={}
  multiply.3 = bf16[2,6,2048]{2,1,0} multiply(convert.9, convert.9)
  divide.3 = bf16[2,6,2048]{2,1,0} divide(broadcast.25, multiply.3)
  broadcast.26 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(divide.3), dimensions={0,1,2}
  multiply.4 = bf16[2,6,2048,2048]{3,2,1,0} multiply(dot.60, broadcast.26)
  multiply.5 = bf16[2,6,2048,2048]{3,2,1,0} multiply(multiply.4, exponential.1)
  reduce.70 = bf16[2,6,2048]{2,1,0} reduce(multiply.5, constant.16), dimensions={3}, to_apply=region_2.66
  negate.2 = bf16[2,6,2048]{2,1,0} negate(reduce.70)
  broadcast.31 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(negate.2), dimensions={0,1,2}
  add.5 = bf16[2,6,2048,2048]{3,2,1,0} add(divide.4, broadcast.31)
  multiply.8 = bf16[2,6,2048,2048]{3,2,1,0} multiply(add.5, exponential.1)
  multiply.9 = bf16[2,6,2048,2048]{3,2,1,0} multiply(multiply.8, broadcast.29)
  dot.90 = bf16[2,6,2048,128]{3,2,1,0} dot(multiply.9, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = bf16[2,6,128,2048]{3,2,1,0} dot(Arg_0.1, multiply.9), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = bf16[2,6,2048,128]{3,2,1,0} dot(divide.5, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.91 = (bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,128,2048]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}) tuple(dot.57, dot.90, dot, dot.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
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
              m::CustomCall(&fmha, {kCudnnfMHASoftmaxCallTarget}), 0)
              .WithShape(BF16, {2, 6, 2048, 128}),
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHASoftmaxBackwardCallTarget}), 0)
              .WithShape(BF16, {2, 6, 2048, 128}),
          m::Transpose(
              m::GetTupleElement(
                  m::CustomCall({kCudnnfMHASoftmaxBackwardCallTarget}), 1))
              .WithShape(BF16, {2, 6, 128, 2048}),
          m::GetTupleElement(
              m::CustomCall({kCudnnfMHASoftmaxBackwardCallTarget}), 2)
              .WithShape(BF16, {2, 6, 2048, 128}))));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_EQ(fmha->operands().size(), 6);
  EXPECT_NEAR(config.dropout_rate(), 0, 1e-2);
  EXPECT_FLOAT_EQ(config.fmha_scale(), 2);
  EXPECT_EQ(config.mask_type(), CudnnfMHABackendConfig::NO_MASK);
}

// GPT3 pattern
TEST_F(CudnnFusedMhaRewriterTestHloTest, FlashAttentionBF16TrainingGPT3_5B) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={((s32[], bf16[32,2048,2048]{1,0,2}, bf16[24,8192]{1,0}, bf16[24,1024,8192]{2,1,0}, bf16[24,1024]{0,1}, /*index=5*/bf16[24,8192,1024]{1,2,0}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, /*index=10*/bf16[24,3,16,128]{3,2,1,0}, bf16[24,3,1024,16,128]{4,3,1,2,0}, bf16[24,1024]{1,0}, bf16[24,1024,16,128]{3,2,1,0}, bf16[24,8192]{1,0}, /*index=15*/bf16[24,1024,8192]{2,1,0}, bf16[24,8192,1024]{1,2,0}, bf16[24,2048]{1,0}, bf16[24,2048]{1,0}, bf16[24,2048]{1,0}, /*index=20*/bf16[24,2048]{1,0}, bf16[24,3,16,128]{3,2,1,0}, bf16[24,3,1024,16,128]{4,3,1,2,0}, bf16[24,1024]{1,0}, bf16[24,1024,16,128]{3,2,1,0}, /*index=25*/bf16[24,32,2048,2048]{2,1,3,0}, bf16[32,1,2048,2048]{3,2,0,1}, bf16[32,2048]{1,0}))->(s32[], bf16[32,2048,2048]{1,0,2}, bf16[24,8192]{1,0}, bf16[24,1024,8192]{2,1,0}, bf16[24,1024]{0,1}, /*index=5*/bf16[24,8192,1024]{1,2,0}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, /*index=10*/bf16[24,3,16,128]{3,2,1,0}, bf16[24,3,1024,16,128]{4,3,1,2,0}, bf16[24,1024]{1,0}, bf16[24,1024,16,128]{3,2,1,0}, bf16[24,8192]{1,0}, /*index=15*/bf16[24,1024,8192]{2,1,0}, bf16[24,8192,1024]{1,2,0}, bf16[24,2048]{1,0}, bf16[24,2048]{1,0}, bf16[24,2048]{1,0}, /*index=20*/bf16[24,2048]{1,0}, bf16[24,3,16,128]{3,2,1,0}, bf16[24,3,1024,16,128]{4,3,1,2,0}, bf16[24,1024]{1,0}, bf16[24,1024,16,128]{3,2,1,0}, /*index=25*/bf16[24,32,2048,2048]{2,1,3,0}, bf16[32,1,2048,2048]{3,2,0,1}, bf16[32,2048]{1,0})}
add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add.580 = bf16[] add(x, y)
}

region_20.962 {
  Arg_0.963 = f32[] parameter(0)
  Arg_1.964 = f32[] parameter(1)
  ROOT add.579 = f32[] add(Arg_0.963, Arg_1.964)
}

region_39.1120 {
  Arg_0.1121 = f32[] parameter(0)
  Arg_1.1122 = f32[] parameter(1)
  ROOT maximum.21 = f32[] maximum(Arg_0.1121, Arg_1.1122)
}

main {
  param.3 = (s32[], bf16[32,2048,2048]{1,0,2}, bf16[24,8192]{1,0}, bf16[24,1024,8192]{2,1,0}, bf16[24,1024]{0,1}, /*index=5*/bf16[24,8192,1024]{1,2,0}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, /*index=10*/bf16[24,3,16,128]{3,2,1,0}, bf16[24,3,1024,16,128]{4,3,1,2,0}, bf16[24,1024]{1,0}, bf16[24,1024,16,128]{3,2,1,0}, bf16[24,8192]{1,0}, /*index=15*/bf16[24,1024,8192]{2,1,0}, bf16[24,8192,1024]{1,2,0}, bf16[24,2048]{1,0}, bf16[24,2048]{1,0}, bf16[24,2048]{1,0}, /*index=20*/bf16[24,2048]{1,0}, bf16[24,3,16,128]{3,2,1,0}, bf16[24,3,1024,16,128]{4,3,1,2,0}, bf16[24,1024]{1,0}, bf16[24,1024,16,128]{3,2,1,0}, /*index=25*/bf16[24,32,2048,2048]{2,1,3,0}, bf16[32,1,2048,2048]{3,2,0,1}, bf16[32,2048]{1,0}) parameter(0)
  get-tuple-element.31 = s32[] get-tuple-element(param.3), index=0
  constant.1961 = s32[] constant(1)
  add.581 = s32[] add(get-tuple-element.31, constant.1961)
  get-tuple-element.32 = bf16[24,32,2048,2048]{2,1,3,0} get-tuple-element(param.3), index=25
  bitcast.187 = bf16[24,2048,32,2048]{3,2,1,0} bitcast(get-tuple-element.32)
  constant.1977 = s32[] constant(23)
  subtract.221 = s32[] subtract(constant.1977, get-tuple-element.31)
  constant.1980 = s32[] constant(0)
  compare.210 = pred[] compare(subtract.221, constant.1980), direction=LT
  constant.1979 = s32[] constant(47)
  subtract.222 = s32[] subtract(constant.1979, get-tuple-element.31)
  select.372 = s32[] select(compare.210, subtract.222, subtract.221)
  dynamic-slice.324 = bf16[1,2048,32,2048]{3,2,1,0} dynamic-slice(bitcast.187, select.372, constant.1980, constant.1980, constant.1980), dynamic_slice_sizes={1,2048,32,2048}
  bitcast.756 = bf16[2048,32,2048]{2,1,0} bitcast(dynamic-slice.324)
  convert.282 = f32[2048,32,2048]{2,1,0} convert(bitcast.756)
  constant.1991 = bf16[] constant(1)
  broadcast.1270 = bf16[32,2048]{1,0} broadcast(constant.1991), dimensions={}
  get-tuple-element.33 = bf16[32,2048]{1,0} get-tuple-element(param.3), index=27
  subtract.229 = bf16[32,2048]{1,0} subtract(broadcast.1270, get-tuple-element.33)
  convert.285 = f32[32,2048]{1,0} convert(subtract.229)
  broadcast.1228 = f32[2048,32,2048]{2,1,0} broadcast(convert.285), dimensions={1,2}
  multiply.656 = f32[2048,32,2048]{2,1,0} multiply(convert.282, broadcast.1228)
  bitcast.367 = f32[32,2048,2048]{1,0,2} bitcast(multiply.656)
  constant.1968 = f32[] constant(0)
  reduce.84 = f32[] reduce(bitcast.367, constant.1968), dimensions={0,1,2}, to_apply=region_20.962
  all-reduce.230 = f32[] all-reduce(reduce.84), channel_id=278, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_20.962
  broadcast.1221 = f32[32,2048,4096]{2,1,0} broadcast(convert.285), dimensions={0,1}
  reduce.85 = f32[] reduce(broadcast.1221, constant.1968), dimensions={0,1,2}, to_apply=region_20.962
  all-reduce.14 = f32[] all-reduce(reduce.85), channel_id=49, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=region_20.962
  constant.2005 = f32[] constant(1)
  maximum.24 = f32[] maximum(all-reduce.14, constant.2005)
  divide.96 = f32[] divide(all-reduce.230, maximum.24)
  broadcast.1223 = f32[2048,32,2048]{2,1,0} broadcast(divide.96), dimensions={}
  subtract.219 = f32[2048,32,2048]{2,1,0} subtract(convert.282, broadcast.1223)
  multiply.644 = f32[2048,32,2048]{2,1,0} multiply(subtract.219, broadcast.1228)
  multiply.645 = f32[2048,32,2048]{2,1,0} multiply(multiply.644, multiply.644)
  bitcast.271 = f32[32,2048,2048]{1,0,2} bitcast(multiply.645)
  reduce.86 = f32[] reduce(bitcast.271, constant.1968), dimensions={0,1,2}, to_apply=region_20.962
  all-reduce.231 = f32[] all-reduce(reduce.86), channel_id=279, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_20.962
  divide.99 = f32[] divide(all-reduce.231, maximum.24)
  rsqrt.16 = f32[] rsqrt(divide.99)
  multiply.650 = f32[] multiply(rsqrt.16, constant.1968)
  divide.100 = f32[] divide(multiply.650, maximum.24)
  constant.1974 = f32[] constant(2)
  multiply.652 = f32[] multiply(divide.100, constant.1974)
  broadcast.1227 = f32[2048,32,2048]{2,1,0} broadcast(multiply.652), dimensions={}
  multiply.653 = f32[2048,32,2048]{2,1,0} multiply(multiply.644, broadcast.1227)
  multiply.654 = f32[2048,32,2048]{2,1,0} multiply(multiply.653, broadcast.1228)
  negate.56 = f32[2048,32,2048]{2,1,0} negate(multiply.654)
  bitcast.321 = f32[32,2048,2048]{1,0,2} bitcast(negate.56)
  reduce.87 = f32[] reduce(bitcast.321, constant.1968), dimensions={0,1,2}, to_apply=region_20.962
  all-reduce.232 = f32[] all-reduce(reduce.87), channel_id=280, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_20.962
  divide.101 = f32[] divide(all-reduce.232, maximum.24)
  broadcast.1229 = f32[32,2048]{1,0} broadcast(divide.101), dimensions={}
  multiply.655 = f32[32,2048]{1,0} multiply(broadcast.1229, convert.285)
  broadcast.1230 = f32[2048,32,2048]{2,1,0} broadcast(multiply.655), dimensions={1,2}
  add.582 = f32[2048,32,2048]{2,1,0} add(multiply.654, broadcast.1230)
  broadcast.1236 = f32[2048,32,2048]{2,1,0} broadcast(constant.1968), dimensions={}
  compare.208 = pred[2048,32,2048]{2,1,0} compare(multiply.656, broadcast.1236), direction=GE
  abs.22 = f32[2048,32,2048]{2,1,0} abs(multiply.656)
  bitcast.373 = f32[32,2048,2048]{1,0,2} bitcast(abs.22)
  constant.1989 = f32[] constant(-inf)
  reduce.88 = f32[] reduce(bitcast.373, constant.1989), dimensions={0,1,2}, to_apply=region_39.1120
  all-reduce.233 = f32[] all-reduce(reduce.88), channel_id=281, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_39.1120
  broadcast.1233 = f32[2048,32,2048]{2,1,0} broadcast(all-reduce.233), dimensions={}
  compare.207 = pred[2048,32,2048]{2,1,0} compare(abs.22, broadcast.1233), direction=EQ
  convert.286 = f32[2048,32,2048]{2,1,0} convert(compare.207)
  bitcast.393 = f32[32,2048,2048]{1,0,2} bitcast(convert.286)
  reduce.89 = f32[] reduce(bitcast.393, constant.1968), dimensions={0,1,2}, to_apply=region_20.962
  all-reduce.234 = f32[] all-reduce(reduce.89), channel_id=282, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_20.962
  divide.103 = f32[] divide(constant.1968, all-reduce.234)
  broadcast.1238 = f32[2048,32,2048]{2,1,0} broadcast(divide.103), dimensions={}
  select.370 = f32[2048,32,2048]{2,1,0} select(compare.207, broadcast.1238, broadcast.1236)
  select.369 = f32[2048,32,2048]{2,1,0} select(compare.208, select.370, broadcast.1236)
  constant.1976 = pred[] constant(false)
  broadcast.1237 = pred[2048,32,2048]{2,1,0} broadcast(constant.1976), dimensions={}
  compare.209 = pred[2048,32,2048]{2,1,0} compare(compare.208, broadcast.1237), direction=EQ
  select.371 = f32[2048,32,2048]{2,1,0} select(compare.209, select.370, broadcast.1236)
  negate.57 = f32[2048,32,2048]{2,1,0} negate(select.371)
  add.583 = f32[2048,32,2048]{2,1,0} add(select.369, negate.57)
  multiply.658 = f32[2048,32,2048]{2,1,0} multiply(add.583, broadcast.1228)
  add.585 = f32[2048,32,2048]{2,1,0} add(add.582, multiply.658)
  convert.287 = bf16[2048,32,2048]{2,1,0} convert(add.585)
  get-tuple-element.34 = bf16[32,2048,2048]{1,0,2} get-tuple-element(param.3), index=1
  bitcast.1652 = bf16[2048,32,2048]{2,1,0} bitcast(get-tuple-element.34)
  get-tuple-element.35 = bf16[24,3,1024,16,128]{4,3,1,2,0} get-tuple-element(param.3), index=22
  bitcast.461 = bf16[24,1024,3,16,128]{4,3,2,1,0} bitcast(get-tuple-element.35)
  dynamic-slice.325 = bf16[1,1024,3,16,128]{4,3,2,1,0} dynamic-slice(bitcast.461, select.372, constant.1980, constant.1980, constant.1980, /*index=5*/constant.1980), dynamic_slice_sizes={1,1024,3,16,128}
  bitcast.485 = bf16[3,1024,16,128]{3,2,0,1} bitcast(dynamic-slice.325)
  all-gather.7 = bf16[3,4096,16,128]{3,2,0,1} all-gather(bitcast.485), channel_id=60, replica_groups={{0,2,4,6},{1,3,5,7}}, dimensions={1}, use_global_device_ids=true
  bitcast.1420 = bf16[6144,4096]{0,1} bitcast(all-gather.7)
  bitcast.500 = f32[32,2048,2048]{1,0,2} bitcast(convert.282)
  reduce.90 = f32[32,2048]{1,0} reduce(bitcast.500, constant.1968), dimensions={2}, to_apply=region_20.962
  all-reduce.23 = f32[32,2048]{1,0} all-reduce(reduce.90), channel_id=58, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, to_apply=region_20.962
  constant.1983 = f32[] constant(0.000244140625)
  broadcast.1243 = f32[32,2048]{1,0} broadcast(constant.1983), dimensions={}
  multiply.660 = f32[32,2048]{1,0} multiply(all-reduce.23, broadcast.1243)
  broadcast.1242 = f32[2048,32,2048]{2,1,0} broadcast(multiply.660), dimensions={1,2}
  subtract.224 = f32[2048,32,2048]{2,1,0} subtract(convert.282, broadcast.1242)
  multiply.661 = f32[2048,32,2048]{2,1,0} multiply(subtract.224, subtract.224)
  bitcast.527 = f32[32,2048,2048]{1,0,2} bitcast(multiply.661)
  reduce.91 = f32[32,2048]{1,0} reduce(bitcast.527, constant.1968), dimensions={2}, to_apply=region_20.962
  all-reduce.24 = f32[32,2048]{1,0} all-reduce(reduce.91), channel_id=59, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, to_apply=region_20.962
  multiply.662 = f32[32,2048]{1,0} multiply(all-reduce.24, broadcast.1243)
  constant.1990 = f32[] constant(1e-05)
  broadcast.1264 = f32[32,2048]{1,0} broadcast(constant.1990), dimensions={}
  add.587 = f32[32,2048]{1,0} add(multiply.662, broadcast.1264)
  bitcast.1447 = f32[1,32,2048]{2,1,0} bitcast(add.587)
  rsqrt.20 = f32[1,32,2048]{2,1,0} rsqrt(bitcast.1447)
  bitcast.1892 = f32[32,2048]{1,0} bitcast(rsqrt.20)
  broadcast.1337 = f32[2048,32,2048]{2,1,0} broadcast(bitcast.1892), dimensions={1,2}
  multiply.754 = f32[2048,32,2048]{2,1,0} multiply(subtract.224, broadcast.1337)
  convert.314 = bf16[2048,32,2048]{2,1,0} convert(multiply.754)
  get-tuple-element.36 = bf16[24,2048]{1,0} get-tuple-element(param.3), index=20
  dynamic-slice.326 = bf16[1,2048]{1,0} dynamic-slice(get-tuple-element.36, select.372, constant.1980), dynamic_slice_sizes={1,2048}
  broadcast.1266 = bf16[1,2048]{1,0} broadcast(constant.1991), dimensions={}
  add.588 = bf16[1,2048]{1,0} add(dynamic-slice.326, broadcast.1266)
  bitcast.1992 = bf16[2048]{0} bitcast(add.588)
  broadcast.1338 = bf16[2048,32,2048]{2,1,0} broadcast(bitcast.1992), dimensions={0}
  multiply.755 = bf16[2048,32,2048]{2,1,0} multiply(convert.314, broadcast.1338)
  get-tuple-element.37 = bf16[24,2048]{1,0} get-tuple-element(param.3), index=19
  dynamic-slice.327 = bf16[1,2048]{1,0} dynamic-slice(get-tuple-element.37, select.372, constant.1980), dynamic_slice_sizes={1,2048}
  bitcast.1998 = bf16[2048]{0} bitcast(dynamic-slice.327)
  broadcast.1339 = bf16[2048,32,2048]{2,1,0} broadcast(bitcast.1998), dimensions={0}
  add.640 = bf16[2048,32,2048]{2,1,0} add(multiply.755, broadcast.1339)
  bitcast.2003 = bf16[32,2048,2048]{1,0,2} bitcast(add.640)
  all-gather.8 = bf16[32,2048,4096]{1,0,2} all-gather(bitcast.2003), channel_id=61, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, use_global_device_ids=true
  bitcast.597 = bf16[4096,65536]{1,0} bitcast(all-gather.8)
  dot.42 = bf16[6144,65536]{1,0} dot(bitcast.1420, bitcast.597), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast.623 = bf16[3,16,128,32,2048]{4,3,2,1,0} bitcast(dot.42)
  transpose.112 = bf16[3,32,16,128,2048]{4,3,2,1,0} transpose(bitcast.623), dimensions={0,3,1,2,4}
  get-tuple-element.38 = bf16[24,3,16,128]{3,2,1,0} get-tuple-element(param.3), index=21
  dynamic-slice.328 = bf16[1,3,16,128]{3,2,1,0} dynamic-slice(get-tuple-element.38, select.372, constant.1980, constant.1980, constant.1980), dynamic_slice_sizes={1,3,16,128}
  bitcast.626 = bf16[3,16,128]{2,1,0} bitcast(dynamic-slice.328)
  broadcast.1250 = bf16[3,32,16,128,2048]{4,3,2,1,0} broadcast(bitcast.626), dimensions={0,2,3}
  add.591 = bf16[3,32,16,128,2048]{4,3,2,1,0} add(transpose.112, broadcast.1250)
  slice.87 = bf16[1,32,16,128,2048]{4,3,2,1,0} slice(add.591), slice={[2:3], [0:32], [0:16], [0:128], [0:2048]}
  bitcast.1280 = bf16[32,16,128,2048]{3,2,1,0} bitcast(slice.87)
  slice.88 = bf16[1,32,16,128,2048]{4,3,2,1,0} slice(add.591), slice={[0:1], [0:32], [0:16], [0:128], [0:2048]}
  constant.2007 = bf16[] constant(0.08838)
  broadcast.1251 = bf16[1,32,16,128,2048]{4,3,2,1,0} broadcast(constant.2007), dimensions={}
  multiply.666 = bf16[1,32,16,128,2048]{4,3,2,1,0} multiply(slice.88, broadcast.1251)
  bitcast.1330 = bf16[32,16,128,2048]{3,2,1,0} bitcast(multiply.666)
  transpose.113 = bf16[32,16,2048,128]{3,2,1,0} transpose(bitcast.1330), dimensions={0,1,3,2}
  slice.89 = bf16[1,32,16,128,2048]{4,3,2,1,0} slice(add.591), slice={[1:2], [0:32], [0:16], [0:128], [0:2048]}
  bitcast.647 = bf16[32,16,128,2048]{3,2,1,0} bitcast(slice.89)
  dot.43 = bf16[32,16,2048,2048]{3,2,1,0} dot(transpose.113, bitcast.647), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  convert.291 = f32[32,16,2048,2048]{3,2,1,0} convert(dot.43)
  get-tuple-element.39 = bf16[32,1,2048,2048]{3,2,0,1} get-tuple-element(param.3), index=26
  bitcast.651 = bf16[1,32,2048,2048]{3,2,1,0} bitcast(get-tuple-element.39)
  iota.38 = s32[2048,2048]{1,0} iota(), iota_dimension=0
  iota.39 = s32[2048,2048]{1,0} iota(), iota_dimension=1
  compare.211 = pred[2048,2048]{1,0} compare(iota.38, iota.39), direction=LT
  constant.1987 = bf16[] constant(-2.366e+38)
  broadcast.1252 = bf16[2048,2048]{1,0} broadcast(constant.1987), dimensions={}
  constant.2006 = bf16[] constant(0)
  broadcast.1253 = bf16[2048,2048]{1,0} broadcast(constant.2006), dimensions={}
  select.373 = bf16[2048,2048]{1,0} select(compare.211, broadcast.1252, broadcast.1253)
  broadcast.1254 = bf16[1,32,2048,2048]{3,2,1,0} broadcast(select.373), dimensions={2,3}
  minimum.5 = bf16[1,32,2048,2048]{3,2,1,0} minimum(bitcast.651, broadcast.1254)
  bitcast.673 = bf16[32,2048,2048]{2,1,0} bitcast(minimum.5)
  convert.292 = f32[32,2048,2048]{2,1,0} convert(bitcast.673)
  broadcast.1256 = f32[32,16,2048,2048]{3,2,1,0} broadcast(convert.292), dimensions={0,2,3}
  add.593 = f32[32,16,2048,2048]{3,2,1,0} add(convert.291, broadcast.1256)
  reduce.92 = f32[32,16,2048]{2,1,0} reduce(add.593, constant.1989), dimensions={3}, to_apply=region_39.1120
  broadcast.1258 = f32[32,16,2048,2048]{3,2,1,0} broadcast(reduce.92), dimensions={0,1,2}
  subtract.226 = f32[32,16,2048,2048]{3,2,1,0} subtract(add.593, broadcast.1258)
  exponential.8 = f32[32,16,2048,2048]{3,2,1,0} exponential(subtract.226)
  reduce.93 = f32[32,16,2048]{2,1,0} reduce(exponential.8, constant.1968), dimensions={3}, to_apply=region_20.962
  broadcast.1309 = f32[32,16,2048,2048]{3,2,1,0} broadcast(reduce.93), dimensions={0,1,2}
  divide.109 = f32[32,16,2048,2048]{3,2,1,0} divide(exponential.8, broadcast.1309)
  convert.306 = bf16[32,16,2048,2048]{3,2,1,0} convert(divide.109)
  dot.44 = bf16[32,16,128,2048]{3,2,1,0} dot(bitcast.1280, convert.306), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.116 = bf16[32,2048,16,128]{3,2,1,0} transpose(dot.44), dimensions={0,3,1,2}
  bitcast.711 = bf16[65536,2048]{1,0} bitcast(transpose.116)
  get-tuple-element.40 = bf16[24,1024,16,128]{3,2,1,0} get-tuple-element(param.3), index=24
  dynamic-slice.329 = bf16[1,1024,16,128]{3,2,1,0} dynamic-slice(get-tuple-element.40, select.372, constant.1980, constant.1980, constant.1980), dynamic_slice_sizes={1,1024,16,128}
  bitcast.724 = bf16[1024,16,128]{2,1,0} bitcast(dynamic-slice.329)
  all-gather.9 = bf16[4096,16,128]{2,1,0} all-gather(bitcast.724), channel_id=62, replica_groups={{0,2,4,6},{1,3,5,7}}, dimensions={0}, use_global_device_ids=true
  bitcast.729 = bf16[2048,4096]{0,1} bitcast(all-gather.9)
  dot.57 = bf16[65536,4096]{0,1} dot(bitcast.711, bitcast.729), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast.733 = bf16[32,2048,4096]{1,0,2} bitcast(dot.57)
  reduce-scatter = bf16[32,2048,2048]{1,0,2} reduce-scatter(bitcast.733), channel_id=322, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={2}, to_apply=add
  bitcast.763 = bf16[2048,32,2048]{2,1,0} bitcast(reduce-scatter)
  get-tuple-element.41 = bf16[24,1024]{1,0} get-tuple-element(param.3), index=23
  dynamic-slice.330 = bf16[1,1024]{1,0} dynamic-slice(get-tuple-element.41, select.372, constant.1980), dynamic_slice_sizes={1,1024}
  bitcast.748 = bf16[1024]{0} bitcast(dynamic-slice.330)
  collective-permute.1 = bf16[1024]{0} collective-permute(bitcast.748), channel_id=64, source_target_pairs={{0,0},{1,2},{2,4},{3,6},{4,1},{5,3},{6,5},{7,7}}
  all-gather.10 = bf16[2048]{0} all-gather(collective-permute.1), channel_id=65, replica_groups={{0,4},{2,6},{1,5},{3,7}}, dimensions={0}, use_global_device_ids=true
  broadcast.1261 = bf16[2048,32,2048]{2,1,0} broadcast(all-gather.10), dimensions={0}
  add.596 = bf16[2048,32,2048]{2,1,0} add(bitcast.763, broadcast.1261)
  add.597 = bf16[2048,32,2048]{2,1,0} add(add.596, bitcast.756)
  convert.295 = f32[2048,32,2048]{2,1,0} convert(add.597)
  bitcast.774 = f32[32,2048,2048]{1,0,2} bitcast(convert.295)
  reduce.94 = f32[32,2048]{1,0} reduce(bitcast.774, constant.1968), dimensions={2}, to_apply=region_20.962
  all-reduce.26 = f32[32,2048]{1,0} all-reduce(reduce.94), channel_id=66, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, to_apply=region_20.962
  multiply.668 = f32[32,2048]{1,0} multiply(all-reduce.26, broadcast.1243)
  broadcast.1263 = f32[2048,32,2048]{2,1,0} broadcast(multiply.668), dimensions={1,2}
  subtract.228 = f32[2048,32,2048]{2,1,0} subtract(convert.295, broadcast.1263)
  multiply.669 = f32[2048,32,2048]{2,1,0} multiply(subtract.228, subtract.228)
  bitcast.809 = f32[32,2048,2048]{1,0,2} bitcast(multiply.669)
  reduce.95 = f32[32,2048]{1,0} reduce(bitcast.809, constant.1968), dimensions={2}, to_apply=region_20.962
  all-reduce.27 = f32[32,2048]{1,0} all-reduce(reduce.95), channel_id=67, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, to_apply=region_20.962
  multiply.670 = f32[32,2048]{1,0} multiply(all-reduce.27, broadcast.1243)
  add.598 = f32[32,2048]{1,0} add(multiply.670, broadcast.1264)
  bitcast.1148 = f32[1,32,2048]{2,1,0} bitcast(add.598)
  rsqrt.19 = f32[1,32,2048]{2,1,0} rsqrt(bitcast.1148)
  bitcast.1602 = f32[32,2048]{1,0} bitcast(rsqrt.19)
  broadcast.1329 = f32[2048,32,2048]{2,1,0} broadcast(bitcast.1602), dimensions={1,2}
  multiply.750 = f32[2048,32,2048]{2,1,0} multiply(subtract.228, broadcast.1329)
  convert.312 = bf16[2048,32,2048]{2,1,0} convert(multiply.750)
  get-tuple-element.42 = bf16[24,2048]{1,0} get-tuple-element(param.3), index=18
  dynamic-slice.331 = bf16[1,2048]{1,0} dynamic-slice(get-tuple-element.42, select.372, constant.1980), dynamic_slice_sizes={1,2048}
  add.599 = bf16[1,2048]{1,0} add(dynamic-slice.331, broadcast.1266)
  bitcast.1609 = bf16[2048]{0} bitcast(add.599)
  broadcast.1330 = bf16[2048,32,2048]{2,1,0} broadcast(bitcast.1609), dimensions={0}
  multiply.745 = bf16[2048,32,2048]{2,1,0} multiply(convert.312, broadcast.1330)
  get-tuple-element.43 = bf16[24,2048]{1,0} get-tuple-element(param.3), index=17
  dynamic-slice.332 = bf16[1,2048]{1,0} dynamic-slice(get-tuple-element.43, select.372, constant.1980), dynamic_slice_sizes={1,2048}
  bitcast.1615 = bf16[2048]{0} bitcast(dynamic-slice.332)
  broadcast.1331 = bf16[2048,32,2048]{2,1,0} broadcast(bitcast.1615), dimensions={0}
  add.636 = bf16[2048,32,2048]{2,1,0} add(multiply.745, broadcast.1331)
  bitcast.1620 = bf16[32,2048,2048]{1,0,2} bitcast(add.636)
  all-gather.12 = bf16[32,2048,4096]{1,0,2} all-gather(bitcast.1620), channel_id=69, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, use_global_device_ids=true
  bitcast.877 = bf16[65536,4096]{0,1} bitcast(all-gather.12)
  get-tuple-element.44 = bf16[24,1024,8192]{2,1,0} get-tuple-element(param.3), index=15
  dynamic-slice.333 = bf16[1,1024,8192]{2,1,0} dynamic-slice(get-tuple-element.44, select.372, constant.1980, constant.1980), dynamic_slice_sizes={1,1024,8192}
  bitcast.890 = bf16[1024,8192]{1,0} bitcast(dynamic-slice.333)
  all-gather.11 = bf16[4096,8192]{1,0} all-gather(bitcast.890), channel_id=68, replica_groups={{0,2,4,6},{1,3,5,7}}, dimensions={0}, use_global_device_ids=true
  dot.45 = bf16[65536,8192]{1,0} dot(bitcast.877, all-gather.11), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  get-tuple-element.45 = bf16[24,8192]{1,0} get-tuple-element(param.3), index=14
  dynamic-slice.334 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.45, select.372, constant.1980), dynamic_slice_sizes={1,8192}
  bitcast.906 = bf16[8192]{0} bitcast(dynamic-slice.334)
  broadcast.1269 = bf16[65536,8192]{1,0} broadcast(bitcast.906), dimensions={1}
  add.601 = bf16[65536,8192]{1,0} add(dot.45, broadcast.1269)
  bitcast.997 = bf16[32,2048,8192]{2,1,0} bitcast(add.601)
  broadcast.1333 = bf16[2048,32,2048]{2,1,0} broadcast(subtract.229), dimensions={1,2}
  multiply.746 = bf16[2048,32,2048]{2,1,0} multiply(bitcast.1652, broadcast.1333)
  bitcast.1739 = bf16[32,2048,2048]{1,0,2} bitcast(multiply.746)
  all-gather.14 = bf16[32,2048,4096]{1,0,2} all-gather(bitcast.1739), channel_id=71, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, use_global_device_ids=true
  bitcast.934 = bf16[65536,4096]{0,1} bitcast(all-gather.14)
  get-tuple-element.46 = bf16[24,8192,1024]{1,2,0} get-tuple-element(param.3), index=16
  bitcast.935 = bf16[24,1024,8192]{2,1,0} bitcast(get-tuple-element.46)
  dynamic-slice.335 = bf16[1,1024,8192]{2,1,0} dynamic-slice(bitcast.935, select.372, constant.1980, constant.1980), dynamic_slice_sizes={1,1024,8192}
  bitcast.947 = bf16[8192,1024]{0,1} bitcast(dynamic-slice.335)
  all-gather.13 = bf16[8192,4096]{0,1} all-gather(bitcast.947), channel_id=70, replica_groups={{0,2,4,6},{1,3,5,7}}, dimensions={1}, use_global_device_ids=true
  dot.46 = bf16[65536,8192]{1,0} dot(bitcast.934, all-gather.13), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bitcast.1092 = bf16[32,2048,8192]{2,1,0} bitcast(dot.46)
  broadcast.1335 = bf16[32,2048,8192]{2,1,0} broadcast(subtract.229), dimensions={0,1}
  multiply.703 = bf16[32,2048,8192]{2,1,0} multiply(bitcast.1092, broadcast.1335)
  multiply.685 = bf16[32,2048,8192]{2,1,0} multiply(bitcast.997, multiply.703)
  constant.2002 = bf16[] constant(0.5)
  broadcast.1288 = bf16[32,2048,8192]{2,1,0} broadcast(constant.2002), dimensions={}
  multiply.686 = bf16[32,2048,8192]{2,1,0} multiply(multiply.685, broadcast.1288)
  broadcast.1287 = bf16[32,2048,8192]{2,1,0} broadcast(constant.1991), dimensions={}
  multiply.700 = bf16[32,2048,8192]{2,1,0} multiply(bitcast.997, bitcast.997)
  multiply.693 = bf16[32,2048,8192]{2,1,0} multiply(bitcast.997, multiply.700)
  constant.1998 = bf16[] constant(0.04468)
  broadcast.1282 = bf16[32,2048,8192]{2,1,0} broadcast(constant.1998), dimensions={}
  multiply.694 = bf16[32,2048,8192]{2,1,0} multiply(multiply.693, broadcast.1282)
  add.605 = bf16[32,2048,8192]{2,1,0} add(bitcast.997, multiply.694)
  constant.2010 = bf16[] constant(0.7969)
  broadcast.1324 = bf16[32,2048,8192]{2,1,0} broadcast(constant.2010), dimensions={}
  multiply.695 = bf16[32,2048,8192]{2,1,0} multiply(add.605, broadcast.1324)
  tanh.7 = bf16[32,2048,8192]{2,1,0} tanh(multiply.695)
  subtract.231 = bf16[32,2048,8192]{2,1,0} subtract(broadcast.1287, tanh.7)
  multiply.691 = bf16[32,2048,8192]{2,1,0} multiply(multiply.686, subtract.231)
  multiply.737 = bf16[32,2048,8192]{2,1,0} multiply(multiply.691, tanh.7)
  add.630 = bf16[32,2048,8192]{2,1,0} add(multiply.691, multiply.737)
  multiply.738 = bf16[32,2048,8192]{2,1,0} multiply(add.630, broadcast.1324)
  constant.2011 = bf16[] constant(0.03564)
  broadcast.1326 = bf16[32,2048,8192]{2,1,0} broadcast(constant.2011), dimensions={}
  multiply.739 = bf16[32,2048,8192]{2,1,0} multiply(add.630, broadcast.1326)
  constant.2012 = bf16[] constant(3)
  broadcast.1327 = bf16[32,2048,8192]{2,1,0} broadcast(constant.2012), dimensions={}
  multiply.740 = bf16[32,2048,8192]{2,1,0} multiply(multiply.700, broadcast.1327)
  multiply.741 = bf16[32,2048,8192]{2,1,0} multiply(multiply.739, multiply.740)
  add.632 = bf16[32,2048,8192]{2,1,0} add(multiply.738, multiply.741)
  add.637 = bf16[32,2048,8192]{2,1,0} add(tanh.7, broadcast.1287)
  multiply.747 = bf16[32,2048,8192]{2,1,0} multiply(add.637, broadcast.1288)
  multiply.743 = bf16[32,2048,8192]{2,1,0} multiply(multiply.703, multiply.747)
  add.635 = bf16[32,2048,8192]{2,1,0} add(add.632, multiply.743)
  bitcast.1629 = bf16[65536,8192]{1,0} bitcast(add.635)
  dot.47 = bf16[65536,4096]{0,1} dot(bitcast.1629, all-gather.11), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bitcast.1130 = bf16[32,2048,4096]{1,0,2} bitcast(dot.47)
  reduce-scatter.1 = bf16[32,2048,2048]{1,0,2} reduce-scatter(bitcast.1130), channel_id=323, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={2}, to_apply=add
  bitcast.1766 = bf16[2048,32,2048]{2,1,0} bitcast(reduce-scatter.1)
  multiply.712 = bf16[2048,32,2048]{2,1,0} multiply(bitcast.1766, broadcast.1330)
  convert.299 = f32[2048,32,2048]{2,1,0} convert(multiply.712)
  multiply.707 = f32[2048,32,2048]{2,1,0} multiply(subtract.228, convert.299)
  bitcast.1135 = f32[32,2048,2048]{1,0,2} bitcast(multiply.707)
  reduce.96 = f32[32,2048]{1,0} reduce(bitcast.1135, constant.1968), dimensions={2}, to_apply=region_20.962
  all-reduce.29 = f32[32,2048]{1,0} all-reduce(reduce.96), channel_id=73, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, to_apply=region_20.962
  bitcast.1140 = f32[1,32,2048]{2,1,0} bitcast(all-reduce.29)
  divide.105 = f32[1,32,2048]{2,1,0} divide(rsqrt.19, bitcast.1148)
  constant.2008 = f32[] constant(-0.5)
  broadcast.1313 = f32[1,32,2048]{2,1,0} broadcast(constant.2008), dimensions={}
  multiply.708 = f32[1,32,2048]{2,1,0} multiply(divide.105, broadcast.1313)
  multiply.709 = f32[1,32,2048]{2,1,0} multiply(bitcast.1140, multiply.708)
  constant.2009 = f32[] constant(0.00048828125)
  broadcast.1315 = f32[1,32,2048]{2,1,0} broadcast(constant.2009), dimensions={}
  multiply.710 = f32[1,32,2048]{2,1,0} multiply(multiply.709, broadcast.1315)
  bitcast.1235 = f32[32,2048]{1,0} bitcast(multiply.710)
  broadcast.1296 = f32[2048,32,2048]{2,1,0} broadcast(bitcast.1235), dimensions={1,2}
  multiply.717 = f32[2048,32,2048]{2,1,0} multiply(subtract.228, broadcast.1296)
  multiply.718 = f32[2048,32,2048]{2,1,0} multiply(convert.299, broadcast.1329)
  add.617 = f32[2048,32,2048]{2,1,0} add(multiply.717, multiply.718)
  negate.58 = f32[2048,32,2048]{2,1,0} negate(multiply.717)
  bitcast.1189 = f32[32,2048,2048]{1,0,2} bitcast(negate.58)
  reduce.97 = f32[32,2048]{1,0} reduce(bitcast.1189, constant.1968), dimensions={2}, to_apply=region_20.962
  negate.59 = f32[2048,32,2048]{2,1,0} negate(multiply.718)
  bitcast.1203 = f32[32,2048,2048]{1,0,2} bitcast(negate.59)
  reduce.98 = f32[32,2048]{1,0} reduce(bitcast.1203, constant.1968), dimensions={2}, to_apply=region_20.962
  add.613 = f32[32,2048]{1,0} add(reduce.97, reduce.98)
  all-reduce.274 = f32[32,2048]{1,0} all-reduce(add.613), channel_id=335, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, to_apply=region_20.962
  multiply.719 = f32[32,2048]{1,0} multiply(all-reduce.274, broadcast.1243)
  broadcast.1297 = f32[2048,32,2048]{2,1,0} broadcast(multiply.719), dimensions={1,2}
  add.618 = f32[2048,32,2048]{2,1,0} add(add.617, broadcast.1297)
  convert.301 = bf16[2048,32,2048]{2,1,0} convert(add.618)
  add.619 = bf16[2048,32,2048]{2,1,0} add(bitcast.1652, convert.301)
  add.616 = bf16[2048,32,2048]{2,1,0} add(convert.287, add.619)
  bitcast.2063 = bf16[32,2048,2048]{1,0,2} bitcast(add.619)
  all-gather.15 = bf16[32,2048,4096]{1,0,2} all-gather(bitcast.2063), channel_id=76, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, use_global_device_ids=true
  bitcast.1263 = bf16[65536,4096]{0,1} bitcast(all-gather.15)
  bitcast.1269 = bf16[4096,2048]{1,0} bitcast(all-gather.9)
  dot.48 = bf16[65536,2048]{1,0} dot(bitcast.1263, bitcast.1269), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast.1381 = bf16[32,2048,16,128]{3,2,1,0} bitcast(dot.48)
  transpose.122 = bf16[32,16,2048,128]{3,2,1,0} transpose(bitcast.1381), dimensions={0,2,1,3}
  dot.49 = bf16[32,16,2048,2048]{3,2,1,0} dot(transpose.122, bitcast.1280), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  convert.303 = f32[32,16,2048,2048]{3,2,1,0} convert(dot.49)
  broadcast.1298 = f32[32,16,2048]{2,1,0} broadcast(constant.2005), dimensions={}
  multiply.720 = f32[32,16,2048]{2,1,0} multiply(reduce.93, reduce.93)
  divide.106 = f32[32,16,2048]{2,1,0} divide(broadcast.1298, multiply.720)
  broadcast.1299 = f32[32,16,2048,2048]{3,2,1,0} broadcast(divide.106), dimensions={0,1,2}
  multiply.721 = f32[32,16,2048,2048]{3,2,1,0} multiply(convert.303, broadcast.1299)
  multiply.722 = f32[32,16,2048,2048]{3,2,1,0} multiply(multiply.721, exponential.8)
  reduce.99 = f32[32,16,2048]{2,1,0} reduce(multiply.722, constant.1968), dimensions={3}, to_apply=region_20.962
  negate.61 = f32[32,16,2048]{2,1,0} negate(reduce.99)
  broadcast.1305 = f32[32,16,2048,2048]{3,2,1,0} broadcast(negate.61), dimensions={0,1,2}
  divide.108 = f32[32,16,2048,2048]{3,2,1,0} divide(convert.303, broadcast.1309)
  add.622 = f32[32,16,2048,2048]{3,2,1,0} add(broadcast.1305, divide.108)
  multiply.724 = f32[32,16,2048,2048]{3,2,1,0} multiply(add.622, exponential.8)
  convert.305 = bf16[32,16,2048,2048]{3,2,1,0} convert(multiply.724)
  dot.50 = bf16[32,16,2048,128]{3,2,1,0} dot(convert.305, transpose.113), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  bitcast.1934 = bf16[1,32,16,2048,128]{4,3,2,1,0} bitcast(dot.50)
  pad.6 = bf16[3,32,16,2048,128]{4,3,2,1,0} pad(bitcast.1934, constant.2006), padding=1_1x0_0x0_0x0_0x0_0
  transpose.120 = bf16[32,16,2048,128]{3,2,1,0} transpose(bitcast.647), dimensions={0,1,3,2}
  dot.51 = bf16[32,16,2048,128]{3,2,1,0} dot(convert.305, transpose.120), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  broadcast.1307 = bf16[32,16,2048,128]{3,2,1,0} broadcast(constant.2007), dimensions={}
  multiply.725 = bf16[32,16,2048,128]{3,2,1,0} multiply(dot.51, broadcast.1307)
  bitcast.1941 = bf16[1,32,16,2048,128]{4,3,2,1,0} bitcast(multiply.725)
  pad.7 = bf16[3,32,16,2048,128]{4,3,2,1,0} pad(bitcast.1941, constant.2006), padding=0_2x0_0x0_0x0_0x0_0
  add.638 = bf16[3,32,16,2048,128]{4,3,2,1,0} add(pad.6, pad.7)
  transpose.123 = bf16[32,16,128,2048]{3,2,1,0} transpose(bitcast.1381), dimensions={0,2,3,1}
  dot.89 = bf16[32,16,2048,128]{3,2,1,0} dot(convert.306, transpose.123), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  bitcast.1949 = bf16[1,32,16,2048,128]{4,3,2,1,0} bitcast(dot.89)
  pad.8 = bf16[3,32,16,2048,128]{4,3,2,1,0} pad(bitcast.1949, constant.2006), padding=2_0x0_0x0_0x0_0x0_0
  add.639 = bf16[3,32,16,2048,128]{4,3,2,1,0} add(add.638, pad.8)
  transpose.127 = bf16[32,2048,3,16,128]{4,3,2,1,0} transpose(add.639), dimensions={1,3,0,2,4}
  bitcast.1416 = bf16[65536,6144]{1,0} bitcast(transpose.127)
  dot.52 = bf16[65536,4096]{0,1} dot(bitcast.1416, bitcast.1420), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast.1424 = bf16[32,2048,4096]{1,0,2} bitcast(dot.52)
  reduce-scatter.2 = bf16[32,2048,2048]{1,0,2} reduce-scatter(bitcast.1424), channel_id=324, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={2}, to_apply=add
  bitcast.1851 = bf16[2048,32,2048]{2,1,0} bitcast(reduce-scatter.2)
  multiply.732 = bf16[2048,32,2048]{2,1,0} multiply(bitcast.1851, broadcast.1338)
  convert.308 = f32[2048,32,2048]{2,1,0} convert(multiply.732)
  multiply.727 = f32[2048,32,2048]{2,1,0} multiply(subtract.224, convert.308)
  bitcast.1434 = f32[32,2048,2048]{1,0,2} bitcast(multiply.727)
  reduce.100 = f32[32,2048]{1,0} reduce(bitcast.1434, constant.1968), dimensions={2}, to_apply=region_20.962
  all-reduce.33 = f32[32,2048]{1,0} all-reduce(reduce.100), channel_id=78, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, to_apply=region_20.962
  bitcast.1439 = f32[1,32,2048]{2,1,0} bitcast(all-reduce.33)
  divide.110 = f32[1,32,2048]{2,1,0} divide(rsqrt.20, bitcast.1447)
  multiply.728 = f32[1,32,2048]{2,1,0} multiply(divide.110, broadcast.1313)
  multiply.729 = f32[1,32,2048]{2,1,0} multiply(bitcast.1439, multiply.728)
  multiply.730 = f32[1,32,2048]{2,1,0} multiply(multiply.729, broadcast.1315)
  bitcast.1485 = f32[32,2048]{1,0} bitcast(multiply.730)
  broadcast.1321 = f32[2048,32,2048]{2,1,0} broadcast(bitcast.1485), dimensions={1,2}
  multiply.734 = f32[2048,32,2048]{2,1,0} multiply(subtract.224, broadcast.1321)
  multiply.735 = f32[2048,32,2048]{2,1,0} multiply(convert.308, broadcast.1337)
  add.625 = f32[2048,32,2048]{2,1,0} add(multiply.734, multiply.735)
  negate.62 = f32[2048,32,2048]{2,1,0} negate(multiply.734)
  bitcast.1491 = f32[32,2048,2048]{1,0,2} bitcast(negate.62)
  reduce.101 = f32[32,2048]{1,0} reduce(bitcast.1491, constant.1968), dimensions={2}, to_apply=region_20.962
  negate.63 = f32[2048,32,2048]{2,1,0} negate(multiply.735)
  bitcast.1505 = f32[32,2048,2048]{1,0,2} bitcast(negate.63)
  reduce.102 = f32[32,2048]{1,0} reduce(bitcast.1505, constant.1968), dimensions={2}, to_apply=region_20.962
  add.626 = f32[32,2048]{1,0} add(reduce.101, reduce.102)
  all-reduce.275 = f32[32,2048]{1,0} all-reduce(add.626), channel_id=336, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, to_apply=region_20.962
  multiply.736 = f32[32,2048]{1,0} multiply(all-reduce.275, broadcast.1243)
  broadcast.1323 = f32[2048,32,2048]{2,1,0} broadcast(multiply.736), dimensions={1,2}
  add.628 = f32[2048,32,2048]{2,1,0} add(add.625, broadcast.1323)
  convert.309 = bf16[2048,32,2048]{2,1,0} convert(add.628)
  add.629 = bf16[2048,32,2048]{2,1,0} add(add.616, convert.309)
  bitcast.1525 = bf16[32,2048,2048]{1,0,2} bitcast(add.629)
  get-tuple-element.47 = bf16[24,8192]{1,0} get-tuple-element(param.3), index=2
  reduce.103 = bf16[8192]{0} reduce(add.635, constant.2006), dimensions={0,1}, to_apply=add
  all-reduce.36 = bf16[8192]{0} all-reduce(reduce.103), channel_id=81, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=add
  bitcast.1583 = bf16[1,8192]{1,0} bitcast(all-reduce.36)
  dynamic-update-slice.28 = bf16[24,8192]{1,0} dynamic-update-slice(get-tuple-element.47, bitcast.1583, select.372, constant.1980)
  get-tuple-element.48 = bf16[24,1024,8192]{2,1,0} get-tuple-element(param.3), index=3
  all-gather.16 = bf16[32,2048,4096]{1,0,2} all-gather(bitcast.1620), channel_id=82, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, use_global_device_ids=true
  bitcast.1625 = bf16[4096,65536]{1,0} bitcast(all-gather.16)
  dot.53 = bf16[4096,8192]{1,0} dot(bitcast.1625, bitcast.1629), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  reduce-scatter.3 = bf16[1024,8192]{1,0} reduce-scatter(dot.53), channel_id=325, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, dimensions={0}, to_apply=add
  bitcast.1634 = bf16[1,1024,8192]{2,1,0} bitcast(reduce-scatter.3)
  dynamic-update-slice.29 = bf16[24,1024,8192]{2,1,0} dynamic-update-slice(get-tuple-element.48, bitcast.1634, select.372, constant.1980, constant.1980)
  get-tuple-element.49 = bf16[24,1024]{0,1} get-tuple-element(param.3), index=4
  collective-permute.2 = bf16[24,1024]{0,1} collective-permute(get-tuple-element.49), channel_id=85, source_target_pairs={{0,0},{1,2},{2,4},{3,6},{4,1},{5,3},{6,5},{7,7}}
  all-gather.17 = bf16[24,2048]{0,1} all-gather(collective-permute.2), channel_id=86, replica_groups={{0,4},{2,6},{1,5},{3,7}}, dimensions={1}, use_global_device_ids=true
  bitcast.1649 = bf16[2048,24]{1,0} bitcast(all-gather.17)
  reduce.104 = bf16[2048]{0} reduce(bitcast.1739, constant.2006), dimensions={0,1}, to_apply=add
  all-reduce.38 = bf16[2048]{0} all-reduce(reduce.104), channel_id=84, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=add
  bitcast.1671 = bf16[2048,1]{1,0} bitcast(all-reduce.38)
  dynamic-update-slice.30 = bf16[2048,24]{1,0} dynamic-update-slice(bitcast.1649, bitcast.1671, constant.1980, select.372)
  constant.2013 = s32[8]{0} constant({0, 2048, 0, 2048, 1024, 3072, 1024, 3072})
  partition-id.3 = u32[] partition-id()
  dynamic-slice.336 = s32[1]{0} dynamic-slice(constant.2013, partition-id.3), dynamic_slice_sizes={1}
  constant.2014 = s32[8]{0} constant({0, 2048, 0, 2048, 0, 2048, 0, 2048})
  dynamic-slice.337 = s32[1]{0} dynamic-slice(constant.2014, partition-id.3), dynamic_slice_sizes={1}
  subtract.232 = s32[1]{0} subtract(dynamic-slice.336, dynamic-slice.337)
  bitcast.2087 = s32[] bitcast(subtract.232)
  dynamic-slice.338 = bf16[1024,24]{1,0} dynamic-slice(dynamic-update-slice.30, bitcast.2087, constant.1980), dynamic_slice_sizes={1024,24}
  bitcast.1695 = bf16[24,1024]{0,1} bitcast(dynamic-slice.338)
  collective-permute.9 = bf16[24,1024]{0,1} collective-permute(bitcast.1695), channel_id=109, source_target_pairs={{0,0},{2,1},{4,2},{6,3},{1,4},{3,5},{5,6},{7,7}}
  get-tuple-element.50 = bf16[24,8192,1024]{1,2,0} get-tuple-element(param.3), index=5
  bitcast.1698 = bf16[24,1024,8192]{2,1,0} bitcast(get-tuple-element.50)
  multiply.748 = bf16[32,2048,8192]{2,1,0} multiply(bitcast.997, multiply.747)
  multiply.749 = bf16[32,2048,8192]{2,1,0} multiply(multiply.748, broadcast.1335)
  bitcast.1735 = bf16[8192,65536]{0,1} bitcast(multiply.749)
  all-gather.18 = bf16[32,2048,4096]{1,0,2} all-gather(bitcast.1739), channel_id=87, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, use_global_device_ids=true
  bitcast.1743 = bf16[65536,4096]{0,1} bitcast(all-gather.18)
  dot.54 = bf16[8192,4096]{0,1} dot(bitcast.1735, bitcast.1743), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  reduce-scatter.4 = bf16[8192,1024]{0,1} reduce-scatter(dot.54), channel_id=326, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
  bitcast.1748 = bf16[1,1024,8192]{2,1,0} bitcast(reduce-scatter.4)
  dynamic-update-slice.31 = bf16[24,1024,8192]{2,1,0} dynamic-update-slice(bitcast.1698, bitcast.1748, select.372, constant.1980, constant.1980)
  bitcast.1758 = bf16[24,8192,1024]{1,2,0} bitcast(dynamic-update-slice.31)
  get-tuple-element.51 = bf16[24,1024]{0,1} get-tuple-element(param.3), index=6
  collective-permute.3 = bf16[24,1024]{0,1} collective-permute(get-tuple-element.51), channel_id=90, source_target_pairs={{0,0},{1,2},{2,4},{3,6},{4,1},{5,3},{6,5},{7,7}}
  all-gather.19 = bf16[24,2048]{0,1} all-gather(collective-permute.3), channel_id=91, replica_groups={{0,4},{2,6},{1,5},{3,7}}, dimensions={1}, use_global_device_ids=true
  bitcast.1763 = bf16[2048,24]{1,0} bitcast(all-gather.19)
  reduce.105 = bf16[2048]{0} reduce(reduce-scatter.1, constant.2006), dimensions={0,1}, to_apply=add
  all-reduce.40 = bf16[2048]{0} all-reduce(reduce.105), channel_id=89, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=add
  bitcast.1779 = bf16[2048,1]{1,0} bitcast(all-reduce.40)
  dynamic-update-slice.32 = bf16[2048,24]{1,0} dynamic-update-slice(bitcast.1763, bitcast.1779, constant.1980, select.372)
  dynamic-slice.339 = bf16[1024,24]{1,0} dynamic-slice(dynamic-update-slice.32, bitcast.2087, constant.1980), dynamic_slice_sizes={1024,24}
  bitcast.1794 = bf16[24,1024]{0,1} bitcast(dynamic-slice.339)
  collective-permute.10 = bf16[24,1024]{0,1} collective-permute(bitcast.1794), channel_id=110, source_target_pairs={{0,0},{2,1},{4,2},{6,3},{1,4},{3,5},{5,6},{7,7}}
  get-tuple-element.52 = bf16[24,1024]{0,1} get-tuple-element(param.3), index=7
  collective-permute.4 = bf16[24,1024]{0,1} collective-permute(get-tuple-element.52), channel_id=93, source_target_pairs={{0,0},{1,2},{2,4},{3,6},{4,1},{5,3},{6,5},{7,7}}
  all-gather.20 = bf16[24,2048]{0,1} all-gather(collective-permute.4), channel_id=94, replica_groups={{0,4},{2,6},{1,5},{3,7}}, dimensions={1}, use_global_device_ids=true
  bitcast.1801 = bf16[2048,24]{1,0} bitcast(all-gather.20)
  multiply.751 = bf16[2048,32,2048]{2,1,0} multiply(convert.312, bitcast.1766)
  bitcast.1817 = bf16[32,2048,2048]{1,0,2} bitcast(multiply.751)
  reduce.106 = bf16[2048]{0} reduce(bitcast.1817, constant.2006), dimensions={0,1}, to_apply=add
  all-reduce.41 = bf16[2048]{0} all-reduce(reduce.106), channel_id=92, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=add
  bitcast.1826 = bf16[2048,1]{1,0} bitcast(all-reduce.41)
  dynamic-update-slice.33 = bf16[2048,24]{1,0} dynamic-update-slice(bitcast.1801, bitcast.1826, constant.1980, select.372)
  dynamic-slice.340 = bf16[1024,24]{1,0} dynamic-slice(dynamic-update-slice.33, bitcast.2087, constant.1980), dynamic_slice_sizes={1024,24}
  bitcast.1841 = bf16[24,1024]{0,1} bitcast(dynamic-slice.340)
  collective-permute.11 = bf16[24,1024]{0,1} collective-permute(bitcast.1841), channel_id=111, source_target_pairs={{0,0},{2,1},{4,2},{6,3},{1,4},{3,5},{5,6},{7,7}}
  get-tuple-element.53 = bf16[24,1024]{0,1} get-tuple-element(param.3), index=8
  collective-permute.5 = bf16[24,1024]{0,1} collective-permute(get-tuple-element.53), channel_id=96, source_target_pairs={{0,0},{1,2},{2,4},{3,6},{4,1},{5,3},{6,5},{7,7}}
  all-gather.21 = bf16[24,2048]{0,1} all-gather(collective-permute.5), channel_id=97, replica_groups={{0,4},{2,6},{1,5},{3,7}}, dimensions={1}, use_global_device_ids=true
  bitcast.1848 = bf16[2048,24]{1,0} bitcast(all-gather.21)
  reduce.107 = bf16[2048]{0} reduce(reduce-scatter.2, constant.2006), dimensions={0,1}, to_apply=add
  all-reduce.42 = bf16[2048]{0} all-reduce(reduce.107), channel_id=95, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=add
  bitcast.1864 = bf16[2048,1]{1,0} bitcast(all-reduce.42)
  dynamic-update-slice.34 = bf16[2048,24]{1,0} dynamic-update-slice(bitcast.1848, bitcast.1864, constant.1980, select.372)
  dynamic-slice.341 = bf16[1024,24]{1,0} dynamic-slice(dynamic-update-slice.34, bitcast.2087, constant.1980), dynamic_slice_sizes={1024,24}
  bitcast.1879 = bf16[24,1024]{0,1} bitcast(dynamic-slice.341)
  collective-permute.12 = bf16[24,1024]{0,1} collective-permute(bitcast.1879), channel_id=112, source_target_pairs={{0,0},{2,1},{4,2},{6,3},{1,4},{3,5},{5,6},{7,7}}
  get-tuple-element.54 = bf16[24,1024]{0,1} get-tuple-element(param.3), index=9
  collective-permute.6 = bf16[24,1024]{0,1} collective-permute(get-tuple-element.54), channel_id=99, source_target_pairs={{0,0},{1,2},{2,4},{3,6},{4,1},{5,3},{6,5},{7,7}}
  all-gather.22 = bf16[24,2048]{0,1} all-gather(collective-permute.6), channel_id=100, replica_groups={{0,4},{2,6},{1,5},{3,7}}, dimensions={1}, use_global_device_ids=true
  bitcast.1886 = bf16[2048,24]{1,0} bitcast(all-gather.22)
  multiply.753 = bf16[2048,32,2048]{2,1,0} multiply(convert.314, bitcast.1851)
  bitcast.1905 = bf16[32,2048,2048]{1,0,2} bitcast(multiply.753)
  reduce.108 = bf16[2048]{0} reduce(bitcast.1905, constant.2006), dimensions={0,1}, to_apply=add
  all-reduce.43 = bf16[2048]{0} all-reduce(reduce.108), channel_id=98, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=add
  bitcast.1914 = bf16[2048,1]{1,0} bitcast(all-reduce.43)
  dynamic-update-slice.35 = bf16[2048,24]{1,0} dynamic-update-slice(bitcast.1886, bitcast.1914, constant.1980, select.372)
  dynamic-slice.342 = bf16[1024,24]{1,0} dynamic-slice(dynamic-update-slice.35, bitcast.2087, constant.1980), dynamic_slice_sizes={1024,24}
  bitcast.1929 = bf16[24,1024]{0,1} bitcast(dynamic-slice.342)
  collective-permute.13 = bf16[24,1024]{0,1} collective-permute(bitcast.1929), channel_id=113, source_target_pairs={{0,0},{2,1},{4,2},{6,3},{1,4},{3,5},{5,6},{7,7}}
  get-tuple-element.55 = bf16[24,3,16,128]{3,2,1,0} get-tuple-element(param.3), index=10
  bitcast.1979 = bf16[3,32,2048,16,128]{4,2,3,1,0} bitcast(add.639)
  reduce.109 = bf16[3,16,128]{2,1,0} reduce(bitcast.1979, constant.2006), dimensions={1,2}, to_apply=add
  all-reduce.44 = bf16[3,16,128]{2,1,0} all-reduce(reduce.109), channel_id=101, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=add
  bitcast.1963 = bf16[1,3,16,128]{3,2,1,0} bitcast(all-reduce.44)
  dynamic-update-slice.36 = bf16[24,3,16,128]{3,2,1,0} dynamic-update-slice(get-tuple-element.55, bitcast.1963, select.372, constant.1980, constant.1980, /*index=5*/constant.1980)
  get-tuple-element.56 = bf16[24,3,1024,16,128]{4,3,1,2,0} get-tuple-element(param.3), index=11
  bitcast.1974 = bf16[24,1024,3,16,128]{4,3,2,1,0} bitcast(get-tuple-element.56)
  transpose.130 = bf16[3,16,128,32,2048]{4,3,2,1,0} transpose(add.639), dimensions={0,2,4,1,3}
  bitcast.1983 = bf16[6144,65536]{1,0} bitcast(transpose.130)
  all-gather.23 = bf16[32,2048,4096]{1,0,2} all-gather(bitcast.2003), channel_id=102, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, use_global_device_ids=true
  bitcast.2007 = bf16[65536,4096]{0,1} bitcast(all-gather.23)
  dot.55 = bf16[6144,4096]{0,1} dot(bitcast.1983, bitcast.2007), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast.2011 = bf16[3,16,128,4096]{2,1,0,3} bitcast(dot.55)
  reduce-scatter.5 = bf16[3,16,128,1024]{2,1,0,3} reduce-scatter(bitcast.2011), channel_id=327, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, dimensions={3}, to_apply=add
  bitcast.2015 = bf16[1,1024,3,16,128]{4,3,2,1,0} bitcast(reduce-scatter.5)
  dynamic-update-slice.37 = bf16[24,1024,3,16,128]{4,3,2,1,0} dynamic-update-slice(bitcast.1974, bitcast.2015, select.372, constant.1980, constant.1980, /*index=5*/constant.1980, constant.1980)
  bitcast.2025 = bf16[24,3,1024,16,128]{4,3,1,2,0} bitcast(dynamic-update-slice.37)
  get-tuple-element.57 = bf16[24,1024]{1,0} get-tuple-element(param.3), index=12
  reduce.110 = bf16[2048]{0} reduce(bitcast.2063, constant.2006), dimensions={0,1}, to_apply=add
  all-reduce.46 = bf16[2048]{0} all-reduce(reduce.110), channel_id=104, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=add
  dynamic-slice.343 = bf16[1024]{0} dynamic-slice(all-reduce.46, bitcast.2087), dynamic_slice_sizes={1024}
  bitcast.2046 = bf16[1,1024]{1,0} bitcast(dynamic-slice.343)
  collective-permute.7 = bf16[1,1024]{1,0} collective-permute(bitcast.2046), channel_id=105, source_target_pairs={{0,0},{2,1},{4,2},{6,3},{1,4},{3,5},{5,6},{7,7}}
  dynamic-update-slice.38 = bf16[24,1024]{1,0} dynamic-update-slice(get-tuple-element.57, collective-permute.7, select.372, constant.1980)
  get-tuple-element.58 = bf16[24,1024,16,128]{3,2,1,0} get-tuple-element(param.3), index=13
  bitcast.2066 = bf16[2048,65536]{1,0} bitcast(add.619)
  transpose.133 = bf16[16,32,2048,128]{3,2,1,0} transpose(dot.44), dimensions={1,0,3,2}
  bitcast.2072 = bf16[32,2048,16,128]{3,1,0,2} bitcast(transpose.133)
  all-gather.24 = bf16[32,2048,32,128]{3,1,0,2} all-gather(bitcast.2072), channel_id=106, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, use_global_device_ids=true
  bitcast.2073 = bf16[32,32,2048,128]{3,2,1,0} bitcast(all-gather.24)
  transpose.134 = bf16[32,2048,32,128]{3,2,1,0} transpose(bitcast.2073), dimensions={1,2,0,3}
  bitcast.2077 = bf16[65536,4096]{1,0} bitcast(transpose.134)
  dot.56 = bf16[2048,4096]{1,0} dot(bitcast.2066, bitcast.2077), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast.2081 = bf16[2048,32,128]{2,1,0} bitcast(dot.56)
  all-reduce.47 = bf16[2048,32,128]{2,1,0} all-reduce(bitcast.2081), channel_id=107, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=add
  constant.2015 = s32[8]{0} constant({0, 0, 16, 16, 0, 0, 16, 16})
  dynamic-slice.344 = s32[1]{0} dynamic-slice(constant.2015, partition-id.3), dynamic_slice_sizes={1}
  bitcast.2095 = s32[] bitcast(dynamic-slice.344)
  dynamic-slice.345 = bf16[1024,16,128]{2,1,0} dynamic-slice(all-reduce.47, bitcast.2087, bitcast.2095, constant.1980), dynamic_slice_sizes={1024,16,128}
  bitcast.2102 = bf16[1,1024,16,128]{3,2,1,0} bitcast(dynamic-slice.345)
  collective-permute.8 = bf16[1,1024,16,128]{3,2,1,0} collective-permute(bitcast.2102), channel_id=108, source_target_pairs={{0,0},{2,1},{4,2},{6,3},{1,4},{3,5},{5,6},{7,7}}
  dynamic-update-slice.39 = bf16[24,1024,16,128]{3,2,1,0} dynamic-update-slice(get-tuple-element.58, collective-permute.8, select.372, constant.1980, constant.1980, /*index=5*/constant.1980)
  ROOT tuple.2 = (s32[], bf16[32,2048,2048]{1,0,2}, bf16[24,8192]{1,0}, bf16[24,1024,8192]{2,1,0}, bf16[24,1024]{0,1}, /*index=5*/bf16[24,8192,1024]{1,2,0}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, bf16[24,1024]{0,1}, /*index=10*/bf16[24,3,16,128]{3,2,1,0}, bf16[24,3,1024,16,128]{4,3,1,2,0}, bf16[24,1024]{1,0}, bf16[24,1024,16,128]{3,2,1,0}, bf16[24,8192]{1,0}, /*index=15*/bf16[24,1024,8192]{2,1,0}, bf16[24,8192,1024]{1,2,0}, bf16[24,2048]{1,0}, bf16[24,2048]{1,0}, bf16[24,2048]{1,0}, /*index=20*/bf16[24,2048]{1,0}, bf16[24,3,16,128]{3,2,1,0}, bf16[24,3,1024,16,128]{4,3,1,2,0}, bf16[24,1024]{1,0}, bf16[24,1024,16,128]{3,2,1,0}, /*index=25*/bf16[24,32,2048,2048]{2,1,3,0}, bf16[32,1,2048,2048]{3,2,0,1}, bf16[32,2048]{1,0}) tuple(add.581, bitcast.1525, dynamic-update-slice.28, dynamic-update-slice.29, collective-permute.9, /*index=5*/bitcast.1758, collective-permute.10, collective-permute.11, collective-permute.12, collective-permute.13, /*index=10*/dynamic-update-slice.36, bitcast.2025, dynamic-update-slice.38, dynamic-update-slice.39, get-tuple-element.45, /*index=15*/get-tuple-element.44, get-tuple-element.46, get-tuple-element.43, get-tuple-element.42, get-tuple-element.37, /*index=20*/get-tuple-element.36, get-tuple-element.38, get-tuple-element.35, get-tuple-element.41, get-tuple-element.40, /*index=25*/get-tuple-element.32, get-tuple-element.39, get-tuple-element.33)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  HloInstruction* fwd_instruction = nullptr;
  HloInstruction* bwd_instruction = nullptr;
  SCOPED_TRACE(m->ToString());
  for (HloInstruction* instr :
       m->entry_computation()->MakeInstructionPostOrder()) {
    if (HloPredicateIsOp<HloOpcode::kCustomCall>(instr) &&
        instr->custom_call_target() == kCudnnfMHASoftmaxCallTarget) {
      fwd_instruction = instr;
    }
    if (HloPredicateIsOp<HloOpcode::kCustomCall>(instr) &&
        instr->custom_call_target() == kCudnnfMHASoftmaxBackwardCallTarget) {
      bwd_instruction = instr;
    }
  }
  EXPECT_NE(fwd_instruction, nullptr);
  EXPECT_NE(bwd_instruction, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fwd_instruction->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_EQ(config.mask_type(), CudnnfMHABackendConfig::CAUSAL);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16TrainingBmm2CanonicalizationRestoreFwdGraph) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  const char* module_str = R"(
HloModule pjit__unnamed_function_, entry_computation_layout={(bf16[2,256,4,64]{3,2,1,0}, bf16[2,256,4,64]{3,2,1,0}, bf16[2,256,4,64]{3,2,1,0}, bf16[2,256,4,64]{3,2,1,0}, bf16[2,4,256,256]{3,2,1,0})->(bf16[4,256,8,64]{3,2,1,0}, bf16[2,256,4,64]{3,2,1,0}, bf16[2,256,4,64]{3,2,1,0}, bf16[2,256,4,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={false,false,false,false}, num_partitions=4

region_0.6 {
  Arg_0.7 = bf16[] parameter(0)
  Arg_1.8 = bf16[] parameter(1)
  ROOT maximum.5 = bf16[] maximum(Arg_0.7, Arg_1.8)
}

region_1.10 {
  Arg_0.11 = f32[] parameter(0)
  Arg_1.12 = f32[] parameter(1)
  ROOT add.14 = f32[] add(Arg_0.11, Arg_1.12)
}

add.clone {
  x.1 = u32[] parameter(0)
  y.1 = u32[] parameter(1)
  ROOT add.15 = u32[] add(x.1, y.1)
}

region_2.65 {
  Arg_0.66 = bf16[] parameter(0)
  Arg_1.67 = bf16[] parameter(1)
  ROOT add.16 = bf16[] add(Arg_0.66, Arg_1.67)
}

ENTRY main.164_spmd {
  param = bf16[2,256,4,64]{3,2,1,0} parameter(2), sharding={devices=[2,1,2,1]<=[4]}
  transpose.26 = bf16[2,4,64,256]{3,2,1,0} transpose(param), dimensions={0,2,3,1}
  param.1 = bf16[2,256,4,64]{3,2,1,0} parameter(0), sharding={devices=[2,1,2,1]<=[4]}
  transpose.27 = bf16[2,4,256,64]{3,2,1,0} transpose(param.1), dimensions={0,2,1,3}
  constant.46 = bf16[] constant(0.5)
  broadcast.126 = bf16[2,4,256,64]{3,2,1,0} broadcast(constant.46), dimensions={}
  multiply.34 = bf16[2,4,256,64]{3,2,1,0} multiply(transpose.27, broadcast.126)
  param.2 = bf16[2,256,4,64]{3,2,1,0} parameter(1), sharding={devices=[2,1,2,1]<=[4]}
  transpose.29 = bf16[2,4,64,256]{3,2,1,0} transpose(param.2), dimensions={0,2,3,1}
  dot.12 = bf16[2,4,256,256]{3,2,1,0} dot(multiply.34, transpose.29), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  param.3 = bf16[2,4,256,256]{3,2,1,0} parameter(4), sharding={devices=[2,2,1,1]<=[4]}
  add.17 = bf16[2,4,256,256]{3,2,1,0} add(dot.12, param.3)
  constant.47 = bf16[] constant(-inf)
  reduce.4 = bf16[2,4,256]{2,1,0} reduce(add.17, constant.47), dimensions={3}, to_apply=region_0.6
  broadcast.127 = bf16[2,4,256,256]{3,2,1,0} broadcast(reduce.4), dimensions={0,1,2}
  subtract.14 = bf16[2,4,256,256]{3,2,1,0} subtract(add.17, broadcast.127)
  exponential.2 = bf16[2,4,256,256]{3,2,1,0} exponential(subtract.14)
  convert.46 = f32[2,4,256,256]{3,2,1,0} convert(exponential.2)
  constant.48 = f32[] constant(0)
  reduce.5 = f32[2,4,256]{2,1,0} reduce(convert.46, constant.48), dimensions={3}, to_apply=region_1.10
  convert.47 = bf16[2,4,256]{2,1,0} convert(reduce.5)
  broadcast.128 = bf16[2,4,256,256]{3,2,1,0} broadcast(convert.47), dimensions={0,1,2}
  divide.7 = bf16[2,4,256,256]{3,2,1,0} divide(exponential.2, broadcast.128)
  broadcast.129 = f32[4096]{0} broadcast(constant.48), dimensions={}
  constant.50 = u32[] constant(0)
  broadcast.131 = u32[8192]{0} broadcast(constant.50), dimensions={}
  broadcast.133 = u32[4096]{0} broadcast(constant.50), dimensions={}
  iota.3 = u32[8192]{0} iota(), iota_dimension=0
  slice.14 = u32[4096]{0} slice(iota.3), slice={[0:4096]}
  slice.15 = u32[4096]{0} slice(iota.3), slice={[4096:8192]}
  custom-call.3 = (u32[4096]{0}, u32[4096]{0}) custom-call(broadcast.133, broadcast.133, slice.14, slice.15), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[4096]{0}, u32[4096]{0}, u32[4096]{0}, u32[4096]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\020\000\000\000\000\000\000"
  get-tuple-element.6 = u32[4096]{0} get-tuple-element(custom-call.3), index=0
  constant.115 = u32[1]{0} constant({0})
  constant.52 = u32[4]{0} constant({0, 0, 1, 1})
  partition-id = u32[] partition-id()
  dynamic-slice.21 = u32[1]{0} dynamic-slice(constant.52, partition-id), dynamic_slice_sizes={1}
  constant.116 = u32[1]{0} constant({1})
  clamp.3 = u32[1]{0} clamp(constant.115, dynamic-slice.21, constant.116)
  convert.48 = s32[1]{0} convert(clamp.3)
  constant.117 = s32[1]{0} constant({2048})
  multiply.35 = s32[1]{0} multiply(convert.48, constant.117)
  bitcast.105 = s32[] bitcast(multiply.35)
  dynamic-slice.22 = u32[2048]{0} dynamic-slice(get-tuple-element.6, bitcast.105), dynamic_slice_sizes={2048}
  constant.58 = s32[4]{0} constant({0, 0, 1, 1})
  dynamic-slice.23 = s32[1]{0} dynamic-slice(constant.58, partition-id), dynamic_slice_sizes={1}
  multiply.36 = s32[1]{0} multiply(dynamic-slice.23, constant.117)
  bitcast.108 = s32[] bitcast(multiply.36)
  dynamic-update-slice.2 = u32[8192]{0} dynamic-update-slice(broadcast.131, dynamic-slice.22, bitcast.108)
  get-tuple-element.7 = u32[4096]{0} get-tuple-element(custom-call.3), index=1
  dynamic-slice.24 = u32[2048]{0} dynamic-slice(get-tuple-element.7, bitcast.105), dynamic_slice_sizes={2048}
  constant.65 = s32[] constant(4096)
  add.18 = s32[] add(bitcast.108, constant.65)
  dynamic-update-slice.3 = u32[8192]{0} dynamic-update-slice(dynamic-update-slice.2, dynamic-slice.24, add.18)
  all-reduce = u32[8192]{0} all-reduce(dynamic-update-slice.3), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=add.clone
  constant.118 = s32[1]{0} constant({4096})
  multiply.37 = s32[1]{0} multiply(dynamic-slice.23, constant.118)
  bitcast.119 = s32[] bitcast(multiply.37)
  dynamic-slice.25 = u32[4096]{0} dynamic-slice(all-reduce, bitcast.119), dynamic_slice_sizes={4096}
  constant.69 = u32[] constant(9)
  broadcast.134 = u32[4096]{0} broadcast(constant.69), dimensions={}
  shift-right-logical.6 = u32[4096]{0} shift-right-logical(dynamic-slice.25, broadcast.134)
  constant.70 = u32[] constant(1065353216)
  broadcast.135 = u32[4096]{0} broadcast(constant.70), dimensions={}
  or.5 = u32[4096]{0} or(shift-right-logical.6, broadcast.135)
  bitcast-convert.5 = f32[4096]{0} bitcast-convert(or.5)
  constant.71 = f32[] constant(-1)
  broadcast.136 = f32[4096]{0} broadcast(constant.71), dimensions={}
  add.19 = f32[4096]{0} add(bitcast-convert.5, broadcast.136)
  maximum.6 = f32[4096]{0} maximum(broadcast.129, add.19)
  constant.72 = f32[] constant(0.5)
  broadcast.137 = f32[4096]{0} broadcast(constant.72), dimensions={}
  compare.4 = pred[4096]{0} compare(maximum.6, broadcast.137), direction=LT
  bitcast.135 = pred[2,8,256]{2,1,0} bitcast(compare.4)
  convert.49 = bf16[2,8,256]{2,1,0} convert(bitcast.135)
  constant.80 = s32[] constant(0)
  constant.78 = s32[4]{0} constant({0, 4, 0, 4})
  dynamic-slice.26 = s32[1]{0} dynamic-slice(constant.78, partition-id), dynamic_slice_sizes={1}
  bitcast.181 = s32[] bitcast(dynamic-slice.26)
  dynamic-slice.27 = bf16[2,4,256]{2,1,0} dynamic-slice(convert.49, constant.80, bitcast.181, constant.80), dynamic_slice_sizes={2,4,256}
  broadcast.139 = bf16[2,4,256,256]{3,2,1,0} broadcast(dynamic-slice.27), dimensions={0,1,3}
  multiply.38 = bf16[2,4,256,256]{3,2,1,0} multiply(divide.7, broadcast.139)
  constant.93 = bf16[] constant(2)
  broadcast.141 = bf16[2,4,256,256]{3,2,1,0} broadcast(constant.93), dimensions={}
  multiply.39 = bf16[2,4,256,256]{3,2,1,0} multiply(multiply.38, broadcast.141)
  dot.13 = bf16[2,4,64,256]{3,2,1,0} dot(transpose.26, multiply.39), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.31 = bf16[4,2,64,256]{3,2,1,0} transpose(dot.13), dimensions={1,0,2,3}
  bitcast.154 = bf16[2,256,4,64]{1,3,0,2} bitcast(transpose.31)
  all-gather = bf16[2,256,8,64]{1,3,0,2} all-gather(bitcast.154), channel_id=2, replica_groups={{0,1},{2,3}}, dimensions={2}, use_global_device_ids=true
  bitcast.155 = bf16[8,2,64,256]{3,2,1,0} bitcast(all-gather)
  transpose.32 = bf16[2,8,64,256]{3,2,1,0} transpose(bitcast.155), dimensions={1,0,2,3}
  bitcast.157 = bf16[2,256,8,64]{1,3,2,0} bitcast(transpose.32)
  all-gather.1 = bf16[4,256,8,64]{1,3,2,0} all-gather(bitcast.157), channel_id=3, replica_groups={{0,2},{1,3}}, dimensions={0}, use_global_device_ids=true
  bitcast.236 = bf16[4,8,64,256]{3,2,1,0} bitcast(all-gather.1)
  transpose.38 = bf16[4,256,8,64]{3,2,1,0} transpose(bitcast.236), dimensions={0,3,1,2}
  param.4 = bf16[2,256,4,64]{3,2,1,0} parameter(3), sharding={devices=[2,1,2,1]<=[4]}
  transpose.33 = bf16[2,4,256,64]{3,2,1,0} transpose(param.4), dimensions={0,2,1,3}
  dot.14 = bf16[2,4,256,256]{3,2,1,0} dot(transpose.33, transpose.26), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  broadcast.142 = bf16[4096]{0} broadcast(constant.93), dimensions={}
  constant.95 = bf16[] constant(0)
  broadcast.143 = bf16[4096]{0} broadcast(constant.95), dimensions={}
  select.4 = bf16[4096]{0} select(compare.4, broadcast.142, broadcast.143)
  bitcast.176 = bf16[2,8,256]{2,1,0} bitcast(select.4)
  dynamic-slice.28 = bf16[2,4,256]{2,1,0} dynamic-slice(bitcast.176, constant.80, bitcast.181, constant.80), dynamic_slice_sizes={2,4,256}
  broadcast.145 = bf16[2,4,256,256]{3,2,1,0} broadcast(dynamic-slice.28), dimensions={0,1,3}
  multiply.40 = bf16[2,4,256,256]{3,2,1,0} multiply(dot.14, broadcast.145)
  divide.8 = bf16[2,4,256,256]{3,2,1,0} divide(multiply.40, broadcast.128)
  constant.106 = bf16[] constant(1)
  broadcast.146 = bf16[2,4,256]{2,1,0} broadcast(constant.106), dimensions={}
  multiply.41 = bf16[2,4,256]{2,1,0} multiply(convert.47, convert.47)
  divide.9 = bf16[2,4,256]{2,1,0} divide(broadcast.146, multiply.41)
  broadcast.147 = bf16[2,4,256,256]{3,2,1,0} broadcast(divide.9), dimensions={0,1,2}
  multiply.42 = bf16[2,4,256,256]{3,2,1,0} multiply(multiply.40, broadcast.147)
  multiply.43 = bf16[2,4,256,256]{3,2,1,0} multiply(multiply.42, exponential.2)
  reduce.6 = bf16[2,4,256]{2,1,0} reduce(multiply.43, constant.95), dimensions={3}, to_apply=region_2.65
  negate.4 = bf16[2,4,256]{2,1,0} negate(reduce.6)
  broadcast.148 = bf16[2,4,256,256]{3,2,1,0} broadcast(negate.4), dimensions={0,1,2}
  add.20 = bf16[2,4,256,256]{3,2,1,0} add(divide.8, broadcast.148)
  multiply.44 = bf16[2,4,256,256]{3,2,1,0} multiply(add.20, exponential.2)
  transpose.34 = bf16[2,4,256,64]{3,2,1,0} transpose(param.2), dimensions={0,2,1,3}
  dot.15 = bf16[2,4,256,64]{3,2,1,0} dot(multiply.44, transpose.34), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  multiply.45 = bf16[2,4,256,64]{3,2,1,0} multiply(dot.15, broadcast.126)
  transpose.39 = bf16[2,256,4,64]{3,2,1,0} transpose(multiply.45), dimensions={0,2,1,3}
  dot.16 = bf16[2,4,256,64]{3,2,1,0} dot(multiply.44, multiply.34), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.40 = bf16[2,256,4,64]{3,2,1,0} transpose(dot.16), dimensions={0,2,1,3}
  transpose.36 = bf16[2,4,64,256]{3,2,1,0} transpose(param.4), dimensions={0,2,3,1}
  dot.11 = bf16[2,4,64,256]{3,2,1,0} dot(transpose.36, multiply.39), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.41 = bf16[2,256,4,64]{3,2,1,0} transpose(dot.11), dimensions={0,3,1,2}
  ROOT tuple.2 = (bf16[4,256,8,64]{3,2,1,0}, bf16[2,256,4,64]{3,2,1,0}, bf16[2,256,4,64]{3,2,1,0}, bf16[2,256,4,64]{3,2,1,0}) tuple(transpose.38, transpose.39, transpose.40, transpose.41)
} // main.164_spmd
)";
  // Dropout bwd pattern not supported, should not lower fwd as well
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  SCOPED_TRACE(m->ToString());
  // check if fwd graph has been restored with cloned activation
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Transpose(), m::Transpose(), m::Transpose(),
          m::Transpose(m::Dot(
              m::Op(), m::Op().WithPredicate([](const HloInstruction* instr) {
                return instr->name() == "multiply.39.fmha_no_match_clone";
              }))))));
}

constexpr absl::string_view hlo_BF16Bmm1BiasSoftmaxBmm2Pattern_dbias = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,1024,4,64]{3,2,1,0}, bf16[2,1024,4,64]{3,2,1,0}, bf16[2,1024,4,64]{3,2,1,0}, bf16[2,1024,4,64]{3,2,1,0}, bf16[4,1024,1024]{2,1,0})->(bf16[2,1024,4,64]{3,2,1,0}, bf16[2,1024,4,64]{3,2,1,0}, bf16[2,1024,4,64]{3,2,1,0}, bf16[2,1024,4,64]{3,2,1,0}, bf16[4,1024,1024]{2,1,0})}, allow_spmd_sharding_propagation_to_parameters={true,true,true,true,true}, allow_spmd_sharding_propagation_to_output={true,true,true,true,true}

region_0.14 {
  Arg_0.15 = bf16[] parameter(0)
  Arg_1.16 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(Arg_0.15, Arg_1.16)
}

region_1.27 {
  Arg_0.28 = f32[] parameter(0)
  Arg_1.29 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.28, Arg_1.29)
}

region_2.56 {
  Arg_0.57 = bf16[] parameter(0)
  Arg_1.58 = bf16[] parameter(1)
  ROOT add.1 = bf16[] add(Arg_0.57, Arg_1.58)
}

ENTRY main.87 {
  Arg_2.3 = bf16[2,1024,4,64]{3,2,1,0} parameter(2)
  transpose.12 = bf16[2,4,64,1024]{3,2,1,0} transpose(Arg_2.3), dimensions={0,2,3,1}
  Arg_0.1 = bf16[2,1024,4,64]{3,2,1,0} parameter(0)
  transpose.13 = bf16[2,4,1024,64]{3,2,1,0} transpose(Arg_0.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[2,1024,4,64]{3,2,1,0} parameter(1)
  transpose.15 = bf16[2,4,64,1024]{3,2,1,0} transpose(Arg_1.2), dimensions={0,2,3,1}
  dot = bf16[2,4,1024,1024]{3,2,1,0} dot(transpose.13, transpose.15), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_4.5 = bf16[4,1024,1024]{2,1,0} parameter(4)
  broadcast.9 = bf16[2,4,1024,1024]{3,2,1,0} broadcast(Arg_4.5), dimensions={1,2,3}
  add.2 = bf16[2,4,1024,1024]{3,2,1,0} add(dot, broadcast.9)
  constant.10 = bf16[] constant(-inf)
  reduce.18 = bf16[2,4,1024]{2,1,0} reduce(add.2, constant.10), dimensions={3}, to_apply=region_0.14
  broadcast.10 = bf16[2,4,1024,1024]{3,2,1,0} broadcast(reduce.18), dimensions={0,1,2}
  subtract = bf16[2,4,1024,1024]{3,2,1,0} subtract(add.2, broadcast.10)
  exponential = bf16[2,4,1024,1024]{3,2,1,0} exponential(subtract)
  convert.5 = f32[2,4,1024,1024]{3,2,1,0} convert(exponential)
  constant.9 = f32[] constant(0)
  reduce.31 = f32[2,4,1024]{2,1,0} reduce(convert.5, constant.9), dimensions={3}, to_apply=region_1.27
  convert.6 = bf16[2,4,1024]{2,1,0} convert(reduce.31)
  broadcast.11 = bf16[2,4,1024,1024]{3,2,1,0} broadcast(convert.6), dimensions={0,1,2}
  divide.2 = bf16[2,4,1024,1024]{3,2,1,0} divide(exponential, broadcast.11)
  dot.1 = bf16[2,4,64,1024]{3,2,1,0} dot(transpose.12, divide.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.22 = bf16[2,1024,4,64]{3,2,1,0} transpose(dot.1), dimensions={0,3,1,2}
  Arg_3.4 = bf16[2,1024,4,64]{3,2,1,0} parameter(3)
  transpose.17 = bf16[2,4,1024,64]{3,2,1,0} transpose(Arg_3.4), dimensions={0,2,1,3}
  dot.2 = bf16[2,4,1024,1024]{3,2,1,0} dot(transpose.17, transpose.12), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  divide.3 = bf16[2,4,1024,1024]{3,2,1,0} divide(dot.2, broadcast.11)
  constant.0 = bf16[] constant(1)
  broadcast.13 = bf16[2,4,1024]{2,1,0} broadcast(constant.0), dimensions={}
  multiply.2 = bf16[2,4,1024]{2,1,0} multiply(convert.6, convert.6)
  divide.4 = bf16[2,4,1024]{2,1,0} divide(broadcast.13, multiply.2)
  broadcast.14 = bf16[2,4,1024,1024]{3,2,1,0} broadcast(divide.4), dimensions={0,1,2}
  multiply.3 = bf16[2,4,1024,1024]{3,2,1,0} multiply(dot.2, broadcast.14)
  multiply.4 = bf16[2,4,1024,1024]{3,2,1,0} multiply(multiply.3, exponential)
  constant.8 = bf16[] constant(0)
  reduce.60 = bf16[2,4,1024]{2,1,0} reduce(multiply.4, constant.8), dimensions={3}, to_apply=region_2.56
  negate.1 = bf16[2,4,1024]{2,1,0} negate(reduce.60)
  broadcast.15 = bf16[2,4,1024,1024]{3,2,1,0} broadcast(negate.1), dimensions={0,1,2}
  add.3 = bf16[2,4,1024,1024]{3,2,1,0} add(divide.3, broadcast.15)
  multiply.5 = bf16[2,4,1024,1024]{3,2,1,0} multiply(add.3, exponential)
  transpose.18 = bf16[2,4,1024,64]{3,2,1,0} transpose(Arg_1.2), dimensions={0,2,1,3}
  dot.4 = bf16[2,4,1024,64]{3,2,1,0} dot(multiply.5, transpose.18), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.23 = bf16[2,1024,4,64]{3,2,1,0} transpose(dot.4), dimensions={0,2,1,3}
  dot.3 = bf16[2,4,1024,64]{3,2,1,0} dot(multiply.5, transpose.13), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.24 = bf16[2,1024,4,64]{3,2,1,0} transpose(dot.3), dimensions={0,2,1,3}
  transpose.20 = bf16[2,4,64,1024]{3,2,1,0} transpose(Arg_3.4), dimensions={0,2,3,1}
  dot.49 = bf16[2,4,64,1024]{3,2,1,0} dot(transpose.20, divide.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.25 = bf16[2,1024,4,64]{3,2,1,0} transpose(dot.49), dimensions={0,3,1,2}
  reduce.81 = bf16[4,1024,1024]{2,1,0} reduce(multiply.5, constant.8), dimensions={0}, to_apply=region_2.56
  ROOT tuple = (bf16[2,1024,4,64]{3,2,1,0}, bf16[2,1024,4,64]{3,2,1,0}, bf16[2,1024,4,64]{3,2,1,0}, bf16[2,1024,4,64]{3,2,1,0}, bf16[4,1024,1024]{2,1,0}) tuple(transpose.22, transpose.23, transpose.24, transpose.25, reduce.81)
} // main.87
)";

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1BiasSoftmaxBmm2PatternDbias) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  TF_ASSERT_OK_AND_ASSIGN(
      auto m,
      ParseAndReturnVerifiedModule(hlo_BF16Bmm1BiasSoftmaxBmm2Pattern_dbias));
  // require cudnn 8.9.6 + hopper for dbias
  CudnnFusedMHARewriter fusedMhaRewriter{se::CudaComputeCapability(9, 0),
                                         se::dnn::VersionInfo(9, 0, 0)};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Transpose(
              m::Transpose(m::GetTupleElement(
                  m::CustomCall(&fmha, {kCudnnfMHAScaleBiasSoftmaxCallTarget}),
                  0)))
              .WithShape(BF16, {2, 1024, 4, 64}),
          m::Transpose(
              m::GetTupleElement(
                  m::CustomCall({kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget}),
                  0))
              .WithShape(BF16, {2, 1024, 4, 64}),
          m::Transpose(
              m::GetTupleElement(
                  m::CustomCall({kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget}),
                  1))
              .WithShape(BF16, {2, 1024, 4, 64}),
          m::Transpose(
              m::Transpose(m::GetTupleElement(
                  m::CustomCall({kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget}),
                  2)))
              .WithShape(BF16, {2, 1024, 4, 64}),
          m::Reshape(
              m::GetTupleElement(
                  m::CustomCall({kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget}),
                  3))
              .WithShape(BF16, {4, 1024, 1024}))));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();
  EXPECT_EQ(fmha->operands().size(), 4);
  EXPECT_EQ(fmha->operand(3)->shape(),
            ShapeUtil::MakeShape(BF16, {1, 4, 1024, 1024}));
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(config.mask_type(), CudnnfMHABackendConfig::NO_MASK);
}
}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
