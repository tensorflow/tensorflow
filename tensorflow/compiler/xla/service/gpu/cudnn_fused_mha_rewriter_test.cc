/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

namespace m = xla::match;

class CudnnFusedMhaRewriterTestHloTest : public HloTestBase {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
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
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Fused MHA is supported with the Nvidia Ampere+ GPUs.";
  }
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[20,40,64]{2,1,0},bf16[20,64,40]{2,1,0},bf16[20,40,64]{2,1,0})->bf16[20,40,64]{2,1,0}}

ENTRY main.6 {
  Arg_0.1 = bf16[20,40,64]{2,1,0} parameter(0)
  Arg_1.2 = bf16[20,64,40]{1,2,0} parameter(1)
  custom-call = bf16[20,40,40]{2,1,0} custom-call(Arg_0.1, Arg_1.2), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\"],\"rhs_batch_dimensions\":[\"0\"]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
  Arg_2.3 = bf16[20,40,64]{2,1,0} parameter(2)
  ROOT custom-call.1 = bf16[20,40,64]{2,1,0} custom-call(custom-call, Arg_2.3), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\"],\"rhs_batch_dimensions\":[\"0\"]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
}


)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {20, 40, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm1_rhs_contracting_dim_not_most_minor) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Fused MHA is supported with the Nvidia Ampere+ GPUs.";
  }
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[20,40,64]{2,1,0},bf16[20,64,40]{2,1,0},bf16[20,40,64]{2,1,0})->bf16[20,40,64]{2,1,0}}

ENTRY main.6 {
  Arg_0.1 = bf16[20,40,64]{2,1,0} parameter(0)
  Arg_1.2 = bf16[20,64,40]{2,1,0} parameter(1)
  custom-call = bf16[20,40,40]{2,1,0} custom-call(Arg_0.1, Arg_1.2), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\"],\"rhs_batch_dimensions\":[\"0\"]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
  Arg_2.3 = bf16[20,40,64]{2,1,0} parameter(2)
  ROOT custom-call.1 = bf16[20,40,64]{2,1,0} custom-call(custom-call, Arg_2.3), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\"],\"rhs_batch_dimensions\":[\"0\"]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
}


)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_FALSE(result);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm1_lhs_contracting_dim_not_most_minor) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Fused MHA is supported with the Nvidia Ampere+ GPUs.";
  }
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[20,40,64]{2,1,0},bf16[20,64,40]{2,1,0},bf16[20,40,64]{2,1,0})->bf16[20,40,64]{2,1,0}}

ENTRY main.6 {
  Arg_0.1 = bf16[20,40,64]{1,2,0} parameter(0)
  Arg_1.2 = bf16[20,64,40]{2,1,0} parameter(1)
  custom-call = bf16[20,40,40]{2,1,0} custom-call(Arg_0.1, Arg_1.2), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\"],\"rhs_batch_dimensions\":[\"0\"]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
  Arg_2.3 = bf16[20,40,64]{2,1,0} parameter(2)
  ROOT custom-call.1 = bf16[20,40,64]{2,1,0} custom-call(custom-call, Arg_2.3), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\"],\"rhs_batch_dimensions\":[\"0\"]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
}


)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_FALSE(result);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm2_non_contracting_dim_not_most_minor) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Fused MHA is supported with the Nvidia Ampere+ GPUs.";
  }
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[20,40,64]{2,1,0},bf16[20,64,40]{2,1,0},bf16[20,40,64]{2,1,0})->bf16[20,40,64]{2,1,0}}

ENTRY main.6 {
  Arg_0.1 = bf16[20,40,64]{2,1,0} parameter(0)
  Arg_1.2 = bf16[20,64,40]{1,2,0} parameter(1)
  custom-call = bf16[20,40,40]{2,1,0} custom-call(Arg_0.1, Arg_1.2), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\"],\"rhs_batch_dimensions\":[\"0\"]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
  Arg_2.3 = bf16[20,40,64]{1,2,0} parameter(2)
  ROOT custom-call.1 = bf16[20,40,64]{2,1,0} custom-call(custom-call, Arg_2.3), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\"],\"rhs_batch_dimensions\":[\"0\"]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
}


)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_FALSE(result);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, F16Bmm1Bmm2Pattern) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Fused MHA is supported with the Nvidia Ampere+ GPUs.";
  }
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(f16[20,40,64]{2,1,0},f16[20,64,40]{2,1,0},f16[20,40,64]{2,1,0})->f16[20,40,64]{2,1,0}}

ENTRY main.6 {
  Arg_0.1 = f16[20,40,64]{2,1,0} parameter(0)
  Arg_1.2 = f16[20,64,40]{1,2,0} parameter(1)
  custom-call = f16[20,40,40]{2,1,0} custom-call(Arg_0.1, Arg_1.2), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\"],\"rhs_batch_dimensions\":[\"0\"]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
  Arg_2.3 = f16[20,40,64]{2,1,0} parameter(2)
  ROOT custom-call.1 = f16[20,40,64]{2,1,0} custom-call(custom-call, Arg_2.3), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\"],\"rhs_batch_dimensions\":[\"0\"]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}"
}


)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(F16, {20, 40, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
}
}  // namespace
}  // namespace gpu
}  // namespace xla
