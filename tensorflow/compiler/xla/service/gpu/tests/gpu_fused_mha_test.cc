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

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

class MultiHeadedAttentionTest : public HloTestBase {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  ErrorSpec error_spec_{2.5E-3, 1e-5};

 protected:
  DebugOptions GetDebugOptionsForTest() override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_xla_runtime_executable(false);
    return debug_options;
  }

  void IsFMHACalled(const std::string &hlo_string,
                    HloModuleConfig &config_with_fmha,
                    const std::string &prefix) {
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> verified_module,
        ParseAndReturnVerifiedModule(hlo_string, config_with_fmha));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> optimized_verified_module,
        GetOptimizedModule(std::move(verified_module)));

    auto count = absl::c_count_if(
        optimized_verified_module->entry_computation()->instructions(),
        [&](const HloInstruction *inst) {
          return inst->opcode() == HloOpcode::kCustomCall &&
                 absl::StrContains(inst->custom_call_target(), prefix);
        });
    EXPECT_EQ(count, 1);
  }

  void ExecuteAndCompare(const std::string hlo_string,
                         Literal &lhs_bmm1_literal, Literal &rhs_bmm1_literal,
                         Literal &rhs_bmm2_literal) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_string, config));

    auto expected_result = ExecuteAndTransfer(
        std::move(module),
        {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});

    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cudnn_fmha(true);

    HloModuleConfig config_with_fmha;
    config_with_fmha.set_debug_options(debug_options);

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> new_module,
        ParseAndReturnVerifiedModule(hlo_string, config_with_fmha));
    auto actual_result = ExecuteAndTransfer(
        std::move(new_module),
        {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
    EXPECT_TRUE(
        LiteralTestUtil::Near(expected_result, actual_result, error_spec_));

    std::string prefix = "__cudnn$fhma";
    IsFMHACalled(hlo_string, config_with_fmha, prefix);
  }

  void ExecuteAndCompareUsingMask(const std::string hlo_string,
                                  Literal &lhs_bmm1_literal,
                                  Literal &rhs_bmm1_literal,
                                  Literal &rhs_bmm2_literal,
                                  Literal &mask_literal) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_string, config));

    auto expected_result = ExecuteAndTransfer(
        std::move(module), {&lhs_bmm1_literal, &rhs_bmm1_literal,
                            &rhs_bmm2_literal, &mask_literal});

    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cudnn_fmha(true);

    HloModuleConfig config_with_fmha;
    config_with_fmha.set_debug_options(debug_options);

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> new_module,
        ParseAndReturnVerifiedModule(hlo_string, config_with_fmha));

    auto actual_result = ExecuteAndTransfer(
        std::move(new_module), {&lhs_bmm1_literal, &rhs_bmm1_literal,
                                &rhs_bmm2_literal, &mask_literal});

    EXPECT_TRUE(
        LiteralTestUtil::Near(expected_result, actual_result, error_spec_));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> verified_module,
        ParseAndReturnVerifiedModule(hlo_string, config_with_fmha));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> optimized_verified_module,
        GetOptimizedModule(std::move(verified_module)));

    std::string prefix = "__cudnn$fhma";
    IsFMHACalled(hlo_string, config_with_fmha, prefix);
  }

  template <typename T>
  Literal GetInput4DLiteral(std::vector<int64_t> dimensions,
                            std::vector<int64_t> minor_to_major) {
    Array4D<T> input_data(dimensions[0], dimensions[1], dimensions[2],
                          dimensions[3]);
    input_data.FillRandom(/*stddev=*/static_cast<T>(0.023), 0.001);

    return LiteralUtil::CreateR4FromArray4DWithLayout(
        input_data, LayoutUtil::MakeLayout(minor_to_major));
  }

  Literal GetMask4DLiteral(std::vector<int64_t> dimensions,
                           std::vector<int64_t> minor_to_major) {
    Array4D<bool> input_data(dimensions[0], dimensions[1], dimensions[2],
                             dimensions[3]);
    input_data.FillRandomBool();

    return LiteralUtil::CreateR4FromArray4DWithLayout(
        input_data, LayoutUtil::MakeLayout(minor_to_major));
  }
};

class MultiHeadedAttentionBMMBMM : public MultiHeadedAttentionTest {
 protected:
  const std::string GetModuleFMHABMM_BMM_vanilla_HloString_F16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0})->f16[16,16,256,64]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
      Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.0 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      ROOT dot.1 = f16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
    }
  )";
    return hlo_text;
  }

  const std::string GetModuleFMHABMM_BMM_vanilla_HloString_BF16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
      Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
    }
  )";
    return hlo_text;
  }

  const std::string GetModuleFMHABMM_BMM_arg_reversal_HloString_F16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0})->f16[16,16,64,256]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
      Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.0 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      ROOT dot.1 = f16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
    }
  )";
    return hlo_text;
  }

  const std::string GetModuleFMHABMM_BMM_arg_reversal_HloString_BF16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,64,256]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
      Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      ROOT dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
    }
  )";
    return hlo_text;
  }

  const std::string
  GetModuleFMHABMM_BMM_arg_layout_manipulation_arg_reversal_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0})->bf16[16,16,64,256]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2)
      Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1)
      dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}, metadata={}
      ROOT dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,2}, lhs_contracting_dims={1}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
    }
  )";
    return hlo_text;
  }

  const std::string
  GetModuleFMHABMM_BMM_arg_layout_manipulation_arg_reversal_HloString_F16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0})->f16[16,16,64,256]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = f16[16,256,16,64]{3,2,1,0} parameter(2)
      Arg_0.1 = f16[16,256,16,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,256,16,64]{3,2,1,0} parameter(1)
      dot.0 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}, metadata={}
      ROOT dot.1 = f16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,2}, lhs_contracting_dims={1}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
    }
  )";
    return hlo_text;
  }

  const std::string GetModuleFMHABMM_BMM_all_canonicalization_HloString_BF16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0})->bf16[16,256,16,64]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2)
      Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1)
      dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}, metadata={}
      dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,2}, lhs_contracting_dims={1}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      ROOT transpose.0 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}, metadata={}
    }
  )";
    return hlo_text;
  }

  const std::string GetModuleFMHABMM_BMM_all_canonicalization_HloString_F16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0})->f16[16,256,16,64]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = f16[16,256,16,64]{3,2,1,0} parameter(2)
      Arg_0.1 = f16[16,256,16,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,256,16,64]{3,2,1,0} parameter(1)
      dot.0 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}, metadata={}
      dot.1 = f16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,2}, lhs_contracting_dims={1}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      ROOT transpose.0 = f16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}, metadata={}
    }
  )";
    return hlo_text;
  }

  const std::string
  GetModuleFMHABMM_BMM1_contracting_dim_stride_not_1_HloString_F16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(f16[16,16,256,64]{2,3,1,0},f16[16,16,256,64]{2,3,1,0},f16[16,16,256,64]{3,2,1,0})->f16[16,16,256,64]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
      Arg_0.1 = f16[16,16,256,64]{2,3,1,0} parameter(0)
      Arg_1.2 = f16[16,16,256,64]{2,3,1,0} parameter(1)
      dot.0 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      ROOT dot.1 = f16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
    }
  )";
    return hlo_text;
  }

  const std::string
  GetModuleFMHABMM_BMM2_non_contracting_dim_stride_not_1_HloString_F16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{2,3,1,0})->f16[16,16,256,64]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = f16[16,16,256,64]{2,3,1,0} parameter(2)
      Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.0 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      ROOT dot.1 = f16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
    }
  )";
    return hlo_text;
  }

  template <typename T>
  void TestImpl_FMHABMM_BMM_vanilla() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string = GetModuleFMHABMM_BMM_vanilla_HloString_F16();
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string = GetModuleFMHABMM_BMM_vanilla_HloString_BF16();
    }

    ExecuteAndCompare(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                      rhs_bmm2_literal);
  }

  template <typename T>
  void TestImpl_FMHABMM_BMM_arg_reversal() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string = GetModuleFMHABMM_BMM_arg_reversal_HloString_F16();
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string = GetModuleFMHABMM_BMM_arg_reversal_HloString_BF16();
    }
    ExecuteAndCompare(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                      rhs_bmm2_literal);
  }

  template <typename T>
  void TestImpl_FMHABMM_BMM_arg_layout_manipulation_arg_reversal_fusion() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 256, 16, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 256, 16, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 256, 16, 64}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string =
          GetModuleFMHABMM_BMM_arg_layout_manipulation_arg_reversal_HloString_F16();  // NOLINT
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string =
          GetModuleFMHABMM_BMM_arg_layout_manipulation_arg_reversal_HloString_BF16();  // NOLINT
    }
    ExecuteAndCompare(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                      rhs_bmm2_literal);
  }

  template <typename T>
  void TestImpl_FMHABMM_BMM_all_canonicalization() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 256, 16, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 256, 16, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 256, 16, 64}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string = GetModuleFMHABMM_BMM_all_canonicalization_HloString_F16();
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string = GetModuleFMHABMM_BMM_all_canonicalization_HloString_BF16();
    }
    ExecuteAndCompare(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                      rhs_bmm2_literal);
  }

  template <typename T>
  void TestImpl_BMM_BMM1_contracting_dim_stride_not_1() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {2, 3, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {2, 3, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string =
          GetModuleFMHABMM_BMM1_contracting_dim_stride_not_1_HloString_F16();
    }

    ExecuteAndCompare(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                      rhs_bmm2_literal);
  }

  template <typename T>
  void TestImpl_BMM_BMM2_non_contracting_dim_stride_not_1() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {2, 3, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string =
          GetModuleFMHABMM_BMM2_non_contracting_dim_stride_not_1_HloString_F16();  // NOLINT
    }

    ExecuteAndCompare(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                      rhs_bmm2_literal);
  }
};

// BMM1 - Scale - Bias - Mask - Softmax - BMM2
class MultiHeadedAttentionBMMScaleBiasMaskSoftmaxBMM
    : public MultiHeadedAttentionTest {
 protected:
  const std::string
  GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_HloString_F16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},pred[16,16,256,256]{3,2,1,0})->f16[16,16,256,64]{3,2,1,0}}

    region_0.17 {
      Arg_0.18 = f16[] parameter(0)
      Arg_1.19 = f16[] parameter(1)
      ROOT maximum.20 = f16[] maximum(Arg_0.18, Arg_1.19)
    }

    region_1.29 {
      Arg_0.30 = f32[] parameter(0)
      Arg_1.31 = f32[] parameter(1)
      ROOT add.32 = f32[] add(Arg_0.30, Arg_1.31)
    }

    ENTRY main.41 {
      constant.12 = pred[16,16,256,256]{3,2,1,0} parameter(3)
      Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.13 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      constant.6 = f16[] constant(2)
      broadcast.7 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
      multiply.14 = f16[16,16,256,256]{3,2,1,0} multiply(dot.13, broadcast.7)
      constant.10 = f16[] constant(1)
      broadcast.11 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.10), dimensions={}
      add.15 = f16[16,16,256,256]{3,2,1,0} add(multiply.14, broadcast.11)
      constant.4 = f16[] constant(0)
      broadcast.5 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
      select.16 = f16[16,16,256,256]{3,2,1,0} select(constant.12, add.15, broadcast.5)
      constant.9 = f16[] constant(-inf)
      reduce.21 = f16[16,16,256]{2,1,0} reduce(select.16, constant.9), dimensions={3}, to_apply=region_0.17
      reshape.22 = f16[16,16,256,1]{3,2,1,0} reshape(reduce.21)
      broadcast.23 = f16[16,16,256,1]{3,2,1,0} broadcast(reshape.22), dimensions={0,1,2,3}
      reshape.24 = f16[16,16,256]{2,1,0} reshape(broadcast.23)
      broadcast.25 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.24), dimensions={0,1,2}
      subtract.26 = f16[16,16,256,256]{3,2,1,0} subtract(select.16, broadcast.25)
      exponential.27 = f16[16,16,256,256]{3,2,1,0} exponential(subtract.26)
      convert.28 = f32[16,16,256,256]{3,2,1,0} convert(exponential.27)
      constant.8 = f32[] constant(0)
      reduce.33 = f32[16,16,256]{2,1,0} reduce(convert.28, constant.8), dimensions={3}, to_apply=region_1.29
      reshape.34 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.33)
      convert.35 = f16[16,16,256,1]{3,2,1,0} convert(reshape.34)
      broadcast.36 = f16[16,16,256,1]{3,2,1,0} broadcast(convert.35), dimensions={0,1,2,3}
      reshape.37 = f16[16,16,256]{2,1,0} reshape(broadcast.36)
      broadcast.38 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.37), dimensions={0,1,2}
      divide.39 = f16[16,16,256,256]{3,2,1,0} divide(exponential.27, broadcast.38)
      Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
      ROOT dot.40 = f16[16,16,256,64]{3,2,1,0} dot(divide.39, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";

    return hlo_text;
  }

  const std::string
  GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_HloString_BF16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},pred[16,16,256,256]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

    region_0.17 {
      Arg_0.18 = bf16[] parameter(0)
      Arg_1.19 = bf16[] parameter(1)
      ROOT maximum.20 = bf16[] maximum(Arg_0.18, Arg_1.19)
    }

    region_1.29 {
      Arg_0.30 = f32[] parameter(0)
      Arg_1.31 = f32[] parameter(1)
      ROOT add.32 = f32[] add(Arg_0.30, Arg_1.31)
    }

    ENTRY main.41 {
      constant.12 = pred[16,16,256,256]{3,2,1,0} parameter(3)
      Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.13 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      constant.6 = bf16[] constant(2)
      broadcast.7 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
      multiply.14 = bf16[16,16,256,256]{3,2,1,0} multiply(dot.13, broadcast.7)
      constant.10 = bf16[] constant(1)
      broadcast.11 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.10), dimensions={}
      add.15 = bf16[16,16,256,256]{3,2,1,0} add(multiply.14, broadcast.11)
      constant.4 = bf16[] constant(0)
      broadcast.5 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
      select.16 = bf16[16,16,256,256]{3,2,1,0} select(constant.12, add.15, broadcast.5)
      constant.9 = bf16[] constant(-inf)
      reduce.21 = bf16[16,16,256]{2,1,0} reduce(select.16, constant.9), dimensions={3}, to_apply=region_0.17
      reshape.22 = bf16[16,16,256,1]{3,2,1,0} reshape(reduce.21)
      broadcast.23 = bf16[16,16,256,1]{3,2,1,0} broadcast(reshape.22), dimensions={0,1,2,3}
      reshape.24 = bf16[16,16,256]{2,1,0} reshape(broadcast.23)
      broadcast.25 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.24), dimensions={0,1,2}
      subtract.26 = bf16[16,16,256,256]{3,2,1,0} subtract(select.16, broadcast.25)
      exponential.27 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.26)
      convert.28 = f32[16,16,256,256]{3,2,1,0} convert(exponential.27)
      constant.8 = f32[] constant(0)
      reduce.33 = f32[16,16,256]{2,1,0} reduce(convert.28, constant.8), dimensions={3}, to_apply=region_1.29
      reshape.34 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.33)
      convert.35 = bf16[16,16,256,1]{3,2,1,0} convert(reshape.34)
      broadcast.36 = bf16[16,16,256,1]{3,2,1,0} broadcast(convert.35), dimensions={0,1,2,3}
      reshape.37 = bf16[16,16,256]{2,1,0} reshape(broadcast.36)
      broadcast.38 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.37), dimensions={0,1,2}
      divide.39 = bf16[16,16,256,256]{3,2,1,0} divide(exponential.27, broadcast.38)
      Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
      ROOT dot.40 = bf16[16,16,256,64]{3,2,1,0} dot(divide.39, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";

    return hlo_text;
  }

  const std::string
  GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_HloString_BF16_smaller() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,40,64]{3,2,1,0},bf16[2,6,64,40]{3,2,1,0},bf16[2,6,40,64]{3,2,1,0},pred[2,6,40,40]{3,2,1,0})->bf16[2,6,40,64]{3,2,1,0}}

    region_0.17 {
      Arg_0.18 = bf16[] parameter(0)
      Arg_1.19 = bf16[] parameter(1)
      ROOT maximum.20 = bf16[] maximum(Arg_0.18, Arg_1.19)
    }

    region_1.29 {
      Arg_0.30 = f32[] parameter(0)
      Arg_1.31 = f32[] parameter(1)
      ROOT add.32 = f32[] add(Arg_0.30, Arg_1.31)
    }

    ENTRY main.41 {
      constant.12 = pred[2,6,40,40]{3,2,1,0} parameter(3)
      Arg_0.1 = bf16[2,6,40,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[2,6,64,40]{3,2,1,0} parameter(1)
      dot.13 = bf16[2,6,40,40]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      constant.6 = bf16[] constant(2)
      broadcast.7 = bf16[2,6,40,40]{3,2,1,0} broadcast(constant.6), dimensions={}
      multiply.14 = bf16[2,6,40,40]{3,2,1,0} multiply(dot.13, broadcast.7)
      constant.10 = bf16[] constant(1)
      broadcast.11 = bf16[2,6,40,40]{3,2,1,0} broadcast(constant.10), dimensions={}
      add.15 = bf16[2,6,40,40]{3,2,1,0} add(multiply.14, broadcast.11)
      constant.4 = bf16[] constant(0)
      broadcast.5 = bf16[2,6,40,40]{3,2,1,0} broadcast(constant.4), dimensions={}
      select.16 = bf16[2,6,40,40]{3,2,1,0} select(constant.12, add.15, broadcast.5)
      constant.9 = bf16[] constant(-inf)
      reduce.21 = bf16[2,6,40]{2,1,0} reduce(select.16, constant.9), dimensions={3}, to_apply=region_0.17
      reshape.22 = bf16[2,6,40,1]{3,2,1,0} reshape(reduce.21)
      broadcast.23 = bf16[2,6,40,1]{3,2,1,0} broadcast(reshape.22), dimensions={0,1,2,3}
      reshape.24 = bf16[2,6,40]{2,1,0} reshape(broadcast.23)
      broadcast.25 = bf16[2,6,40,40]{3,2,1,0} broadcast(reshape.24), dimensions={0,1,2}
      subtract.26 = bf16[2,6,40,40]{3,2,1,0} subtract(select.16, broadcast.25)
      exponential.27 = bf16[2,6,40,40]{3,2,1,0} exponential(subtract.26)
      convert.28 = f32[2,6,40,40]{3,2,1,0} convert(exponential.27)
      constant.8 = f32[] constant(0)
      reduce.33 = f32[2,6,40]{2,1,0} reduce(convert.28, constant.8), dimensions={3}, to_apply=region_1.29
      reshape.34 = f32[2,6,40,1]{3,2,1,0} reshape(reduce.33)
      convert.35 = bf16[2,6,40,1]{3,2,1,0} convert(reshape.34)
      broadcast.36 = bf16[2,6,40,1]{3,2,1,0} broadcast(convert.35), dimensions={0,1,2,3}
      reshape.37 = bf16[2,6,40]{2,1,0} reshape(broadcast.36)
      broadcast.38 = bf16[2,6,40,40]{3,2,1,0} broadcast(reshape.37), dimensions={0,1,2}
      divide.39 = bf16[2,6,40,40]{3,2,1,0} divide(exponential.27, broadcast.38)
      Arg_2.3 = bf16[2,6,40,64]{3,2,1,0} parameter(2)
      ROOT dot.40 = bf16[2,6,40,64]{3,2,1,0} dot(divide.39, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";

    return hlo_text;
  }

  const std::string
  GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_HloString_F16_smaller() {  // NOLINT
    const std::string hlo_text = R"(
  HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,40,64]{3,2,1,0},f16[2,6,64,40]{3,2,1,0},f16[2,6,40,64]{3,2,1,0},pred[2,6,40,40]{3,2,1,0})->f16[2,6,40,64]{3,2,1,0}}

  region_0.17 {
    Arg_0.18 = f16[] parameter(0)
    Arg_1.19 = f16[] parameter(1)
    ROOT maximum.20 = f16[] maximum(Arg_0.18, Arg_1.19)
  }

  region_1.29 {
    Arg_0.30 = f32[] parameter(0)
    Arg_1.31 = f32[] parameter(1)
    ROOT add.32 = f32[] add(Arg_0.30, Arg_1.31)
  }

  ENTRY main.41 {
    constant.12 = pred[2,6,40,40]{3,2,1,0} parameter(3)
    Arg_0.1 = f16[2,6,40,64]{3,2,1,0} parameter(0)
    Arg_1.2 = f16[2,6,64,40]{3,2,1,0} parameter(1)
    dot.13 = f16[2,6,40,40]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    constant.6 = f16[] constant(2)
    broadcast.7 = f16[2,6,40,40]{3,2,1,0} broadcast(constant.6), dimensions={}
    multiply.14 = f16[2,6,40,40]{3,2,1,0} multiply(dot.13, broadcast.7)
    constant.10 = f16[] constant(1)
    broadcast.11 = f16[2,6,40,40]{3,2,1,0} broadcast(constant.10), dimensions={}
    add.15 = f16[2,6,40,40]{3,2,1,0} add(multiply.14, broadcast.11)
    constant.4 = f16[] constant(0)
    broadcast.5 = f16[2,6,40,40]{3,2,1,0} broadcast(constant.4), dimensions={}
    select.16 = f16[2,6,40,40]{3,2,1,0} select(constant.12, add.15, broadcast.5)
    constant.9 = f16[] constant(-inf)
    reduce.21 = f16[2,6,40]{2,1,0} reduce(select.16, constant.9), dimensions={3}, to_apply=region_0.17
    reshape.22 = f16[2,6,40,1]{3,2,1,0} reshape(reduce.21)
    broadcast.23 = f16[2,6,40,1]{3,2,1,0} broadcast(reshape.22), dimensions={0,1,2,3}
    reshape.24 = f16[2,6,40]{2,1,0} reshape(broadcast.23)
    broadcast.25 = f16[2,6,40,40]{3,2,1,0} broadcast(reshape.24), dimensions={0,1,2}
    subtract.26 = f16[2,6,40,40]{3,2,1,0} subtract(select.16, broadcast.25)
    exponential.27 = f16[2,6,40,40]{3,2,1,0} exponential(subtract.26)
    convert.28 = f32[2,6,40,40]{3,2,1,0} convert(exponential.27)
    constant.8 = f32[] constant(0)
    reduce.33 = f32[2,6,40]{2,1,0} reduce(convert.28, constant.8), dimensions={3}, to_apply=region_1.29
    reshape.34 = f32[2,6,40,1]{3,2,1,0} reshape(reduce.33)
    convert.35 = f16[2,6,40,1]{3,2,1,0} convert(reshape.34)
    broadcast.36 = f16[2,6,40,1]{3,2,1,0} broadcast(convert.35), dimensions={0,1,2,3}
    reshape.37 = f16[2,6,40]{2,1,0} reshape(broadcast.36)
    broadcast.38 = f16[2,6,40,40]{3,2,1,0} broadcast(reshape.37), dimensions={0,1,2}
    divide.39 = f16[2,6,40,40]{3,2,1,0} divide(exponential.27, broadcast.38)
    Arg_2.3 = f16[2,6,40,64]{3,2,1,0} parameter(2)
    ROOT dot.40 = f16[2,6,40,64]{3,2,1,0} dot(divide.39, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  }
)";

    return hlo_text;
  }

  const std::string
  GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_arg_reversal_HloString_F16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},pred[16,16,256,256]{3,2,1,0})->f16[16,16,64,256]{3,2,1,0}}

    region_0.17 {
      Arg_0.18 = f16[] parameter(0)
      Arg_1.19 = f16[] parameter(1)
      ROOT maximum.20 = f16[] maximum(Arg_0.18, Arg_1.19)
    }

    region_1.29 {
      Arg_0.30 = f32[] parameter(0)
      Arg_1.31 = f32[] parameter(1)
      ROOT add.32 = f32[] add(Arg_0.30, Arg_1.31)
    }

    ENTRY main.41 {
      constant.12 = pred[16,16,256,256]{3,2,1,0} parameter(3)
      Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.13 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      constant.6 = f16[] constant(2)
      broadcast.7 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
      multiply.14 = f16[16,16,256,256]{3,2,1,0} multiply(dot.13, broadcast.7)
      constant.10 = f16[] constant(1)
      broadcast.11 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.10), dimensions={}
      add.15 = f16[16,16,256,256]{3,2,1,0} add(multiply.14, broadcast.11)
      constant.4 = f16[] constant(0)
      broadcast.5 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
      select.16 = f16[16,16,256,256]{3,2,1,0} select(constant.12, add.15, broadcast.5)
      constant.9 = f16[] constant(-inf)
      reduce.21 = f16[16,16,256]{2,1,0} reduce(select.16, constant.9), dimensions={3}, to_apply=region_0.17
      reshape.22 = f16[16,16,256,1]{3,2,1,0} reshape(reduce.21)
      broadcast.23 = f16[16,16,256,1]{3,2,1,0} broadcast(reshape.22), dimensions={0,1,2,3}
      reshape.24 = f16[16,16,256]{2,1,0} reshape(broadcast.23)
      broadcast.25 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.24), dimensions={0,1,2}
      subtract.26 = f16[16,16,256,256]{3,2,1,0} subtract(select.16, broadcast.25)
      exponential.27 = f16[16,16,256,256]{3,2,1,0} exponential(subtract.26)
      convert.28 = f32[16,16,256,256]{3,2,1,0} convert(exponential.27)
      constant.8 = f32[] constant(0)
      reduce.33 = f32[16,16,256]{2,1,0} reduce(convert.28, constant.8), dimensions={3}, to_apply=region_1.29
      reshape.34 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.33)
      convert.35 = f16[16,16,256,1]{3,2,1,0} convert(reshape.34)
      broadcast.36 = f16[16,16,256,1]{3,2,1,0} broadcast(convert.35), dimensions={0,1,2,3}
      reshape.37 = f16[16,16,256]{2,1,0} reshape(broadcast.36)
      broadcast.38 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.37), dimensions={0,1,2}
      divide.39 = f16[16,16,256,256]{3,2,1,0} divide(exponential.27, broadcast.38)
      Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
      ROOT dot.40 = f16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, divide.39), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
    }
  )";

    return hlo_text;
  }

  const std::string
  GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_arg_reversal_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
  HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},pred[16,16,256,256]{3,2,1,0})->bf16[16,16,64,256]{3,2,1,0}}

  region_0.17 {
    Arg_0.18 = bf16[] parameter(0)
    Arg_1.19 = bf16[] parameter(1)
    ROOT maximum.20 = bf16[] maximum(Arg_0.18, Arg_1.19)
  }

  region_1.29 {
    Arg_0.30 = f32[] parameter(0)
    Arg_1.31 = f32[] parameter(1)
    ROOT add.32 = f32[] add(Arg_0.30, Arg_1.31)
  }

  ENTRY main.41 {
    constant.12 = pred[16,16,256,256]{3,2,1,0} parameter(3)
    Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
    Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
    dot.13 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
    constant.6 = bf16[] constant(2)
    broadcast.7 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
    multiply.14 = bf16[16,16,256,256]{3,2,1,0} multiply(dot.13, broadcast.7)
    constant.10 = bf16[] constant(1)
    broadcast.11 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.10), dimensions={}
    add.15 = bf16[16,16,256,256]{3,2,1,0} add(multiply.14, broadcast.11)
    constant.4 = bf16[] constant(0)
    broadcast.5 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
    select.16 = bf16[16,16,256,256]{3,2,1,0} select(constant.12, add.15, broadcast.5)
    constant.9 = bf16[] constant(-inf)
    reduce.21 = bf16[16,16,256]{2,1,0} reduce(select.16, constant.9), dimensions={3}, to_apply=region_0.17
    reshape.22 = bf16[16,16,256,1]{3,2,1,0} reshape(reduce.21)
    broadcast.23 = bf16[16,16,256,1]{3,2,1,0} broadcast(reshape.22), dimensions={0,1,2,3}
    reshape.24 = bf16[16,16,256]{2,1,0} reshape(broadcast.23)
    broadcast.25 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.24), dimensions={0,1,2}
    subtract.26 = bf16[16,16,256,256]{3,2,1,0} subtract(select.16, broadcast.25)
    exponential.27 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.26)
    convert.28 = f32[16,16,256,256]{3,2,1,0} convert(exponential.27)
    constant.8 = f32[] constant(0)
    reduce.33 = f32[16,16,256]{2,1,0} reduce(convert.28, constant.8), dimensions={3}, to_apply=region_1.29
    reshape.34 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.33)
    convert.35 = bf16[16,16,256,1]{3,2,1,0} convert(reshape.34)
    broadcast.36 = bf16[16,16,256,1]{3,2,1,0} broadcast(convert.35), dimensions={0,1,2,3}
    reshape.37 = bf16[16,16,256]{2,1,0} reshape(broadcast.36)
    broadcast.38 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.37), dimensions={0,1,2}
    divide.39 = bf16[16,16,256,256]{3,2,1,0} divide(exponential.27, broadcast.38)
    Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
    ROOT dot.40 = bf16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, divide.39), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  }
)";

    return hlo_text;
  }

  // BMM1 - Scale - Bias - Mask - Softmax - BMM2
  template <typename T>
  void TestImpl_FMHABMM1_Scale_Bias_Mask_Softmax_BMM2_vanilla() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto mask_literal = GetMask4DLiteral({16, 16, 256, 256}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string =
          GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_HloString_F16();
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string =
          GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_HloString_BF16();
    }

    ExecuteAndCompareUsingMask(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                               rhs_bmm2_literal, mask_literal);
  }

  template <typename T>
  void TestImpl_FMHABMM1_Scale_Bias_Mask_Softmax_BMM2_vanilla_smaller() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal = GetInput4DLiteral<T>({2, 6, 40, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal = GetInput4DLiteral<T>({2, 6, 64, 40}, {3, 2, 1, 0});
    auto rhs_bmm2_literal = GetInput4DLiteral<T>({2, 6, 40, 64}, {3, 2, 1, 0});
    auto mask_literal = GetMask4DLiteral({2, 6, 40, 40}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string =
          GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_HloString_F16_smaller();  // NOLINT
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string =
          GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_HloString_BF16_smaller();  // NOLINT
    }

    ExecuteAndCompareUsingMask(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                               rhs_bmm2_literal, mask_literal);
  }

  template <typename T>
  void TestImpl_FMHABMM1_Scale_Bias_Mask_Softmax_BMM2_arg_reversal() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto mask_literal = GetMask4DLiteral({16, 16, 256, 256}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string =
          GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_arg_reversal_HloString_F16();  // NOLINT
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string =
          GetModuleFMHABMM1_Scale_Bias_Mask_Softmax_BMM2_arg_reversal_HloString_BF16();  // NOLINT
    }

    ExecuteAndCompareUsingMask(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                               rhs_bmm2_literal, mask_literal);
  }
};

// BMM1 - Scale - Mask - Softmax - BMM2
class MultiHeadedAttentionBMMScaleMaskSoftmaxBMM
    : public MultiHeadedAttentionTest {
 protected:
  const std::string GetModuleFMHABMM1_Scale_Mask_Softmax_BMM2_HloString_F16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},pred[16,16,256,256]{3,2,1,0})->f16[16,16,256,64]{3,2,1,0}}

    region_0.14 {
      Arg_0.15 = f16[] parameter(0)
      Arg_1.16 = f16[] parameter(1)
      ROOT maximum.17 = f16[] maximum(Arg_0.15, Arg_1.16)
    }

    region_1.26 {
      Arg_0.27 = f32[] parameter(0)
      Arg_1.28 = f32[] parameter(1)
      ROOT add.29 = f32[] add(Arg_0.27, Arg_1.28)
    }

    ENTRY main.38 {
      constant.10 = pred[16,16,256,256]{3,2,1,0} parameter(3)
      Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.11 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      constant.6 = f16[] constant(2)
      broadcast.7 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
      multiply.12 = f16[16,16,256,256]{3,2,1,0} multiply(dot.11, broadcast.7)
      constant.4 = f16[] constant(0)
      broadcast.5 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
      select.13 = f16[16,16,256,256]{3,2,1,0} select(constant.10, multiply.12, broadcast.5)
      constant.9 = f16[] constant(-inf)
      reduce.18 = f16[16,16,256]{2,1,0} reduce(select.13, constant.9), dimensions={3}, to_apply=region_0.14
      reshape.19 = f16[16,16,256,1]{3,2,1,0} reshape(reduce.18)
      broadcast.20 = f16[16,16,256,1]{3,2,1,0} broadcast(reshape.19), dimensions={0,1,2,3}
      reshape.21 = f16[16,16,256]{2,1,0} reshape(broadcast.20)
      broadcast.22 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.21), dimensions={0,1,2}
      subtract.23 = f16[16,16,256,256]{3,2,1,0} subtract(select.13, broadcast.22)
      exponential.24 = f16[16,16,256,256]{3,2,1,0} exponential(subtract.23)
      convert.25 = f32[16,16,256,256]{3,2,1,0} convert(exponential.24)
      constant.8 = f32[] constant(0)
      reduce.30 = f32[16,16,256]{2,1,0} reduce(convert.25, constant.8), dimensions={3}, to_apply=region_1.26
      reshape.31 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.30)
      convert.32 = f16[16,16,256,1]{3,2,1,0} convert(reshape.31)
      broadcast.33 = f16[16,16,256,1]{3,2,1,0} broadcast(convert.32), dimensions={0,1,2,3}
      reshape.34 = f16[16,16,256]{2,1,0} reshape(broadcast.33)
      broadcast.35 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.34), dimensions={0,1,2}
      divide.36 = f16[16,16,256,256]{3,2,1,0} divide(exponential.24, broadcast.35)
      Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
      ROOT dot.37 = f16[16,16,256,64]{3,2,1,0} dot(divide.36, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";

    return hlo_text;
  }

  const std::string GetModuleFMHABMM1_Scale_Mask_Softmax_BMM2_HloString_BF16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},pred[16,16,256,256]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

    region_0.14 {
      Arg_0.15 = bf16[] parameter(0)
      Arg_1.16 = bf16[] parameter(1)
      ROOT maximum.17 = bf16[] maximum(Arg_0.15, Arg_1.16)
    }

    region_1.26 {
      Arg_0.27 = f32[] parameter(0)
      Arg_1.28 = f32[] parameter(1)
      ROOT add.29 = f32[] add(Arg_0.27, Arg_1.28)
    }

    ENTRY main.38 {
      constant.10 = pred[16,16,256,256]{3,2,1,0} parameter(3)
      Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.11 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      constant.6 = bf16[] constant(2)
      broadcast.7 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
      multiply.12 = bf16[16,16,256,256]{3,2,1,0} multiply(dot.11, broadcast.7)
      constant.4 = bf16[] constant(0)
      broadcast.5 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
      select.13 = bf16[16,16,256,256]{3,2,1,0} select(constant.10, multiply.12, broadcast.5)
      constant.9 = bf16[] constant(-inf)
      reduce.18 = bf16[16,16,256]{2,1,0} reduce(select.13, constant.9), dimensions={3}, to_apply=region_0.14
      reshape.19 = bf16[16,16,256,1]{3,2,1,0} reshape(reduce.18)
      broadcast.20 = bf16[16,16,256,1]{3,2,1,0} broadcast(reshape.19), dimensions={0,1,2,3}
      reshape.21 = bf16[16,16,256]{2,1,0} reshape(broadcast.20)
      broadcast.22 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.21), dimensions={0,1,2}
      subtract.23 = bf16[16,16,256,256]{3,2,1,0} subtract(select.13, broadcast.22)
      exponential.24 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.23)
      convert.25 = f32[16,16,256,256]{3,2,1,0} convert(exponential.24)
      constant.8 = f32[] constant(0)
      reduce.30 = f32[16,16,256]{2,1,0} reduce(convert.25, constant.8), dimensions={3}, to_apply=region_1.26
      reshape.31 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.30)
      convert.32 = bf16[16,16,256,1]{3,2,1,0} convert(reshape.31)
      broadcast.33 = bf16[16,16,256,1]{3,2,1,0} broadcast(convert.32), dimensions={0,1,2,3}
      reshape.34 = bf16[16,16,256]{2,1,0} reshape(broadcast.33)
      broadcast.35 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.34), dimensions={0,1,2}
      divide.36 = bf16[16,16,256,256]{3,2,1,0} divide(exponential.24, broadcast.35)
      Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
      ROOT dot.37 = bf16[16,16,256,64]{3,2,1,0} dot(divide.36, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";

    return hlo_text;
  }

  const std::string
  GetModuleFMHABMM1_Scale_Mask_Softmax_BMM2_arg_reversal_HloString_F16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},pred[16,16,256,256]{3,2,1,0})->f16[16,16,64,256]{3,2,1,0}}

    region_0.14 {
      Arg_0.15 = f16[] parameter(0)
      Arg_1.16 = f16[] parameter(1)
      ROOT maximum.17 = f16[] maximum(Arg_0.15, Arg_1.16)
    }

    region_1.26 {
      Arg_0.27 = f32[] parameter(0)
      Arg_1.28 = f32[] parameter(1)
      ROOT add.29 = f32[] add(Arg_0.27, Arg_1.28)
    }

    ENTRY main.38 {
      constant.10 = pred[16,16,256,256]{3,2,1,0} parameter(3)
      Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.11 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      constant.6 = f16[] constant(2)
      broadcast.7 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
      multiply.12 = f16[16,16,256,256]{3,2,1,0} multiply(dot.11, broadcast.7)
      constant.4 = f16[] constant(0)
      broadcast.5 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
      select.13 = f16[16,16,256,256]{3,2,1,0} select(constant.10, multiply.12, broadcast.5)
      constant.9 = f16[] constant(-inf)
      reduce.18 = f16[16,16,256]{2,1,0} reduce(select.13, constant.9), dimensions={3}, to_apply=region_0.14
      reshape.19 = f16[16,16,256,1]{3,2,1,0} reshape(reduce.18)
      broadcast.20 = f16[16,16,256,1]{3,2,1,0} broadcast(reshape.19), dimensions={0,1,2,3}
      reshape.21 = f16[16,16,256]{2,1,0} reshape(broadcast.20)
      broadcast.22 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.21), dimensions={0,1,2}
      subtract.23 = f16[16,16,256,256]{3,2,1,0} subtract(select.13, broadcast.22)
      exponential.24 = f16[16,16,256,256]{3,2,1,0} exponential(subtract.23)
      convert.25 = f32[16,16,256,256]{3,2,1,0} convert(exponential.24)
      constant.8 = f32[] constant(0)
      reduce.30 = f32[16,16,256]{2,1,0} reduce(convert.25, constant.8), dimensions={3}, to_apply=region_1.26
      reshape.31 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.30)
      convert.32 = f16[16,16,256,1]{3,2,1,0} convert(reshape.31)
      broadcast.33 = f16[16,16,256,1]{3,2,1,0} broadcast(convert.32), dimensions={0,1,2,3}
      reshape.34 = f16[16,16,256]{2,1,0} reshape(broadcast.33)
      broadcast.35 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.34), dimensions={0,1,2}
      divide.36 = f16[16,16,256,256]{3,2,1,0} divide(exponential.24, broadcast.35)
      Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
      ROOT dot.37 = f16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, divide.36), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
    }
  )";

    return hlo_text;
  }

  const std::string
  GetModuleFMHABMM1_Scale_Mask_Softmax_BMM2_arg_reversal_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},pred[16,16,256,256]{3,2,1,0})->bf16[16,16,64,256]{3,2,1,0}}

    region_0.14 {
      Arg_0.15 = bf16[] parameter(0)
      Arg_1.16 = bf16[] parameter(1)
      ROOT maximum.17 = bf16[] maximum(Arg_0.15, Arg_1.16)
    }

    region_1.26 {
      Arg_0.27 = f32[] parameter(0)
      Arg_1.28 = f32[] parameter(1)
      ROOT add.29 = f32[] add(Arg_0.27, Arg_1.28)
    }

    ENTRY main.38 {
      constant.10 = pred[16,16,256,256]{3,2,1,0} parameter(3)
      Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.11 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      constant.6 = bf16[] constant(2)
      broadcast.7 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
      multiply.12 = bf16[16,16,256,256]{3,2,1,0} multiply(dot.11, broadcast.7)
      constant.4 = bf16[] constant(0)
      broadcast.5 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
      select.13 = bf16[16,16,256,256]{3,2,1,0} select(constant.10, multiply.12, broadcast.5)
      constant.9 = bf16[] constant(-inf)
      reduce.18 = bf16[16,16,256]{2,1,0} reduce(select.13, constant.9), dimensions={3}, to_apply=region_0.14
      reshape.19 = bf16[16,16,256,1]{3,2,1,0} reshape(reduce.18)
      broadcast.20 = bf16[16,16,256,1]{3,2,1,0} broadcast(reshape.19), dimensions={0,1,2,3}
      reshape.21 = bf16[16,16,256]{2,1,0} reshape(broadcast.20)
      broadcast.22 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.21), dimensions={0,1,2}
      subtract.23 = bf16[16,16,256,256]{3,2,1,0} subtract(select.13, broadcast.22)
      exponential.24 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.23)
      convert.25 = f32[16,16,256,256]{3,2,1,0} convert(exponential.24)
      constant.8 = f32[] constant(0)
      reduce.30 = f32[16,16,256]{2,1,0} reduce(convert.25, constant.8), dimensions={3}, to_apply=region_1.26
      reshape.31 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.30)
      convert.32 = bf16[16,16,256,1]{3,2,1,0} convert(reshape.31)
      broadcast.33 = bf16[16,16,256,1]{3,2,1,0} broadcast(convert.32), dimensions={0,1,2,3}
      reshape.34 = bf16[16,16,256]{2,1,0} reshape(broadcast.33)
      broadcast.35 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.34), dimensions={0,1,2}
      divide.36 = bf16[16,16,256,256]{3,2,1,0} divide(exponential.24, broadcast.35)
      Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
      ROOT dot.37 = bf16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, divide.36), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
    }
  )";

    return hlo_text;
  }

  // BMM1 - Scale - Mask - Softmax - BMM2
  template <typename T>
  void TestImpl_FMHABMM1_Scale_Mask_Softmax_BMM2_vanilla() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto mask_literal = GetMask4DLiteral({16, 16, 256, 256}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string = GetModuleFMHABMM1_Scale_Mask_Softmax_BMM2_HloString_F16();
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string = GetModuleFMHABMM1_Scale_Mask_Softmax_BMM2_HloString_BF16();
    }

    ExecuteAndCompareUsingMask(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                               rhs_bmm2_literal, mask_literal);
  }

  template <typename T>
  void TestImpl_FMHABMM1_Scale_Mask_Softmax_BMM2_arg_reversal() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 16, 64, 256}, {3, 2, 1, 0});
    auto mask_literal = GetMask4DLiteral({16, 16, 256, 256}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string =
          GetModuleFMHABMM1_Scale_Mask_Softmax_BMM2_arg_reversal_HloString_F16();  // NOLINT
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string =
          GetModuleFMHABMM1_Scale_Mask_Softmax_BMM2_arg_reversal_HloString_BF16();  // NOLINT
    }

    ExecuteAndCompareUsingMask(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                               rhs_bmm2_literal, mask_literal);
  }
};

class MultiHeadedAttentionBMMSoftmaxBMM : public MultiHeadedAttentionTest {
  // Bmm1 - Softmax - Bmm2
 protected:
  const std::string GetModuleFMHABMM1_Softmax_BMM2_HloString_F16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0})->f16[16,16,256,64]{3,2,1,0}}

    region_0.7 {
      Arg_0.8 = f16[] parameter(0)
      Arg_1.9 = f16[] parameter(1)
      ROOT maximum.10 = f16[] maximum(Arg_0.8, Arg_1.9)
    }

    region_1.19 {
      Arg_0.20 = f32[] parameter(0)
      Arg_1.21 = f32[] parameter(1)
      ROOT add.22 = f32[] add(Arg_0.20, Arg_1.21)
    }

    ENTRY main.31 {
      Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.6 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      constant.5 = f16[] constant(-inf)
      reduce.11 = f16[16,16,256]{2,1,0} reduce(dot.6, constant.5), dimensions={3}, to_apply=region_0.7
      reshape.12 = f16[16,16,256,1]{3,2,1,0} reshape(reduce.11)
      broadcast.13 = f16[16,16,256,1]{3,2,1,0} broadcast(reshape.12), dimensions={0,1,2,3}
      reshape.14 = f16[16,16,256]{2,1,0} reshape(broadcast.13)
      broadcast.15 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.14), dimensions={0,1,2}
      subtract.16 = f16[16,16,256,256]{3,2,1,0} subtract(dot.6, broadcast.15)
      exponential.17 = f16[16,16,256,256]{3,2,1,0} exponential(subtract.16)
      convert.18 = f32[16,16,256,256]{3,2,1,0} convert(exponential.17)
      constant.4 = f32[] constant(0)
      reduce.23 = f32[16,16,256]{2,1,0} reduce(convert.18, constant.4), dimensions={3}, to_apply=region_1.19
      reshape.24 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.23)
      convert.25 = f16[16,16,256,1]{3,2,1,0} convert(reshape.24)
      broadcast.26 = f16[16,16,256,1]{3,2,1,0} broadcast(convert.25), dimensions={0,1,2,3}
      reshape.27 = f16[16,16,256]{2,1,0} reshape(broadcast.26)
      broadcast.28 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.27), dimensions={0,1,2}
      divide.29 = f16[16,16,256,256]{3,2,1,0} divide(exponential.17, broadcast.28)
      Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
      ROOT dot.30 = f16[16,16,256,64]{3,2,1,0} dot(divide.29, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";

    return hlo_text;
  }

  const std::string GetModuleFMHABMM1_Softmax_BMM2_HloString_BF16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

    region_0.7 {
      Arg_0.8 = bf16[] parameter(0)
      Arg_1.9 = bf16[] parameter(1)
      ROOT maximum.10 = bf16[] maximum(Arg_0.8, Arg_1.9)
    }

    region_1.19 {
      Arg_0.20 = f32[] parameter(0)
      Arg_1.21 = f32[] parameter(1)
      ROOT add.22 = f32[] add(Arg_0.20, Arg_1.21)
    }

    ENTRY main.31 {
      Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.6 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      constant.5 = bf16[] constant(-inf)
      reduce.11 = bf16[16,16,256]{2,1,0} reduce(dot.6, constant.5), dimensions={3}, to_apply=region_0.7
      reshape.12 = bf16[16,16,256,1]{3,2,1,0} reshape(reduce.11)
      broadcast.13 = bf16[16,16,256,1]{3,2,1,0} broadcast(reshape.12), dimensions={0,1,2,3}
      reshape.14 = bf16[16,16,256]{2,1,0} reshape(broadcast.13)
      broadcast.15 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.14), dimensions={0,1,2}
      subtract.16 = bf16[16,16,256,256]{3,2,1,0} subtract(dot.6, broadcast.15)
      exponential.17 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.16)
      convert.18 = f32[16,16,256,256]{3,2,1,0} convert(exponential.17)
      constant.4 = f32[] constant(0)
      reduce.23 = f32[16,16,256]{2,1,0} reduce(convert.18, constant.4), dimensions={3}, to_apply=region_1.19
      reshape.24 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.23)
      convert.25 = bf16[16,16,256,1]{3,2,1,0} convert(reshape.24)
      broadcast.26 = bf16[16,16,256,1]{3,2,1,0} broadcast(convert.25), dimensions={0,1,2,3}
      reshape.27 = bf16[16,16,256]{2,1,0} reshape(broadcast.26)
      broadcast.28 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.27), dimensions={0,1,2}
      divide.29 = bf16[16,16,256,256]{3,2,1,0} divide(exponential.17, broadcast.28)
      Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
      ROOT dot.30 = bf16[16,16,256,64]{3,2,1,0} dot(divide.29, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";

    return hlo_text;
  }

  // BMM1 - Softmax - BMM2
  template <typename T>
  void TestImpl_FMHABMM1_Softmax_BMM2_vanilla() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string = GetModuleFMHABMM1_Softmax_BMM2_HloString_F16();
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string = GetModuleFMHABMM1_Softmax_BMM2_HloString_BF16();
    }

    ExecuteAndCompare(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                      rhs_bmm2_literal);
  }
};

class MultiHeadedAttentionBMMScaleBiasSoftmaxBMM
    : public MultiHeadedAttentionTest {
 protected:
  // Bmm1 - Scale - Bias - Softmax - Bmm2
  const std::string GetModuleFMHABMM1_Scale_Bias_Softmax_BMM2_HloString_F16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0})->f16[16,16,256,64]{3,2,1,0}}

    region_0.13 {
      Arg_0.14 = f16[] parameter(0)
      Arg_1.15 = f16[] parameter(1)
      ROOT maximum.16 = f16[] maximum(Arg_0.14, Arg_1.15)
    }

    region_1.25 {
      Arg_0.26 = f32[] parameter(0)
      Arg_1.27 = f32[] parameter(1)
      ROOT add.28 = f32[] add(Arg_0.26, Arg_1.27)
    }

    ENTRY main.37 {
      Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.10 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      constant.4 = f16[] constant(2)
      broadcast.5 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
      multiply.11 = f16[16,16,256,256]{3,2,1,0} multiply(dot.10, broadcast.5)
      constant.8 = f16[] constant(1)
      broadcast.9 = f16[16,16,256,256]{3,2,1,0} broadcast(constant.8), dimensions={}
      add.12 = f16[16,16,256,256]{3,2,1,0} add(multiply.11, broadcast.9)
      constant.7 = f16[] constant(-inf)
      reduce.17 = f16[16,16,256]{2,1,0} reduce(add.12, constant.7), dimensions={3}, to_apply=region_0.13
      reshape.18 = f16[16,16,256,1]{3,2,1,0} reshape(reduce.17)
      broadcast.19 = f16[16,16,256,1]{3,2,1,0} broadcast(reshape.18), dimensions={0,1,2,3}
      reshape.20 = f16[16,16,256]{2,1,0} reshape(broadcast.19)
      broadcast.21 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.20), dimensions={0,1,2}
      subtract.22 = f16[16,16,256,256]{3,2,1,0} subtract(add.12, broadcast.21)
      exponential.23 = f16[16,16,256,256]{3,2,1,0} exponential(subtract.22)
      convert.24 = f32[16,16,256,256]{3,2,1,0} convert(exponential.23)
      constant.6 = f32[] constant(0)
      reduce.29 = f32[16,16,256]{2,1,0} reduce(convert.24, constant.6), dimensions={3}, to_apply=region_1.25
      reshape.30 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.29)
      convert.31 = f16[16,16,256,1]{3,2,1,0} convert(reshape.30)
      broadcast.32 = f16[16,16,256,1]{3,2,1,0} broadcast(convert.31), dimensions={0,1,2,3}
      reshape.33 = f16[16,16,256]{2,1,0} reshape(broadcast.32)
      broadcast.34 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.33), dimensions={0,1,2}
      divide.35 = f16[16,16,256,256]{3,2,1,0} divide(exponential.23, broadcast.34)
      Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
      ROOT dot.36 = f16[16,16,256,64]{3,2,1,0} dot(divide.35, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";

    return hlo_text;
  }

  const std::string GetModuleFMHABMM1_Scale_Bias_Softmax_BMM2_HloString_BF16() {
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

    region_0.13 {
      Arg_0.14 = bf16[] parameter(0)
      Arg_1.15 = bf16[] parameter(1)
      ROOT maximum.16 = bf16[] maximum(Arg_0.14, Arg_1.15)
    }

    region_1.25 {
      Arg_0.26 = f32[] parameter(0)
      Arg_1.27 = f32[] parameter(1)
      ROOT add.28 = f32[] add(Arg_0.26, Arg_1.27)
    }

    ENTRY main.37 {
      Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.10 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      constant.4 = bf16[] constant(2)
      broadcast.5 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
      multiply.11 = bf16[16,16,256,256]{3,2,1,0} multiply(dot.10, broadcast.5)
      constant.8 = bf16[] constant(1)
      broadcast.9 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.8), dimensions={}
      add.12 = bf16[16,16,256,256]{3,2,1,0} add(multiply.11, broadcast.9)
      constant.7 = bf16[] constant(-inf)
      reduce.17 = bf16[16,16,256]{2,1,0} reduce(add.12, constant.7), dimensions={3}, to_apply=region_0.13
      reshape.18 = bf16[16,16,256,1]{3,2,1,0} reshape(reduce.17)
      broadcast.19 = bf16[16,16,256,1]{3,2,1,0} broadcast(reshape.18), dimensions={0,1,2,3}
      reshape.20 = bf16[16,16,256]{2,1,0} reshape(broadcast.19)
      broadcast.21 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.20), dimensions={0,1,2}
      subtract.22 = bf16[16,16,256,256]{3,2,1,0} subtract(add.12, broadcast.21)
      exponential.23 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.22)
      convert.24 = f32[16,16,256,256]{3,2,1,0} convert(exponential.23)
      constant.6 = f32[] constant(0)
      reduce.29 = f32[16,16,256]{2,1,0} reduce(convert.24, constant.6), dimensions={3}, to_apply=region_1.25
      reshape.30 = f32[16,16,256,1]{3,2,1,0} reshape(reduce.29)
      convert.31 = bf16[16,16,256,1]{3,2,1,0} convert(reshape.30)
      broadcast.32 = bf16[16,16,256,1]{3,2,1,0} broadcast(convert.31), dimensions={0,1,2,3}
      reshape.33 = bf16[16,16,256]{2,1,0} reshape(broadcast.32)
      broadcast.34 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.33), dimensions={0,1,2}
      divide.35 = bf16[16,16,256,256]{3,2,1,0} divide(exponential.23, broadcast.34)
      Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
      ROOT dot.36 = bf16[16,16,256,64]{3,2,1,0} dot(divide.35, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";

    return hlo_text;
  }

  // BMM1 - Scale - bias - Softmax - BMM2
  template <typename T>
  void TestImpl_FMHABMM1_Scale_Bias_Softmax_BMM2_vanilla() {
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    if (!(cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0)) {
      GTEST_SKIP() << "Fused MHA is supported with the Nvidia AMPERE+ GPUs.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string = GetModuleFMHABMM1_Scale_Bias_Softmax_BMM2_HloString_F16();
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string = GetModuleFMHABMM1_Scale_Bias_Softmax_BMM2_HloString_BF16();
    }

    ExecuteAndCompare(hlo_string, lhs_bmm1_literal, rhs_bmm1_literal,
                      rhs_bmm2_literal);
  }
};

// BMM1 - BMM2
XLA_TEST_F(MultiHeadedAttentionBMMBMM, FMHABMM_BMM_vanilla_F16) {
  TestImpl_FMHABMM_BMM_vanilla<Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMBMM, FMHABMM_BMM_vanilla_BF16) {
  TestImpl_FMHABMM_BMM_vanilla<bfloat16>();
}

XLA_TEST_F(MultiHeadedAttentionBMMBMM, FMHABMM_BMM_arg_reversal_F16) {
  TestImpl_FMHABMM_BMM_arg_reversal<Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMBMM, FMHABMM_BMM_arg_reversal_BF16) {
  TestImpl_FMHABMM_BMM_arg_reversal<bfloat16>();
}

XLA_TEST_F(MultiHeadedAttentionBMMBMM,
           FMHABMM_BMM_arg_layout_manipulation_arg_reversal_F16) {
  TestImpl_FMHABMM_BMM_arg_layout_manipulation_arg_reversal_fusion<
      Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMBMM,
           FMHABMM_BMM_arg_layout_manipulation_arg_reversal_fusion_BF16) {
  TestImpl_FMHABMM_BMM_arg_layout_manipulation_arg_reversal_fusion<bfloat16>();
}

XLA_TEST_F(MultiHeadedAttentionBMMBMM, FMHABMM_BMM_all_canonicalization_F16) {
  TestImpl_FMHABMM_BMM_all_canonicalization<Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMBMM, FMHABMM_BMM_all_canonicalization_BF16) {
  TestImpl_FMHABMM_BMM_all_canonicalization<bfloat16>();
}

XLA_TEST_F(MultiHeadedAttentionBMMBMM,
           FMHABMM_BMM_BMM1_contracting_dim_stride_not_1_F16) {
  TestImpl_BMM_BMM1_contracting_dim_stride_not_1<Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMBMM,
           FMHABMM_BMM_BMM2_non_contracting_dim_stride_not_1_F16) {
  TestImpl_BMM_BMM2_non_contracting_dim_stride_not_1<Eigen::half>();
}

// BMM1 - Scale - Bias - Mask - Softmax - BMM2
XLA_TEST_F(MultiHeadedAttentionBMMScaleBiasMaskSoftmaxBMM,
           FMHABMM1_Scale_Bias_Mask_Softmax_BMM2_vanilla_F16) {
  TestImpl_FMHABMM1_Scale_Bias_Mask_Softmax_BMM2_vanilla<Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMScaleBiasMaskSoftmaxBMM,
           FMHABMM1_Scale_Bias_Mask_Softmax_BMM2_vanilla_BF16) {
  TestImpl_FMHABMM1_Scale_Bias_Mask_Softmax_BMM2_vanilla<bfloat16>();
}

XLA_TEST_F(MultiHeadedAttentionBMMScaleBiasMaskSoftmaxBMM,
           FMHABMM1_Scale_Bias_Mask_Softmax_BMM2_vanilla_BF16_smaller) {
  TestImpl_FMHABMM1_Scale_Bias_Mask_Softmax_BMM2_vanilla_smaller<bfloat16>();
}

XLA_TEST_F(MultiHeadedAttentionBMMScaleBiasMaskSoftmaxBMM,
           FMHABMM1_Scale_Bias_Mask_Softmax_BMM2_arg_reversal_F16) {
  TestImpl_FMHABMM1_Scale_Bias_Mask_Softmax_BMM2_arg_reversal<Eigen::half>();
}

// BMM1 - Scale - Mask - Softmax - BMM2
XLA_TEST_F(MultiHeadedAttentionBMMScaleMaskSoftmaxBMM,
           FMHABMM1_Scale_Mask_Softmax_BMM2_vanilla_F16) {
  TestImpl_FMHABMM1_Scale_Mask_Softmax_BMM2_vanilla<Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMScaleMaskSoftmaxBMM,
           FMHABMM1_Scale_Mask_Softmax_BMM2_vanilla_BF16) {
  TestImpl_FMHABMM1_Scale_Mask_Softmax_BMM2_vanilla<bfloat16>();
}

XLA_TEST_F(MultiHeadedAttentionBMMScaleMaskSoftmaxBMM,
           FMHABMM1_Scale_Mask_Softmax_BMM2_arg_reversal_F16) {
  TestImpl_FMHABMM1_Scale_Mask_Softmax_BMM2_arg_reversal<Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMScaleMaskSoftmaxBMM,
           FMHABMM1_Scale_Mask_Softmax_BMM2_arg_reversal_BF16) {
  TestImpl_FMHABMM1_Scale_Mask_Softmax_BMM2_arg_reversal<bfloat16>();
}

// BMM1 - Softmax - BMM2
XLA_TEST_F(MultiHeadedAttentionBMMSoftmaxBMM,
           FMHABMM1_Softmax_BMM2_vanilla_F16) {
  TestImpl_FMHABMM1_Softmax_BMM2_vanilla<Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMSoftmaxBMM,
           FMHABMM1_Softmax_BMM2_vanilla_BF16) {
  TestImpl_FMHABMM1_Softmax_BMM2_vanilla<Eigen::half>();
}

// BMM1 - Scale - bias - Softmax - BMM2
XLA_TEST_F(MultiHeadedAttentionBMMScaleBiasSoftmaxBMM,
           FMHABMM1_Scale_Bias_Softmax_BMM2_vanilla_F16) {
  TestImpl_FMHABMM1_Scale_Bias_Softmax_BMM2_vanilla<Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMScaleBiasSoftmaxBMM,
           FMHABMM1_Scale_Bias_Softmax_BMM2_vanilla_BF16) {
  TestImpl_FMHABMM1_Scale_Bias_Softmax_BMM2_vanilla<bfloat16>();
}

}  // namespace
}  // namespace xla
