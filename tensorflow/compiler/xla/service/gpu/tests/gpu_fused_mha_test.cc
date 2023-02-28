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

  ErrorSpec error_spec_{0.0001, 1e-5};
};

class MultiHeadedAttentionBMMBMM : public MultiHeadedAttentionTest {
 protected:
  DebugOptions GetDebugOptionsForTest() override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_xla_runtime_executable(false);
    return debug_options;
  }

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
};

XLA_TEST_F(MultiHeadedAttentionBMMBMM, FMHABMM_BMM_vanilla_F16) {
  TestImpl_FMHABMM_BMM_vanilla<Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMBMM, FMHABMM_BMM_vanilla_BF16) {
  TestImpl_FMHABMM_BMM_vanilla<bfloat16>();
}

}
} 
