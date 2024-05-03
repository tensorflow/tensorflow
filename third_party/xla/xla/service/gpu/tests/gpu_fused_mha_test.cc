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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/array4d.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

namespace xla {
namespace gpu {
namespace {

class MultiHeadedAttentionTest : public GpuCodegenTest {
 public:
  MultiHeadedAttentionTest() {
#if !defined(GOOGLE_CUDA) || CUDA_VERSION < 12000
    skip_reason_ = "cuDNN Fused MHA requires CUDA 12 or later.";
    return;
#endif
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    // Enforce capability minor == 0 because hardware with a non-zero minor
    // number typically has insufficient shared memory for cuDNN FMHA.
    if (!cc.IsAtLeastAmpere() || cc.minor != 0) {
      skip_reason_ =
          "cuDNN Fused MHA requires Nvidia AMPERE+ GPUs with minor "
          "compute capability == 0.";
      return;
    }
    if (GetDnnVersionInfo(backend().default_stream_executor()) <
        se::dnn::VersionInfo(8, 8, 0)) {
      skip_reason_ = "cuDNN Fused MHA requires cuDNN 8.8.0 or later.";
      return;
    }
  }

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
    debug_options.set_xla_gpu_enable_cudnn_fmha(false);
    return debug_options;
  }

  absl::StatusOr<int> CountFMHACalls(absl::string_view hlo_string,
                                     const HloModuleConfig &config) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> verified_module,
                        ParseAndReturnVerifiedModule(hlo_string, config));

    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_verified_module,
                        GetOptimizedModule(std::move(verified_module)));

    return absl::c_count_if(
        optimized_verified_module->entry_computation()->instructions(),
        [&](const HloInstruction *inst) {
          return inst->opcode() == HloOpcode::kCustomCall &&
                 absl::StrContains(inst->custom_call_target(), "__cudnn$fmha");
        });
  }

  void ExecuteAndCompare(absl::string_view hlo_string,
                         const std::vector<Literal *> &literals,
                         int expected_num_fmha_calls = 1) {
    HloModuleConfig config;
    DebugOptions debug_options = GetDebugOptionsForTest();
    config.set_debug_options(debug_options);
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_string, config));
    auto expected_result = ExecuteAndTransfer(std::move(module), literals);

    // Sanity check to ensure the first computation doesn't use FMHA.
    TF_ASSERT_OK_AND_ASSIGN(int num_fmha_calls,
                            CountFMHACalls(hlo_string, config));
    EXPECT_EQ(num_fmha_calls, 0);

    debug_options.set_xla_gpu_enable_cudnn_fmha(true);
    HloModuleConfig config_with_fmha;
    config_with_fmha.set_debug_options(debug_options);

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> new_module,
        ParseAndReturnVerifiedModule(hlo_string, config_with_fmha));
    auto actual_result = ExecuteAndTransfer(std::move(new_module), literals);
    EXPECT_TRUE(
        LiteralTestUtil::Near(expected_result, actual_result, error_spec_));

    TF_ASSERT_OK_AND_ASSIGN(num_fmha_calls,
                            CountFMHACalls(hlo_string, config_with_fmha));
    EXPECT_EQ(num_fmha_calls, expected_num_fmha_calls);
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

  // Centralize skip checks in the constructor. Unfortunately we cannot call
  // GTEST_SKIP from the constructor. Instead, we set (if needed) `skip_reason`,
  // and then check it from all test fixtures.
  // An alternative would be to use the SetUp() override, but for this to be
  // correct we'd have to ensure that all the parents' SetUp() methods are
  // called, which is error prone.
  std::optional<absl::string_view> skip_reason_;
};

class MultiHeadedAttentionBMMBMM : public MultiHeadedAttentionTest {
 protected:
  std::string GetModuleFMHABMM_BMM_vanilla_HloString_F16() {
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

  std::string GetModuleFMHABMM_BMM_vanilla_HloString_BF16() {
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

  const std::string                                    // NOLINT
  GetModuleFMHABMM_BMM_arg_reversal_HloString_F16() {  // NOLINT
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

  const std::string                                     // NOLINT
  GetModuleFMHABMM_BMM_arg_reversal_HloString_BF16() {  // NOLINT
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

  std::string
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

  std::string
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

  const std::string  // NOLINT
  GetModuleFMHABMM_BMM_arg_reversal_epilogue_transpose_fusion_HloString_F16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0})->f16[16,256,16,64]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
      Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.0 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      dot.1 = f16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      ROOT transpose.0 = f16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}, metadata={}
    }
  )";
    return hlo_text;
  }

  const std::string  // NOLINT
  GetModuleFMHABMM_BMM_arg_layout_manipulation_arg_reversal_prologue_transpose_fusion_HloString_F16() {  // NOLINT
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

  const std::string  // NOLINT
  GetModuleFMHABMM_BMM_arg_layout_manipulation_arg_reversal_prologue_transpose_fusion_HloString_BF16() {  // NOLINT
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

  const std::string                                             // NOLINT
  GetModuleFMHABMM_BMM_all_canonicalization_HloString_BF16() {  // NOLINT
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

  const std::string                                            // NOLINT
  GetModuleFMHABMM_BMM_all_canonicalization_HloString_F16() {  // NOLINT
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

  const std::string  // NOLINT
  GetModuleFMHABMM_BMM_all_canonicalization_transpose_fusion_HloString_F16() {  // NOLINT
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

  const std::string  // NOLINT
  GetModuleFMHABMM_BMM_all_canonicalization_transpose_fusion_HloString_BF16() {  // NOLINT
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

  const std::string  // NOLINT
  GetModuleFMHABMM_BMM_all_canonicalization_transpose_fusion_HloString_F16_small() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(f16[2,4,2,64]{3,2,1,0},f16[2,4,2,64]{3,2,1,0},f16[2,4,2,64]{3,2,1,0})->f16[2,4,2,64]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = f16[2,4,2,64]{3,2,1,0} parameter(2)
      Arg_0.1 = f16[2,4,2,64]{3,2,1,0} parameter(0)
      Arg_1.2 = f16[2,4,2,64]{3,2,1,0} parameter(1)
      dot.0 = f16[2,2,4,4]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}, metadata={}
      dot.1 = f16[2,2,64,4]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,2}, lhs_contracting_dims={1}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      ROOT transpose.0 = f16[2,4,2,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}, metadata={}
    }
  )";
    return hlo_text;
  }

  const std::string  // NOLINT
  GetModuleFMHABMM_BMM1_contracting_dim_stride_not_1_HloString_F16() {  // NOLINT
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

  std::string
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

  const std::string  // NOLINT
  GetModuleFMHABMM_BMM_arg_reversal_epilogue_transpose_fusion_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_.10, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,256,16,64]{3,2,1,0}}

    ENTRY main.15 {
      Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
      Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
      dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
      ROOT transpose.0 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}, metadata={}
    }
  )";
    return hlo_text;
  }

  template <typename T>
  void TestImpl_FMHABMM_BMM_vanilla() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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

    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }

  template <typename T>
  void TestImpl_FMHABMM_BMM_arg_reversal() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }

  template <typename T>
  void TestImpl_FMHABMM_BMM_arg_layout_manipulation_arg_reversal_fusion() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }

  template <typename T>
  void TestImpl_FMHABMM_BMM_arg_reversal_epilogue_transpose_fusion() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({16, 16, 256, 64}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string =
          GetModuleFMHABMM_BMM_arg_reversal_epilogue_transpose_fusion_HloString_F16();  // NOLINT
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string =
          GetModuleFMHABMM_BMM_arg_reversal_epilogue_transpose_fusion_HloString_BF16();  // NOLINT
    }
    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }

  template <typename T>
  void
  TestImpl_FMHABMM_BMM_arg_layout_manipulation_arg_reversal_prologue_transpose_fusion() {  // NOLINT
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
          GetModuleFMHABMM_BMM_arg_layout_manipulation_arg_reversal_prologue_transpose_fusion_HloString_F16();  // NOLINT
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string =
          GetModuleFMHABMM_BMM_arg_layout_manipulation_arg_reversal_prologue_transpose_fusion_HloString_BF16();  // NOLINT
    }
    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }

  template <typename T>
  void TestImpl_FMHABMM_BMM_all_canonicalization_transpose_fusion() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
          GetModuleFMHABMM_BMM_all_canonicalization_transpose_fusion_HloString_F16();  // NOLINT
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string =
          GetModuleFMHABMM_BMM_all_canonicalization_transpose_fusion_HloString_BF16();  // NOLINT
    }
    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }

  template <typename T>
  void TestImpl_FMHABMM_BMM_all_canonicalization() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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
    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }

  template <typename T>
  void TestImpl_FMHABMM_BMM_all_canonicalization_transpose_fusion_small() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal = GetInput4DLiteral<T>({2, 4, 2, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal = GetInput4DLiteral<T>({2, 4, 2, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal = GetInput4DLiteral<T>({2, 4, 2, 64}, {3, 2, 1, 0});

    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string =
          GetModuleFMHABMM_BMM_all_canonicalization_transpose_fusion_HloString_F16_small();  // NOLINT
    }
    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }

  template <typename T>
  void TestImpl_BMM_BMM1_contracting_dim_stride_not_1() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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

    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }

  template <typename T>
  void TestImpl_BMM_BMM2_non_contracting_dim_stride_not_1() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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

    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }
};

// Bmm1 - Softmax - Bmm2
class MultiHeadedAttentionBMMSoftmaxBMM : public MultiHeadedAttentionTest {
 protected:
  std::string GetModuleFMHABMM1_Softmax_BMM2_HloString_F16() {
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

  std::string GetModuleFMHABMM1_Softmax_BMM2_HloString_BF16() {
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
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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

    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }
};

class MultiHeadedAttentionBMMScaleBiasSoftmaxBMM
    : public MultiHeadedAttentionTest {
 protected:
  // Bmm1 - Scale - Bias - Softmax - Bmm2
  std::string GetModuleFMHABMM1_Scale_Bias_Softmax_BMM2_HloString_F16() {
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

  const std::string  // NOLINT
  GetModuleFMHA_Training_BMM1_Scale_Bias_Softmax_BMM2_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[8,128,8,64]{3,2,1,0},bf16[8,128,8,64]{3,2,1,0},bf16[8,128,8,64]{3,2,1,0},bf16[1,8,128,128]{3,2,1,0},pred[8,1,128,128]{3,2,1,0},bf16[8,128,8,64]{3,2,1,0})->(bf16[8,128,8,64]{3,2,1,0}, bf16[8,128,8,64]{3,2,1,0}, bf16[8,128,8,64]{3,2,1,0}, bf16[8,128,8,64]{3,2,1,0}, bf16[1,8,128,128]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true,true}

    region_0.35 {
      Arg_0.36 = bf16[] parameter(0)
      Arg_1.37 = bf16[] parameter(1)
      ROOT maximum.38 = bf16[] maximum(Arg_0.36, Arg_1.37)
    }

    region_1.47 {
      Arg_0.48 = f32[] parameter(0)
      Arg_1.49 = f32[] parameter(1)
      ROOT add.50 = f32[] add(Arg_0.48, Arg_1.49)
    }

    region_2.71 {
      Arg_0.72 = bf16[] parameter(0)
      Arg_1.73 = bf16[] parameter(1)
      ROOT add.74 = bf16[] add(Arg_0.72, Arg_1.73)
    }

    region_3.83 {
      Arg_0.84 = f32[] parameter(0)
      Arg_1.85 = f32[] parameter(1)
      ROOT add.86 = f32[] add(Arg_0.84, Arg_1.85)
    }

    region_4.92 {
      Arg_0.93 = bf16[] parameter(0)
      Arg_1.94 = bf16[] parameter(1)
      ROOT add.95 = bf16[] add(Arg_0.93, Arg_1.94)
    }

    ENTRY main.104 {
      Arg_2.3 = bf16[8,128,8,64]{3,2,1,0} parameter(2)
      Arg_0.1 = bf16[8,128,8,64]{3,2,1,0} parameter(0)
      constant.7 = bf16[] constant(2)
      broadcast.8 = bf16[8,128,8,64]{3,2,1,0} broadcast(constant.7), dimensions={}
      divide.32 = bf16[8,128,8,64]{3,2,1,0} divide(Arg_0.1, broadcast.8)
      Arg_1.2 = bf16[8,128,8,64]{3,2,1,0} parameter(1)
      dot.33 = bf16[8,8,128,128]{3,2,1,0} dot(divide.32, Arg_1.2), lhs_batch_dims={0,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}
      Arg_4.5 = pred[8,1,128,128]{3,2,1,0} parameter(4)
      convert.20 = s32[8,1,128,128]{3,2,1,0} convert(Arg_4.5)
      constant.15 = s32[] constant(0)
      broadcast.16 = s32[8,1,128,128]{3,2,1,0} broadcast(constant.15), dimensions={}
      compare.21 = pred[8,1,128,128]{3,2,1,0} compare(convert.20, broadcast.16), direction=GT
      constant.13 = f32[] constant(0)
      broadcast.14 = f32[8,1,128,128]{3,2,1,0} broadcast(constant.13), dimensions={}
      convert.22 = bf16[8,1,128,128]{3,2,1,0} convert(broadcast.14)
      constant.11 = f32[] constant(-1e+10)
      broadcast.12 = f32[8,1,128,128]{3,2,1,0} broadcast(constant.11), dimensions={}
      convert.23 = bf16[8,1,128,128]{3,2,1,0} convert(broadcast.12)
      select.24 = bf16[8,1,128,128]{3,2,1,0} select(compare.21, convert.22, convert.23)
      broadcast.25 = bf16[8,1,128,128]{3,2,1,0} broadcast(select.24), dimensions={0,1,2,3}
      reshape.26 = bf16[8,128,128]{2,1,0} reshape(broadcast.25)
      broadcast.27 = bf16[8,8,128,128]{3,2,1,0} broadcast(reshape.26), dimensions={0,2,3}
      Arg_3.4 = bf16[1,8,128,128]{3,2,1,0} parameter(3)
      broadcast.28 = bf16[1,8,128,128]{3,2,1,0} broadcast(Arg_3.4), dimensions={0,1,2,3}
      reshape.29 = bf16[8,128,128]{2,1,0} reshape(broadcast.28)
      broadcast.30 = bf16[8,8,128,128]{3,2,1,0} broadcast(reshape.29), dimensions={1,2,3}
      add.31 = bf16[8,8,128,128]{3,2,1,0} add(broadcast.27, broadcast.30)
      add.34 = bf16[8,8,128,128]{3,2,1,0} add(dot.33, add.31)
      constant.18 = bf16[] constant(-inf)
      reduce.39 = bf16[8,8,128]{2,1,0} reduce(add.34, constant.18), dimensions={3}, to_apply=region_0.35
      reshape.40 = bf16[8,8,128,1]{3,2,1,0} reshape(reduce.39)
      broadcast.41 = bf16[8,8,128,1]{3,2,1,0} broadcast(reshape.40), dimensions={0,1,2,3}
      reshape.42 = bf16[8,8,128]{2,1,0} reshape(broadcast.41)
      broadcast.43 = bf16[8,8,128,128]{3,2,1,0} broadcast(reshape.42), dimensions={0,1,2}
      subtract.44 = bf16[8,8,128,128]{3,2,1,0} subtract(add.34, broadcast.43)
      exponential.45 = bf16[8,8,128,128]{3,2,1,0} exponential(subtract.44)
      convert.46 = f32[8,8,128,128]{3,2,1,0} convert(exponential.45)
      constant.19 = f32[] constant(0)
      reduce.51 = f32[8,8,128]{2,1,0} reduce(convert.46, constant.19), dimensions={3}, to_apply=region_1.47
      reshape.52 = f32[8,8,128,1]{3,2,1,0} reshape(reduce.51)
      convert.53 = bf16[8,8,128,1]{3,2,1,0} convert(reshape.52)
      broadcast.54 = bf16[8,8,128,1]{3,2,1,0} broadcast(convert.53), dimensions={0,1,2,3}
      reshape.55 = bf16[8,8,128]{2,1,0} reshape(broadcast.54)
      broadcast.56 = bf16[8,8,128,128]{3,2,1,0} broadcast(reshape.55), dimensions={0,1,2}
      divide.57 = bf16[8,8,128,128]{3,2,1,0} divide(exponential.45, broadcast.56)
      dot.60 = bf16[8,8,64,128]{3,2,1,0} dot(Arg_2.3, divide.57), lhs_batch_dims={0,2}, lhs_contracting_dims={1}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      transpose.61 = bf16[8,128,8,64]{1,3,2,0} transpose(dot.60), dimensions={0,3,1,2}
      Arg_5.6 = bf16[8,128,8,64]{3,2,1,0} parameter(5)
      transpose.62 = bf16[8,8,64,128]{2,1,3,0} transpose(Arg_5.6), dimensions={0,2,3,1}
      dot.63 = bf16[8,8,128,128]{3,2,1,0} dot(transpose.62, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}
      broadcast.78 = bf16[8,8,128,1]{3,2,1,0} broadcast(convert.53), dimensions={0,1,2,3}
      reshape.79 = bf16[8,8,128]{2,1,0} reshape(broadcast.78)
      broadcast.80 = bf16[8,8,128,128]{3,2,1,0} broadcast(reshape.79), dimensions={0,1,2}
      divide.81 = bf16[8,8,128,128]{3,2,1,0} divide(dot.63, broadcast.80)
      constant.9 = bf16[] constant(1)
      broadcast.10 = bf16[8,8,128,1]{3,2,1,0} broadcast(constant.9), dimensions={}
      multiply.58 = bf16[8,8,128,1]{3,2,1,0} multiply(convert.53, convert.53)
      divide.59 = bf16[8,8,128,1]{3,2,1,0} divide(broadcast.10, multiply.58)
      broadcast.66 = bf16[8,8,128,1]{3,2,1,0} broadcast(divide.59), dimensions={0,1,2,3}
      reshape.67 = bf16[8,8,128]{2,1,0} reshape(broadcast.66)
      broadcast.68 = bf16[8,8,128,128]{3,2,1,0} broadcast(reshape.67), dimensions={0,1,2}
      multiply.69 = bf16[8,8,128,128]{3,2,1,0} multiply(dot.63, broadcast.68)
      multiply.70 = bf16[8,8,128,128]{3,2,1,0} multiply(multiply.69, exponential.45)
      constant.17 = bf16[] constant(0)
      reduce.75 = bf16[8,8,128]{2,1,0} reduce(multiply.70, constant.17), dimensions={3}, to_apply=region_2.71
      reshape.76 = bf16[8,8,128,1]{3,2,1,0} reshape(reduce.75)
      negate.77 = bf16[8,8,128,1]{3,2,1,0} negate(reshape.76)
      convert.82 = f32[8,8,128,1]{3,2,1,0} convert(negate.77)
      reduce.87 = f32[8,8,128]{2,1,0} reduce(convert.82, constant.19), dimensions={3}, to_apply=region_3.83
      broadcast.88 = f32[8,8,128,128]{3,2,1,0} broadcast(reduce.87), dimensions={0,1,2}
      convert.89 = bf16[8,8,128,128]{3,2,1,0} convert(broadcast.88)
      add.90 = bf16[8,8,128,128]{3,2,1,0} add(divide.81, convert.89)
      multiply.91 = bf16[8,8,128,128]{3,2,1,0} multiply(add.90, exponential.45)
      dot.100 = bf16[8,8,128,64]{3,2,1,0} dot(multiply.91, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={1}
      transpose.101 = bf16[8,128,8,64]{3,1,2,0} transpose(dot.100), dimensions={0,2,1,3}
      divide.102 = bf16[8,128,8,64]{3,1,2,0} divide(transpose.101, broadcast.8)
      dot.98 = bf16[8,8,128,64]{3,2,1,0} dot(multiply.91, divide.32), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,2}, rhs_contracting_dims={1}
      transpose.99 = bf16[8,128,8,64]{3,1,2,0} transpose(dot.98), dimensions={0,2,1,3}
      dot.64 = bf16[8,8,64,128]{3,2,1,0} dot(transpose.62, divide.57), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      transpose.65 = bf16[8,128,8,64]{1,3,2,0} transpose(dot.64), dimensions={0,3,1,2}
      reduce.96 = bf16[8,128,128]{2,1,0} reduce(multiply.91, constant.17), dimensions={0}, to_apply=region_4.92
      reshape.97 = bf16[1,8,128,128]{3,2,1,0} reshape(reduce.96)
      ROOT tuple.103 = (bf16[8,128,8,64]{1,3,2,0}, bf16[8,128,8,64]{3,1,2,0}, bf16[8,128,8,64]{3,1,2,0}, bf16[8,128,8,64]{1,3,2,0}, bf16[1,8,128,128]{3,2,1,0}) tuple(transpose.61, divide.102, transpose.99, transpose.65, reshape.97)
    }
  )";

    return hlo_text;
  }

  const std::string  // NOLINT
  GetModuleFMHA_Training_BMM1_Scale_Bias_Softmax_BMM2_HloString_F16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[8,128,8,64]{3,2,1,0},f16[8,128,8,64]{3,2,1,0},f16[8,128,8,64]{3,2,1,0},f16[1,8,128,128]{3,2,1,0},pred[8,1,128,128]{3,2,1,0},f16[8,128,8,64]{3,2,1,0})->(f16[8,128,8,64]{3,2,1,0}, f16[8,128,8,64]{3,2,1,0}, f16[8,128,8,64]{3,2,1,0}, f16[8,128,8,64]{3,2,1,0}, f16[1,8,128,128]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true,true}

    region_0.35 {
      Arg_0.36 = f16[] parameter(0)
      Arg_1.37 = f16[] parameter(1)
      ROOT maximum.38 = f16[] maximum(Arg_0.36, Arg_1.37)
    }

    region_1.47 {
      Arg_0.48 = f32[] parameter(0)
      Arg_1.49 = f32[] parameter(1)
      ROOT add.50 = f32[] add(Arg_0.48, Arg_1.49)
    }

    region_2.71 {
      Arg_0.72 = f16[] parameter(0)
      Arg_1.73 = f16[] parameter(1)
      ROOT add.74 = f16[] add(Arg_0.72, Arg_1.73)
    }

    region_3.83 {
      Arg_0.84 = f32[] parameter(0)
      Arg_1.85 = f32[] parameter(1)
      ROOT add.86 = f32[] add(Arg_0.84, Arg_1.85)
    }

    region_4.92 {
      Arg_0.93 = f16[] parameter(0)
      Arg_1.94 = f16[] parameter(1)
      ROOT add.95 = f16[] add(Arg_0.93, Arg_1.94)
    }

    ENTRY main.104 {
      Arg_2.3 = f16[8,128,8,64]{3,2,1,0} parameter(2)
      Arg_0.1 = f16[8,128,8,64]{3,2,1,0} parameter(0)
      constant.7 = f16[] constant(2)
      broadcast.8 = f16[8,128,8,64]{3,2,1,0} broadcast(constant.7), dimensions={}
      divide.32 = f16[8,128,8,64]{3,2,1,0} divide(Arg_0.1, broadcast.8)
      Arg_1.2 = f16[8,128,8,64]{3,2,1,0} parameter(1)
      dot.33 = f16[8,8,128,128]{3,2,1,0} dot(divide.32, Arg_1.2), lhs_batch_dims={0,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}
      Arg_4.5 = pred[8,1,128,128]{3,2,1,0} parameter(4)
      convert.20 = s32[8,1,128,128]{3,2,1,0} convert(Arg_4.5)
      constant.15 = s32[] constant(0)
      broadcast.16 = s32[8,1,128,128]{3,2,1,0} broadcast(constant.15), dimensions={}
      compare.21 = pred[8,1,128,128]{3,2,1,0} compare(convert.20, broadcast.16), direction=GT
      constant.13 = f32[] constant(0)
      broadcast.14 = f32[8,1,128,128]{3,2,1,0} broadcast(constant.13), dimensions={}
      convert.22 = f16[8,1,128,128]{3,2,1,0} convert(broadcast.14)
      constant.11 = f32[] constant(-1e+10)
      broadcast.12 = f32[8,1,128,128]{3,2,1,0} broadcast(constant.11), dimensions={}
      convert.23 = f16[8,1,128,128]{3,2,1,0} convert(broadcast.12)
      select.24 = f16[8,1,128,128]{3,2,1,0} select(compare.21, convert.22, convert.23)
      broadcast.25 = f16[8,1,128,128]{3,2,1,0} broadcast(select.24), dimensions={0,1,2,3}
      reshape.26 = f16[8,128,128]{2,1,0} reshape(broadcast.25)
      broadcast.27 = f16[8,8,128,128]{3,2,1,0} broadcast(reshape.26), dimensions={0,2,3}
      Arg_3.4 = f16[1,8,128,128]{3,2,1,0} parameter(3)
      broadcast.28 = f16[1,8,128,128]{3,2,1,0} broadcast(Arg_3.4), dimensions={0,1,2,3}
      reshape.29 = f16[8,128,128]{2,1,0} reshape(broadcast.28)
      broadcast.30 = f16[8,8,128,128]{3,2,1,0} broadcast(reshape.29), dimensions={1,2,3}
      add.31 = f16[8,8,128,128]{3,2,1,0} add(broadcast.27, broadcast.30)
      add.34 = f16[8,8,128,128]{3,2,1,0} add(dot.33, add.31)
      constant.18 = f16[] constant(-inf)
      reduce.39 = f16[8,8,128]{2,1,0} reduce(add.34, constant.18), dimensions={3}, to_apply=region_0.35
      reshape.40 = f16[8,8,128,1]{3,2,1,0} reshape(reduce.39)
      broadcast.41 = f16[8,8,128,1]{3,2,1,0} broadcast(reshape.40), dimensions={0,1,2,3}
      reshape.42 = f16[8,8,128]{2,1,0} reshape(broadcast.41)
      broadcast.43 = f16[8,8,128,128]{3,2,1,0} broadcast(reshape.42), dimensions={0,1,2}
      subtract.44 = f16[8,8,128,128]{3,2,1,0} subtract(add.34, broadcast.43)
      exponential.45 = f16[8,8,128,128]{3,2,1,0} exponential(subtract.44)
      convert.46 = f32[8,8,128,128]{3,2,1,0} convert(exponential.45)
      constant.19 = f32[] constant(0)
      reduce.51 = f32[8,8,128]{2,1,0} reduce(convert.46, constant.19), dimensions={3}, to_apply=region_1.47
      reshape.52 = f32[8,8,128,1]{3,2,1,0} reshape(reduce.51)
      convert.53 = f16[8,8,128,1]{3,2,1,0} convert(reshape.52)
      broadcast.54 = f16[8,8,128,1]{3,2,1,0} broadcast(convert.53), dimensions={0,1,2,3}
      reshape.55 = f16[8,8,128]{2,1,0} reshape(broadcast.54)
      broadcast.56 = f16[8,8,128,128]{3,2,1,0} broadcast(reshape.55), dimensions={0,1,2}
      divide.57 = f16[8,8,128,128]{3,2,1,0} divide(exponential.45, broadcast.56)
      dot.60 = f16[8,8,64,128]{3,2,1,0} dot(Arg_2.3, divide.57), lhs_batch_dims={0,2}, lhs_contracting_dims={1}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      transpose.61 = f16[8,128,8,64]{1,3,2,0} transpose(dot.60), dimensions={0,3,1,2}
      Arg_5.6 = f16[8,128,8,64]{3,2,1,0} parameter(5)
      transpose.62 = f16[8,8,64,128]{2,1,3,0} transpose(Arg_5.6), dimensions={0,2,3,1}
      dot.63 = f16[8,8,128,128]{3,2,1,0} dot(transpose.62, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}
      broadcast.78 = f16[8,8,128,1]{3,2,1,0} broadcast(convert.53), dimensions={0,1,2,3}
      reshape.79 = f16[8,8,128]{2,1,0} reshape(broadcast.78)
      broadcast.80 = f16[8,8,128,128]{3,2,1,0} broadcast(reshape.79), dimensions={0,1,2}
      divide.81 = f16[8,8,128,128]{3,2,1,0} divide(dot.63, broadcast.80)
      constant.9 = f16[] constant(1)
      broadcast.10 = f16[8,8,128,1]{3,2,1,0} broadcast(constant.9), dimensions={}
      multiply.58 = f16[8,8,128,1]{3,2,1,0} multiply(convert.53, convert.53)
      divide.59 = f16[8,8,128,1]{3,2,1,0} divide(broadcast.10, multiply.58)
      broadcast.66 = f16[8,8,128,1]{3,2,1,0} broadcast(divide.59), dimensions={0,1,2,3}
      reshape.67 = f16[8,8,128]{2,1,0} reshape(broadcast.66)
      broadcast.68 = f16[8,8,128,128]{3,2,1,0} broadcast(reshape.67), dimensions={0,1,2}
      multiply.69 = f16[8,8,128,128]{3,2,1,0} multiply(dot.63, broadcast.68)
      multiply.70 = f16[8,8,128,128]{3,2,1,0} multiply(multiply.69, exponential.45)
      constant.17 = f16[] constant(0)
      reduce.75 = f16[8,8,128]{2,1,0} reduce(multiply.70, constant.17), dimensions={3}, to_apply=region_2.71
      reshape.76 = f16[8,8,128,1]{3,2,1,0} reshape(reduce.75)
      negate.77 = f16[8,8,128,1]{3,2,1,0} negate(reshape.76)
      convert.82 = f32[8,8,128,1]{3,2,1,0} convert(negate.77)
      reduce.87 = f32[8,8,128]{2,1,0} reduce(convert.82, constant.19), dimensions={3}, to_apply=region_3.83
      broadcast.88 = f32[8,8,128,128]{3,2,1,0} broadcast(reduce.87), dimensions={0,1,2}
      convert.89 = f16[8,8,128,128]{3,2,1,0} convert(broadcast.88)
      add.90 = f16[8,8,128,128]{3,2,1,0} add(divide.81, convert.89)
      multiply.91 = f16[8,8,128,128]{3,2,1,0} multiply(add.90, exponential.45)
      dot.100 = f16[8,8,128,64]{3,2,1,0} dot(multiply.91, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={1}
      transpose.101 = f16[8,128,8,64]{3,1,2,0} transpose(dot.100), dimensions={0,2,1,3}
      divide.102 = f16[8,128,8,64]{3,1,2,0} divide(transpose.101, broadcast.8)
      dot.98 = f16[8,8,128,64]{3,2,1,0} dot(multiply.91, divide.32), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,2}, rhs_contracting_dims={1}
      transpose.99 = f16[8,128,8,64]{3,1,2,0} transpose(dot.98), dimensions={0,2,1,3}
      dot.64 = f16[8,8,64,128]{3,2,1,0} dot(transpose.62, divide.57), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      transpose.65 = f16[8,128,8,64]{1,3,2,0} transpose(dot.64), dimensions={0,3,1,2}
      reduce.96 = f16[8,128,128]{2,1,0} reduce(multiply.91, constant.17), dimensions={0}, to_apply=region_4.92
      reshape.97 = f16[1,8,128,128]{3,2,1,0} reshape(reduce.96)
      ROOT tuple.103 = (f16[8,128,8,64]{1,3,2,0}, f16[8,128,8,64]{3,1,2,0}, f16[8,128,8,64]{3,1,2,0}, f16[8,128,8,64]{1,3,2,0}, f16[1,8,128,128]{3,2,1,0}) tuple(transpose.61, divide.102, transpose.99, transpose.65, reshape.97)
    }
  )";

    return hlo_text;
  }

  const std::string                                             // NOLINT
  GetModuleFMHABMM1_Scale_Bias_Softmax_BMM2_HloString_BF16() {  // NOLINT
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
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
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

    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }
  // Training BMM1 - Scale - bias - Softmax - BMM2
  template <typename T>
  void TestImpl_FMHA_Training_BMM1_Scale_Bias_Softmax_BMM2_vanilla() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfo(backend().default_stream_executor()) <
        se::dnn::VersionInfo(8, 9, 1)) {
      GTEST_SKIP() << "Backward fused MHA requires cuDNN >= 8.9.1.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal = GetInput4DLiteral<T>({8, 128, 8, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal = GetInput4DLiteral<T>({8, 128, 8, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal = GetInput4DLiteral<T>({8, 128, 8, 64}, {3, 2, 1, 0});
    auto bias_literal = GetInput4DLiteral<T>({1, 8, 128, 128}, {3, 2, 1, 0});
    auto mask_literal = GetMask4DLiteral({8, 1, 128, 128}, {3, 2, 1, 0});
    auto do_literal = GetInput4DLiteral<T>({8, 128, 8, 64}, {3, 2, 1, 0});
    std::string hlo_string = "";
    if (std::is_same<T, Eigen::half>::value) {
      hlo_string =
          GetModuleFMHA_Training_BMM1_Scale_Bias_Softmax_BMM2_HloString_F16();
    } else if (std::is_same<T, bfloat16>::value) {
      hlo_string =
          GetModuleFMHA_Training_BMM1_Scale_Bias_Softmax_BMM2_HloString_BF16();
    }

    ExecuteAndCompare(hlo_string,
                      {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal,
                       &bias_literal, &mask_literal, &do_literal},
                      /*expected_num_fmha_calls=*/2);
  }
};

class FlashAttentionBMMScaleCausalMaskSoftmaxBMM
    : public MultiHeadedAttentionTest {
 protected:
  const std::string  // NOLINT
  GetModuleFlash_Attention_BMM1_CausalMask_Softmax_BMM2_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,128,2048]{3,2,1,0},bf16[2,6,2048,128]{3,2,1,0})->bf16[2,6,2048,128]{3,2,1,0}}, allow_spmd_sharding_propagation_to_output={true}

    region_0.28 {
      Arg_0.29 = bf16[] parameter(0)
      Arg_1.30 = bf16[] parameter(1)
      ROOT maximum.31 = bf16[] maximum(Arg_0.29, Arg_1.30)
    }

    region_1.40 {
      Arg_0.41 = f32[] parameter(0)
      Arg_1.42 = f32[] parameter(1)
      ROOT add.43 = f32[] add(Arg_0.41, Arg_1.42)
    }

    ENTRY main.52 {
      Arg_0.1 = bf16[2,6,2048,128]{3,2,1,0} parameter(0), sharding={replicated}
      Arg_1.2 = bf16[2,6,128,2048]{3,2,1,0} parameter(1), sharding={replicated}
      dot.10 = bf16[2,6,2048,2048]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      constant.6 = bf16[] constant(2)
      broadcast.7 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(constant.6), dimensions={}
      multiply.11 = bf16[2,6,2048,2048]{3,2,1,0} multiply(dot.10, broadcast.7)
      iota.16 = s32[2048]{0} iota(), iota_dimension=0
      reshape.17 = s32[1,2048,1]{2,1,0} reshape(iota.16)
      broadcast.18 = s32[1,2048,2048,1]{3,2,1,0} broadcast(reshape.17), dimensions={0,1,3}
      reshape.19 = s32[2048,2048]{1,0} reshape(broadcast.18)
      iota.12 = s32[2048]{0} iota(), iota_dimension=0
      reshape.13 = s32[1,1,2048]{2,1,0} reshape(iota.12)
      broadcast.14 = s32[2048,1,1,2048]{3,2,1,0} broadcast(reshape.13), dimensions={1,2,3}
      reshape.15 = s32[2048,2048]{1,0} reshape(broadcast.14)
      compare.20 = pred[2048,2048]{1,0} compare(reshape.19, reshape.15), direction=LT
      convert.21 = bf16[2048,2048]{1,0} convert(compare.20)
      constant.4 = bf16[] constant(-2.366e+38)
      broadcast.5 = bf16[2048,2048]{1,0} broadcast(constant.4), dimensions={}
      multiply.22 = bf16[2048,2048]{1,0} multiply(convert.21, broadcast.5)
      reshape.23 = bf16[1,1,2048,2048]{3,2,1,0} reshape(multiply.22)
      broadcast.24 = bf16[1,1,2048,2048]{3,2,1,0} broadcast(reshape.23), dimensions={0,1,2,3}
      reshape.25 = bf16[2048,2048]{1,0} reshape(broadcast.24)
      broadcast.26 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(reshape.25), dimensions={2,3}
      add.27 = bf16[2,6,2048,2048]{3,2,1,0} add(multiply.11, broadcast.26)
      constant.9 = bf16[] constant(-inf)
      reduce.32 = bf16[2,6,2048]{2,1,0} reduce(add.27, constant.9), dimensions={3}, to_apply=region_0.28
      reshape.33 = bf16[2,6,2048,1]{3,2,1,0} reshape(reduce.32)
      broadcast.34 = bf16[2,6,2048,1]{3,2,1,0} broadcast(reshape.33), dimensions={0,1,2,3}
      reshape.35 = bf16[2,6,2048]{2,1,0} reshape(broadcast.34)
      broadcast.36 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(reshape.35), dimensions={0,1,2}
      subtract.37 = bf16[2,6,2048,2048]{3,2,1,0} subtract(add.27, broadcast.36)
      exponential.38 = bf16[2,6,2048,2048]{3,2,1,0} exponential(subtract.37)
      convert.39 = f32[2,6,2048,2048]{3,2,1,0} convert(exponential.38)
      constant.8 = f32[] constant(0)
      reduce.44 = f32[2,6,2048]{2,1,0} reduce(convert.39, constant.8), dimensions={3}, to_apply=region_1.40
      reshape.45 = f32[2,6,2048,1]{3,2,1,0} reshape(reduce.44)
      convert.46 = bf16[2,6,2048,1]{3,2,1,0} convert(reshape.45)
      broadcast.47 = bf16[2,6,2048,1]{3,2,1,0} broadcast(convert.46), dimensions={0,1,2,3}
      reshape.48 = bf16[2,6,2048]{2,1,0} reshape(broadcast.47)
      broadcast.49 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(reshape.48), dimensions={0,1,2}
      divide.50 = bf16[2,6,2048,2048]{3,2,1,0} divide(exponential.38, broadcast.49)
      Arg_2.3 = bf16[2,6,2048,128]{3,2,1,0} parameter(2), sharding={replicated}
      ROOT dot.51 = bf16[2,6,2048,128]{3,2,1,0} dot(divide.50, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";
    return hlo_text;
  }

  const std::string  // NOLINT
  GetModuleFlash_Attention_Training_BMM1_CausalMask_Softmax_BMM2_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,64,1024]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0})->(bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,64,1024]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

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

    region_2.63 {
      Arg_0.64 = bf16[] parameter(0)
      Arg_1.65 = bf16[] parameter(1)
      ROOT add.66 = bf16[] add(Arg_0.64, Arg_1.65)
    }

    region_3.75 {
      Arg_0.76 = f32[] parameter(0)
      Arg_1.77 = f32[] parameter(1)
      ROOT add.78 = f32[] add(Arg_0.76, Arg_1.77)
    }

    ENTRY main.88 {
      Arg_0.1 = bf16[2,6,1024,64]{3,2,1,0} parameter(0), sharding={replicated}
      Arg_1.2 = bf16[2,6,64,1024]{3,2,1,0} parameter(1), sharding={replicated}
      dot.12 = bf16[2,6,1024,1024]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      iota.17 = s32[1024]{0} iota(), iota_dimension=0
      reshape.18 = s32[1,1024,1]{2,1,0} reshape(iota.17)
      broadcast.19 = s32[1,1024,1024,1]{3,2,1,0} broadcast(reshape.18), dimensions={0,1,3}
      reshape.20 = s32[1024,1024]{1,0} reshape(broadcast.19)
      iota.13 = s32[1024]{0} iota(), iota_dimension=0
      reshape.14 = s32[1,1,1024]{2,1,0} reshape(iota.13)
      broadcast.15 = s32[1024,1,1,1024]{3,2,1,0} broadcast(reshape.14), dimensions={1,2,3}
      reshape.16 = s32[1024,1024]{1,0} reshape(broadcast.15)
      compare.21 = pred[1024,1024]{1,0} compare(reshape.20, reshape.16), direction=LT
      convert.22 = bf16[1024,1024]{1,0} convert(compare.21)
      constant.7 = bf16[] constant(-2.366e+38)
      broadcast.8 = bf16[1024,1024]{1,0} broadcast(constant.7), dimensions={}
      multiply.23 = bf16[1024,1024]{1,0} multiply(convert.22, broadcast.8)
      reshape.24 = bf16[1,1,1024,1024]{3,2,1,0} reshape(multiply.23)
      broadcast.25 = bf16[1,1,1024,1024]{3,2,1,0} broadcast(reshape.24), dimensions={0,1,2,3}
      reshape.26 = bf16[1024,1024]{1,0} reshape(broadcast.25)
      broadcast.27 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.26), dimensions={2,3}
      add.28 = bf16[2,6,1024,1024]{3,2,1,0} add(dot.12, broadcast.27)
      constant.11 = bf16[] constant(-inf)
      reduce.33 = bf16[2,6,1024]{2,1,0} reduce(add.28, constant.11), dimensions={3}, to_apply=region_0.29
      reshape.34 = bf16[2,6,1024,1]{3,2,1,0} reshape(reduce.33)
      broadcast.35 = bf16[2,6,1024,1]{3,2,1,0} broadcast(reshape.34), dimensions={0,1,2,3}
      reshape.36 = bf16[2,6,1024]{2,1,0} reshape(broadcast.35)
      broadcast.37 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.36), dimensions={0,1,2}
      subtract.38 = bf16[2,6,1024,1024]{3,2,1,0} subtract(add.28, broadcast.37)
      exponential.39 = bf16[2,6,1024,1024]{3,2,1,0} exponential(subtract.38)
      convert.40 = f32[2,6,1024,1024]{3,2,1,0} convert(exponential.39)
      constant.10 = f32[] constant(0)
      reduce.45 = f32[2,6,1024]{2,1,0} reduce(convert.40, constant.10), dimensions={3}, to_apply=region_1.41
      reshape.46 = f32[2,6,1024,1]{3,2,1,0} reshape(reduce.45)
      convert.47 = bf16[2,6,1024,1]{3,2,1,0} convert(reshape.46)
      broadcast.48 = bf16[2,6,1024,1]{3,2,1,0} broadcast(convert.47), dimensions={0,1,2,3}
      reshape.49 = bf16[2,6,1024]{2,1,0} reshape(broadcast.48)
      broadcast.50 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.49), dimensions={0,1,2}
      divide.51 = bf16[2,6,1024,1024]{3,2,1,0} divide(exponential.39, broadcast.50)
      Arg_2.3 = bf16[2,6,1024,64]{3,2,1,0} parameter(2), sharding={replicated}
      dot.54 = bf16[2,6,1024,64]{3,2,1,0} dot(divide.51, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      Arg_3.4 = bf16[2,6,1024,64]{3,2,1,0} parameter(3), sharding={replicated}
      dot.57 = bf16[2,6,1024,1024]{3,2,1,0} dot(Arg_3.4, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      broadcast.70 = bf16[2,6,1024,1]{3,2,1,0} broadcast(convert.47), dimensions={0,1,2,3}
      reshape.71 = bf16[2,6,1024]{2,1,0} reshape(broadcast.70)
      broadcast.72 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.71), dimensions={0,1,2}
      divide.73 = bf16[2,6,1024,1024]{3,2,1,0} divide(dot.57, broadcast.72)
      constant.5 = bf16[] constant(1)
      broadcast.6 = bf16[2,6,1024,1]{3,2,1,0} broadcast(constant.5), dimensions={}
      multiply.52 = bf16[2,6,1024,1]{3,2,1,0} multiply(convert.47, convert.47)
      divide.53 = bf16[2,6,1024,1]{3,2,1,0} divide(broadcast.6, multiply.52)
      broadcast.58 = bf16[2,6,1024,1]{3,2,1,0} broadcast(divide.53), dimensions={0,1,2,3}
      reshape.59 = bf16[2,6,1024]{2,1,0} reshape(broadcast.58)
      broadcast.60 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.59), dimensions={0,1,2}
      multiply.61 = bf16[2,6,1024,1024]{3,2,1,0} multiply(dot.57, broadcast.60)
      multiply.62 = bf16[2,6,1024,1024]{3,2,1,0} multiply(multiply.61, exponential.39)
      constant.9 = bf16[] constant(0)
      reduce.67 = bf16[2,6,1024]{2,1,0} reduce(multiply.62, constant.9), dimensions={3}, to_apply=region_2.63
      reshape.68 = bf16[2,6,1024,1]{3,2,1,0} reshape(reduce.67)
      negate.69 = bf16[2,6,1024,1]{3,2,1,0} negate(reshape.68)
      convert.74 = f32[2,6,1024,1]{3,2,1,0} convert(negate.69)
      reduce.79 = f32[2,6,1024]{2,1,0} reduce(convert.74, constant.10), dimensions={3}, to_apply=region_3.75
      broadcast.80 = f32[2,6,1024,1024]{3,2,1,0} broadcast(reduce.79), dimensions={0,1,2}
      convert.81 = bf16[2,6,1024,1024]{3,2,1,0} convert(broadcast.80)
      add.82 = bf16[2,6,1024,1024]{3,2,1,0} add(divide.73, convert.81)
      multiply.83 = bf16[2,6,1024,1024]{3,2,1,0} multiply(add.82, exponential.39)
      dot.86 = bf16[2,6,1024,64]{3,2,1,0} dot(multiply.83, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      dot.84 = bf16[2,6,1024,64]{3,2,1,0} dot(multiply.83, Arg_0.1), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      transpose.85 = bf16[2,6,64,1024]{2,3,1,0} transpose(dot.84), dimensions={0,1,3,2}
      dot.55 = bf16[2,6,64,1024]{3,2,1,0} dot(Arg_3.4, divide.51), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      transpose.56 = bf16[2,6,1024,64]{2,3,1,0} transpose(dot.55), dimensions={0,1,3,2}
      ROOT tuple.87 = (bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,64,1024]{2,3,1,0}, bf16[2,6,1024,64]{2,3,1,0}) tuple(dot.54, dot.86, transpose.85, transpose.56)
    }
  )";
    return hlo_text;
  }

  template <typename T>
  void TestImpl_Flash_Attention_BMM1_CausalMask_Softmax_BMM2() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfo(backend().default_stream_executor()) <
        se::dnn::VersionInfo(8, 9, 3)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 8.9.3.";
    }
    XlaBuilder builder(TestName());
    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 2048, 128}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 128, 2048}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({2, 6, 2048, 128}, {3, 2, 1, 0});
    std::string hlo_string = "";
    hlo_string =
        GetModuleFlash_Attention_BMM1_CausalMask_Softmax_BMM2_HloString_BF16();  // NOLINT
    ExecuteAndCompare(
        hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal});
  }

  template <typename T>
  void TestImpl_Flash_Attention_Training_BMM1_CausalMask_Softmax_BMM2() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfo(backend().default_stream_executor()) <
        se::dnn::VersionInfo(8, 9, 3)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 8.9.3.";
    }
    XlaBuilder builder(TestName());
    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 1024, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 64, 1024}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({2, 6, 1024, 64}, {3, 2, 1, 0});
    auto do_literal = GetInput4DLiteral<T>({2, 6, 1024, 64}, {3, 2, 1, 0});
    std::string hlo_string = "";
    hlo_string =
        GetModuleFlash_Attention_Training_BMM1_CausalMask_Softmax_BMM2_HloString_BF16();  // NOLINT
    ExecuteAndCompare(
        hlo_string,
        {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal, &do_literal},
        /*expected_num_fmha_calls=*/2);
  }
};

class FlashAttentionBMMScaleBiasSoftmaxBMM : public MultiHeadedAttentionTest {
 protected:
  const std::string                                                   // NOLINT
  GetModuleFlash_Attention_BMM1_Bias_Softmax_BMM2_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,128,2048]{3,2,1,0},bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,2048,2048]{3,2,1,0})->bf16[2,6,2048,128]{3,2,1,0}}, allow_spmd_sharding_propagation_to_output={true}

    region_0.28 {
      Arg_0.29 = bf16[] parameter(0)
      Arg_1.30 = bf16[] parameter(1)
      ROOT maximum.31 = bf16[] maximum(Arg_0.29, Arg_1.30)
    }

    region_1.40 {
      Arg_0.41 = f32[] parameter(0)
      Arg_1.42 = f32[] parameter(1)
      ROOT add.43 = f32[] add(Arg_0.41, Arg_1.42)
    }

    ENTRY main.52 {
      Arg_0.1 = bf16[2,6,2048,128]{3,2,1,0} parameter(0), sharding={replicated}
      Arg_1.2 = bf16[2,6,128,2048]{3,2,1,0} parameter(1), sharding={replicated}
      dot.10 = bf16[2,6,2048,2048]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      constant.6 = bf16[] constant(2)
      broadcast.7 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(constant.6), dimensions={}
      multiply.11 = bf16[2,6,2048,2048]{3,2,1,0} multiply(dot.10, broadcast.7)
      Arg_3.4 = bf16[2,6,2048,2048]{3,2,1,0} parameter(3), sharding={replicated}
      add.27 = bf16[2,6,2048,2048]{3,2,1,0} add(multiply.11, Arg_3.4)
      constant.9 = bf16[] constant(-inf)
      reduce.32 = bf16[2,6,2048]{2,1,0} reduce(add.27, constant.9), dimensions={3}, to_apply=region_0.28
      reshape.33 = bf16[2,6,2048,1]{3,2,1,0} reshape(reduce.32)
      broadcast.34 = bf16[2,6,2048,1]{3,2,1,0} broadcast(reshape.33), dimensions={0,1,2,3}
      reshape.35 = bf16[2,6,2048]{2,1,0} reshape(broadcast.34)
      broadcast.36 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(reshape.35), dimensions={0,1,2}
      subtract.37 = bf16[2,6,2048,2048]{3,2,1,0} subtract(add.27, broadcast.36)
      exponential.38 = bf16[2,6,2048,2048]{3,2,1,0} exponential(subtract.37)
      convert.39 = f32[2,6,2048,2048]{3,2,1,0} convert(exponential.38)
      constant.8 = f32[] constant(0)
      reduce.44 = f32[2,6,2048]{2,1,0} reduce(convert.39, constant.8), dimensions={3}, to_apply=region_1.40
      reshape.45 = f32[2,6,2048,1]{3,2,1,0} reshape(reduce.44)
      convert.46 = bf16[2,6,2048,1]{3,2,1,0} convert(reshape.45)
      broadcast.47 = bf16[2,6,2048,1]{3,2,1,0} broadcast(convert.46), dimensions={0,1,2,3}
      reshape.48 = bf16[2,6,2048]{2,1,0} reshape(broadcast.47)
      broadcast.49 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(reshape.48), dimensions={0,1,2}
      divide.50 = bf16[2,6,2048,2048]{3,2,1,0} divide(exponential.38, broadcast.49)
      Arg_2.3 = bf16[2,6,2048,128]{3,2,1,0} parameter(2), sharding={replicated}
      ROOT dot.51 = bf16[2,6,2048,128]{3,2,1,0} dot(divide.50, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";
    return hlo_text;
  }

  const std::string  // NOLINT
  GetModuleFlash_Attention_Training_BMM1_Bias_Softmax_BMM2_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,64,1024]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,1024,1024]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0})->(bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,64,1024]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

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

    region_2.47 {
      Arg_0.48 = bf16[] parameter(0)
      Arg_1.49 = bf16[] parameter(1)
      ROOT add.50 = bf16[] add(Arg_0.48, Arg_1.49)
    }

    region_3.59 {
      Arg_0.60 = f32[] parameter(0)
      Arg_1.61 = f32[] parameter(1)
      ROOT add.62 = f32[] add(Arg_0.60, Arg_1.61)
    }

    ENTRY main.72 {
      Arg_0.1 = bf16[2,6,1024,64]{3,2,1,0} parameter(0), sharding={replicated}
      Arg_1.2 = bf16[2,6,64,1024]{3,2,1,0} parameter(1), sharding={replicated}
      dot.11 = bf16[2,6,1024,1024]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      Arg_3.4 = bf16[2,6,1024,1024]{3,2,1,0} parameter(3), sharding={replicated}
      add.12 = bf16[2,6,1024,1024]{3,2,1,0} add(dot.11, Arg_3.4)
      constant.9 = bf16[] constant(-inf)
      reduce.17 = bf16[2,6,1024]{2,1,0} reduce(add.12, constant.9), dimensions={3}, to_apply=region_0.13
      reshape.18 = bf16[2,6,1024,1]{3,2,1,0} reshape(reduce.17)
      broadcast.19 = bf16[2,6,1024,1]{3,2,1,0} broadcast(reshape.18), dimensions={0,1,2,3}
      reshape.20 = bf16[2,6,1024]{2,1,0} reshape(broadcast.19)
      broadcast.21 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.20), dimensions={0,1,2}
      subtract.22 = bf16[2,6,1024,1024]{3,2,1,0} subtract(add.12, broadcast.21)
      exponential.23 = bf16[2,6,1024,1024]{3,2,1,0} exponential(subtract.22)
      convert.24 = f32[2,6,1024,1024]{3,2,1,0} convert(exponential.23)
      constant.8 = f32[] constant(0)
      reduce.29 = f32[2,6,1024]{2,1,0} reduce(convert.24, constant.8), dimensions={3}, to_apply=region_1.25
      reshape.30 = f32[2,6,1024,1]{3,2,1,0} reshape(reduce.29)
      convert.31 = bf16[2,6,1024,1]{3,2,1,0} convert(reshape.30)
      broadcast.32 = bf16[2,6,1024,1]{3,2,1,0} broadcast(convert.31), dimensions={0,1,2,3}
      reshape.33 = bf16[2,6,1024]{2,1,0} reshape(broadcast.32)
      broadcast.34 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.33), dimensions={0,1,2}
      divide.35 = bf16[2,6,1024,1024]{3,2,1,0} divide(exponential.23, broadcast.34)
      Arg_2.3 = bf16[2,6,1024,64]{3,2,1,0} parameter(2), sharding={replicated}
      dot.38 = bf16[2,6,1024,64]{3,2,1,0} dot(divide.35, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      Arg_4.5 = bf16[2,6,1024,64]{3,2,1,0} parameter(4), sharding={replicated}
      dot.41 = bf16[2,6,1024,1024]{3,2,1,0} dot(Arg_4.5, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      broadcast.54 = bf16[2,6,1024,1]{3,2,1,0} broadcast(convert.31), dimensions={0,1,2,3}
      reshape.55 = bf16[2,6,1024]{2,1,0} reshape(broadcast.54)
      broadcast.56 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.55), dimensions={0,1,2}
      divide.57 = bf16[2,6,1024,1024]{3,2,1,0} divide(dot.41, broadcast.56)
      constant.5 = bf16[] constant(1)
      broadcast.6 = bf16[2,6,1024,1]{3,2,1,0} broadcast(constant.5), dimensions={}
      multiply.36 = bf16[2,6,1024,1]{3,2,1,0} multiply(convert.31, convert.31)
      divide.37 = bf16[2,6,1024,1]{3,2,1,0} divide(broadcast.6, multiply.36)
      broadcast.42 = bf16[2,6,1024,1]{3,2,1,0} broadcast(divide.37), dimensions={0,1,2,3}
      reshape.43 = bf16[2,6,1024]{2,1,0} reshape(broadcast.42)
      broadcast.44 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.43), dimensions={0,1,2}
      multiply.45 = bf16[2,6,1024,1024]{3,2,1,0} multiply(dot.41, broadcast.44)
      multiply.46 = bf16[2,6,1024,1024]{3,2,1,0} multiply(multiply.45, exponential.23)
      constant.7 = bf16[] constant(0)
      reduce.51 = bf16[2,6,1024]{2,1,0} reduce(multiply.46, constant.7), dimensions={3}, to_apply=region_2.47
      reshape.52 = bf16[2,6,1024,1]{3,2,1,0} reshape(reduce.51)
      negate.53 = bf16[2,6,1024,1]{3,2,1,0} negate(reshape.52)
      convert.58 = f32[2,6,1024,1]{3,2,1,0} convert(negate.53)
      reduce.63 = f32[2,6,1024]{2,1,0} reduce(convert.58, constant.8), dimensions={3}, to_apply=region_3.59
      broadcast.64 = f32[2,6,1024,1024]{3,2,1,0} broadcast(reduce.63), dimensions={0,1,2}
      convert.65 = bf16[2,6,1024,1024]{3,2,1,0} convert(broadcast.64)
      add.66 = bf16[2,6,1024,1024]{3,2,1,0} add(divide.57, convert.65)
      multiply.67 = bf16[2,6,1024,1024]{3,2,1,0} multiply(add.66, exponential.23)
      dot.70 = bf16[2,6,1024,64]{3,2,1,0} dot(multiply.67, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      dot.68 = bf16[2,6,1024,64]{3,2,1,0} dot(multiply.67, Arg_0.1), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      transpose.69 = bf16[2,6,64,1024]{2,3,1,0} transpose(dot.68), dimensions={0,1,3,2}
      dot.39 = bf16[2,6,64,1024]{3,2,1,0} dot(Arg_4.5, divide.35), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      transpose.40 = bf16[2,6,1024,64]{2,3,1,0} transpose(dot.39), dimensions={0,1,3,2}
      ROOT tuple.71 = (bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,64,1024]{2,3,1,0}, bf16[2,6,1024,64]{2,3,1,0}) tuple(dot.38, dot.70, transpose.69, transpose.40)
    }
  )";
    return hlo_text;
  }

  const std::string  // NOLINT
  GetModuleFlash_Attention_BMM1_Bias_Softmax_BMM2_Cross_Attention_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,128,1024]{3,2,1,0},bf16[2,6,1024,128]{3,2,1,0},bf16[2,6,2048,1024]{3,2,1,0})->bf16[2,6,2048,128]{3,2,1,0}}, allow_spmd_sharding_propagation_to_output={true}

    region_0.28 {
      Arg_0.29 = bf16[] parameter(0)
      Arg_1.30 = bf16[] parameter(1)
      ROOT maximum.31 = bf16[] maximum(Arg_0.29, Arg_1.30)
    }

    region_1.40 {
      Arg_0.41 = f32[] parameter(0)
      Arg_1.42 = f32[] parameter(1)
      ROOT add.43 = f32[] add(Arg_0.41, Arg_1.42)
    }

    ENTRY main.52 {
      Arg_0.1 = bf16[2,6,2048,128]{3,2,1,0} parameter(0), sharding={replicated}
      Arg_1.2 = bf16[2,6,128,1024]{3,2,1,0} parameter(1), sharding={replicated}
      dot.10 = bf16[2,6,2048,1024]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      constant.6 = bf16[] constant(2)
      broadcast.7 = bf16[2,6,2048,1024]{3,2,1,0} broadcast(constant.6), dimensions={}
      multiply.11 = bf16[2,6,2048,1024]{3,2,1,0} multiply(dot.10, broadcast.7)
      Arg_3.4 = bf16[2,6,2048,1024]{3,2,1,0} parameter(3), sharding={replicated}
      add.27 = bf16[2,6,2048,1024]{3,2,1,0} add(multiply.11, Arg_3.4)
      constant.9 = bf16[] constant(-inf)
      reduce.32 = bf16[2,6,2048]{2,1,0} reduce(add.27, constant.9), dimensions={3}, to_apply=region_0.28
      reshape.33 = bf16[2,6,2048,1]{3,2,1,0} reshape(reduce.32)
      broadcast.34 = bf16[2,6,2048,1]{3,2,1,0} broadcast(reshape.33), dimensions={0,1,2,3}
      reshape.35 = bf16[2,6,2048]{2,1,0} reshape(broadcast.34)
      broadcast.36 = bf16[2,6,2048,1024]{3,2,1,0} broadcast(reshape.35), dimensions={0,1,2}
      subtract.37 = bf16[2,6,2048,1024]{3,2,1,0} subtract(add.27, broadcast.36)
      exponential.38 = bf16[2,6,2048,1024]{3,2,1,0} exponential(subtract.37)
      convert.39 = f32[2,6,2048,1024]{3,2,1,0} convert(exponential.38)
      constant.8 = f32[] constant(0)
      reduce.44 = f32[2,6,2048]{2,1,0} reduce(convert.39, constant.8), dimensions={3}, to_apply=region_1.40
      reshape.45 = f32[2,6,2048,1]{3,2,1,0} reshape(reduce.44)
      convert.46 = bf16[2,6,2048,1]{3,2,1,0} convert(reshape.45)
      broadcast.47 = bf16[2,6,2048,1]{3,2,1,0} broadcast(convert.46), dimensions={0,1,2,3}
      reshape.48 = bf16[2,6,2048]{2,1,0} reshape(broadcast.47)
      broadcast.49 = bf16[2,6,2048,1024]{3,2,1,0} broadcast(reshape.48), dimensions={0,1,2}
      divide.50 = bf16[2,6,2048,1024]{3,2,1,0} divide(exponential.38, broadcast.49)
      Arg_2.3 = bf16[2,6,1024,128]{3,2,1,0} parameter(2), sharding={replicated}
      ROOT dot.51 = bf16[2,6,2048,128]{3,2,1,0} dot(divide.50, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
  )";
    return hlo_text;
  }
  template <typename T>
  void TestImpl_Flash_Attention_BMM1_Bias_Softmax_BMM2() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfo(backend().default_stream_executor()) <
        se::dnn::VersionInfo(8, 9, 3)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 8.9.3.";
    }
    XlaBuilder builder(TestName());
    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 2048, 128}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 128, 2048}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({2, 6, 2048, 128}, {3, 2, 1, 0});
    auto bias_literal = GetInput4DLiteral<T>({2, 6, 2048, 2048}, {3, 2, 1, 0});
    std::string hlo_string = "";
    hlo_string =
        GetModuleFlash_Attention_BMM1_Bias_Softmax_BMM2_HloString_BF16();
    ExecuteAndCompare(hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal,
                                   &rhs_bmm2_literal, &bias_literal});
  }

  template <typename T>
  void TestImpl_Flash_Attention_Training_BMM1_Bias_Softmax_BMM2() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfo(backend().default_stream_executor()) <
        se::dnn::VersionInfo(8, 9, 3)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 8.9.3.";
    }
    XlaBuilder builder(TestName());
    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 1024, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 64, 1024}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({2, 6, 1024, 64}, {3, 2, 1, 0});
    auto bias_literal = GetInput4DLiteral<T>({2, 6, 1024, 1024}, {3, 2, 1, 0});
    auto do_literal = GetInput4DLiteral<T>({2, 6, 1024, 64}, {3, 2, 1, 0});
    std::string hlo_string = "";
    hlo_string =
        GetModuleFlash_Attention_Training_BMM1_Bias_Softmax_BMM2_HloString_BF16();  // NOLINT
    ExecuteAndCompare(hlo_string,
                      {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal,
                       &bias_literal, &do_literal},
                      /*expected_num_fmha_calls=*/2);
  }

  template <typename T>
  void TestImpl_Flash_Attention_BMM1_Bias_Softmax_BMM2_Cross_Attention() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfo(backend().default_stream_executor()) <
        se::dnn::VersionInfo(8, 9, 4)) {
      GTEST_SKIP() << "Flash Attention cross attention requires "
                      "cuDNN >= 8.9.4.";
    }
    XlaBuilder builder(TestName());
    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 2048, 128}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 128, 1024}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({2, 6, 1024, 128}, {3, 2, 1, 0});
    auto bias_literal = GetInput4DLiteral<T>({2, 6, 2048, 1024}, {3, 2, 1, 0});
    std::string hlo_string = "";
    hlo_string =
        GetModuleFlash_Attention_BMM1_Bias_Softmax_BMM2_Cross_Attention_HloString_BF16();  // NOLINT
    ExecuteAndCompare(hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal,
                                   &rhs_bmm2_literal, &bias_literal});
  }
};

class FlashAttentionBMMScaleSoftmaxBMM : public MultiHeadedAttentionTest {
 protected:
  const std::string  // NOLINT
  GetModuleFlash_Attention_Training_BMM1_Softmax_BMM2_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,64,1024]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0})->(bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,64,1024]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

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

    region_2.47 {
      Arg_0.48 = bf16[] parameter(0)
      Arg_1.49 = bf16[] parameter(1)
      ROOT add.50 = bf16[] add(Arg_0.48, Arg_1.49)
    }

    region_3.59 {
      Arg_0.60 = f32[] parameter(0)
      Arg_1.61 = f32[] parameter(1)
      ROOT add.62 = f32[] add(Arg_0.60, Arg_1.61)
    }

    ENTRY main.72 {
      Arg_0.1 = bf16[2,6,1024,64]{3,2,1,0} parameter(0), sharding={replicated}
      Arg_1.2 = bf16[2,6,64,1024]{3,2,1,0} parameter(1), sharding={replicated}
      dot.11 = bf16[2,6,1024,1024]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      constant.17 = bf16[] constant(37)
      broadcast.29 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(constant.17), dimensions={}
      multiply.2 = bf16[2,6,1024,1024]{3,2,1,0} multiply(dot.11, broadcast.29)
      constant.9 = bf16[] constant(-inf)
      reduce.17 = bf16[2,6,1024]{2,1,0} reduce(multiply.2, constant.9), dimensions={3}, to_apply=region_0.13
      reshape.18 = bf16[2,6,1024,1]{3,2,1,0} reshape(reduce.17)
      broadcast.19 = bf16[2,6,1024,1]{3,2,1,0} broadcast(reshape.18), dimensions={0,1,2,3}
      reshape.20 = bf16[2,6,1024]{2,1,0} reshape(broadcast.19)
      broadcast.21 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.20), dimensions={0,1,2}
      subtract.22 = bf16[2,6,1024,1024]{3,2,1,0} subtract(multiply.2, broadcast.21)
      exponential.23 = bf16[2,6,1024,1024]{3,2,1,0} exponential(subtract.22)
      convert.24 = f32[2,6,1024,1024]{3,2,1,0} convert(exponential.23)
      constant.8 = f32[] constant(0)
      reduce.29 = f32[2,6,1024]{2,1,0} reduce(convert.24, constant.8), dimensions={3}, to_apply=region_1.25
      reshape.30 = f32[2,6,1024,1]{3,2,1,0} reshape(reduce.29)
      convert.31 = bf16[2,6,1024,1]{3,2,1,0} convert(reshape.30)
      broadcast.32 = bf16[2,6,1024,1]{3,2,1,0} broadcast(convert.31), dimensions={0,1,2,3}
      reshape.33 = bf16[2,6,1024]{2,1,0} reshape(broadcast.32)
      broadcast.34 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.33), dimensions={0,1,2}
      divide.35 = bf16[2,6,1024,1024]{3,2,1,0} divide(exponential.23, broadcast.34)
      Arg_2.3 = bf16[2,6,1024,64]{3,2,1,0} parameter(2), sharding={replicated}
      dot.38 = bf16[2,6,1024,64]{3,2,1,0} dot(divide.35, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      Arg_4.5 = bf16[2,6,1024,64]{3,2,1,0} parameter(3), sharding={replicated}
      dot.41 = bf16[2,6,1024,1024]{3,2,1,0} dot(Arg_4.5, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      broadcast.54 = bf16[2,6,1024,1]{3,2,1,0} broadcast(convert.31), dimensions={0,1,2,3}
      reshape.55 = bf16[2,6,1024]{2,1,0} reshape(broadcast.54)
      broadcast.56 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.55), dimensions={0,1,2}
      divide.57 = bf16[2,6,1024,1024]{3,2,1,0} divide(dot.41, broadcast.56)
      constant.5 = bf16[] constant(1)
      broadcast.6 = bf16[2,6,1024,1]{3,2,1,0} broadcast(constant.5), dimensions={}
      multiply.36 = bf16[2,6,1024,1]{3,2,1,0} multiply(convert.31, convert.31)
      divide.37 = bf16[2,6,1024,1]{3,2,1,0} divide(broadcast.6, multiply.36)
      broadcast.42 = bf16[2,6,1024,1]{3,2,1,0} broadcast(divide.37), dimensions={0,1,2,3}
      reshape.43 = bf16[2,6,1024]{2,1,0} reshape(broadcast.42)
      broadcast.44 = bf16[2,6,1024,1024]{3,2,1,0} broadcast(reshape.43), dimensions={0,1,2}
      multiply.45 = bf16[2,6,1024,1024]{3,2,1,0} multiply(dot.41, broadcast.44)
      multiply.46 = bf16[2,6,1024,1024]{3,2,1,0} multiply(multiply.45, exponential.23)
      constant.7 = bf16[] constant(0)
      reduce.51 = bf16[2,6,1024]{2,1,0} reduce(multiply.46, constant.7), dimensions={3}, to_apply=region_2.47
      reshape.52 = bf16[2,6,1024,1]{3,2,1,0} reshape(reduce.51)
      negate.53 = bf16[2,6,1024,1]{3,2,1,0} negate(reshape.52)
      convert.58 = f32[2,6,1024,1]{3,2,1,0} convert(negate.53)
      reduce.63 = f32[2,6,1024]{2,1,0} reduce(convert.58, constant.8), dimensions={3}, to_apply=region_3.59
      broadcast.64 = f32[2,6,1024,1024]{3,2,1,0} broadcast(reduce.63), dimensions={0,1,2}
      convert.65 = bf16[2,6,1024,1024]{3,2,1,0} convert(broadcast.64)
      add.66 = bf16[2,6,1024,1024]{3,2,1,0} add(divide.57, convert.65)
      multiply.67 = bf16[2,6,1024,1024]{3,2,1,0} multiply(add.66, exponential.23)
      dot.70 = bf16[2,6,1024,64]{3,2,1,0} dot(multiply.67, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      dot.68 = bf16[2,6,1024,64]{3,2,1,0} dot(multiply.67, Arg_0.1), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      transpose.69 = bf16[2,6,64,1024]{2,3,1,0} transpose(dot.68), dimensions={0,1,3,2}
      dot.39 = bf16[2,6,64,1024]{3,2,1,0} dot(Arg_4.5, divide.35), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      transpose.40 = bf16[2,6,1024,64]{2,3,1,0} transpose(dot.39), dimensions={0,1,3,2}
      ROOT tuple.71 = (bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,1024,64]{3,2,1,0}, bf16[2,6,64,1024]{2,3,1,0}, bf16[2,6,1024,64]{2,3,1,0}) tuple(dot.38, dot.70, transpose.69, transpose.40)
    }
  )";
    return hlo_text;
  }

  template <typename T>
  void TestImpl_Flash_Attention_Training_BMM1_Softmax_BMM2() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfo(backend().default_stream_executor()) <
        se::dnn::VersionInfo(8, 9, 3)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 8.9.3.";
    }
    XlaBuilder builder(TestName());
    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 1024, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 6, 64, 1024}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({2, 6, 1024, 64}, {3, 2, 1, 0});
    auto do_literal = GetInput4DLiteral<T>({2, 6, 1024, 64}, {3, 2, 1, 0});
    std::string hlo_string = "";
    hlo_string =
        GetModuleFlash_Attention_Training_BMM1_Softmax_BMM2_HloString_BF16();  // NOLINT
    ExecuteAndCompare(
        hlo_string,
        {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal, &do_literal},
        /*expected_num_fmha_calls=*/2);
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

XLA_TEST_F(MultiHeadedAttentionBMMScaleBiasSoftmaxBMM,
           FMHA_Training_BMM1_Scale_Bias_Softmax_BMM2_vanilla_F16) {
  TestImpl_FMHA_Training_BMM1_Scale_Bias_Softmax_BMM2_vanilla<Eigen::half>();
}

XLA_TEST_F(MultiHeadedAttentionBMMScaleBiasSoftmaxBMM,
           FMHA_Training_BMM1_Scale_Bias_Softmax_BMM2_vanilla_BF16) {
  TestImpl_FMHA_Training_BMM1_Scale_Bias_Softmax_BMM2_vanilla<bfloat16>();
}

// flash attention
// BMM1 - Scale - CausalMask - Softmax - BMM2
XLA_TEST_F(FlashAttentionBMMScaleCausalMaskSoftmaxBMM,
           Flash_Attention_BMM1_CausalMask_Softmax_BMM2_BF16) {
  TestImpl_Flash_Attention_BMM1_CausalMask_Softmax_BMM2<bfloat16>();
}

XLA_TEST_F(FlashAttentionBMMScaleCausalMaskSoftmaxBMM,
           Flash_Attention_Training_BMM1_CausalMask_Softmax_BMM2_BF16) {
  TestImpl_Flash_Attention_Training_BMM1_CausalMask_Softmax_BMM2<bfloat16>();
}

// BMM1 - Scale - Bias - Softmax - BMM2
XLA_TEST_F(FlashAttentionBMMScaleBiasSoftmaxBMM,
           Flash_Attention_BMM1_Bias_Softmax_BMM2_BF16) {
  TestImpl_Flash_Attention_BMM1_Bias_Softmax_BMM2<bfloat16>();
}

XLA_TEST_F(FlashAttentionBMMScaleBiasSoftmaxBMM,
           Flash_Attention_Training_BMM1_Bias_Softmax_BMM2_BF16) {
  TestImpl_Flash_Attention_Training_BMM1_Bias_Softmax_BMM2<bfloat16>();
}

XLA_TEST_F(FlashAttentionBMMScaleBiasSoftmaxBMM,
           Flash_Attention_BMM1_Bias_Softmax_BMM2_BF16_Cross_Attention) {
  TestImpl_Flash_Attention_BMM1_Bias_Softmax_BMM2_Cross_Attention<bfloat16>();
}

// BMM1 - Scale - Softmax - BMM2
XLA_TEST_F(FlashAttentionBMMScaleSoftmaxBMM,
           Flash_Attention_Training_BMM1_Softmax_BMM2_BF16) {
  TestImpl_Flash_Attention_Training_BMM1_Softmax_BMM2<bfloat16>();
}
}  // namespace
}  // namespace gpu
}  // namespace xla
