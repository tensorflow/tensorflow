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
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class MultiHeadedAttentionTest : public GpuCodegenTest {
 public:
  MultiHeadedAttentionTest() {
    if (backend().platform()->id() != stream_executor::cuda::kCudaPlatformId ||
        backend()
                .default_stream_executor()
                ->GetDeviceDescription()
                .runtime_version() <
            stream_executor::SemanticVersion{12, 0, 0}) {
      skip_reason_ = "cuDNN Fused MHA requires CUDA 12 or later.";
      return;
    }
    stream_executor::CudaComputeCapability cc = GetCudaComputeCapability();
    // Enforce capability minor == 0 because hardware with a non-zero minor
    // number typically has insufficient shared memory for cuDNN FMHA.
    if (!cc.IsAtLeastAmpere() || cc.minor != 0) {
      skip_reason_ =
          "cuDNN Fused MHA requires Nvidia AMPERE+ GPUs with minor "
          "compute capability == 0.";
      return;
    }
  }

  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  ErrorSpec mha_error_spec_{2.5E-3, 1e-5};

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cudnn_fmha(true);
    debug_options.clear_xla_gpu_enable_command_buffer();
    return debug_options;
  }

  absl::StatusOr<int> CountFMHACalls(
      std::unique_ptr<HloModule> unoptimized_module) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_module,
                        GetOptimizedModule(std::move(unoptimized_module)));

    return absl::c_count_if(
        optimized_module->entry_computation()->instructions(),
        [&](const HloInstruction *inst) {
          return inst->opcode() == HloOpcode::kCustomCall &&
                 absl::StrContains(inst->custom_call_target(), "__cudnn$fmha");
        });
  }

  void ExecuteAndCompare(absl::string_view hlo_string,
                         const std::vector<Literal *> &literals,
                         int expected_num_fmha_calls = 1) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> reference_module,
                            ParseAndReturnVerifiedModule(hlo_string));
    {
      DebugOptions debug_options = GetDebugOptionsForTest();
      debug_options.set_xla_gpu_enable_cudnn_fmha(false);
      reference_module->mutable_config().set_debug_options(debug_options);
    }
    // Sanity check to ensure the first computation doesn't use FMHA.
    TF_ASSERT_OK_AND_ASSIGN(int num_fmha_calls,
                            CountFMHACalls(reference_module->Clone()));
    EXPECT_EQ(num_fmha_calls, 0);
    const Literal expected_result =
        ExecuteAndTransfer(std::move(reference_module), literals);

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> test_module,
                            ParseAndReturnVerifiedModule(hlo_string));
    TF_ASSERT_OK_AND_ASSIGN(num_fmha_calls,
                            CountFMHACalls(test_module->Clone()));
    EXPECT_EQ(num_fmha_calls, expected_num_fmha_calls);
    const Literal actual_result =
        ExecuteAndTransfer(std::move(test_module), literals);

    EXPECT_TRUE(
        LiteralTestUtil::Near(expected_result, actual_result, mha_error_spec_));

    // Run through command buffer
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUDNN);
    debug_options.set_xla_gpu_graph_min_graph_size(1);
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> test_cmd_buffer_module,
                            ParseAndReturnVerifiedModule(hlo_string));
    test_cmd_buffer_module->mutable_config().set_debug_options(debug_options);
    const Literal cmd_buffer_actual_result =
        ExecuteAndTransfer(std::move(test_cmd_buffer_module), literals);
    EXPECT_TRUE(LiteralTestUtil::Near(expected_result, cmd_buffer_actual_result,
                                      mha_error_spec_));
  }

  void VerifyBackwardDeterminism(absl::string_view hlo_string,
                                 const std::vector<Literal *> &literals) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> reference_module,
                            ParseAndReturnVerifiedModule(hlo_string));
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cudnn_fmha(true);
    debug_options.set_xla_gpu_exclude_nondeterministic_ops(true);

    reference_module->mutable_config().set_debug_options(debug_options);
    const Literal first_run_result =
        ExecuteAndTransfer(reference_module->Clone(), literals);

    const Literal second_run_result =
        ExecuteAndTransfer(std::move(reference_module), literals);

    ErrorSpec error_spec{1E-8, 1e-8};
    EXPECT_TRUE(
        LiteralTestUtil::Near(first_run_result, second_run_result, error_spec));
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
  Literal GetInput3DLiteral(std::vector<int64_t> dimensions,
                            std::vector<int64_t> minor_to_major) {
    Array3D<T> input_data(dimensions[0], dimensions[1], dimensions[2]);
    input_data.FillRandom(/*stddev=*/static_cast<T>(0.023), 0.001);

    return LiteralUtil::CreateR3FromArray3DWithLayout(
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
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
        se::dnn::VersionInfo(9, 0, 0)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 9.0.0.";
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
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
        se::dnn::VersionInfo(9, 0, 0)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 9.0.0.";
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

  const std::string  // NOLINT
  GetModuleFlash_Attention_BMM1_Bias_Softmax_BMM2_Dbias_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
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
    return hlo_text;
  }
  template <typename T>
  void TestImpl_Flash_Attention_BMM1_Bias_Softmax_BMM2() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
        se::dnn::VersionInfo(9, 0, 0)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 9.0.0.";
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
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
        se::dnn::VersionInfo(9, 0, 0)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 9.0.0.";
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
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
        se::dnn::VersionInfo(9, 0, 0)) {
      GTEST_SKIP() << "Flash Attention cross attention requires "
                      "cuDNN >= 9.0.0.";
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

  template <typename T>
  void TestImpl_Flash_Attention_BMM1_Bias_Softmax_BMM2_Dbias() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    auto cc = GetCudaComputeCapability();
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
            se::dnn::VersionInfo(9, 0, 0) ||
        !cc.IsAtLeastHopper() || cc.minor != 0) {
      GTEST_SKIP()
          << "Flash Attention dbias requires cuDNN >= 9.0.0 and Hopper arch.";
    }
    XlaBuilder builder(TestName());
    auto lhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 1024, 4, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<T>({2, 1024, 4, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<T>({2, 1024, 4, 64}, {3, 2, 1, 0});
    auto bias_literal = GetInput3DLiteral<T>({4, 1024, 1024}, {2, 1, 0});
    auto do_literal = GetInput4DLiteral<T>({2, 1024, 4, 64}, {3, 2, 1, 0});
    std::string hlo_string =
        GetModuleFlash_Attention_BMM1_Bias_Softmax_BMM2_Dbias_HloString_BF16();  // NOLINT
    ExecuteAndCompare(hlo_string,
                      {&lhs_bmm1_literal, &rhs_bmm1_literal, &rhs_bmm2_literal,
                       &do_literal, &bias_literal},
                      /*expected_num_fmha_calls=*/2);
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
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
        se::dnn::VersionInfo(9, 0, 0)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 9.0.0.";
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

  template <typename T>
  void TestImpl_Flash_Attention_Training_BMM1_Softmax_BMM2_Deterministic() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    auto cc = GetCudaComputeCapability();
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
            se::dnn::VersionInfo(9, 0, 0) ||
        !cc.IsAtLeastHopper() || cc.minor != 0) {
      GTEST_SKIP() << "Flash Attention deterministic kernels requires cuDNN >= "
                      "9.0.0 and Hopper arch.";
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
    VerifyBackwardDeterminism(hlo_string, {&lhs_bmm1_literal, &rhs_bmm1_literal,
                                           &rhs_bmm2_literal, &do_literal});
  }
};

class FlashAttentionBMMScalePaddingMaskSoftmaxBMM
    : public MultiHeadedAttentionTest {
 protected:
  const std::string  // NOLINT
  GetModuleFlash_Attention_Training_BMM1_PaddingMask_As_Bias_Softmax_BMM2_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0})->(bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_parameters={true,true,true,true}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

    region_0.32 {
      Arg_0.33 = bf16[] parameter(0)
      Arg_1.34 = bf16[] parameter(1)
      ROOT maximum = bf16[] maximum(Arg_0.33, Arg_1.34)
    }

    region_1.45 {
      Arg_0.46 = f32[] parameter(0)
      Arg_1.47 = f32[] parameter(1)
      ROOT add = f32[] add(Arg_0.46, Arg_1.47)
    }

    region_2.80 {
      Arg_0.81 = bf16[] parameter(0)
      Arg_1.82 = bf16[] parameter(1)
      ROOT add.1 = bf16[] add(Arg_0.81, Arg_1.82)
    }

    ENTRY main.106 {
      Arg_2.3 = bf16[4,1024,4,64]{3,2,1,0} parameter(2)
      transpose.16 = bf16[4,4,64,1024]{3,2,1,0} transpose(Arg_2.3), dimensions={0,2,3,1}
      Arg_0.1 = bf16[4,1024,4,64]{3,2,1,0} parameter(0)
      transpose.17 = bf16[4,4,1024,64]{3,2,1,0} transpose(Arg_0.1), dimensions={0,2,1,3}
      Arg_1.2 = bf16[4,1024,4,64]{3,2,1,0} parameter(1)
      transpose.19 = bf16[4,4,64,1024]{3,2,1,0} transpose(Arg_1.2), dimensions={0,2,3,1}
      dot = bf16[4,4,1024,1024]{3,2,1,0} dot(transpose.17, transpose.19), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      iota = s32[1024]{0} iota(), iota_dimension=0
      constant.9 = s32[] constant(512)
      broadcast.15 = s32[1024]{0} broadcast(constant.9), dimensions={}
      compare = pred[1024]{0} compare(iota, broadcast.15), direction=GE
      broadcast.16 = pred[1024,1024]{1,0} broadcast(compare), dimensions={0}
      broadcast.17 = pred[1024,1024]{1,0} broadcast(compare), dimensions={1}
      or = pred[1024,1024]{1,0} or(broadcast.16, broadcast.17)
      constant.7 = bf16[] constant(-2.199e+12)
      broadcast.18 = bf16[1024,1024]{1,0} broadcast(constant.7), dimensions={}
      constant.11 = bf16[] constant(0)
      broadcast.19 = bf16[1024,1024]{1,0} broadcast(constant.11), dimensions={}
      select.1 = bf16[1024,1024]{1,0} select(or, broadcast.18, broadcast.19)
      broadcast.20 = bf16[4,4,1024,1024]{3,2,1,0} broadcast(select.1), dimensions={2,3}
      add.2 = bf16[4,4,1024,1024]{3,2,1,0} add(dot, broadcast.20)
      constant.13 = bf16[] constant(-inf)
      reduce.36 = bf16[4,4,1024]{2,1,0} reduce(add.2, constant.13), dimensions={3}, to_apply=region_0.32
      broadcast.22 = bf16[4,4,1024,1024]{3,2,1,0} broadcast(reduce.36), dimensions={0,1,2}
      subtract = bf16[4,4,1024,1024]{3,2,1,0} subtract(add.2, broadcast.22)
      exponential = bf16[4,4,1024,1024]{3,2,1,0} exponential(subtract)
      convert.5 = f32[4,4,1024,1024]{3,2,1,0} convert(exponential)
      constant.12 = f32[] constant(0)
      reduce.49 = f32[4,4,1024]{2,1,0} reduce(convert.5, constant.12), dimensions={3}, to_apply=region_1.45
      convert.6 = bf16[4,4,1024]{2,1,0} convert(reduce.49)
      broadcast.25 = bf16[4,4,1024,1024]{3,2,1,0} broadcast(convert.6), dimensions={0,1,2}
      divide.2 = bf16[4,4,1024,1024]{3,2,1,0} divide(exponential, broadcast.25)
      dot.1 = bf16[4,4,64,1024]{3,2,1,0} dot(transpose.16, divide.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
      iota.1 = s32[1024]{0} iota(), iota_dimension=0
      compare.1 = pred[1024]{0} compare(iota.1, broadcast.15), direction=LT
      convert.7 = bf16[1024]{0} convert(compare.1)
      broadcast.27 = bf16[4,4,64,1024]{3,2,1,0} broadcast(convert.7), dimensions={3}
      multiply.3 = bf16[4,4,64,1024]{3,2,1,0} multiply(dot.1, broadcast.27)
      transpose.26 = bf16[4,1024,4,64]{3,2,1,0} transpose(multiply.3), dimensions={0,3,1,2}
      Arg_3.4 = bf16[4,1024,4,64]{3,2,1,0} parameter(3)
      transpose.21 = bf16[4,4,1024,64]{3,2,1,0} transpose(Arg_3.4), dimensions={0,2,1,3}
      broadcast.28 = bf16[4,4,1024,64]{3,2,1,0} broadcast(convert.7), dimensions={2}
      multiply.4 = bf16[4,4,1024,64]{3,2,1,0} multiply(transpose.21, broadcast.28)
      dot.2 = bf16[4,4,1024,1024]{3,2,1,0} dot(multiply.4, transpose.16), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      divide.3 = bf16[4,4,1024,1024]{3,2,1,0} divide(dot.2, broadcast.25)
      constant.0 = bf16[] constant(1)
      broadcast.29 = bf16[4,4,1024]{2,1,0} broadcast(constant.0), dimensions={}
      multiply.5 = bf16[4,4,1024]{2,1,0} multiply(convert.6, convert.6)
      divide.4 = bf16[4,4,1024]{2,1,0} divide(broadcast.29, multiply.5)
      broadcast.31 = bf16[4,4,1024,1024]{3,2,1,0} broadcast(divide.4), dimensions={0,1,2}
      multiply.6 = bf16[4,4,1024,1024]{3,2,1,0} multiply(dot.2, broadcast.31)
      multiply.7 = bf16[4,4,1024,1024]{3,2,1,0} multiply(multiply.6, exponential)
      reduce.84 = bf16[4,4,1024]{2,1,0} reduce(multiply.7, constant.11), dimensions={3}, to_apply=region_2.80
      negate.1 = bf16[4,4,1024]{2,1,0} negate(reduce.84)
      broadcast.32 = bf16[4,4,1024,1024]{3,2,1,0} broadcast(negate.1), dimensions={0,1,2}
      add.3 = bf16[4,4,1024,1024]{3,2,1,0} add(divide.3, broadcast.32)
      multiply.8 = bf16[4,4,1024,1024]{3,2,1,0} multiply(add.3, exponential)
      transpose.22 = bf16[4,4,1024,64]{3,2,1,0} transpose(Arg_1.2), dimensions={0,2,1,3}
      dot.4 = bf16[4,4,1024,64]{3,2,1,0} dot(multiply.8, transpose.22), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      transpose.27 = bf16[4,1024,4,64]{3,2,1,0} transpose(dot.4), dimensions={0,2,1,3}
      dot.3 = bf16[4,4,1024,64]{3,2,1,0} dot(multiply.8, transpose.17), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      transpose.28 = bf16[4,1024,4,64]{3,2,1,0} transpose(dot.3), dimensions={0,2,1,3}
      transpose.24 = bf16[4,4,64,1024]{3,2,1,0} transpose(multiply.4), dimensions={0,1,3,2}
      dot.73 = bf16[4,4,64,1024]{3,2,1,0} dot(transpose.24, divide.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      transpose.29 = bf16[4,1024,4,64]{3,2,1,0} transpose(dot.73), dimensions={0,3,1,2}
      ROOT tuple = (bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}) tuple(transpose.26, transpose.27, transpose.28, transpose.29)
    } // main.106
  )";
    return hlo_text;
  }

  const std::string  // NOLINT
  GetModuleFlash_Attention_Training_BMM1_PaddingMask_Generation_Softmax_BMM2_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0})->(bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_parameters={true,true,true,true}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

    ENTRY main.21 {
      Arg_0.1 = bf16[4,1024,4,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[4,1024,4,64]{3,2,1,0} parameter(1)
      Arg_2.3 = bf16[4,1024,4,64]{3,2,1,0} parameter(2)
      constant.5 = s32[] constant(512)
      broadcast.6 = s32[4]{0} broadcast(constant.5), dimensions={}
      custom-call.7 = (bf16[4,4,1024,64]{3,1,2,0}, f32[4,4,1024]{2,1,0}, u8[0]{0}) custom-call(Arg_0.1, Arg_1.2, Arg_2.3, broadcast.6, broadcast.6), custom_call_target="__cudnn$fmhaSoftmax", operand_layout_constraints={bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, s32[4]{0}, s32[4]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config={"operation_queue_id": "0", "wait_on_operation_queues": [], "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": 1.0, "dropout_rate": 0, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["4", "4", "1024", "1024"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "seed": 42, "is_flash_attention": true, "mask_type": "PADDING", "bmm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}}}
      get-tuple-element.9 = u8[0]{0} get-tuple-element(custom-call.7), index=2
      get-tuple-element.10 = f32[4,4,1024]{2,1,0} get-tuple-element(custom-call.7), index=1
      Arg_3.4 = bf16[4,1024,4,64]{3,2,1,0} parameter(3)
      get-tuple-element.8 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.7), index=0
      transpose.11 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.8), dimensions={0,2,1,3}
      custom-call.12 = (bf16[4,4,1024,64]{3,1,2,0}, bf16[4,4,1024,64]{3,1,2,0}, bf16[4,4,1024,64]{3,1,2,0}, u8[0]{0}) custom-call(Arg_0.1, Arg_1.2, Arg_2.3, get-tuple-element.10, Arg_3.4, /*index=5*/transpose.11, broadcast.6, broadcast.6), custom_call_target="__cudnn$fmhaSoftmaxBackward", operand_layout_constraints={bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, f32[4,4,1024]{2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, s32[4]{0}, s32[4]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config={"operation_queue_id": "0", "wait_on_operation_queues": [], "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": 1.0, "dropout_rate": 0, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["4", "4", "1024", "1024"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "seed": 42, "is_flash_attention": true, "mask_type": "PADDING", "bmm1_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm1_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}}}
      get-tuple-element.16 = u8[0]{0} get-tuple-element(custom-call.12), index=3
      get-tuple-element.13 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.12), index=0
      transpose.17 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.13), dimensions={0,2,1,3}
      get-tuple-element.14 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.12), index=1
      transpose.18 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.14), dimensions={0,2,1,3}
      get-tuple-element.15 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.12), index=2
      transpose.19 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.15), dimensions={0,2,1,3}
      ROOT tuple.20 = (bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}) tuple(transpose.11, transpose.17, transpose.18, transpose.19)
    } // main.21
    )";
    return hlo_text;
  }

  template <typename T>
  void TestImpl_Flash_Attention_Training_BMM1_PaddingMask_Softmax_BMM2() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
        se::dnn::VersionInfo(9, 0, 0)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 9.0.0.";
    }
    XlaBuilder builder(TestName());
    // pass padding mask as bias
    std::string hlo_string =
        GetModuleFlash_Attention_Training_BMM1_PaddingMask_As_Bias_Softmax_BMM2_HloString_BF16();  // NOLINT
    // generate padding mask in cuDNN directly
    // XLA pattern match does not support pattern matching padding mask
    // so directly lower to custom call instead for reference
    std::string hlo_string_ref =
        GetModuleFlash_Attention_Training_BMM1_PaddingMask_Generation_Softmax_BMM2_HloString_BF16();  // NOLINT
    EXPECT_TRUE(RunAndCompareTwoModules(hlo_string, hlo_string_ref,
                                        ErrorSpec{1e-3, 1e-3}));
  }
};

class FlashAttentionBMMScaleSlidingWindowMaskSoftmaxBMM
    : public MultiHeadedAttentionTest {
 protected:
  const std::string  // NOLINT
  GetModuleFlash_Attention_Training_BMM1_SlidingWindowMask_As_Bias_Softmax_BMM2_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
  HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0})->(bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_parameters={true,true,true,true}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

  region_0.30 {
    Arg_0.31 = bf16[] parameter(0)
    Arg_1.32 = bf16[] parameter(1)
    ROOT maximum.33 = bf16[] maximum(Arg_0.31, Arg_1.32)
  }

  region_1.43 {
    Arg_0.44 = f32[] parameter(0)
    Arg_1.45 = f32[] parameter(1)
    ROOT add.46 = f32[] add(Arg_0.44, Arg_1.45)
  }

  integer_pow.54 {
    constant.56 = bf16[] constant(1)
    broadcast.57 = bf16[4,4,1024,1]{3,2,1,0} broadcast(constant.56), dimensions={}
    Arg_0.55 = bf16[4,4,1024,1]{3,2,1,0} parameter(0)
    multiply.58 = bf16[4,4,1024,1]{3,2,1,0} multiply(Arg_0.55, Arg_0.55)
    ROOT divide.59 = bf16[4,4,1024,1]{3,2,1,0} divide(broadcast.57, multiply.58)
  }

  region_2.72 {
    Arg_0.73 = bf16[] parameter(0)
    Arg_1.74 = bf16[] parameter(1)
    ROOT add.75 = bf16[] add(Arg_0.73, Arg_1.74)
  }

  region_3.84 {
    Arg_0.85 = f32[] parameter(0)
    Arg_1.86 = f32[] parameter(1)
    ROOT add.87 = f32[] add(Arg_0.85, Arg_1.86)
  }

  ENTRY main.98 {
    Arg_2.3 = bf16[4,1024,4,64]{3,2,1,0} parameter(2)
    Arg_0.1 = bf16[4,1024,4,64]{3,2,1,0} parameter(0)
    Arg_1.2 = bf16[4,1024,4,64]{3,2,1,0} parameter(1)
    dot.14 = bf16[4,4,1024,1024]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}
    iota.17 = s32[1024]{0} iota(), iota_dimension=0
    broadcast.18 = s32[1024,1024]{1,0} broadcast(iota.17), dimensions={0}
    iota.15 = s32[1024]{0} iota(), iota_dimension=0
    broadcast.16 = s32[1024,1024]{1,0} broadcast(iota.15), dimensions={1}
    compare.19 = pred[1024,1024]{1,0} compare(broadcast.18, broadcast.16), direction=LT
    constant.9 = s32[] constant(64)
    broadcast.10 = s32[1024,1024]{1,0} broadcast(constant.9), dimensions={}
    subtract.20 = s32[1024,1024]{1,0} subtract(broadcast.18, broadcast.10)
    compare.21 = pred[1024,1024]{1,0} compare(broadcast.16, subtract.20), direction=LE
    or.22 = pred[1024,1024]{1,0} or(compare.19, compare.21)
    convert.23 = bf16[1024,1024]{1,0} convert(or.22)
    constant.7 = bf16[] constant(-2.199e+12)
    broadcast.8 = bf16[1024,1024]{1,0} broadcast(constant.7), dimensions={}
    multiply.24 = bf16[1024,1024]{1,0} multiply(convert.23, broadcast.8)
    reshape.25 = bf16[1,1,1024,1024]{3,2,1,0} reshape(multiply.24)
    broadcast.26 = bf16[1,1,1024,1024]{3,2,1,0} broadcast(reshape.25), dimensions={0,1,2,3}
    reshape.27 = bf16[1024,1024]{1,0} reshape(broadcast.26)
    broadcast.28 = bf16[4,4,1024,1024]{3,2,1,0} broadcast(reshape.27), dimensions={2,3}
    add.29 = bf16[4,4,1024,1024]{3,2,1,0} add(dot.14, broadcast.28)
    constant.13 = bf16[] constant(-inf)
    reduce.34 = bf16[4,4,1024]{2,1,0} reduce(add.29, constant.13), dimensions={3}, to_apply=region_0.30
    constant.5 = bf16[] constant(-inf)
    broadcast.6 = bf16[4,4,1024]{2,1,0} broadcast(constant.5), dimensions={}
    maximum.35 = bf16[4,4,1024]{2,1,0} maximum(reduce.34, broadcast.6)
    reshape.36 = bf16[4,4,1024,1]{3,2,1,0} reshape(maximum.35)
    broadcast.37 = bf16[4,4,1024,1]{3,2,1,0} broadcast(reshape.36), dimensions={0,1,2,3}
    reshape.38 = bf16[4,4,1024]{2,1,0} reshape(broadcast.37)
    broadcast.39 = bf16[4,4,1024,1024]{3,2,1,0} broadcast(reshape.38), dimensions={0,1,2}
    subtract.40 = bf16[4,4,1024,1024]{3,2,1,0} subtract(add.29, broadcast.39)
    exponential.41 = bf16[4,4,1024,1024]{3,2,1,0} exponential(subtract.40)
    convert.42 = f32[4,4,1024,1024]{3,2,1,0} convert(exponential.41)
    constant.12 = f32[] constant(0)
    reduce.47 = f32[4,4,1024]{2,1,0} reduce(convert.42, constant.12), dimensions={3}, to_apply=region_1.43
    reshape.48 = f32[4,4,1024,1]{3,2,1,0} reshape(reduce.47)
    convert.49 = bf16[4,4,1024,1]{3,2,1,0} convert(reshape.48)
    broadcast.50 = bf16[4,4,1024,1]{3,2,1,0} broadcast(convert.49), dimensions={0,1,2,3}
    reshape.51 = bf16[4,4,1024]{2,1,0} reshape(broadcast.50)
    broadcast.52 = bf16[4,4,1024,1024]{3,2,1,0} broadcast(reshape.51), dimensions={0,1,2}
    divide.53 = bf16[4,4,1024,1024]{3,2,1,0} divide(exponential.41, broadcast.52)
    dot.61 = bf16[4,4,64,1024]{3,2,1,0} dot(Arg_2.3, divide.53), lhs_batch_dims={0,2}, lhs_contracting_dims={1}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
    transpose.62 = bf16[4,1024,4,64]{1,3,2,0} transpose(dot.61), dimensions={0,3,1,2}
    Arg_3.4 = bf16[4,1024,4,64]{3,2,1,0} parameter(3)
    transpose.63 = bf16[4,4,64,1024]{2,1,3,0} transpose(Arg_3.4), dimensions={0,2,3,1}
    dot.64 = bf16[4,4,1024,1024]{3,2,1,0} dot(transpose.63, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}
    broadcast.79 = bf16[4,4,1024,1]{3,2,1,0} broadcast(convert.49), dimensions={0,1,2,3}
    reshape.80 = bf16[4,4,1024]{2,1,0} reshape(broadcast.79)
    broadcast.81 = bf16[4,4,1024,1024]{3,2,1,0} broadcast(reshape.80), dimensions={0,1,2}
    divide.82 = bf16[4,4,1024,1024]{3,2,1,0} divide(dot.64, broadcast.81)
    call.60 = bf16[4,4,1024,1]{3,2,1,0} call(convert.49), to_apply=integer_pow.54
    broadcast.67 = bf16[4,4,1024,1]{3,2,1,0} broadcast(call.60), dimensions={0,1,2,3}
    reshape.68 = bf16[4,4,1024]{2,1,0} reshape(broadcast.67)
    broadcast.69 = bf16[4,4,1024,1024]{3,2,1,0} broadcast(reshape.68), dimensions={0,1,2}
    multiply.70 = bf16[4,4,1024,1024]{3,2,1,0} multiply(dot.64, broadcast.69)
    multiply.71 = bf16[4,4,1024,1024]{3,2,1,0} multiply(multiply.70, exponential.41)
    constant.11 = bf16[] constant(0)
    reduce.76 = bf16[4,4,1024]{2,1,0} reduce(multiply.71, constant.11), dimensions={3}, to_apply=region_2.72
    reshape.77 = bf16[4,4,1024,1]{3,2,1,0} reshape(reduce.76)
    negate.78 = bf16[4,4,1024,1]{3,2,1,0} negate(reshape.77)
    convert.83 = f32[4,4,1024,1]{3,2,1,0} convert(negate.78)
    reduce.88 = f32[4,4,1024]{2,1,0} reduce(convert.83, constant.12), dimensions={3}, to_apply=region_3.84
    broadcast.89 = f32[4,4,1024,1024]{3,2,1,0} broadcast(reduce.88), dimensions={0,1,2}
    convert.90 = bf16[4,4,1024,1024]{3,2,1,0} convert(broadcast.89)
    add.91 = bf16[4,4,1024,1024]{3,2,1,0} add(divide.82, convert.90)
    multiply.92 = bf16[4,4,1024,1024]{3,2,1,0} multiply(add.91, exponential.41)
    dot.95 = bf16[4,4,1024,64]{3,2,1,0} dot(multiply.92, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={1}
    transpose.96 = bf16[4,1024,4,64]{3,1,2,0} transpose(dot.95), dimensions={0,2,1,3}
    dot.93 = bf16[4,4,1024,64]{3,2,1,0} dot(multiply.92, Arg_0.1), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,2}, rhs_contracting_dims={1}
    transpose.94 = bf16[4,1024,4,64]{3,1,2,0} transpose(dot.93), dimensions={0,2,1,3}
    dot.65 = bf16[4,4,64,1024]{3,2,1,0} dot(transpose.63, divide.53), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    transpose.66 = bf16[4,1024,4,64]{1,3,2,0} transpose(dot.65), dimensions={0,3,1,2}
    ROOT tuple.97 = (bf16[4,1024,4,64]{1,3,2,0}, bf16[4,1024,4,64]{3,1,2,0}, bf16[4,1024,4,64]{3,1,2,0}, bf16[4,1024,4,64]{1,3,2,0}) tuple(transpose.62, transpose.96, transpose.94, transpose.66)
  } // main.98
  )";
    return hlo_text;
  }

  const std::string  // NOLINT
  GetModuleFlash_Attention_Training_BMM1_SlidingWindowMask_Generation_Softmax_BMM2_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0})->(bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_parameters={true,true,true,true}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

    ENTRY main.19 {
      Arg_0.1 = bf16[4,1024,4,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[4,1024,4,64]{3,2,1,0} parameter(1)
      Arg_2.3 = bf16[4,1024,4,64]{3,2,1,0} parameter(2)
      custom-call.5 = (bf16[4,4,1024,64]{3,1,2,0}, f32[4,4,1024]{2,1,0}, u8[0]{0}) custom-call(Arg_0.1, Arg_1.2, Arg_2.3), custom_call_target="__cudnn$fmhaSoftmax", operand_layout_constraints={bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config={"operation_queue_id": "0", "wait_on_operation_queues": [], "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": 1.0, "dropout_rate": 0, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["4", "4", "1024", "1024"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "seed": 42, "is_flash_attention": true, "mask_type": "CAUSAL", "sliding_window_length": 64, "bmm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}}}
      get-tuple-element.8 = u8[0]{0} get-tuple-element(custom-call.5), index=2
      get-tuple-element.7 = f32[4,4,1024]{2,1,0} get-tuple-element(custom-call.5), index=1
      Arg_3.4 = bf16[4,1024,4,64]{3,2,1,0} parameter(3)
      get-tuple-element.6 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.5), index=0
      transpose.9 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.6), dimensions={0,2,1,3}
      custom-call.10 = (bf16[4,4,1024,64]{3,1,2,0}, bf16[4,4,1024,64]{3,1,2,0}, bf16[4,4,1024,64]{3,1,2,0}, u8[0]{0}) custom-call(Arg_0.1, Arg_1.2, Arg_2.3, get-tuple-element.7, Arg_3.4, /*index=5*/transpose.9), custom_call_target="__cudnn$fmhaSoftmaxBackward", operand_layout_constraints={bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, f32[4,4,1024]{2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config={"operation_queue_id": "0", "wait_on_operation_queues": [], "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": 1.0, "dropout_rate": 0, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["4", "4", "1024", "1024"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "seed": 42, "is_flash_attention": true, "mask_type": "CAUSAL", "sliding_window_length": 64, "bmm1_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm1_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}}}
      get-tuple-element.14 = u8[0]{0} get-tuple-element(custom-call.10), index=3
      get-tuple-element.11 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.10), index=0
      transpose.15 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.11), dimensions={0,2,1,3}
      get-tuple-element.12 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.10), index=1
      transpose.16 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.12), dimensions={0,2,1,3}
      get-tuple-element.13 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.10), index=2
      transpose.17 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.13), dimensions={0,2,1,3}
      ROOT tuple.18 = (bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}) tuple(transpose.9, transpose.15, transpose.16, transpose.17)
    } // main.19
    )";
    return hlo_text;
  }

  template <typename T>
  void TestImpl_Flash_Attention_Training_BMM1_SlidingWindowMask_Softmax_BMM2() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
        se::dnn::VersionInfo(9, 2, 0)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 9.2.0.";
    }
    XlaBuilder builder(TestName());
    // pass sliding window mask as bias
    std::string hlo_string =
        GetModuleFlash_Attention_Training_BMM1_SlidingWindowMask_As_Bias_Softmax_BMM2_HloString_BF16();  // NOLINT
    // generate sliding window mask in cuDNN directly
    // XLA pattern match does not support pattern matching padding mask
    // so directly lower to custom call instead for reference
    std::string hlo_string_ref =
        GetModuleFlash_Attention_Training_BMM1_SlidingWindowMask_Generation_Softmax_BMM2_HloString_BF16();  // NOLINT
    EXPECT_TRUE(RunAndCompareTwoModules(hlo_string, hlo_string_ref,
                                        ErrorSpec{1e-3, 1e-3}));
  }
};

class FlashAttentionBMMScaleSoftmaxBMMF8 : public MultiHeadedAttentionTest {};

class FlashAttentionBMMScaleSoftmaxDropoutBMM
    : public MultiHeadedAttentionTest {
 protected:
  static constexpr absl::string_view
      kModuleFlashAttentionTrainingBMM1SoftmaxDropoutBMM2HloStringBF16 = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0})->(bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_parameters={true,true,true,true}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

    ENTRY main.21 {
      Arg_0.1 = bf16[4,1024,4,64]{3,2,1,0} parameter(0)
      Arg_1.2 = bf16[4,1024,4,64]{3,2,1,0} parameter(1)
      Arg_2.3 = bf16[4,1024,4,64]{3,2,1,0} parameter(2)
      constant.5 = s32[] constant(512)
      broadcast.6 = s32[4]{0} broadcast(constant.5), dimensions={}
      custom-call.7 = (bf16[4,4,1024,64]{3,1,2,0}, f32[4,4,1024]{2,1,0}, u8[0]{0}) custom-call(Arg_0.1, Arg_1.2, Arg_2.3, broadcast.6, broadcast.6), custom_call_target="__cudnn$fmhaSoftmaxDropout", operand_layout_constraints={bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, s32[4]{0}, s32[4]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config={"operation_queue_id": "0", "wait_on_operation_queues": [], "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": 1.0, "dropout_rate": 0.5, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["4", "4", "1024", "1024"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "seed": 42, "is_flash_attention": true, "mask_type": "PADDING", "bmm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}}}
      get-tuple-element.9 = u8[0]{0} get-tuple-element(custom-call.7), index=2
      get-tuple-element.10 = f32[4,4,1024]{2,1,0} get-tuple-element(custom-call.7), index=1
      Arg_3.4 = bf16[4,1024,4,64]{3,2,1,0} parameter(3)
      get-tuple-element.8 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.7), index=0
      transpose.11 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.8), dimensions={0,2,1,3}
      custom-call.12 = (bf16[4,4,1024,64]{3,1,2,0}, bf16[4,4,1024,64]{3,1,2,0}, bf16[4,4,1024,64]{3,1,2,0}, u8[0]{0}) custom-call(Arg_0.1, Arg_1.2, Arg_2.3, get-tuple-element.10, Arg_3.4, /*index=5*/transpose.11, broadcast.6, broadcast.6), custom_call_target="__cudnn$fmhaSoftmaxDropoutBackward", operand_layout_constraints={bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, f32[4,4,1024]{2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, s32[4]{0}, s32[4]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config={"operation_queue_id": "0", "wait_on_operation_queues": [], "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": 1.0, "dropout_rate": 0.5, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["4", "4", "1024", "1024"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "seed": 42, "is_flash_attention": true, "mask_type": "PADDING", "bmm1_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm1_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}}}
      get-tuple-element.16 = u8[0]{0} get-tuple-element(custom-call.12), index=3
      get-tuple-element.13 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.12), index=0
      transpose.17 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.13), dimensions={0,2,1,3}
      get-tuple-element.14 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.12), index=1
      transpose.18 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.14), dimensions={0,2,1,3}
      get-tuple-element.15 = bf16[4,4,1024,64]{3,1,2,0} get-tuple-element(custom-call.12), index=2
      transpose.19 = bf16[4,1024,4,64]{3,2,1,0} transpose(get-tuple-element.15), dimensions={0,2,1,3}
      ROOT tuple.20 = (bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}, bf16[4,1024,4,64]{3,2,1,0}) tuple(transpose.11, transpose.17, transpose.18, transpose.19)
    } // main.21
    )";

  void TestImpl_Flash_Attention_Training_BMM1_Softmax_Dropout_BMM2() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
        se::dnn::VersionInfo(9, 0, 0)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 9.0.0.";
    }
    XlaBuilder builder(TestName());

    auto lhs_bmm1_literal =
        GetInput4DLiteral<bfloat16>({4, 1024, 4, 64}, {3, 2, 1, 0});
    auto rhs_bmm1_literal =
        GetInput4DLiteral<bfloat16>({4, 1024, 4, 64}, {3, 2, 1, 0});
    auto rhs_bmm2_literal =
        GetInput4DLiteral<bfloat16>({4, 1024, 4, 64}, {3, 2, 1, 0});
    auto do_literal =
        GetInput4DLiteral<bfloat16>({4, 1024, 4, 64}, {3, 2, 1, 0});

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnVerifiedModule(
            kModuleFlashAttentionTrainingBMM1SoftmaxDropoutBMM2HloStringBF16));
    ExecuteAndTransfer(std::move(module), {&lhs_bmm1_literal, &rhs_bmm1_literal,
                                           &rhs_bmm2_literal, &do_literal});
  }
};

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

XLA_TEST_F(FlashAttentionBMMScaleBiasSoftmaxBMM,
           Flash_Attention_BMM1_Bias_Softmax_BMM2_BF16_Dbias) {
  TestImpl_Flash_Attention_BMM1_Bias_Softmax_BMM2_Dbias<bfloat16>();
}

// BMM1 - Scale - Softmax - BMM2
XLA_TEST_F(FlashAttentionBMMScaleSoftmaxBMM,
           Flash_Attention_Training_BMM1_Softmax_BMM2_BF16) {
  TestImpl_Flash_Attention_Training_BMM1_Softmax_BMM2<bfloat16>();
}

XLA_TEST_F(FlashAttentionBMMScaleSoftmaxBMM,
           Flash_Attention_Training_BMM1_Softmax_BMM2_Deterministic_BF16) {
  TestImpl_Flash_Attention_Training_BMM1_Softmax_BMM2_Deterministic<bfloat16>();
}

// BMM1 - Scale - PaddingMask - Softmax - BMM2
XLA_TEST_F(FlashAttentionBMMScalePaddingMaskSoftmaxBMM,
           Flash_Attention_Training_BMM1_PaddingMask_Softmax_BMM2_BF16) {
  TestImpl_Flash_Attention_Training_BMM1_PaddingMask_Softmax_BMM2<bfloat16>();
}

// BMM1 - Scale - PaddingMask - Softmax - BMM2
XLA_TEST_F(FlashAttentionBMMScaleSlidingWindowMaskSoftmaxBMM,
           Flash_Attention_Training_BMM1_SlidingWindowMask_Softmax_BMM2_BF16) {
  TestImpl_Flash_Attention_Training_BMM1_SlidingWindowMask_Softmax_BMM2<
      bfloat16>();  // NOLINT
}

absl::string_view GetModuleFlashAttentionBMMScaleSoftmaxBMMCommonRef() {
  static constexpr absl::string_view hlo_text =
      R"(
  HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0})->bf16[4,16,4,16]{3,2,1,0}}, allow_spmd_sharding_propagation_to_parameters={true,true,true}

  clip.33 {
    Arg_2.36 = bf16[] parameter(2)
    broadcast.39 = bf16[4,16,4,16]{3,2,1,0} broadcast(Arg_2.36), dimensions={}
    Arg_1.35 = bf16[] parameter(1)
    broadcast.37 = bf16[4,16,4,16]{3,2,1,0} broadcast(Arg_1.35), dimensions={}
    Arg_0.34 = bf16[4,16,4,16]{3,2,1,0} parameter(0)
    maximum.38 = bf16[4,16,4,16]{3,2,1,0} maximum(broadcast.37, Arg_0.34)
    ROOT minimum.40 = bf16[4,16,4,16]{3,2,1,0} minimum(broadcast.39, maximum.38)
  }

  ENTRY main.106 {
    Arg_0.1 = bf16[4,16,4,16]{3,2,1,0} parameter(0)
    Arg_1.2 = bf16[4,16,4,16]{3,2,1,0} parameter(1)
    Arg_2.3 = bf16[4,16,4,16]{3,2,1,0} parameter(2)

    constant.6 = bf16[] constant(1)
    broadcast.7 = bf16[4,16,4,16]{3,2,1,0} broadcast(constant.6), dimensions={}

    divide.8 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_0.1, broadcast.7)
    call.17 = bf16[4,16,4,16]{3,2,1,0} call(divide.8, bf16[] constant(-448), bf16[] constant(448)), to_apply=clip.33
    convert.18 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.17)
    convert.19 = bf16[4,16,4,16]{3,2,1,0} convert(convert.18)

    divide.20 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_1.2, broadcast.7)
    call.29 = bf16[4,16,4,16]{3,2,1,0} call(divide.20, bf16[] constant(-448), bf16[] constant(448)), to_apply=clip.33
    convert.30 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.29)
    convert.31 = bf16[4,16,4,16]{3,2,1,0} convert(convert.30)

    divide.32 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_2.3, broadcast.7)
    call.41 = bf16[4,16,4,16]{3,2,1,0} call(divide.32, bf16[] constant(-448), bf16[] constant(448)), to_apply=clip.33
    convert.42 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.41)
    convert.43 = bf16[4,16,4,16]{3,2,1,0} convert(convert.42)
  )";
  return hlo_text;
}

absl::string_view GetModuleFlashAttentionBMMScaleSoftmaxBMMCommonF8() {
  static constexpr absl::string_view hlo_text = R"(
  HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0})->bf16[4,16,4,16]{3,2,1,0}}, allow_spmd_sharding_propagation_to_parameters={true,true,true}
  clip.33 {
    Arg_2.36 = bf16[] parameter(2)
    broadcast.39 = bf16[4,16,4,16]{3,2,1,0} broadcast(Arg_2.36), dimensions={}
    Arg_1.35 = bf16[] parameter(1)
    broadcast.37 = bf16[4,16,4,16]{3,2,1,0} broadcast(Arg_1.35), dimensions={}
    Arg_0.34 = bf16[4,16,4,16]{3,2,1,0} parameter(0)
    maximum.38 = bf16[4,16,4,16]{3,2,1,0} maximum(broadcast.37, Arg_0.34)
    ROOT minimum.40 = bf16[4,16,4,16]{3,2,1,0} minimum(broadcast.39, maximum.38)
  } // clip.33
  ENTRY main.106 {
    constant.99 = f32[] constant(1)
    broadcast.99 = f32[1,1,1,1]{3,2,1,0} broadcast(constant.99), dimensions={}
    Arg_0.1 = bf16[4,16,4,16]{3,2,1,0} parameter(0)
    constant.6 = bf16[] constant(1)
    broadcast.7 = bf16[4,16,4,16]{3,2,1,0} broadcast(constant.6), dimensions={}
    divide.8 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_0.1, broadcast.7)
    constant.5 = bf16[] constant(-448)
    constant.4 = bf16[] constant(448)
    call.17 = bf16[4,16,4,16]{3,2,1,0} call(divide.8, constant.5, constant.4), to_apply=clip.33
    convert.18 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.17)
    convert.19 = bf16[4,16,4,16]{3,2,1,0} convert(convert.18)
    Arg_1.2 = bf16[4,16,4,16]{3,2,1,0} parameter(1)
    divide.20 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_1.2, broadcast.7)
    call.29 = bf16[4,16,4,16]{3,2,1,0} call(divide.20, constant.5, constant.4), to_apply=clip.33
    convert.30 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.29)
    convert.31 = bf16[4,16,4,16]{3,2,1,0} convert(convert.30)
    Arg_2.3 = bf16[4,16,4,16]{3,2,1,0} parameter(2)
    divide.32 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_2.3, broadcast.7)
    call.41 = bf16[4,16,4,16]{3,2,1,0} call(divide.32, constant.5, constant.4), to_apply=clip.33
    convert.42 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.41)
    convert.43 = bf16[4,16,4,16]{3,2,1,0} convert(convert.42)
    )";
  return hlo_text;
}
// BMM1 - Scale - Softmax - BMM2 fp8
XLA_TEST_F(FlashAttentionBMMScaleSoftmaxBMMF8,
           Flash_Attention_Inference_BMM1_NoMask_Softmax_BMM2_BNTH_F8) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
      se::dnn::VersionInfo(9, 1, 0)) {
    GTEST_SKIP() << "Flash Attention requires cuDNN >= 9.1.0.";
  }
  auto cc = GetCudaComputeCapability();
  if (!cc.IsAtLeastHopper()) {
    GTEST_SKIP() << "Flash Attention fp8 requires at least Hopper.";
  }
  XlaBuilder builder(TestName());
  std::string ref_bnth = R"(
    custom-call.4.0 = (
        bf16[4,4,16,16]{3,1,2,0}
    ) custom-call(
        convert.19,
        convert.31,
        convert.43
    ),
    custom_call_target="__cudnn$fmhaSoftmax",
    operand_layout_constraints={
        bf16[4,16,4,16]{3,2,1,0},
        bf16[4,16,4,16]{3,2,1,0},
        bf16[4,16,4,16]{3,2,1,0}
    },
    api_version=API_VERSION_STATUS_RETURNING,
    backend_config={
        "operation_queue_id": "0",
        "wait_on_operation_queues": [],
        "cudnn_fmha_backend_config": {
            "algorithm": {
                "algo_id": "0",
                "math_type": "TENSOR_OP_MATH",
                "tuning_knobs": {
                    "17": "1",
                    "24": "0"
                },
                "is_cudnn_frontend": true,
                "workspace_size": "0"
            },
            "fmha_scale": 0.75,
            "dropout_rate": 0.0,
            "intermediate_tensor_shape": {
                "element_type": "BF16",
                "dimensions": ["4", "4", "16", "16"],
                "tuple_shapes": [],
                "layout": {
                    "dim_level_types": [],
                    "dim_unique": [],
                    "dim_ordered": [],
                    "minor_to_major": ["3", "2", "1", "0"],
                    "tiles": [],
                    "element_size_in_bits": "0",
                    "memory_space": "0",
                    "index_primitive_type": "PRIMITIVE_TYPE_INVALID",
                    "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID",
                    "dynamic_shape_metadata_prefix_bytes": "0"
                },
                "is_dynamic_dimension": [false, false, false, false]
            },
            "seed": 42,
            "is_flash_attention": true,
            "mask_type": "NO_MASK",
            "sliding_window_length": 0,
            "bmm1_dot_dimension_numbers": {
                "lhs_contracting_dimensions": ["3"],
                "rhs_contracting_dimensions": ["3"],
                "lhs_batch_dimensions": ["0", "2"],
                "rhs_batch_dimensions": ["0", "2"]
            },
            "bmm2_dot_dimension_numbers": {
                "lhs_contracting_dimensions": ["3"],
                "rhs_contracting_dimensions": ["1"],
                "lhs_batch_dimensions": ["0", "1"],
                "rhs_batch_dimensions": ["0", "2"]
            }
        }
    }
    get-tuple-element.5.0 = bf16[4,4,16,16]{3,1,2,0} get-tuple-element(custom-call.4.0), index=0
    ROOT transpose.7 = bf16[4,16,4,16]{3,2,1,0} transpose(get-tuple-element.5.0), dimensions={0,2,1,3}
    }
)";

  std::string fp8_bnth = R"(
    custom-call.21.0 = (
        f8e4m3fn[4,4,16,16]{3,1,2,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0}
    ) custom-call(
        convert.18,
        convert.30,
        convert.42,
        broadcast.99,
        broadcast.99,
        /*index=5*/broadcast.99,
        broadcast.99,
        broadcast.99,
        broadcast.99
    ),
    custom_call_target="__cudnn$fmhaSoftmaxF8",
    operand_layout_constraints={
        f8e4m3fn[4,16,4,16]{3,2,1,0},
        f8e4m3fn[4,16,4,16]{3,2,1,0},
        f8e4m3fn[4,16,4,16]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0}
    },
    api_version=API_VERSION_STATUS_RETURNING,
    backend_config={
        "operation_queue_id": "0",
        "wait_on_operation_queues": [],
        "cudnn_fmha_backend_config": {
            "algorithm": {
                "algo_id": "0",
                "math_type": "TENSOR_OP_MATH",
                "tuning_knobs": {
                    "17": "1",
                    "24": "0"
                },
                "is_cudnn_frontend": true,
                "workspace_size": "0"
            },
            "fmha_scale": 0.75,
            "intermediate_tensor_shape": {
                "element_type": "BF16",
                "dimensions": ["4", "4", "16", "16"],
                "tuple_shapes": [],
                "layout": {
                    "dim_level_types": [],
                    "dim_unique": [],
                    "dim_ordered": [],
                    "minor_to_major": ["3", "2", "1", "0"],
                    "tiles": [],
                    "element_size_in_bits": "0",
                    "memory_space": "0",
                    "index_primitive_type": "PRIMITIVE_TYPE_INVALID",
                    "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID",
                    "dynamic_shape_metadata_prefix_bytes": "0"
                },
                "is_dynamic_dimension": [false, false, false, false]
            },
            "is_flash_attention": true,
            "mask_type": "NO_MASK",
            "bmm1_dot_dimension_numbers": {
                "lhs_contracting_dimensions": ["3"],
                "rhs_contracting_dimensions": ["3"],
                "lhs_batch_dimensions": ["0", "2"],
                "rhs_batch_dimensions": ["0", "2"]
            },
            "bmm2_dot_dimension_numbers": {
                "lhs_contracting_dimensions": ["3"],
                "rhs_contracting_dimensions": ["1"],
                "lhs_batch_dimensions": ["0", "1"],
                "rhs_batch_dimensions": ["0", "2"]
            }
        }
    }
    get-tuple-element.5.0 = f8e4m3fn[4,4,16,16]{3,1,2,0} get-tuple-element(custom-call.21.0), index=0
    transpose.26 = f8e4m3fn[4,16,4,16]{3,2,1,0} transpose(get-tuple-element.5.0), dimensions={0,2,1,3}
    ROOT out = bf16[4,16,4,16]{3,2,1,0} convert(transpose.26)
    }
    )";

  std::string hlo_string =
      std::string(GetModuleFlashAttentionBMMScaleSoftmaxBMMCommonF8()) +
      fp8_bnth;
  std::string hlo_string_ref =
      std::string(GetModuleFlashAttentionBMMScaleSoftmaxBMMCommonRef()) +
      ref_bnth;
  EXPECT_TRUE(RunAndCompareTwoModules(hlo_string, hlo_string_ref,
                                      ErrorSpec{5e-2, 5e-2}));
}

XLA_TEST_F(FlashAttentionBMMScaleSoftmaxBMMF8,
           Flash_Attention_Inference_BMM1_NoMask_Softmax_BMM2_BTNH_F8) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
      se::dnn::VersionInfo(9, 1, 0)) {
    GTEST_SKIP() << "Flash Attention requires cuDNN >= 9.1.0.";
  }
  auto cc = GetCudaComputeCapability();
  if (!cc.IsAtLeastHopper()) {
    GTEST_SKIP() << "Flash Attention fp8 requires at least Hopper.";
  }
  XlaBuilder builder(TestName());

  std::string ref_btnh = R"(
    custom-call.4.0 = (
        bf16[4,16,4,16]{3,2,1,0}
    ) custom-call(
        convert.19,
        convert.31,
        convert.43
    ),
    custom_call_target="__cudnn$fmhaSoftmax",
    operand_layout_constraints={
        bf16[4,16,4,16]{3,2,1,0},
        bf16[4,16,4,16]{3,2,1,0},
        bf16[4,16,4,16]{3,2,1,0}
    },
    api_version=API_VERSION_STATUS_RETURNING,
    backend_config={
        "operation_queue_id": "0",
        "wait_on_operation_queues": [],
        "cudnn_fmha_backend_config": {
            "algorithm": {
                "algo_id": "0",
                "math_type": "TENSOR_OP_MATH",
                "tuning_knobs": {
                    "17": "1",
                    "24": "0"
                },
                "is_cudnn_frontend": true,
                "workspace_size": "0"
            },
            "fmha_scale": 0.75,
            "dropout_rate": 0.0,
            "intermediate_tensor_shape": {
                "element_type": "BF16",
                "dimensions": ["4", "16", "4", "4"],
                "tuple_shapes": [],
                "layout": {
                    "dim_level_types": [],
                    "dim_unique": [],
                    "dim_ordered": [],
                    "minor_to_major": ["3", "2", "1", "0"],
                    "tiles": [],
                    "element_size_in_bits": "0",
                    "memory_space": "0",
                    "index_primitive_type": "PRIMITIVE_TYPE_INVALID",
                    "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID",
                    "dynamic_shape_metadata_prefix_bytes": "0"
                },
                "is_dynamic_dimension": [false, false, false, false]
            },
            "seed": 42,
            "is_flash_attention": true,
            "mask_type": "NO_MASK",
            "sliding_window_length": 0,
            "bmm1_dot_dimension_numbers": {
                "lhs_contracting_dimensions": ["3"],
                "rhs_contracting_dimensions": ["3"],
                "lhs_batch_dimensions": ["0", "1"],
                "rhs_batch_dimensions": ["0", "1"]
            },
            "bmm2_dot_dimension_numbers": {
                "lhs_contracting_dimensions": ["3"],
                "rhs_contracting_dimensions": ["2"],
                "lhs_batch_dimensions": ["0", "1"],
                "rhs_batch_dimensions": ["0", "1"]
            }
        }
    }
    ROOT get-tuple-element.5.0 = bf16[4,16,4,16]{3,2,1,0} get-tuple-element(custom-call.4.0), index=0
    }
)";

  std::string fp8_btnh = R"(
    custom-call.21.0 = (
        f8e4m3fn[4,16,4,16]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0}
    ) custom-call(
        convert.18,
        convert.30,
        convert.42,
        broadcast.99,
        broadcast.99,
        /*index=5*/broadcast.99,
        broadcast.99,
        broadcast.99,
        broadcast.99
    ),
    custom_call_target="__cudnn$fmhaSoftmaxF8",
    operand_layout_constraints={
        f8e4m3fn[4,16,4,16]{3,2,1,0},
        f8e4m3fn[4,16,4,16]{3,2,1,0},
        f8e4m3fn[4,16,4,16]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0}
    },
    api_version=API_VERSION_STATUS_RETURNING,
    backend_config={
        "operation_queue_id": "0",
        "wait_on_operation_queues": [],
        "cudnn_fmha_backend_config": {
            "algorithm": {
                "algo_id": "0",
                "math_type": "TENSOR_OP_MATH",
                "tuning_knobs": {
                    "17": "1",
                    "24": "0"
                },
                "is_cudnn_frontend": true,
                "workspace_size": "0"
            },
            "fmha_scale": 0.75,
            "intermediate_tensor_shape": {
                "element_type": "BF16",
                "dimensions": ["4", "16", "4", "4"],
                "tuple_shapes": [],
                "layout": {
                    "dim_level_types": [],
                    "dim_unique": [],
                    "dim_ordered": [],
                    "minor_to_major": ["3", "2", "1", "0"],
                    "tiles": [],
                    "element_size_in_bits": "0",
                    "memory_space": "0",
                    "index_primitive_type": "PRIMITIVE_TYPE_INVALID",
                    "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID",
                    "dynamic_shape_metadata_prefix_bytes": "0"
                },
                "is_dynamic_dimension": [false, false, false, false]
            },
            "is_flash_attention": true,
            "mask_type": "NO_MASK",
            "bmm1_dot_dimension_numbers": {
                "lhs_contracting_dimensions": ["3"],
                "rhs_contracting_dimensions": ["3"],
                "lhs_batch_dimensions": ["0", "1"],
                "rhs_batch_dimensions": ["0", "1"]
            },
            "bmm2_dot_dimension_numbers": {
                "lhs_contracting_dimensions": ["3"],
                "rhs_contracting_dimensions": ["2"],
                "lhs_batch_dimensions": ["0", "1"],
                "rhs_batch_dimensions": ["0", "1"]
            }
        }
    }
    get-tuple-element.5.0 = f8e4m3fn[4,16,4,16]{3,2,1,0} get-tuple-element(custom-call.21.0), index=0
    ROOT out = bf16[4,16,4,16]{3,2,1,0} convert(get-tuple-element.5.0)
    }
    )";

  std::string hlo_string =
      std::string(GetModuleFlashAttentionBMMScaleSoftmaxBMMCommonF8()) +
      fp8_btnh;
  std::string hlo_string_ref =
      std::string(GetModuleFlashAttentionBMMScaleSoftmaxBMMCommonRef()) +
      ref_btnh;
  EXPECT_TRUE(RunAndCompareTwoModules(hlo_string, hlo_string_ref,
                                      ErrorSpec{5e-2, 5e-2}));
}

// BMM1 - Scale - Softmax - BMM2 fp8
XLA_TEST_F(FlashAttentionBMMScaleSoftmaxDropoutBMM,
           Flash_Attention_Training_BMM1_Softmax_Dropout_BMM2) {
  TestImpl_Flash_Attention_Training_BMM1_Softmax_Dropout_BMM2();
}

XLA_TEST_F(FlashAttentionBMMScaleSoftmaxBMMF8,
           Flash_Attention_Bwd_BMM1_NoMask_Softmax_BMM2_F8) {
  if (skip_reason_) GTEST_SKIP() << *skip_reason_;
  if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
      se::dnn::VersionInfo(9, 1, 0)) {
    GTEST_SKIP() << "Flash Attention requires cuDNN >= 9.1.0.";
  }
  auto cc = GetCudaComputeCapability();
  if (!cc.IsAtLeastHopper()) {
    GTEST_SKIP() << "Flash Attention fp8 requires at least Hopper.";
  }
  XlaBuilder builder(TestName());
  std::string hlo_string_ref = R"(
    HloModule fmha_cudnn_custom_call_bwd
    // Process inputs: clip, convert to f8e4m3fn, and convert back to bf16
    cast_to_representable {
        // Parameters
        input = bf16[1,1,128,128] parameter(0)
        min_val = bf16[] parameter(1)
        max_val = bf16[] parameter(2)

        // Broadcasting min and max values
        min_broadcast = bf16[1,1,128,128] broadcast(min_val), dimensions={}
        max_broadcast = bf16[1,1,128,128] broadcast(max_val), dimensions={}

        // Clipping the scaled input
        clipped_min = bf16[1,1,128,128] maximum(min_broadcast, input)
        clipped = bf16[1,1,128,128] minimum(max_broadcast, clipped_min)

        // Converting to f8e4m3fn and back to bf16
        converted_f8 = f8e4m3fn[1,1,128,128] convert(clipped)
        ROOT converted_bf16 = bf16[1,1,128,128] convert(converted_f8)
    }
    // Main function
    ENTRY main {
      // Input parameters
      query = bf16[1,1,128,128] parameter(0)
      key = bf16[1,1,128,128] parameter(1)
      value = bf16[1,1,128,128] parameter(2)
      grad_output = bf16[1,1,128,128] parameter(3)
      fwd_output = bf16[1,1,128,128] parameter(4)
      score = f32[1,1,128] parameter(5)

      // Constants
      one_f32 = f32[] constant(1)
      one_f32_broadcast = f32[1,1,1,1] broadcast(one_f32), dimensions={}
      min_clip_val = bf16[] constant(-448)
      max_clip_val = bf16[] constant(448)

      query_processed = bf16[1,1,128,128] call(query, min_clip_val, max_clip_val), to_apply=cast_to_representable
      key_processed = bf16[1,1,128,128] call(key, min_clip_val, max_clip_val), to_apply=cast_to_representable
      value_processed = bf16[1,1,128,128] call(value, min_clip_val, max_clip_val), to_apply=cast_to_representable
      grad_output_processed = bf16[1,1,128,128] call(grad_output, min_clip_val, max_clip_val), to_apply=cast_to_representable
      fwd_output_processed = bf16[1,1,128,128] call(fwd_output, min_clip_val, max_clip_val), to_apply=cast_to_representable

      // FMHA Forward Backward custom call
      fmha_result = (bf16[1,1,128,128], bf16[1,1,128,128], bf16[1,1,128,128], u8[0]) custom-call(
        query_processed, key_processed, value_processed,
        score, fwd_output_processed, grad_output_processed
      ),
      custom_call_target="__cudnn$fmhaSoftmaxBackward",
      operand_layout_constraints={
        bf16[1,1,128,128]{3,2,1,0}, bf16[1,1,128,128]{3,2,1,0},
        bf16[1,1,128,128]{3,2,1,0}, f32[1,1,128]{2,1,0},
        bf16[1,1,128,128]{3,2,1,0}, bf16[1,1,128,128]{3,2,1,0}
      },
      api_version=API_VERSION_STATUS_RETURNING,
      backend_config={
        "operation_queue_id": "0",
        "wait_on_operation_queues": [],
        "cudnn_fmha_backend_config": {
          "algorithm": {
            "algo_id": "0",
            "math_type": "TENSOR_OP_MATH",
            "tuning_knobs": {"17": "1", "24": "0"},
            "is_cudnn_frontend": true,
            "workspace_size": "0"
          },
          "fmha_scale": 1.0,
          "dropout_rate": 0.0,
          "intermediate_tensor_shape": {
            "element_type": "BF16",
            "dimensions": ["1", "1", "128", "128"],
            "tuple_shapes": [],
            "layout": {
              "dim_level_types": [],
              "dim_unique": [],
              "dim_ordered": [],
              "minor_to_major": ["3", "2", "1", "0"],
              "tiles": [],
              "element_size_in_bits": "0",
              "memory_space": "0",
              "index_primitive_type": "PRIMITIVE_TYPE_INVALID",
              "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID",
              "dynamic_shape_metadata_prefix_bytes": "0"
            },
            "is_dynamic_dimension": [false, false, false, false]
          },
          "seed": 42,
          "is_flash_attention": true,
          "mask_type": "NO_MASK",
          "sliding_window_length": 0,
          "bmm1_grad_gemm1_dot_dimension_numbers": {
            "lhs_contracting_dimensions": ["2"],
            "rhs_contracting_dimensions": ["2"],
            "lhs_batch_dimensions": ["0", "1"],
            "rhs_batch_dimensions": ["0", "1"]
          },
          "bmm1_grad_gemm2_dot_dimension_numbers": {
            "lhs_contracting_dimensions": ["3"],
            "rhs_contracting_dimensions": ["2"],
            "lhs_batch_dimensions": ["0", "1"],
            "rhs_batch_dimensions": ["0", "1"]
          },
          "bmm2_grad_gemm1_dot_dimension_numbers": {
            "lhs_contracting_dimensions": ["2"],
            "rhs_contracting_dimensions": ["2"],
            "lhs_batch_dimensions": ["0", "1"],
            "rhs_batch_dimensions": ["0", "1"]
          },
          "bmm2_grad_gemm2_dot_dimension_numbers": {
            "lhs_contracting_dimensions": ["3"],
            "rhs_contracting_dimensions": ["3"],
            "lhs_batch_dimensions": ["0", "1"],
            "rhs_batch_dimensions": ["0", "1"]
          }
        }
      }

      ROOT output = bf16[1,1,128,128] get-tuple-element(fmha_result), index=0
  })";

  std::string hlo_string = R"(
    HloModule fmha_cudnn_custom_call_bwd_f8
    // Process inputs: clip, convert to f8e4m3fn
    cast_to_representable {
      // Parameters
      input = bf16[1,1,128,128] parameter(0)
      min_val = bf16[] parameter(1)
      max_val = bf16[] parameter(2)

      // Broadcasting min and max values
      min_broadcast = bf16[1,1,128,128] broadcast(min_val), dimensions={}
      max_broadcast = bf16[1,1,128,128] broadcast(max_val), dimensions={}

      // Clipping the scaled input
      clipped_min = bf16[1,1,128,128] maximum(min_broadcast, input)
      clipped = bf16[1,1,128,128] minimum(max_broadcast, clipped_min)

      // Converting to f8e4m3fn and back to bf16
      ROOT converted_f8 = f8e4m3fn[1,1,128,128] convert(clipped)
    }

    // Main function
    ENTRY main {
      // Input parameters
      query = bf16[1,1,128,128] parameter(0)
      key = bf16[1,1,128,128] parameter(1)
      value = bf16[1,1,128,128] parameter(2)
      grad_output = bf16[1,1,128,128] parameter(3)
      fwd_output = bf16[1,1,128,128] parameter(4)
      score = f32[1,1,128] parameter(5)

      // Constants
      one_f32 = f32[] constant(1)
      one_f32_broadcast = f32[1,1,1,1] broadcast(one_f32), dimensions={}
      min_clip_val = bf16[] constant(-448)
      max_clip_val = bf16[] constant(448)

      query_processed = f8e4m3fn[1,1,128,128] call(query, min_clip_val, max_clip_val), to_apply=cast_to_representable
      key_processed = f8e4m3fn[1,1,128,128] call(key, min_clip_val, max_clip_val), to_apply=cast_to_representable
      value_processed = f8e4m3fn[1,1,128,128] call(value, min_clip_val, max_clip_val), to_apply=cast_to_representable
      grad_output_processed = f8e4m3fn[1,1,128,128] call(grad_output, min_clip_val, max_clip_val), to_apply=cast_to_representable
      fwd_output_processed = f8e4m3fn[1,1,128,128] call(fwd_output, min_clip_val, max_clip_val), to_apply=cast_to_representable

      // FMHA Softmax Backward custom call
      fmha_result = (f8e4m3fn[1,1,128,128], f8e4m3fn[1,1,128,128], f8e4m3fn[1,1,128,128],
                     f32[1,1,1,1], f32[1,1,1,1], f32[1,1,1,1], f32[1,1,1,1], u8[0]) custom-call(
        query_processed, key_processed, value_processed,
        grad_output_processed, fwd_output_processed, score,
        one_f32_broadcast, one_f32_broadcast, one_f32_broadcast, one_f32_broadcast,
        one_f32_broadcast, one_f32_broadcast, one_f32_broadcast, one_f32_broadcast,
        one_f32_broadcast, one_f32_broadcast, one_f32_broadcast, one_f32_broadcast
      ),
      custom_call_target="__cudnn$fmhaSoftmaxBackwardF8",
      operand_layout_constraints={
        f8e4m3fn[1,1,128,128]{3,2,1,0}, f8e4m3fn[1,1,128,128]{3,2,1,0},
        f8e4m3fn[1,1,128,128]{3,2,1,0}, f8e4m3fn[1,1,128,128]{3,2,1,0},
        f8e4m3fn[1,1,128,128]{3,2,1,0}, f32[1,1,128]{2,1,0},
        f32[1,1,1,1]{3,2,1,0}, f32[1,1,1,1]{3,2,1,0}, f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0}, f32[1,1,1,1]{3,2,1,0}, f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0}, f32[1,1,1,1]{3,2,1,0}, f32[1,1,1,1]{3,2,1,0},
        f32[1,1,1,1]{3,2,1,0}, f32[1,1,1,1]{3,2,1,0}, f32[1,1,1,1]{3,2,1,0}
      },
      api_version=API_VERSION_STATUS_RETURNING,
      backend_config={
        "operation_queue_id": "0",
        "wait_on_operation_queues": [],
        "cudnn_fmha_backend_config": {
          "algorithm": {
            "algo_id": "0",
            "math_type": "TENSOR_OP_MATH",
            "tuning_knobs": {"17": "1", "24": "0"},
            "is_cudnn_frontend": true,
            "workspace_size": "0"
          },
          "fmha_scale": 1.0,
          "intermediate_tensor_shape": {
            "element_type": "BF16",
            "dimensions": ["1", "1", "128", "128"],
            "tuple_shapes": [],
            "layout": {
              "dim_level_types": [],
              "dim_unique": [],
              "dim_ordered": [],
              "minor_to_major": ["3", "2", "1", "0"],
              "tiles": [],
              "element_size_in_bits": "0",
              "memory_space": "0",
              "index_primitive_type": "PRIMITIVE_TYPE_INVALID",
              "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID",
              "dynamic_shape_metadata_prefix_bytes": "0"
            },
            "is_dynamic_dimension": [false, false, false, false]
          },
          "is_flash_attention": true,
          "mask_type": "NO_MASK",
          "bmm1_grad_gemm1_dot_dimension_numbers": {
            "lhs_contracting_dimensions": ["2"],
            "rhs_contracting_dimensions": ["2"],
            "lhs_batch_dimensions": ["0", "1"],
            "rhs_batch_dimensions": ["0", "1"]
          },
          "bmm1_grad_gemm2_dot_dimension_numbers": {
            "lhs_contracting_dimensions": ["3"],
            "rhs_contracting_dimensions": ["2"],
            "lhs_batch_dimensions": ["0", "1"],
            "rhs_batch_dimensions": ["0", "1"]
          },
          "bmm2_grad_gemm1_dot_dimension_numbers": {
            "lhs_contracting_dimensions": ["2"],
            "rhs_contracting_dimensions": ["2"],
            "lhs_batch_dimensions": ["0", "1"],
            "rhs_batch_dimensions": ["0", "1"]
          },
          "bmm2_grad_gemm2_dot_dimension_numbers": {
            "lhs_contracting_dimensions": ["3"],
            "rhs_contracting_dimensions": ["3"],
            "lhs_batch_dimensions": ["0", "1"],
            "rhs_batch_dimensions": ["0", "1"]
          }
        }
      }

      fmha_output = f8e4m3fn[1,1,128,128] get-tuple-element(fmha_result), index=0
      ROOT output = bf16[1,1,128,128] convert(fmha_output)
    })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_string_ref, hlo_string,
                                      ErrorSpec{2e-1, 2e-1}));
}
}  // namespace
}  // namespace gpu
}  // namespace xla
