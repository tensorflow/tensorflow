/* Copyright 2024 The OpenXLA Authors.

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

// TODO(b/343158720): Simplify the tests in this file after a generic emitter
// has landed.
#include "xla/service/gpu/triton_support.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emitter_triton.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/gpu/triton_test_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using DotTest = TritonSupportTestWithParam;

TEST_P(DotTest, IsTritonSupportedDot) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string kHloTestTemplate = R"(
triton_computation {
  parameter_0 = $0[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  ROOT dot = $0[92,63]{1,0} $1(parameter_0, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  parameter_0 = $0[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  ROOT triton_op = $0[92,63]{1,0} fusion(parameter_0, parameter_1), kind=kCustom,
    calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  // TODO(b/345763510): Change the kind above to "__triton" once dots are
  // supported.
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));
  const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  const HloComputation* computation = fusion->fused_instructions_computation();
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    TF_EXPECT_OK(ApplyFloatNormalization(module.get()));
    // TODO(b/345763510): Change the the line below to a file check on generated
    // code once dots are supported.
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(module), ErrorSpec{/*aabs=*/2e-4, /*arel=*/2e-4}));
  } else {
    const se::DeviceDescription dev_info =
        TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
    EXPECT_THAT(TritonWrapper(*TritonFusionAnalysis::Execute(*computation),
                              "test_fn", fusion, GetCudaComputeCapability(),
                              dev_info, config_, /*output_tile_sizes=*/{},
                              &llvm_module_, &EmitMatMul, mlir_context_),
                tsl::testing::StatusIs(
                    absl::StatusCode::kInternal,
                    ::testing::HasSubstr("Failed to compile Triton kernel")));
  }
}

INSTANTIATE_TEST_SUITE_P(DotTestTestSuite, DotTest,
                         ::testing::Combine(::testing::Values(F16, F32, BF16),
                                            ::testing::Values(HloOpcode::kDot)),
                         TritonSupportTestParamsToString);

struct DynamicSliceTestParam {
  PrimitiveType data_type;
  PrimitiveType index_type;
  bool is_the_majormost_dim_being_sliced;

  using TupleType = std::tuple<PrimitiveType, PrimitiveType, bool>;

  explicit DynamicSliceTestParam(const TupleType& tuple)
      : data_type(std::get<0>(tuple)),
        index_type(std::get<1>(tuple)),
        is_the_majormost_dim_being_sliced(std::get<2>(tuple)) {}
};

std::string DynamicSliceTestParamToString(
    const ::testing::TestParamInfo<DynamicSliceTestParam::TupleType>& info) {
  const DynamicSliceTestParam param(info.param);
  return absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(param.data_type), "_",
      primitive_util::LowercasePrimitiveTypeName(param.index_type), "_",
      param.is_the_majormost_dim_being_sliced ? "majormost" : "not_majormost");
}

// We pass the tuple type here instead of the struct, to avoid the usage of
// ::testing::ConvertGenerator, which broke the build in some OSS
// configurations.
class DynamicSliceTest
    : public TritonSupportTest,
      public ::testing::WithParamInterface<DynamicSliceTestParam::TupleType> {};

TEST_P(DynamicSliceTest, IsTritonSupportedDynamicSlice) {
  const DynamicSliceTestParam param(GetParam());

  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      param.data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  constexpr absl::string_view kHloTestTemplate =
      R"(
HloModule m

triton_computation {
  dynamic_slice_input = $0[$2,$3] parameter(0)
  dot_rhs = f32[2,4] parameter(1)
  start_index0 = $1[] parameter(2)
  start_index1 = $1[] parameter(3)
  dynamic_slice = $0[5,2] dynamic-slice(dynamic_slice_input, start_index0, start_index1),
                  dynamic_slice_sizes={5,2}
  convert = f32[5,2] convert(dynamic_slice)
  ROOT dot = f32[5, 4] dot(convert, dot_rhs),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  dynamic_slice_input = $0[$2,$3] parameter(0)
  dot_rhs = f32[2,4] parameter(1)
  start_index0 = $1[] constant($4)
  start_index1 = $1[] constant($5)
  ROOT fusion = f32[5,4] fusion(dynamic_slice_input, dot_rhs, start_index0, start_index1),
       kind=kCustom, calls=triton_computation,
       backend_config={
         "fusion_backend_config":{
           "kind":"__triton_gemm","triton_gemm_config":{
             "block_m":"32","block_n":"32","block_k":"32","split_k":"1",
             "num_stages":"1","num_warps":"4","num_ctas":"1"}}}
})";

  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate,
      primitive_util::LowercasePrimitiveTypeName(param.data_type),
      primitive_util::LowercasePrimitiveTypeName(param.index_type),
      param.is_the_majormost_dim_being_sliced ? 7 : 5,  // input dim0
      param.is_the_majormost_dim_being_sliced ? 2 : 4,  // input dim1
      param.is_the_majormost_dim_being_sliced ? 1 : 0,  // start_index0
      param.is_the_majormost_dim_being_sliced ? 0 : 1   // start_index1
  );
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));
  const HloComputation* computation =
      module->GetComputationWithName("triton_computation");
  ASSERT_NE(computation, nullptr);
  const HloInstruction* instr = hlo_query::GetFirstInstructionWithOpcode(
      *computation, HloOpcode::kDynamicSlice);

  const bool is_supported_instruction =
      IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
          .CanFuse();
  const bool is_supported_dynamic_slice =
      IsTritonSupportedDynamicSlice(*Cast<HloDynamicSliceInstruction>(instr))
          .CanFuse();
  EXPECT_EQ(is_supported_instruction, is_supported_dynamic_slice);

  if (is_supported_instruction) {
    TF_EXPECT_OK(ApplyFloatNormalization(module.get()));
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(module), ErrorSpec{/*aabs=*/2e-4, /*arel=*/2e-4}));
  } else {
    EXPECT_THAT(TritonFusionAnalysis::Execute(*computation),
                tsl::testing::StatusIs(absl::StatusCode::kFailedPrecondition));
  }
}

INSTANTIATE_TEST_SUITE_P(
    All, DynamicSliceTest,
    ::testing::Combine(::testing::Values(F16, BF16, F32),
                       ::testing::Values(S8, S16, S32, S64, U8, U16, U32, U64),
                       ::testing::Bool()),
    DynamicSliceTestParamToString);

TEST_F(TritonSupportTest, UnsupportedDotOutputTypeFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
triton_computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = f32[11,63]{1,0} parameter(1)
  ROOT dot = pred[92,63]{1,0} dot(parameter_0, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = f32[11,63]{1,0} parameter(1)
  ROOT triton_op = pred[92,63]{1,0} fusion(parameter_0, parameter_1), kind=kCustom,
    calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  const HloComputation* computation = fusion->fused_instructions_computation();
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr("Unsupported output data type for Dot op."));
  EXPECT_THAT(
      TritonWrapper(*TritonFusionAnalysis::Execute(*computation), "test_fn",
                    fusion, GetCudaComputeCapability(), dev_info, config_,
                    /*output_tile_sizes=*/{}, &llvm_module_, &EmitMatMul,
                    mlir_context_),
      tsl::testing::StatusIs(
          absl::StatusCode::kInternal,
          ::testing::HasSubstr("pm.run(triton_module.get()).succeeded()")));
}

TEST_F(TritonSupportTest,
       UnsupportedDotWithMultipleBatchDimensionsFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
triton_computation {
  parameter_0 = f32[2,2,2,2]{3,2,1,0} parameter(0)
  parameter_1 = f32[2,2,2,2]{3,2,1,0} parameter(1)
  ROOT dot = f32[2,2,2,2]{3,2,1,0} dot(parameter_0, parameter_1),
    lhs_contracting_dims={3}, lhs_batch_dims={1,0}, rhs_contracting_dims={2},
    rhs_batch_dims={1,0}
}

ENTRY e {
  parameter_0 = f32[2,2,2,2]{3,2,1,0} parameter(0)
  parameter_1 = f32[2,2,2,2]{3,2,1,0} parameter(1)
  ROOT triton_op = f32[2,2,2,2]{3,2,1,0} fusion(parameter_0, parameter_1),
    kind=kCustom, calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  const HloComputation* computation = fusion->fused_instructions_computation();
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr("Multiple batch dimensions"));
  EXPECT_THAT(
      TritonWrapper(*TritonFusionAnalysis::Execute(*computation), "test_fn",
                    fusion, GetCudaComputeCapability(), dev_info, config_,
                    /*output_tile_sizes=*/{}, &llvm_module_, &EmitMatMul,
                    mlir_context_),
      tsl::testing::StatusIs(absl::StatusCode::kInternal,
                             ::testing::HasSubstr("num_batch_dims <= 1")));
}

TEST_F(TritonSupportTest,
       UnsupportedDotWithNoNonContractingDimensionsFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
triton_computation {
  parameter_0 = f32[2]{0} parameter(0)
  parameter_1 = f32[2]{0} parameter(1)
  ROOT dot = f32[] dot(parameter_0, parameter_1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  parameter_0 = f32[2]{0} parameter(0)
  parameter_1 = f32[2]{0} parameter(1)
  ROOT triton_op = f32[] fusion(parameter_0, parameter_1), kind=kCustom,
    calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  EXPECT_THAT(IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr("No non-contracting dimensions."));
  EXPECT_THAT(TritonFusionAnalysis::Execute(*computation),
              tsl::testing::StatusIs(
                  absl::StatusCode::kInternal,
                  ::testing::HasSubstr("non_contracting_dims.size() == 1")));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
