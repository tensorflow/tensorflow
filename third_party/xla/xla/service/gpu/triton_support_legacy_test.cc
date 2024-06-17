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
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emitter_triton.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/gpu/triton_support.h"
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

bool CombinationCrashesTriton(
    PrimitiveType lhs_type, PrimitiveType rhs_type, PrimitiveType output_type,
    se::CudaComputeCapability cuda_compute_capability) {
  if (!cuda_compute_capability.IsAtLeastHopper() &&
      (lhs_type == F8E4M3FN || rhs_type == F8E4M3FN ||
       output_type == F8E4M3FN)) {
    return true;
  }
  return false;
}

class DotTest : public TritonSupportTestWithParam {
 protected:
  void TestDotWithTypes(PrimitiveType lhs_type, PrimitiveType rhs_type,
                        PrimitiveType output_type) {
    if (lhs_type == BF16 && SkipBF16Tests()) {
      GTEST_SKIP();
    }
    const HloOpcode opcode = HloOpcode::kDot;
    const std::string lhs =
        primitive_util::LowercasePrimitiveTypeName(lhs_type);
    const std::string rhs =
        primitive_util::LowercasePrimitiveTypeName(rhs_type);
    const std::string output =
        primitive_util::LowercasePrimitiveTypeName(output_type);

    const std::string kHloTestTemplate = R"(
triton_computation {
  parameter_0 = $0[92,11]{1,0} parameter(0)
  parameter_1 = $1[11,63]{1,0} parameter(1)
  ROOT dot = $2[92,63]{1,0} $3(parameter_0, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  parameter_0 = $0[92,11]{1,0} parameter(0)
  parameter_1 = $1[11,63]{1,0} parameter(1)
  ROOT triton_op = $2[92,63]{1,0} fusion(parameter_0, parameter_1), kind=kCustom,
    calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm",
      triton_gemm_config:
        {"block_m":16,"block_n":32,"block_k":512,
         "split_k":1,"num_stages":4,"num_warps":8,
         "num_ctas":1}}}
})";
    // TODO(b/345763510): Change the kind above to "__triton" once dots are
    // supported.
    const std::string hlo_test = absl::Substitute(
        kHloTestTemplate, lhs, rhs, output, HloOpcodeString(opcode));
    TF_ASSERT_OK_AND_ASSIGN(
        TestedInstruction ti,
        ParseTemplateAndGetInstruction(hlo_test, /*data_type=*/{}, opcode));
    if (legacy_triton::IsTritonSupportedInstruction(
            ti.Instruction(), GetCudaComputeCapability())) {
      TF_EXPECT_OK(ApplyFloatNormalization(ti.Module().get()));
      EXPECT_TRUE(RunAndCompareNoHloPasses(
          std::move(ti.Module()),
          ErrorSpec{/*aabs=*/primitive_util::IsF8Type(lhs_type) ? 1.0 : 2e-4,
                    /*arel=*/2e-4}));
    } else {
      if (CombinationCrashesTriton(lhs_type, rhs_type, output_type,
                                   GetCudaComputeCapability())) {
        return;
      }
      const se::DeviceDescription dev_info =
          TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
      BlockLevelParameters block_level_parameters;
      block_level_parameters.num_ctas = 1;
      block_level_parameters.num_stages = 4;
      block_level_parameters.num_warps = 8;
      EXPECT_THAT(
          TritonWrapper("test_fn", &ti.TritonFusion(),
                        GetCudaComputeCapability(), dev_info,
                        block_level_parameters, &llvm_module_, mlir_context_),
          tsl::testing::StatusIs(
              absl::StatusCode::kInternal,
              ::testing::HasSubstr("Failed to compile Triton kernel")));
    }
  }
};

TEST_P(DotTest, IsTritonSupportedExecutesCorrectlyForDot) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  CHECK_EQ(opcode, HloOpcode::kDot);
  TestDotWithTypes(data_type, data_type, data_type);

  switch (data_type) {
    case F8E5M2:
      TestDotWithTypes(F8E5M2, F8E4M3FN, F32);
      TestDotWithTypes(F8E5M2, F8E5M2, F16);
      TestDotWithTypes(F8E5M2, F8E5M2, F32);
      break;
    case F8E4M3FN:
      TestDotWithTypes(F8E4M3FN, F8E5M2, F32);
      TestDotWithTypes(F8E4M3FN, F8E4M3FN, F16);
      TestDotWithTypes(F8E4M3FN, F8E4M3FN, F32);
      break;
    default:
      break;
  }
}

INSTANTIATE_TEST_SUITE_P(DotTestTestSuite, DotTest,
                         ::testing::Combine(::testing::Values(F16, F32, BF16,
                                                              F8E5M2, F8E4M3FN),
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
  if (param.data_type == BF16 && SkipBF16Tests()) {
    GTEST_SKIP();
  }

  constexpr absl::string_view kHloTestTemplate =
      R"(
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
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    hlo_test, /*data_type=*/{},
                                                    HloOpcode::kDynamicSlice));

  const bool is_supported_instruction =
      legacy_triton::IsTritonSupportedInstruction(ti.Instruction(),
                                                  GetCudaComputeCapability())
          .CanFuse();
  const bool is_supported_dynamic_slice =
      legacy_triton::IsTritonSupportedDynamicSlice(
          *Cast<HloDynamicSliceInstruction>(&ti.Instruction()))
          .CanFuse();
  EXPECT_EQ(is_supported_instruction, is_supported_dynamic_slice);

  if (is_supported_instruction) {
    TF_EXPECT_OK(ApplyFloatNormalization(ti.Module().get()));
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(ti.Module()), ErrorSpec{/*aabs=*/2e-4, /*arel=*/2e-4}));
  } else {
    EXPECT_THAT(TritonFusionAnalysis::Execute(ti.TritonComputation()),
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
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm",
      triton_gemm_config:
        {"block_m":16,"block_n":32,"block_k":512,
         "split_k":1,"num_stages":4,"num_warps":8,
         "num_ctas":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTest, /*data_type=*/{}, HloOpcode::kDot));
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(legacy_triton::IsTritonSupportedInstruction(
                  ti.Instruction(), GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr("Unsupported output data type for Dot op."));
  BlockLevelParameters block_level_parameters;
  block_level_parameters.num_ctas = 1;
  block_level_parameters.num_stages = 4;
  block_level_parameters.num_warps = 8;
  EXPECT_THAT(
      TritonWrapper("test_fn", &ti.TritonFusion(), GetCudaComputeCapability(),
                    dev_info, block_level_parameters, &llvm_module_,
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
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm",
      triton_gemm_config:
        {"block_m":16,"block_n":32,"block_k":512,
         "split_k":1,"num_stages":4,"num_warps":8,
         "num_ctas":1}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTest, /*data_type=*/{}, HloOpcode::kDot));
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(legacy_triton::IsTritonSupportedInstruction(
                  ti.Instruction(), GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr("Multiple batch dimensions"));
  BlockLevelParameters block_level_parameters;
  block_level_parameters.num_ctas = 1;
  block_level_parameters.num_stages = 4;
  block_level_parameters.num_warps = 8;
  EXPECT_THAT(
      TritonWrapper("test_fn", &ti.TritonFusion(), GetCudaComputeCapability(),
                    dev_info, block_level_parameters, &llvm_module_,
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
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTest, /*data_type=*/{}, HloOpcode::kDot));
  EXPECT_THAT(legacy_triton::IsTritonSupportedInstruction(
                  ti.Instruction(), GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr("No non-contracting dimensions."));
  EXPECT_THAT(TritonFusionAnalysis::Execute(ti.TritonComputation()),
              tsl::testing::StatusIs(
                  absl::StatusCode::kInternal,
                  ::testing::HasSubstr("non_contracting_dims.size() == 1")));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
