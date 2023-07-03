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
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/tensor_float_32_utils.h"

namespace xla {
namespace gpu {
namespace {

struct MixTypeParams {
  PrimitiveType lhs_ty;
  PrimitiveType rhs_ty;
  int m;
  int k;
  int n;
  float aabs = 1e-6;
  float arel = 1e-6;
};

class MixedTypeTest : public GpuCodegenTest,
                      public ::testing::WithParamInterface<MixTypeParams> {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

TEST_P(MixedTypeTest, MixedTypeDotProducesCorrectResult) {
  MixTypeParams params = GetParam();
  if ((params.lhs_ty == BF16 || params.rhs_ty == BF16) &&
      !GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }
  const std::string hlo_string_template = R"(
HloModule m

ENTRY e {
  p0 = $0[$2,$3] parameter(0)
  p0c = $1[$2,$3] convert(p0)
  p1 = $1[$3,$4] parameter(1)
  ROOT _ = $1[$2,$4] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  std::string hlo_string = absl::Substitute(
      hlo_string_template,
      primitive_util::LowercasePrimitiveTypeName(params.lhs_ty),
      primitive_util::LowercasePrimitiveTypeName(params.rhs_ty), params.m,
      params.k, params.n);
  MatchOptimizedHlo(hlo_string, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: kCustom
)");

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{params.aabs, params.arel}));
}

std::string GemmTestParamsParamsToString(
    const ::testing::TestParamInfo<MixTypeParams>& data) {
  return absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(data.param.lhs_ty), "_",
      primitive_util::LowercasePrimitiveTypeName(data.param.rhs_ty), "_",
      data.param.m, "_", data.param.k, "_", data.param.n);
}

INSTANTIATE_TEST_SUITE_P(RewriteTestSuite, MixedTypeTest,
                         ::testing::ValuesIn({
                             MixTypeParams{PRED, F16, 16, 32, 8},
                             MixTypeParams{PRED, BF16, 16, 32, 8},
                             MixTypeParams{PRED, F32, 16, 32, 8, 1e-4, 1e-3},
                             MixTypeParams{S8, F16, 16, 32, 8},
                             MixTypeParams{S8, BF16, 16, 32, 8},
                             MixTypeParams{S8, F32, 16, 32, 8, 5e-2, 1e-2},
                             MixTypeParams{S8, F32, 101, 7, 303, 0.1, 0.1},
                             MixTypeParams{S8, F32, 101, 32, 303, 0.1, 0.1},
                             MixTypeParams{S8, F32, 101, 2048, 303, 0.5, 0.1},
                             MixTypeParams{S8, F32, 101, 2555, 303, 0.5, 0.1},
                             // Is supported but overflows.
                             //  GemmTestParams{S32, F16},
                             MixTypeParams{S16, F16, 30, 19, 12},
                             MixTypeParams{S32, F32, 4, 4, 4, 1, 1e-2},
                             MixTypeParams{F16, BF16, 16, 32, 8},
                             MixTypeParams{F16, F32, 16, 32, 8, 1e-3, 1e-6},
                             MixTypeParams{BF16, F16, 16, 32, 8, 1e-3, 1e-6},
                             MixTypeParams{BF16, F32, 16, 32, 8, 1e-3, 1e-6},
                             // Supported but disabled because narrowing
                             // converts should rather belong to producers.
                             // TODO(b/266862493): Move these to CompareTest.
                             // TritonRewriteTest2Params{S32, BF16},
                             //  TritonRewriteTest2Params{F32, F16},
                             //  TritonRewriteTest2Params{F32, BF16},
                             MixTypeParams{S8, BF16, 24, 40, 8},
                             MixTypeParams{S8, F16, 80, 16, 32},
                             MixTypeParams{F16, F32, 127, 3, 300, 1e-2, 1e-2},
                             MixTypeParams{F16, BF16, 544, 96, 16, 1e-3, 1e-3},
                             MixTypeParams{BF16, F32, 77, 500, 333, 3e-3, 3e-3},
                         }),
                         GemmTestParamsParamsToString);

class NoTF32Test : public GpuCodegenTest {
 public:
  void SetUp() override {
    tf32_state_ = tsl::tensor_float_32_execution_enabled();
    // Elementwise operations do not use tf32 anyway but disabling it on the
    // dot allows verification in higher precision for f32.
    tsl::enable_tensor_float_32_execution(false);
  }
  void TearDown() override {
    tsl::enable_tensor_float_32_execution(tf32_state_);
  }

 private:
  bool tf32_state_;
};

class ElementwiseTest : public NoTF32Test,
                        public ::testing::WithParamInterface<
                            std::tuple<PrimitiveType, HloOpcode, float>> {};

std::string ElementwiseTestParamsToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, HloOpcode, float>>&
        data) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = data.param;
  return absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(data_type), "_",
      absl::StrReplaceAll(HloOpcodeString(opcode), {{"-", "_"}}));
}

using UnaryElementwiseTest = ElementwiseTest;

TEST_P(UnaryElementwiseTest, ElementwiseFusionExecutesCorrectly) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = GetParam();

  const std::string hHloTestTemplate = R"(
HloModule m, is_scheduled=true

triton_gemm___computation {
  parameter_0 = f32[15,33]{1,0} parameter(0)
  parameter_1 = $0[33,68]{1,0} parameter(1)
  f1.1 = $0[33,68]{1,0} $1(parameter_1)
  c.1 = f32[33,68]{1,0} convert(f1.1)
  ROOT _.1 = f32[15,68]{1,0} dot(parameter_0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p1 = $0[33,68]{1,0} parameter(1)
  p0 = f32[15,33]{1,0} parameter(0)
  ROOT triton_gemm__ = f32[15,68]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"kind":"__triton_gemm",
                    "triton_gemm_config":{"block_m":"32","block_n":"32",
                                          "block_k":"32","split_k":"1",
                                          "num_stages":"1","num_warps":"4"}}
})";
  const std::string hlo_test = absl::Substitute(
      hHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  const std::string hHloRefTemplate = R"(
HloModule m, is_scheduled=true

fused_computation {
  param_0.1 = $0[33,68]{1,0} parameter(0)
  f.1 = $0[33,68]{1,0} $1(param_0.1)
  ROOT convert.1 = f32[33,68]{1,0} convert(f.1)
}

ENTRY e {
  p1 = $0[33,68]{1,0} parameter(1)
  p0 = f32[15,33]{1,0} parameter(0)
  fusion = f32[33,68]{1,0} fusion(p1), kind=kLoop, calls=fused_computation
  ROOT custom-call = f32[15,68]{1,0} custom-call(p0, fusion),
    custom_call_target="__cublas$$gemm",
    backend_config={"alpha_real":1,"beta":0,"dot_dimension_numbers":
      {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],
      "lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},
      "alpha_imag":0,"precision_config":
      {"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}
})";
  const std::string hlo_ref = absl::Substitute(
      hHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_test, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance},
      /*run_hlo_passes=*/false));
}

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuitePRED, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(PRED),
        ::testing::ValuesIn(TritonSupportedUnaryElementwise(PRED)),
        ::testing::Values(3e-2)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS8, UnaryElementwiseTest,
    ::testing::Combine(::testing::Values(S8),
                       ::testing::ValuesIn(TritonSupportedUnaryElementwise(S8)),
                       ::testing::Values(3e-2)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS16, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(S16),
        ::testing::ValuesIn(TritonSupportedUnaryElementwise(S16)),
        ::testing::Values(1e-3)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS32, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(S32),
        ::testing::ValuesIn(TritonSupportedUnaryElementwise(S32)),
        ::testing::Values(1e-3)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteF16, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(F16),
        ::testing::ValuesIn(TritonSupportedUnaryElementwise(F16)),
        ::testing::Values(2e-4)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteF32, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(F32),
        ::testing::ValuesIn(TritonSupportedUnaryElementwise(F32)),
        ::testing::Values(1e-6)),
    ElementwiseTestParamsToString);

using BinaryElementwiseTest = ElementwiseTest;

TEST_P(BinaryElementwiseTest, ElementwiseFusionExecutesCorrectly) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = GetParam();

  const std::string hHloTestTemplate = R"(
HloModule m, is_scheduled=true

triton_gemm___computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  parameter_2 = $0[11,63]{1,0} parameter(2)
  f1.1 = $0[11,63]{1,0} $1(parameter_1, parameter_2)
  c.1 = f32[11,63]{1,0} convert(f1.1)
  ROOT _.1 = f32[92,63]{1,0} dot(parameter_0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[92,11]{1,0} parameter(0)
  p1 = $0[11,63]{1,0} parameter(1)
  p2 = $0[11,63]{1,0} parameter(2)
  ROOT triton_gemm__ = f32[92,63]{1,0} fusion(p0, p1, p2), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"kind":"__triton_gemm",
                    "triton_gemm_config":{"block_m":"64","block_n":"32",
                                          "block_k":"64","split_k":"1",
                                          "num_stages":"2","num_warps":"2"}}
})";
  const std::string hlo_test = absl::Substitute(
      hHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  const std::string hHloRefTemplate = R"(
HloModule m, is_scheduled=true

fused_computation {
  p0 = $0[11,63]{1,0} parameter(0)
  p1 = $0[11,63]{1,0} parameter(1)
  f.1 = $0[11,63]{1,0} $1(p0, p1)
  ROOT convert.1 = f32[11,63]{1,0} convert(f.1)
}

ENTRY e {
  p2 = $0[11,63]{1,0} parameter(2)
  p1 = $0[11,63]{1,0} parameter(1)
  p0 = f32[92,11]{1,0} parameter(0)
  fusion = f32[11,63]{1,0} fusion(p1, p2), kind=kLoop, calls=fused_computation
  ROOT custom-call = f32[92,63]{1,0} custom-call(p0, fusion),
    custom_call_target="__cublas$$gemm",
    backend_config={"alpha_real":1,"beta":0,"dot_dimension_numbers":
      {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],
      "lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},
      "alpha_imag":0,"precision_config":
      {"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}
})";
  const std::string hlo_ref = absl::Substitute(
      hHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_test, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance},
      /*run_hlo_passes=*/false));
}

std::vector<HloOpcode> TestedBinaryElementwise(PrimitiveType element_type) {
  std::vector<HloOpcode> ret = TritonSupportedBinaryElementwise(element_type);
  // Comparison requires an additional property.
  ret.erase(std::remove_if(ret.begin(), ret.end(), HloOpcodeIsComparison),
            ret.end());
  return ret;
}

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuitePRED, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(PRED),
                       ::testing::ValuesIn(TestedBinaryElementwise(PRED)),
                       ::testing::Values(3e-2)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS8, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(S8),
                       ::testing::ValuesIn(TestedBinaryElementwise(S8)),
                       ::testing::Values(3e-2)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS16, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(S16),
                       ::testing::ValuesIn(TestedBinaryElementwise(S16)),
                       ::testing::Values(1e-3)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS32, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(S32),
                       ::testing::ValuesIn(TestedBinaryElementwise(S32)),
                       ::testing::Values(1e-5)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteF16, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(F16),
                       ::testing::ValuesIn(TestedBinaryElementwise(F16)),
                       ::testing::Values(2e-4)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteF32, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(F32),
                       ::testing::ValuesIn(TestedBinaryElementwise(F32)),
                       ::testing::Values(1e-6)),
    ElementwiseTestParamsToString);

class CompareTest : public NoTF32Test,
                    public ::testing::WithParamInterface<
                        std::tuple<PrimitiveType, Comparison::Direction>> {};

std::string CompareTestParamsToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, Comparison::Direction>>& data) {
  PrimitiveType data_type;
  Comparison::Direction direction;
  std::tie(data_type, direction) = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type),
                      "_", ComparisonDirectionToString(direction));
}

TEST_P(CompareTest, CompareFusionExecutesCorrectly) {
  PrimitiveType data_type;
  Comparison::Direction direction;
  std::tie(data_type, direction) = GetParam();

  const std::string hHloTestTemplate = R"(
HloModule m, is_scheduled=true

triton_gemm___computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  parameter_2 = $0[11,63]{1,0} parameter(2)
  f1.1 = pred[11,63]{1,0} compare(parameter_1, parameter_2), direction=$1
  c.1 = f32[11,63]{1,0} convert(f1.1)
  ROOT _.1 = f32[92,63]{1,0} dot(parameter_0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[92,11]{1,0} parameter(0)
  p1 = $0[11,63]{1,0} parameter(1)
  p2 = $0[11,63]{1,0} parameter(2)
  ROOT triton_gemm__ = f32[92,63]{1,0} fusion(p0, p1, p2), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"kind":"__triton_gemm",
                    "triton_gemm_config":{"block_m":"16","block_n":"64",
                                          "block_k":"16","split_k":"1",
                                          "num_stages":"3","num_warps":"2"}}
})";
  const std::string hlo_test = absl::Substitute(
      hHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      ComparisonDirectionToString(direction));

  const std::string hHloRefTemplate = R"(
HloModule m, is_scheduled=true

fused_computation {
  p0 = $0[11,63]{1,0} parameter(0)
  p1 = $0[11,63]{1,0} parameter(1)
  f.1 = pred[11,63]{1,0} compare(p0, p1), direction=$1
  ROOT convert.1 = f32[11,63]{1,0} convert(f.1)
}

ENTRY e {
  p2 = $0[11,63]{1,0} parameter(2)
  p1 = $0[11,63]{1,0} parameter(1)
  p0 = f32[92,11]{1,0} parameter(0)
  fusion = f32[11,63]{1,0} fusion(p1, p2), kind=kLoop, calls=fused_computation
  ROOT custom-call = f32[92,63]{1,0} custom-call(p0, fusion),
    custom_call_target="__cublas$$gemm",
    backend_config={"alpha_real":1,"beta":0,"dot_dimension_numbers":
      {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],
      "lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},
      "alpha_imag":0,"precision_config":
      {"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}
})";
  const std::string hlo_ref = absl::Substitute(
      hHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      ComparisonDirectionToString(direction));

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case PRED:
    case S8:
      tolerance = 3e-2;
      break;
    case S16:
      tolerance = 1e-3;
      break;
    case S32:
      tolerance = 1e-5;
      break;
    default:
      CHECK(false);
  }
  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_test, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance},
      /*run_hlo_passes=*/false));
}

using cd = Comparison::Direction;

INSTANTIATE_TEST_SUITE_P(
    CompareTestSuite, CompareTest,
    ::testing::Combine(::testing::Values(PRED, S8, S16, S32, F16, F32),
                       ::testing::Values(cd::kEq, cd::kNe, cd::kGe, cd::kGt,
                                         cd::kLe, cd::kLt)),
    CompareTestParamsToString);

class SelectTest : public NoTF32Test,
                   public ::testing::WithParamInterface<
                       std::tuple<PrimitiveType, PrimitiveType>> {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

TEST_P(SelectTest, SelectFusionExecutesCorrectly) {
  PrimitiveType data_type1;
  PrimitiveType data_type2;
  std::tie(data_type1, data_type2) = GetParam();

  if ((data_type1 == BF16 || data_type2 == BF16) &&
      !GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string hHloTestTemplate = R"(
HloModule m, is_scheduled=true

triton_gemm___computation {
  parameter_0 = $1[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  parameter_2 = $0[11,63]{1,0} parameter(2)
  parameter_3 = pred[11,63]{1,0} parameter(3)
  f1.1 = $0[11,63]{1,0} select(parameter_3, parameter_1, parameter_2)
  c.1 = $1[11,63]{1,0} convert(f1.1)
  ROOT _.1 = $1[92,63]{1,0} dot(parameter_0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = $1[92,11]{1,0} parameter(0)
  p1 = $0[11,63]{1,0} parameter(1)
  p2 = $0[11,63]{1,0} parameter(2)
  p3 = pred[11,63]{1,0} parameter(3)
  ROOT triton_gemm__ = $1[92,63]{1,0} fusion(p0, p1, p2, p3), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"kind":"__triton_gemm",
                    "triton_gemm_config":{"block_m":"16","block_n":"64",
                                          "block_k":"16","split_k":"1",
                                          "num_stages":"3","num_warps":"2"}}
})";
  const std::string hlo_test = absl::Substitute(
      hHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type1),
      primitive_util::LowercasePrimitiveTypeName(data_type2));

  const std::string hHloRefTemplate = R"(
HloModule m, is_scheduled=true

fused_computation {
  p0 = $0[11,63]{1,0} parameter(0)
  p1 = $0[11,63]{1,0} parameter(1)
  p2 = pred[11,63]{1,0} parameter(2)
  f.1 = $0[11,63]{1,0} select(p2, p0, p1)
  ROOT convert.1 = $1[11,63]{1,0} convert(f.1)
}

ENTRY e {
  p3 = pred[11,63]{1,0} parameter(3)
  p2 = $0[11,63]{1,0} parameter(2)
  p1 = $0[11,63]{1,0} parameter(1)
  p0 = $1[92,11]{1,0} parameter(0)
  fusion = $1[11,63]{1,0} fusion(p1, p2, p3), kind=kLoop,
    calls=fused_computation
  ROOT custom-call = $1[92,63]{1,0} custom-call(p0, fusion),
    custom_call_target="__cublas$$gemm",
    backend_config={"alpha_real":1,"beta":0,"dot_dimension_numbers":
      {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],
      "lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},
      "alpha_imag":0,"precision_config":
      {"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}
})";
  const std::string hlo_ref = absl::Substitute(
      hHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type1),
      primitive_util::LowercasePrimitiveTypeName(data_type2));

  float tolerance;
  switch (data_type1) {
    case F32:
      tolerance = 1e-6;
      break;
    case BF16:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case PRED:
    case S8:
      tolerance = 3e-2;
      break;
    case S16:
      tolerance = 1e-3;
      break;
    case S32:
      tolerance = 1e-5;
      break;
    default:
      CHECK(false);
  }
  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_test, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance},
      /*run_hlo_passes=*/false));
}

std::string SelectTestParamsToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, PrimitiveType>>&
        data) {
  PrimitiveType data_type1;
  PrimitiveType data_type2;
  std::tie(data_type1, data_type2) = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type1),
                      "_",
                      primitive_util::LowercasePrimitiveTypeName(data_type2));
}

INSTANTIATE_TEST_SUITE_P(
    SelectTestSuite, SelectTest,
    ::testing::Combine(::testing::Values(PRED, S8, S16, S32, F16, BF16, F32),
                       ::testing::Values(F16, BF16, F32)),
    SelectTestParamsToString);

class ConstantTest : public NoTF32Test,
                     public ::testing::WithParamInterface<PrimitiveType> {};

TEST_P(ConstantTest, ConstantFusionExecutesCorrectly) {
  PrimitiveType data_type = GetParam();

  const std::string hHloTestTemplate = R"(
HloModule m, is_scheduled=true

triton_gemm___computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = f32[11,63]{1,0} parameter(1)
  c = $0[] constant(123)
  b = $0[11,63] broadcast(c)
  cv = f32[11,63] convert(b)
  m = f32[11,63] multiply(cv, parameter_1)
  ROOT _.1 = f32[92,63]{1,0} dot(parameter_0, m),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[92,11]{1,0} parameter(0)
  p1 = f32[11,63]{1,0} parameter(1)
  ROOT triton_gemm__ = f32[92,63]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"kind":"__triton_gemm",
                    "triton_gemm_config":{"block_m":"16","block_n":"64",
                                          "block_k":"16","split_k":"1",
                                          "num_stages":"3","num_warps":"2"}}
})";
  const std::string hlo_test = absl::Substitute(
      hHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hHloRefTemplate = R"(
HloModule m, is_scheduled=true

fused_computation {
  p0 = f32[11,63]{1,0} parameter(0)
  c = $0[] constant(123)
  b = $0[11,63] broadcast(c)
  cv = f32[11,63] convert(b)
  ROOT m = f32[11,63] multiply(cv, p0)
}

ENTRY e {
  p1 = f32[11,63]{1,0} parameter(1)
  p0 = f32[92,11]{1,0} parameter(0)
  fusion = f32[11,63]{1,0} fusion(p1), kind=kLoop,
    calls=fused_computation
  ROOT custom-call = f32[92,63]{1,0} custom-call(p0, fusion),
    custom_call_target="__cublas$$gemm",
    backend_config={"alpha_real":1,"beta":0,"dot_dimension_numbers":
      {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],
      "lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},
      "alpha_imag":0,"precision_config":
      {"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}
})";
  const std::string hlo_ref = absl::Substitute(
      hHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type));

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case PRED:
    case S8:
      tolerance = 3e-2;
      break;
    case S16:
      tolerance = 1e-3;
      break;
    case S32:
      tolerance = 1e-5;
      break;
    default:
      CHECK(false);
  }
  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_test, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance},
      /*run_hlo_passes=*/false));
}

INSTANTIATE_TEST_SUITE_P(ConstantTestSuite, ConstantTest,
                         ::testing::Values(PRED, S8, S16, S32, F16, F32));

}  // namespace
}  // namespace gpu
}  // namespace xla
