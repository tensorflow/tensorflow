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

#include <algorithm>
#include <array>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/optimization.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/comparison_util.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/fusions/triton/triton_support_legacy.h"
#include "xla/service/gpu/fusions/triton/triton_test_utils.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

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
  se::GpuComputeCapability GetGpuComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability();
  }

  void SetUp() override {
    if (std::holds_alternative<se::RocmComputeCapability>(
            GetGpuComputeCapability())) {
      GTEST_SKIP()
          << "Related fusions are not performed on ROCm without Triton.";
    }
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // We are testing Triton, remove cuBLAS fallback for these tests.
    debug_options.set_xla_gpu_cublas_fallback(false);
    // Always rewrite Gemms with Triton regardless of size.
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
    return debug_options;
  }
};

TEST_P(MixedTypeTest, MixedTypeDotProducesCorrectResult) {
  MixTypeParams params = GetParam();
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
                             MixTypeParams{S8, F16, 80, 16, 32, 1e-3, 1e-6},
                             MixTypeParams{F16, F32, 127, 3, 300, 1e-2, 1e-2},
                             MixTypeParams{F16, BF16, 544, 96, 16, 1e-3, 1e-3},
                             MixTypeParams{BF16, F32, 77, 500, 333, 3e-3, 3e-3},
                         }),
                         GemmTestParamsParamsToString);

class TritonTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_triton_gemm_any(true);
    debug_options.set_xla_gpu_cublas_fallback(false);
    // Always rewrite Gemms with Triton regardless of size.
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
    return debug_options;
  }

  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

class ElementwiseTest : public TritonTest,
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

  const std::string kHloTestTemplate = R"(
triton_gemm___computation {
  parameter_0 = f32[15,33]{1,0} parameter(0)
  parameter_1 = $0[33,68]{1,0} parameter(1)
  f1.1 = $0[33,68]{1,0} $1(parameter_1)
  c.1 = f32[33,68]{1,0} convert(f1.1)
  ROOT _.1 = f32[15,68]{1,0} dot(parameter_0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  p1 = $0[33,68]{1,0} parameter(1)
  p0 = f32[15,33]{1,0} parameter(0)
  ROOT triton_gemm__ = f32[15,68]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm",
                    "triton_gemm_config":
                      {"block_m":"32",
                       "block_n":"32",
                       "block_k":"32",
                       "split_k":"1",
                       "num_stages":"1",
                       "num_warps":"4",
                       "num_ctas":"1"}}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  const std::string kHloRefTemplate = R"(
fused_computation {
  param_0.1 = $0[33,68]{1,0} parameter(0)
  f.1 = $0[33,68]{1,0} $1(param_0.1)
  ROOT convert.1 = f32[33,68]{1,0} convert(f.1)
}

ENTRY e {
  p1 = $0[33,68]{1,0} parameter(1)
  p0 = f32[15,33]{1,0} parameter(0)
  fusion = f32[33,68]{1,0} fusion(p1), kind=kLoop, calls=fused_computation
  gemm = (f32[15,68]{1,0}, s8[0]{0}) custom-call(p0, fusion),
    custom_call_target="__cublas$$gemm",
    backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":
      {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],
      "lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},
      "alpha_imag":0,"precision_config":
      {"operand_precision":["HIGHEST","HIGHEST"]},"epilogue":"DEFAULT"}}
   ROOT get-tuple-element = f32[15,68]{1,0} get-tuple-element((f32[15,68]{1,0}, s8[0]{0}) gemm), index=0
})";
  const std::string hlo_ref = absl::Substitute(
      kHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_test, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance},
      /*run_hlo_passes=*/false));
}

TEST_P(UnaryElementwiseTest, ElementwiseUnaryOpExecutesCorrectly) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = GetParam();

  const std::string kHloTestTemplate = R"(
triton_computation {
  parameter_0 = $0[33,68]{1,0} parameter(0)
  output = $0[33,68]{1,0} $1(parameter_0)
  ROOT convert = f32[33,68]{1,0} convert(output)
}

ENTRY e {
  p0 = $0[33,68]{1,0} parameter(0)
  ROOT triton_fusion = f32[33,68]{1,0} fusion(p0), kind=kCustom,
    calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton",
                    "block_level_fusion_config":{"output_tile_sizes":["1", "1"],"num_warps":"1"}}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  const std::string kHloRefTemplate = R"(
fused_computation {
  param_0.1 = $0[33,68]{1,0} parameter(0)
  output = $0[33,68]{1,0} $1(param_0.1)
  ROOT convert = f32[33,68]{1,0} convert(output)
}

ENTRY e {
  p0 = $0[33,68]{1,0} parameter(0)
  ROOT fusion = f32[33,68]{1,0} fusion(p0), kind=kLoop, calls=fused_computation
})";
  const std::string hlo_ref = absl::Substitute(
      kHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_test, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance},
      /*run_hlo_passes=*/false));
}

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuitePRED, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(PRED),
        ::testing::ValuesIn(
            legacy_triton::
                TritonSupportedUnaryElementwiseUpToFloatNormalization(PRED)),
        ::testing::Values(3e-2)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS8, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(S8),
        ::testing::ValuesIn(
            legacy_triton::
                TritonSupportedUnaryElementwiseUpToFloatNormalization(S8)),
        ::testing::Values(3e-2)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS16, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(S16),
        ::testing::ValuesIn(
            legacy_triton::
                TritonSupportedUnaryElementwiseUpToFloatNormalization(S16)),
        ::testing::Values(1e-3)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS32, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(S32),
        ::testing::ValuesIn(
            legacy_triton::
                TritonSupportedUnaryElementwiseUpToFloatNormalization(S32)),
        ::testing::Values(1e-3)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteF16, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(F16),
        ::testing::ValuesIn(
            legacy_triton::
                TritonSupportedUnaryElementwiseUpToFloatNormalization(F16)),
        ::testing::Values(2e-4)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteF32, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(F32),
        ::testing::ValuesIn(
            legacy_triton::
                TritonSupportedUnaryElementwiseUpToFloatNormalization(F32)),
        ::testing::Values(1e-6)),
    ElementwiseTestParamsToString);

using BinaryElementwiseTest = ElementwiseTest;

TEST_P(BinaryElementwiseTest, ElementwiseFusionExecutesCorrectly) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = GetParam();

  const std::string kHloTestTemplate = R"(
triton_gemm___computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  parameter_2 = $0[11,63]{1,0} parameter(2)
  f1.1 = $0[11,63]{1,0} $1(parameter_1, parameter_2)
  c.1 = f32[11,63]{1,0} convert(f1.1)
  ROOT _.1 = f32[92,63]{1,0} dot(parameter_0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  p0 = f32[92,11]{1,0} parameter(0)
  p1 = $0[11,63]{1,0} parameter(1)
  p2 = $0[11,63]{1,0} parameter(2)
  ROOT triton_gemm__ = f32[92,63]{1,0} fusion(p0, p1, p2), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm",
                    "triton_gemm_config":
                      {"block_m":"64",
                       "block_n":"32",
                       "block_k":"64",
                       "split_k":"1",
                       "num_stages":"2",
                       "num_warps":"2",
                       "num_ctas":"1"}}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  const std::string kHloRefTemplate = R"(
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
  gemm = (f32[92,63]{1,0}, s8[0]{0}) custom-call(p0, fusion),
    custom_call_target="__cublas$$gemm",
    backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":
      {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],
      "lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},
      "alpha_imag":0,"precision_config":
      {"operand_precision":["HIGHEST","HIGHEST"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f32[92,63]{1,0} get-tuple-element((f32[92,63]{1,0}, s8[0]{0}) gemm), index=0
})";
  const std::string hlo_ref = absl::Substitute(
      kHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_test, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance},
      /*run_hlo_passes=*/false, /*args_max_bits_of_precision=*/6));
}

TEST_P(BinaryElementwiseTest, ElementwiseBinaryOpExecutesCorrectly) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = GetParam();

  const std::string kHloTestTemplate = R"(
triton_computation {
  parameter_0 = $0[11,63]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  output = $0[11,63]{1,0} $1(parameter_0, parameter_1)
  ROOT c.1 = f32[11,63]{1,0} convert(output)
}

ENTRY e {
  p0 = $0[11,63]{1,0} parameter(0)
  p1 = $0[11,63]{1,0} parameter(1)
  ROOT triton_fusion = f32[11,63]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton",
                    "block_level_fusion_config":{"output_tile_sizes":["1", "1"],"num_warps":"1"}}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  const std::string kHloRefTemplate = R"(
fused_computation {
  p0 = $0[11,63]{1,0} parameter(0)
  p1 = $0[11,63]{1,0} parameter(1)
  output = $0[11,63]{1,0} $1(p0, p1)
  ROOT convert.1 = f32[11,63]{1,0} convert(output)
}

ENTRY e {
  p1 = $0[11,63]{1,0} parameter(1)
  p0 = $0[11,63]{1,0} parameter(0)
  ROOT fusion = f32[11,63]{1,0} fusion(p0, p1), kind=kLoop, calls=fused_computation
})";
  const std::string hlo_ref = absl::Substitute(
      kHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_test, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance},
      /*run_hlo_passes=*/false, /*args_max_bits_of_precision=*/6));
}

std::vector<HloOpcode> TestedBinaryElementwise(PrimitiveType element_type) {
  std::vector<HloOpcode> ret =
      legacy_triton::TritonSupportedBinaryElementwiseUpToFloatNormalization(
          element_type);
  // Comparison requires an additional property.
  ret.erase(std::remove_if(ret.begin(), ret.end(), HloOpcodeIsComparison),
            ret.end());
  return ret;
}

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuitePRED, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(PRED),
                       ::testing::ValuesIn(TestedBinaryElementwise(PRED)),
                       ::testing::Values(0)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS8, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(S8),
                       ::testing::ValuesIn(TestedBinaryElementwise(S8)),
                       ::testing::Values(0)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS16, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(S16),
                       ::testing::ValuesIn(TestedBinaryElementwise(S16)),
                       ::testing::Values(0)),
    ElementwiseTestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteS32, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(S32),
                       ::testing::ValuesIn(TestedBinaryElementwise(S32)),
                       ::testing::Values(0)),
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

class CompareTest : public TritonTest,
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

  const std::string kHloTestTemplate = R"(
triton_gemm___computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  parameter_2 = $0[11,63]{1,0} parameter(2)
  f1.1 = pred[11,63]{1,0} compare(parameter_1, parameter_2), direction=$1
  c.1 = f32[11,63]{1,0} convert(f1.1)
  ROOT _.1 = f32[92,63]{1,0} dot(parameter_0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  p0 = f32[92,11]{1,0} parameter(0)
  p1 = $0[11,63]{1,0} parameter(1)
  p2 = $0[11,63]{1,0} parameter(2)
  ROOT triton_gemm__ = f32[92,63]{1,0} fusion(p0, p1, p2), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm",
                    "triton_gemm_config":
                      {"block_m":"16",
                       "block_n":"64",
                       "block_k":"16",
                       "split_k":"1",
                       "num_stages":"3",
                       "num_warps":"2",
                       "num_ctas":"1"}}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      ComparisonDirectionToString(direction));

  const std::string kHloRefTemplate = R"(
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
  gemm = (f32[92,63]{1,0}, s8[0]{0}) custom-call(p0, fusion),
    custom_call_target="__cublas$$gemm",
    backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":
      {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],
      "lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},
      "alpha_imag":0,"precision_config":
      {"operand_precision":["HIGHEST","HIGHEST"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f32[92,63]{1,0} get-tuple-element((f32[92,63]{1,0}, s8[0]{0}) gemm), index=0
})";
  const std::string hlo_ref = absl::Substitute(
      kHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
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
      ABSL_UNREACHABLE();
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

class SelectTest : public TritonTest,
                   public ::testing::WithParamInterface<
                       std::tuple<PrimitiveType, PrimitiveType>> {};

TEST_P(SelectTest, SelectFusionExecutesCorrectly) {
  PrimitiveType data_type1, data_type2;
  std::tie(data_type1, data_type2) = GetParam();
  for (const PrimitiveType type : {data_type1, data_type2}) {
    if (!legacy_triton::IsTritonSupportedDataType(type,
                                                  GetCudaComputeCapability())) {
      GTEST_SKIP() << absl::Substitute(
          "Unsupported data type: $0",
          primitive_util::LowercasePrimitiveTypeName(type));
    }
  }

  const std::string kHloTestTemplate = R"(
triton_gemm___computation {
  parameter_0 = $1[92,13]{1,0} parameter(0)
  parameter_1 = $0[13,63]{1,0} parameter(1)
  parameter_2 = $0[13,63]{1,0} parameter(2)
  parameter_3 = pred[13,63]{1,0} parameter(3)
  f1.1 = $0[13,63]{1,0} select(parameter_3, parameter_1, parameter_2)
  c.1 = $1[13,63]{1,0} convert(f1.1)
  ROOT _.1 = $1[92,63]{1,0} dot(parameter_0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  p0 = $1[92,13]{1,0} parameter(0)
  p1 = $0[13,63]{1,0} parameter(1)
  p2 = $0[13,63]{1,0} parameter(2)
  p3 = pred[13,63]{1,0} parameter(3)
  ROOT triton_gemm__ = $1[92,63]{1,0} fusion(p0, p1, p2, p3), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm",
                    "triton_gemm_config":
                      {"block_m":"16",
                       "block_n":"64",
                       "block_k":"16",
                       "split_k":"1",
                       "num_stages":"3",
                       "num_warps":"2",
                       "num_ctas":"1"}}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type1),
      primitive_util::LowercasePrimitiveTypeName(data_type2));

  const std::string kHloRefTemplate = R"(
fused_computation {
  p0 = $0[13,63]{1,0} parameter(0)
  p1 = $0[13,63]{1,0} parameter(1)
  p2 = pred[13,63]{1,0} parameter(2)
  f.1 = $0[13,63]{1,0} select(p2, p0, p1)
  ROOT convert.1 = $1[13,63]{1,0} convert(f.1)
}

ENTRY e {
  p3 = pred[13,63]{1,0} parameter(3)
  p2 = $0[13,63]{1,0} parameter(2)
  p1 = $0[13,63]{1,0} parameter(1)
  p0 = $1[92,13]{1,0} parameter(0)
  fusion = $1[13,63]{1,0} fusion(p1, p2, p3), kind=kLoop,
    calls=fused_computation
  gemm = ($1[92,63]{1,0}, s8[0]{0}) custom-call(p0, fusion),
    custom_call_target="__cublas$$gemm",
    backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":
      {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],
      "lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},
      "alpha_imag":0,"precision_config":
      {"operand_precision":["HIGHEST","HIGHEST"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = $1[92,63]{1,0} get-tuple-element(($1[92,63]{1,0}, s8[0]{0}) gemm), index=0
})";
  const std::string hlo_ref = absl::Substitute(
      kHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type1),
      primitive_util::LowercasePrimitiveTypeName(data_type2));

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_test, ErrorSpec{/*aabs=*/0, /*arel=*/0},
      /*run_hlo_passes=*/false, /*args_max_bits_of_precision=*/9));
}

std::string TwoPrimitiveTypesToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, PrimitiveType>>&
        data) {
  PrimitiveType data_type1;
  PrimitiveType data_type2;
  std::tie(data_type1, data_type2) = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type1),
                      "_",
                      primitive_util::LowercasePrimitiveTypeName(data_type2));
}

// BF16: depending on the GPU generation.
constexpr std::array<PrimitiveType, 7> kSupportedDataTypes{PRED, S8,  S16, S32,
                                                           F16,  F32, BF16};

INSTANTIATE_TEST_SUITE_P(
    SelectTestSuite, SelectTest,
    ::testing::Combine(::testing::ValuesIn(kSupportedDataTypes),
                       ::testing::Values(F16, BF16, F32)),
    TwoPrimitiveTypesToString);

class ConstantTest : public TritonTest,
                     public ::testing::WithParamInterface<PrimitiveType> {};

TEST_P(ConstantTest, ConstantFusionExecutesCorrectly) {
  const PrimitiveType data_type = GetParam();
  if (!legacy_triton::IsTritonSupportedDataType(data_type,
                                                GetCudaComputeCapability())) {
    GTEST_SKIP() << absl::Substitute(
        "Unsupported data type: $0",
        primitive_util::LowercasePrimitiveTypeName(data_type));
  }

  const std::string kHloTestTemplate = R"(
triton_gemm___computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = f32[11,63]{1,0} parameter(1)
  c = $0[] constant(123)
  b = $0[11,63] broadcast(c)
  cv = f32[11,63] convert(b)
  m = f32[11,63] multiply(cv, parameter_1)
  ROOT _.1 = f32[92,63]{1,0} dot(parameter_0, m),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  p0 = f32[92,11]{1,0} parameter(0)
  p1 = f32[11,63]{1,0} parameter(1)
  ROOT triton_gemm__ = f32[92,63]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm",
                    "triton_gemm_config":
                      {"block_m":"16",
                       "block_n":"64",
                       "block_k":"16",
                       "split_k":"1",
                       "num_stages":"3",
                       "num_warps":"2",
                       "num_ctas":"1"}}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string kHloRefTemplate = R"(
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
  gemm = (f32[92,63]{1,0}, s8[0]{0}) custom-call(p0, fusion),
    custom_call_target="__cublas$$gemm",
    backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":
      {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],
      "lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},
      "alpha_imag":0,"precision_config":
      {"operand_precision":["HIGHEST","HIGHEST"]},"epilogue":"DEFAULT"}}
  ROOT get-tuple-element = f32[92,63]{1, 0} get-tuple-element((f32[92,63]{1, 0}, s8[0]{0}) gemm), index=0
})";
  const std::string hlo_ref = absl::Substitute(
      kHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type));

  float tolerance;
  switch (data_type) {
    case F32:
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
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_test, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance},
      /*run_hlo_passes=*/false));
}

INSTANTIATE_TEST_SUITE_P(ConstantTestSuite, ConstantTest,
                         ::testing::ValuesIn(kSupportedDataTypes),
                         TritonSupportTestTypeToString);

class ConvertTest : public TritonTest,
                    public ::testing::WithParamInterface<
                        std::tuple<PrimitiveType, PrimitiveType>> {};

TEST_P(ConvertTest, ConvertFusionExecutesCorrectly) {
  PrimitiveType data_type1, data_type2;
  std::tie(data_type1, data_type2) = GetParam();
  for (const PrimitiveType type : {data_type1, data_type2}) {
    if (!legacy_triton::IsTritonSupportedDataType(type,
                                                  GetCudaComputeCapability())) {
      GTEST_SKIP() << absl::Substitute(
          "Unsupported data type: $0",
          primitive_util::LowercasePrimitiveTypeName(type));
    }
  }

  const std::string hlo_text = absl::Substitute(
      R"(
t {
  p0 = $0[2,2] parameter(0)
  p0c = $1[2,2] convert(p0)
  p0cc = f32[2,2] convert(p0c)
  p1 = f32[2,2] parameter(1)
  ROOT r = f32[2,2] dot(p0cc, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  p0 = $0[2,2] parameter(0)
  p1 = f32[2,2] parameter(1)
  ROOT r = f32[2,2] fusion(p0, p1), kind=kCustom, calls=t,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})",
      primitive_util::LowercasePrimitiveTypeName(data_type1),
      primitive_util::LowercasePrimitiveTypeName(data_type2));

  MatchOptimizedHlo(hlo_text, R"(
CHECK: block_m
  )");
}

INSTANTIATE_TEST_SUITE_P(
    ConvertTestSuite, ConvertTest,
    ::testing::Combine(::testing::ValuesIn(kSupportedDataTypes),
                       ::testing::ValuesIn(kSupportedDataTypes)),
    TwoPrimitiveTypesToString);

class TritonNormalizationTest
    : public GpuCodegenTest,
      public ::testing::WithParamInterface<PrimitiveType> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // TODO(b/38354253): Remove once HloTestBase does not remove constant
    // folding.
    debug_options.clear_xla_disable_hlo_passes();
    return debug_options;
  }
};

TEST_P(TritonNormalizationTest, CanFuseAndEmitExactSoftmax) {
  PrimitiveType data_type = GetParam();

  if (data_type == F16) {
    GTEST_SKIP() << "Exponential op does not support F16.";
  }

  const std::string hlo_text_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  exponential = $0[127,125]{1,0} exponential(subtract)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = $0[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[param_0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[param_0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case BF16:
      tolerance = 2e-4;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(TritonNormalizationTest, CanFuseAndEmitFirstSoftmaxDiamond) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_text_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
    case BF16:
      tolerance = 2e-4;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(TritonNormalizationTest, CanFuseAndEmitSoftmaxDiamondWithSmallRows) {
  PrimitiveType data_type = GetParam();
  constexpr absl::string_view kHloTextTemplate = R"(
HloModule softmax
min_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT minimum = $0[] minimum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,7]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=min_computation
  broadcast = $0[127,7]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,7]{1,0} subtract(param_0, broadcast)
}
)";

  const std::string hlo_text = absl::Substitute(
      kHloTextTemplate, primitive_util::LowercasePrimitiveTypeName(data_type));

  constexpr absl::string_view kHloRefTemplate = R"(
; CHECK:    ENTRY
; CHECK:      %[[param_0:.*]] = $0[127,7]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[param_0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      kHloRefTemplate, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec(/*aabs=*/0, /*arel=*/0)));
}

TEST_F(TritonNormalizationTest, CanFuseAndEmitDiamondWithBF16Converts) {
  const std::string hlo_text = R"(
HloModule softmax
max_computation {
  arg_0 = bf16[] parameter(0)
  arg_1 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = bf16[127,125]{1,0} parameter(0)
  constant_neg_inf = bf16[] constant(-inf)
  reduce = bf16[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = bf16[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = bf16[127,125]{1,0} subtract(param_0, broadcast)
}
)";

  const std::string hlo_ref = R"(
; CHECK:      %[[P0_FUSION:.*]] = bf16[127,125]{1,0} parameter(0)
; CHECK:      %[[convert:.*]] = f32[127,125]{1,0} convert(%[[P0_FUSION]])
; CHECK:    ENTRY
; CHECK:      %[[P0_ENTRY:.*]] = bf16[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0_ENTRY]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance = 2e-4;
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(TritonNormalizationTest,
       CanFuseAndEmitDiamondWithMultipleBroadcastDimensions) {
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[1,3,125,125]{3,2,1,0} parameter(0)
  reshape = $0[3,125,125]{2,1,0} reshape($0[1,3,125,125]{3,2,1,0} param_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[3,125]{1,0} reduce($0[3,125,125]{2,1,0} reshape, $0[] constant_neg_inf), dimensions={2}, to_apply=max_computation
  broadcast = $0[1,3,125,125]{3,2,1,0} broadcast($0[3,125]{1,0} reduce), dimensions={1,2}
  ROOT subtract = $0[1,3,125,125]{3,2,1,0} subtract($0[1,3,125,125]{3,2,1,0} param_0, $0[1,3,125,125]{3,2,1,0} broadcast)
})";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[1,3,125,125]{3,2,1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-3;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(TritonNormalizationTest,
       CanFuseAndEmitSoftmaxWithIntermediateUnaryElementwise) {
  PrimitiveType data_type = GetParam();

  if (data_type == F16) {
    GTEST_SKIP() << "Exponential op does not support F16.";
  }

  const std::string hlo_text_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  abs = $0[127,125]{1,0} abs(subtract)
  exponential = $0[127,125]{1,0} exponential(abs)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = $0[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case BF16:
      tolerance = 2e-3;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitTwoDiamondsWithSecondDiamondProducerEqualToFirstDiamondRoot) {
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(subtract, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT multiply = $0[127,125]{1,0} multiply(subtract, second_broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
    case BF16:
      tolerance = 2e-2;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(TritonNormalizationTest,
       CanFuseAndEmitDiamondWithTrailingUnaryElementwiseAtTheRoot) {
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  ROOT abs = $0[127,125]{1,0} abs(subtract)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-3;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(TritonNormalizationTest,
       CanFuseAndEmitDiamondWithUnaryElementwisePrefix) {
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  abs = $0[127,125]{1,0} abs(param_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(abs, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-3;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(TritonNormalizationTest,
       CanFuseAndEmitSoftmaxDiamondWithLastDimensionBitcastAfterReduce) {
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = $0[3,127,125]{2,1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[3,127]{1,0} reduce(param_0, constant_neg_inf), dimensions={2}, to_apply=max_computation
  bitcasted_reduce = $0[381]{0} reshape(reduce)
  broadcast = $0[381,125]{1,0} broadcast(bitcasted_reduce), dimensions={0}
  bitcasted_broadcast = $0[3,127,125]{2,1,0} reshape(broadcast)
  ROOT subtract = $0[3,127,125]{2,1,0} subtract(param_0, bitcasted_broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[3,127,125]{2,1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-3;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(TritonNormalizationTest,
       CanFuseAndEmitConvertInvolvingBF16InputIntoSoftmaxDiamondCorrectly) {
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = bf16[127,125]{1,0} parameter(0)
  param_0_$0 = $0[127,125]{1,0} convert(param_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0_$0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0_$0, broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = bf16[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-3;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitBinaryElementwiseProducerIntoDiamondWhenBothOperandsAreTheSame) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule fusible_diamond
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  multiply =  $0[127,125]{1,0} multiply(param_0, param_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(multiply, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(multiply, broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 3e-3;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitIntermediateBinaryElementwiseWithinDiamondWhenBothOperandsAreTheSame) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule fusible_diamond
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  multiply =  $0[127]{0} multiply(reduce, reduce)
  broadcast = $0[127,125]{1,0} broadcast(multiply), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-3;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitBinaryElementwiseWhenBothOperandsAreTheSameBetweenDiamonds) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule fusible_diamonds
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  add = $0[127,125]{1,0} add(subtract, subtract)
  second_reduce = $0[127]{0} reduce(add, constant_neg_inf), dimensions={1}, to_apply=max_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT add_root = $0[127,125]{1,0} add(add, second_broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  // The precision-changing ops in the kernel above are add & subtract, meaning
  // a value can be X*2(first add)*2(subtract)*2(second add) larger than it was
  // originally. In order to fit this into a datatype, we do:
  // X*2^3 <= 2^(fraction bits of the data type)
  // 2^(max_bits_of_precision)*2^3 <= 2^(fraction bits of the data type)
  // max_bits_of_precision = fraction_bits - 3.
  uint max_bits_of_precision;
  switch (data_type) {
    case F32:
      max_bits_of_precision = 20;
      break;
    case F16:
      max_bits_of_precision = 7;
      break;
    case BF16:
      max_bits_of_precision = 4;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec(/*aabs=*/0, /*arel=*/0),
                            /*reference_preprocessor=*/nullptr,
                            /*test_preprocessor=*/nullptr,
                            max_bits_of_precision));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitBinaryElementwiseConsumerWhereBothOperandsAreTheSameIntoDiamond) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule fusible_diamond
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  ROOT multiply = $0[127,125]{1,0} multiply(subtract, subtract)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-3;
      break;
    default:
      ABSL_UNREACHABLE();
      // ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitTwoBinaryElementwiseWhereBothOperandsAreTheSameBetweenDiamonds) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule fusible_diamonds
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  add = $0[127,125]{1,0} add(subtract, subtract)
  multiply = $0[127,125]{1,0} multiply(add, add)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(multiply, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT multiply_root = $0[127,125]{1,0} multiply(multiply, second_broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
    case BF16:
      tolerance = 2e-2;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(TritonNormalizationTest, DiamondEmitterIsNumericallyStable) {
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
min_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT minimum = $0[] minimum(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  exponential = $0[127,125]{1,0} exponential(subtract)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=min_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = $0[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec(/*aabs=*/0, /*arel=*/0)));
}

TEST_P(TritonNormalizationTest, CanFuseAndEmitRMSNormDiamond) {
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule rms_norm
add_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT add.1 = $0[] add(arg_0, arg_1)
}
ENTRY main.30 {
  param_0 = $0[10,10,10,128]{3,2,1,0} parameter(0)
  multiply_param = $0[10,10,10,128]{3,2,1,0} multiply(param_0, param_0)
  constant_0 = $0[] constant(0)
  reduce = $0[10,10,10]{2,1,0} reduce(multiply_param, constant_0), dimensions={3}, to_apply=add_computation
  constant_1 = $0[] constant(0.333333343)
  splat = $0[10,10,10]{2,1,0} broadcast(constant_1), dimensions={}
  multiply_splat = $0[10,10,10]{2,1,0} multiply(reduce, splat)
  epsilon = $0[] constant(1e-06)
  splat_epsilon = $0[10,10,10]{2,1,0} broadcast(epsilon), dimensions={}
  add = $0[10,10,10]{2,1,0} add(multiply_splat, splat_epsilon)
  rsqrt = $0[10,10,10]{2,1,0} rsqrt(add)
  broadcast = $0[10,10,10,128]{3,2,1,0} broadcast(rsqrt), dimensions={0,1,2}
  ROOT multiply = $0[10,10,10,128]{3,2,1,0} multiply(param_0, broadcast)
}
)";

  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[10,10,10,128]{3,2,1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 5e-4;
      break;
    case BF16:
      tolerance = 4e-2;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitBinaryElementwiseWhereTheFirstOperandIsASplatConstantBetweenDiamonds) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule fusible_diamonds
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=add_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  constant = $0[] constant(0.333333343)
  broadcast_splat = $0[127,125]{1,0} broadcast(constant), dimensions={}
  multiply = $0[127,125]{1,0} multiply(broadcast_splat, subtract)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(multiply, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT second_subtract = $0[127,125]{1,0} subtract(multiply, second_broadcast)
}
)";

  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-2;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitBinaryElementwiseWhereTheSecondOperandIsASplatConstantBetweenDiamonds) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule fusible_diamonds
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=add_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  constant = $0[] constant(0.333333343)
  broadcast_splat = $0[127,125]{1,0} broadcast(constant), dimensions={}
  multiply = $0[127,125]{1,0} multiply(subtract, broadcast_splat)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(multiply, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT second_subtract = $0[127,125]{1,0} subtract(multiply, second_broadcast)
}
)";

  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-2;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitBinaryElementwiseWhereTheFirstOperandIsASplatConstantWithinDiamond) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule fusible_diamond
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  constant = $0[] constant(0.333333343)
  broadcast_splat = $0[127]{0} broadcast(constant), dimensions={}
  multiply = $0[127]{0} multiply(broadcast_splat, reduce)
  broadcast = $0[127,125]{1,0} broadcast(multiply), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";

  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";

  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-2;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitBinaryElementwiseConsumerWhereTheFirstOperandIsASplatConstantIntoDiamond) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule fusible_diamond
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=add_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  constant = $0[] constant(0.333333343)
  broadcast_splat = $0[127,125]{1,0} broadcast(constant), dimensions={}
  ROOT multiply = $0[127,125]{1,0} multiply(broadcast_splat, subtract)
}
)";

  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";
  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-2;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitBinaryElementwiseProducerWhereTheFirstOperandIsASplatConstantIntoDiamond) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule fusible_diamond
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {

  param_0 = $0[127,125]{1,0} parameter(0)
  constant = $0[] constant(0.333333343)
  broadcast_splat = $0[127,125]{1,0} broadcast(constant), dimensions={}
  multiply = $0[127,125]{1,0} multiply(broadcast_splat, param_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(multiply, constant_neg_inf), dimensions={1}, to_apply=add_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(multiply, broadcast)
}
)";

  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";
  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance;
  switch (data_type) {
    case F32:
      tolerance = 1e-6;
      break;
    case F16:
      tolerance = 2e-4;
      break;
    case BF16:
      tolerance = 2e-2;
      break;
    default:
      ABSL_UNREACHABLE();
  }
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_P(
    TritonNormalizationTest,
    CanFuseAndEmitBinaryElementwiseOperationWhereOneOperandIsASharedSplatProducerIntoDiamond) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();

  const std::string hlo_text_template = R"(
HloModule nonfusible_diamond
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT max = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  param_1 = $0[127,125]{1,0} parameter(1)
  constant_2 = $0[] constant(2)
  broadcast_splat = $0[127,125]{1,0} broadcast(constant_2), dimensions={}
  multiply = $0[127,125]{1,0} multiply(param_0, broadcast_splat)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(multiply, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_text = absl::Substitute(
      hlo_text_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  const std::string hlo_ref_template = R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = $0[127,125]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)";
  const std::string hlo_ref = absl::Substitute(
      hlo_ref_template, primitive_util::LowercasePrimitiveTypeName(data_type));

  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance = 0.0;
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

INSTANTIATE_TEST_SUITE_P(TritonNormalizationTestSuite, TritonNormalizationTest,
                         ::testing::Values(F32, F16, BF16));

TEST_F(TritonNormalizationTest, CanFuseAndEmitTritonSoftmaxWithTwoParameters) {
  const std::string hlo_text = R"(
HloModule layernorm

add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[127]{0} parameter(1)
  broadcast_0 = f32[125,127]{1,0} broadcast(param_1), dimensions={1}
  multiply_0 = f32[125,127]{1,0} multiply(param_0, broadcast_0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}
)";

  const std::string hlo_ref = R"(
; CHECK:    ENTRY
; CHECK-DAG:    %[[param_0:.*]] = f32[125,127]{1,0} parameter(0)
; CHECK-DAG:    %[[param_1:.*]] = f32[127]{0} parameter(1)
; CHECK:      ROOT
; CHECK-SAME:   f32[125,127]{1,0} fusion
; CHECK-SAME:    %[[param_0]]
; CHECK-SAME:    %[[param_1]]
; CHECK-SAME:          kind=kCustom
; CHECK-SAME:          triton_softmax
)";
  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance = 2e-6;
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

TEST_F(TritonNormalizationTest, CanFuseAndEmitTritonSoftmaxWithNonBatchReduce) {
  const std::string hlo_text = R"(
HloModule layernorm

add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

ENTRY main {
  param_0 = f32[125,127]{1,0} parameter(0)
  param_1 = f32[10,125,127]{2,1,0} parameter(1)
  constant = f32[] constant(0)
  reduce_0 = f32[125,127]{1,0} reduce(param_1, constant), dimensions={0}, to_apply=add
  multiply_0 = f32[125,127]{1,0} multiply(param_0, reduce_0)
  constant_0 = f32[] constant(0)
  reduce_1 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast_4 = f32[125,127]{1,0} broadcast(reduce_1), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
}
)";

  // We expect to not fuse everything into the triton softmax, because of the
  // reduce over the non-row dimension.
  const std::string hlo_ref = R"(
; CHECK:      ENTRY
; CHECK-DAG:    %[[P0:.*]] = f32[125,127]{1,0} parameter(0)
; CHECK-DAG:    %[[P1:.*]] = f32[10,125,127]{2,1,0} parameter(1)
; CHECK:        ROOT %[[FUSION:.*]] = f32[125,127]{1,0} fusion(%[[P0]], %[[P1]])
; CHECK-SAME:       kind=kCustom
; CHECK-SAME:       __triton
)";
  MatchOptimizedHlo(hlo_text, hlo_ref);

  float tolerance = 2e-6;
  EXPECT_TRUE(RunAndCompare(hlo_text,
                            ErrorSpec(/*aabs=*/tolerance, /*arel=*/tolerance)));
}

class ReductionTypeTest : public TritonTest,
                          public ::testing::WithParamInterface<PrimitiveType> {
};

TEST_P(ReductionTypeTest, DifferentReductionTypes) {
  PrimitiveType data_type = GetParam();

  const std::string kHloTestTemplate = R"(
max {
  p0 = $0[] parameter(0)
  p1 = $0[] parameter(1)
  ROOT max = $0[] maximum(p0, p1)
}

triton_computation {
  p = $0[400,16] parameter(0)
  zero = $0[] constant(0)
  ROOT reduce = $0[400] reduce(p, zero), dimensions={1}, to_apply=max
}

ENTRY entry_computation {
  p = $0[400,16] parameter(0)
  ROOT fusion = $0[400] fusion(p), kind=kCustom, calls=triton_computation,
    backend_config={ "operation_queue_id":"0", "wait_on_operation_queues":[],
      "fusion_backend_config":{ "kind":"__triton", "block_level_fusion_config":{
          "output_tile_sizes":["400"], "num_warps":"1"}},
      "force_earliest_schedule":false}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type));
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(hlo_test, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

constexpr std::array<PrimitiveType, 9> kReductionSupportedDataTypes{
    PRED, S8, S16, S32, S64, F16, F32, F64, BF16};

INSTANTIATE_TEST_SUITE_P(ReductionTypeTestSuite, ReductionTypeTest,
                         ::testing::ValuesIn(kReductionSupportedDataTypes),
                         TritonSupportTestTypeToString);

}  // namespace
}  // namespace gpu
}  // namespace xla
