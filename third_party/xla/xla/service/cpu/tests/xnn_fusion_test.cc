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

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {
namespace {

using ::testing::HasSubstr;

struct XnnFusionTestParams {
  std::string in_dtype;
  std::string out_dtype;  // Only used for mixed input/output types.
};

class XnnFusionTest
    : public HloTestBase,
      public ::testing::WithParamInterface<XnnFusionTestParams> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<XnnFusionTestParams>& info) {
    return absl::StrCat(info.param.in_dtype, "_", info.param.out_dtype);
  }

 protected:
  void RunTest(absl::string_view hlo_template, absl::string_view check_str) {
    XnnFusionTestParams params = GetParam();
    std::string hlo_text =
        absl::StrReplaceAll(hlo_template, {{"$dtype", params.in_dtype},
                                           {"$in_dtype", params.in_dtype},
                                           {"$out_dtype", params.out_dtype}});
    bool bf16_compute = params.in_dtype == "bf16" || params.out_dtype == "bf16";
    double tolerance = bf16_compute ? 1e-2 : 1e-7;
    EXPECT_TRUE(RunAndCompare(
        hlo_text, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));

    if (bf16_compute && !check_str.empty()) {
      std::string check_text =
          absl::StrReplaceAll(check_str, {{"$dtype", params.in_dtype},
                                          {"$in_dtype", params.in_dtype},
                                          {"$out_dtype", params.out_dtype}});
      MatchOptimizedHlo(hlo_text, check_text);
    }
  }
};

bool ShouldSkipDotBf16Test(absl::string_view in_dtype) {
  return in_dtype == "bf16" &&
         !tsl::port::TestCPUFeature(tsl::port::AVX512_BF16);
}

absl::string_view GetOutputTypeSupportedByXnnBatchMatMul(
    absl::string_view in_dtype) {
  static const auto* kSupportedOutputTypes =
      new absl::flat_hash_map<absl::string_view, absl::string_view>(
          {{"f32", "f32"}, {"bf16", "f32"}});

  return kSupportedOutputTypes->at(in_dtype);
}

std::string InsertConvertIfNecessary(absl::string_view hlo_text,
                                     absl::string_view in_dtype,
                                     absl::string_view out_dtype,
                                     absl::string_view convert_text) {
  absl::string_view supported_dtype =
      GetOutputTypeSupportedByXnnBatchMatMul(in_dtype);
  bool need_convert = out_dtype != supported_dtype;
  return absl::StrReplaceAll(
      hlo_text, {{"$root ", need_convert ? "" : "ROOT "},
                 {"$dot_dtype", need_convert ? supported_dtype : "$out_dtype"},
                 {"$convert_if_necessary", need_convert ? convert_text : ""},
                 {"$dot_or_convert", need_convert ? "%convert" : "%dot"}});
}

// For tests that always have same input/output types.
using SameTypeTest = XnnFusionTest;

TEST_P(SameTypeTest, AddAndMultiply) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule add_and_multiply

    xnn_fusion {
      %lhs = $dtype[4] parameter(0)
      %rhs = $dtype[4] parameter(1)
      %add = $dtype[4] add(%lhs, %rhs)
      ROOT %mul = $in_dtype[4] multiply(%add, %add)
    }

    ENTRY entry {
      %p0 = $dtype[4] parameter(0)
      %p1 = $dtype[4] parameter(1)
      ROOT %fusion = $dtype[4] fusion(%p0, %p1), kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  // Optimized HLO shouldn't have any convert.
  constexpr absl::string_view kCheckStr = R"(
    CHECK:      %xnn_fusion
    CHECK-NOT:  convert
    CHECK:      multiply
  )";

  RunTest(kModuleStr, kCheckStr);
}

TEST_P(SameTypeTest, DotAddMultiply) {
  XnnFusionTestParams params = GetParam();
  if (ShouldSkipDotBf16Test(params.in_dtype)) {
    GTEST_SKIP() << "XNNPACK bf16 matmul requires AVX512_BF16 which this CPU "
                    "doesn't have.";
  }

  constexpr absl::string_view kModuleStr = R"(
    HloModule dot_add_multiply

    xnn_fusion {
      %lhs = $dtype[4,5] parameter(0)
      %rhs = $dtype[5,6] parameter(1)
      %addend = $dtype[4,6] parameter(2)
      %multiplier = $dtype[4,6] parameter(3)
      %dot = $dot_dtype[4,6] dot(%lhs, %rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      $convert_if_necessary
      %add = $dtype[4,6] add($dot_or_convert, %addend)
      ROOT %mul = $dtype[4,6] multiply(%add, %multiplier)
    }

    ENTRY entry {
      %lhs = $dtype[4,5] parameter(0)
      %rhs = $dtype[5,6] parameter(1)
      %addend = $dtype[4, 6] parameter(2)
      %multiplier = $dtype[4, 6] parameter(3)
      ROOT %fusion = $dtype[4,6] fusion(%lhs, %rhs, %addend, %multiplier),
        kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  constexpr absl::string_view kConvertStr =
      "%convert = $dtype[4,6] convert(%dot)";

  // Optimized HLO shouldn't have any convert before the dot.
  constexpr absl::string_view kCheckStr = R"(
    CHECK:      %xnn_fusion
    CHECK-NOT:  convert
    CHECK:      dot
  )";

  RunTest(InsertConvertIfNecessary(kModuleStr, params.in_dtype,
                                   params.out_dtype, kConvertStr),
          kCheckStr);
}

TEST_P(SameTypeTest, DotRhsTransposedAndMultiply) {
  XnnFusionTestParams params = GetParam();
  if (ShouldSkipDotBf16Test(params.in_dtype)) {
    GTEST_SKIP() << "XNNPACK bf16 matmul requires AVX512_BF16 which this CPU "
                    "doesn't have.";
  }

  constexpr absl::string_view kModuleStr = R"(
    HloModule dot_rhs_transposed_and_multiply

    xnn_fusion {
      %lhs = $dtype[4,5] parameter(0)
      %rhs = $dtype[6,5] parameter(1)
      %multiplier = $dtype[4,6] parameter(2)
      %dot = $dot_dtype[4,6] dot(%lhs, %rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
      $convert_if_necessary
      ROOT %mul = $dtype[4,6] multiply($dot_or_convert, %multiplier)
    }

    ENTRY entry {
      %lhs = $dtype[4,5] parameter(0)
      %rhs = $dtype[6,5] parameter(1)
      %multiplier = $dtype[4, 6] parameter(2)
      ROOT %fusion = $dtype[4,6] fusion(%lhs, %rhs, %multiplier),
        kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  constexpr absl::string_view kConvertStr =
      "%convert = $dtype[4,6] convert(%dot)";

  // Optimized HLO shouldn't have any convert before the dot.
  constexpr absl::string_view kCheckStr = R"(
    CHECK:      %xnn_fusion
    CHECK-NOT:  convert
    CHECK:      dot
  )";

  RunTest(InsertConvertIfNecessary(kModuleStr, params.in_dtype,
                                   params.out_dtype, kConvertStr),
          kCheckStr);
}

std::vector<XnnFusionTestParams> GetSameTypeTestCases() {
  return std::vector<XnnFusionTestParams>({
      XnnFusionTestParams{"f32", "f32" /*unused*/},
      XnnFusionTestParams{"bf16", "bf16" /*unused*/},
  });
}

INSTANTIATE_TEST_SUITE_P(SameTypeTestInstantiation, SameTypeTest,
                         ::testing::ValuesIn(GetSameTypeTestCases()),
                         XnnFusionTest::Name);

// For tests that we might want to use different input/output types.
using MixedTypesTest = XnnFusionTest;

TEST_P(MixedTypesTest, BatchedDot) {
  XnnFusionTestParams params = GetParam();
  if (ShouldSkipDotBf16Test(params.in_dtype)) {
    GTEST_SKIP() << "XNNPACK bf16 matmul requires AVX512_BF16 which this CPU"
                    "doesn't have.";
  }

  constexpr absl::string_view kModuleStr = R"(
    HloModule dot_add_multiply

    xnn_fusion {
      %lhs = $in_dtype[2,3,4,5] parameter(0)
      %rhs = $in_dtype[2,3,5,6] parameter(1)
      $root %dot = $dot_dtype[2,3,4,6] dot(%lhs, %rhs),
        lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
        lhs_contracting_dims={3}, rhs_contracting_dims={2}
      $convert_if_necessary
    }

    ENTRY entry {
      %lhs = $in_dtype[2,3,4,5] parameter(0)
      %rhs = $in_dtype[2,3,5,6] parameter(1)
      ROOT %fusion = $out_dtype[2,3,4,6] fusion(%lhs, %rhs),
        kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  constexpr absl::string_view kConvertStr =
      "ROOT %convert = $out_dtype[2,3,4,6] convert(%dot)";

  // Optimized HLO shouldn't have any convert before the dot.
  constexpr absl::string_view kCheckStr = R"(
    CHECK:      %xnn_fusion
    CHECK-NOT:  convert
    CHECK:      dot
  )";

  RunTest(InsertConvertIfNecessary(kModuleStr, params.in_dtype,
                                   params.out_dtype, kConvertStr),
          kCheckStr);
}

std::vector<XnnFusionTestParams> GetMixedTypesTestCases() {
  return std::vector<XnnFusionTestParams>({
      XnnFusionTestParams{"f32", "f32"},
      XnnFusionTestParams{"bf16", "f32"},
      XnnFusionTestParams{"bf16", "bf16"},
  });
}

INSTANTIATE_TEST_SUITE_P(MixedTypesTestInstantiation, MixedTypesTest,
                         ::testing::ValuesIn(GetMixedTypesTestCases()),
                         XnnFusionTest::Name);

TEST_F(XnnFusionTest, ConvertF32ToBF16) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule convert

    xnn_fusion {
      %input = f32[2,3,4,5] parameter(0)
      ROOT %dot = bf16[2,3,4,5] convert(%input)
    }

    ENTRY entry {
      %input = f32[2,3,4,5] parameter(0)
      ROOT %fusion = bf16[2,3,4,5] fusion(%input),
        kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-2}));
}

// The following tests don't need to be run with different data types.
TEST_F(XnnFusionTest, UnsupportedDot) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule unsupported_dot

    xnn_fusion {
      %lhs = f32[5,4] parameter(0)
      %rhs = f32[5,6] parameter(1)
      ROOT %dot = f32[4,6] dot(%lhs, %rhs),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }

    ENTRY entry {
      %lhs = f32[5,4] parameter(0)
      %rhs = f32[5,6] parameter(1)
      ROOT %fusion = f32[4,6] fusion(%lhs, %rhs),
        kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  auto status = RunAndCompare(kModuleStr, ErrorSpec{0.0});
  EXPECT_FALSE(status);
  EXPECT_THAT(status.message(),
              HasSubstr("Unsupported XNNPACK Dot op variation"));
}

TEST_F(XnnFusionTest, UnsupportedOp) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule unsupported_sqrt

    xnn_fusion {
      %x = f32[10] parameter(0)
      ROOT %sqrt = f32[10] sqrt(%x)
    }

    ENTRY entry {
      %x = f32[10] parameter(0)
      ROOT %sqrt = f32[10] fusion(%x), kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  auto status = RunAndCompare(kModuleStr, ErrorSpec{0.0});
  EXPECT_FALSE(status);
  EXPECT_THAT(status.message(),
              HasSubstr("Unsupported XNNPACK fusion instruction"));
}

}  // namespace
}  // namespace xla::cpu
