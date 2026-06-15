/* Copyright 2025 The OpenXLA Authors.

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

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

struct YnnFusionTestParams {
  std::string in_dtype;
  std::string out_dtype;  // Only used for mixed input/output types.
};

class YnnFusionTest
    : public HloPjRtInterpreterReferenceMixin<HloTestBase>,
      public ::testing::WithParamInterface<YnnFusionTestParams> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<YnnFusionTestParams>& info) {
    return absl::StrCat(info.param.in_dtype, "x", info.param.out_dtype);
  }

 protected:
  void RunTest(absl::string_view hlo_template) {
    YnnFusionTestParams params = GetParam();
    std::string hlo_text =
        absl::StrReplaceAll(hlo_template, {{"$dtype", params.in_dtype},
                                           {"$in_dtype", params.in_dtype},
                                           {"$out_dtype", params.out_dtype}});
    bool bf16_compute = absl::StrContains(hlo_text, "bf16");
    double tolerance = bf16_compute ? 1e-2 : 2e-7;
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        hlo_text, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
  }
};

TEST_P(YnnFusionTest, AddAndMultiply) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule add_and_multiply

    ynn_fusion {
      %lhs = $dtype[100] parameter(0)
      %rhs = $dtype[100] parameter(1)
      %add = $dtype[100] add(%lhs, %rhs)
      ROOT %mul = $in_dtype[100] multiply(%add, %add)
    }

    ENTRY entry {
      %p0 = $dtype[100] parameter(0)
      %p1 = $dtype[100] parameter(1)
      ROOT %fusion = $dtype[100] fusion(%p0, %p1), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, Pad) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule pad

    ynn_fusion {
      %input = $dtype[8, 10] parameter(0)
      %padding_value = $dtype[] parameter(1)
      ROOT %pad = $dtype[10, 14] pad(%input, %padding_value),
        padding=1_1_0x2_2_0
    }

    ENTRY entry {
      %p0 = $dtype[8, 10] parameter(0)
      %p1 = $dtype[] parameter(1)
      ROOT %fusion = $dtype[10, 14] fusion(%p0, %p1), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, Transpose) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule transpose

    ynn_fusion {
      %input = $dtype[8, 10] parameter(0)
      ROOT %transpose = $dtype[10, 8] transpose(%input), dimensions={1, 0}
    }

    ENTRY entry {
      %p0 = $dtype[8, 10] parameter(0)
      ROOT %fusion = $dtype[10, 8] fusion(%p0), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, Broadcast) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule broadcast

    ynn_fusion {
      %input = $dtype[10] parameter(0)
      ROOT %broadcast = $dtype[8, 10] broadcast(%input), dimensions={1}
    }

    ENTRY entry {
      %p0 = $dtype[10] parameter(0)
      ROOT %fusion = $dtype[8, 10] fusion(%p0), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, Concatenate) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule concatenate

    ynn_fusion {
      %p0 = $dtype[8, 10] parameter(0)
      %p1 = $dtype[8, 10] parameter(1)
      ROOT %concatenate = $dtype[16, 10] concatenate(%p0, %p1), dimensions={0}
    }

    ENTRY entry {
      %p0 = $dtype[8, 10] parameter(0)
      %p1 = $dtype[8, 10] parameter(1)
      ROOT %fusion = $dtype[16, 10] fusion(%p0, %p1), kind=kCustom,
        calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, Slice) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule slice

    ynn_fusion {
      %input = $dtype[16, 16] parameter(0)
      ROOT %slice = $dtype[8, 8] slice(%input), slice={[4:12], [4:12]}
    }

    ENTRY entry {
      %p0 = $dtype[16, 16] parameter(0)
      ROOT %fusion = $dtype[8, 8] fusion(%p0), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, IotaRank1) {
  const std::string& in_dtype = GetParam().in_dtype;
  if (in_dtype == "bf16" || in_dtype == "f64") {
    GTEST_SKIP() << "Iota not supported for " << in_dtype;
  }
  constexpr absl::string_view kModuleStr = R"(
    HloModule iota

    ynn_fusion {
      ROOT %iota = $dtype[8] iota(), iota_dimension=0
    }

    ENTRY entry {
      ROOT %fusion = $dtype[8] fusion(), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, IotaDim0) {
  const std::string& in_dtype = GetParam().in_dtype;
  if (in_dtype == "bf16" || in_dtype == "f64") {
    GTEST_SKIP() << "Iota not supported for " << in_dtype;
  }
  constexpr absl::string_view kModuleStr = R"(
    HloModule iota

    ynn_fusion {
      ROOT %iota = $dtype[8, 10] iota(), iota_dimension=0
    }

    ENTRY entry {
      ROOT %fusion = $dtype[8, 10] fusion(), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, IotaDim1) {
  const std::string& in_dtype = GetParam().in_dtype;
  if (in_dtype == "bf16" || in_dtype == "f64") {
    GTEST_SKIP() << "Iota not supported for " << in_dtype;
  }
  constexpr absl::string_view kModuleStr = R"(
    HloModule iota

    ynn_fusion {
      ROOT %iota = $dtype[8, 10] iota(), iota_dimension=1
    }

    ENTRY entry {
      ROOT %fusion = $dtype[8, 10] fusion(), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, ConcatenateAddMul) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule concatenate_add_mul

    ynn_fusion {
      %p0 = $dtype[8, 10] parameter(0)
      %p1 = $dtype[8, 10] parameter(1)
      %add = $dtype[8, 10] add(%p0, %p1)
      %mul = $dtype[8, 10] multiply(%p0, %p1)
      ROOT %concatenate = $dtype[16, 10] concatenate(%add, %mul), dimensions={0}
    }

    ENTRY entry {
      %p0 = $dtype[8, 10] parameter(0)
      %p1 = $dtype[8, 10] parameter(1)
      ROOT %fusion = $dtype[16, 10] fusion(%p0, %p1), kind=kCustom,
        calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

struct AddWithBroadcastConfig {
  std::string operand_shape;
  std::string broadcast_dims;
};

class AddWithBroadcastTest
    : public HloPjRtInterpreterReferenceMixin<HloTestBase>,
      public ::testing::WithParamInterface<
          std::tuple<YnnFusionTestParams, AddWithBroadcastConfig>> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<
          std::tuple<YnnFusionTestParams, AddWithBroadcastConfig>>& info) {
    const auto& [type_params, config] = info.param;
    std::string dims_str =
        config.broadcast_dims.empty() ? "all" : config.broadcast_dims;
    return absl::StrCat(type_params.in_dtype, "_dims_", dims_str);
  }

 protected:
  void RunTest(absl::string_view hlo_template) {
    const auto& [type_params, config] = GetParam();
    std::string hlo_text = absl::StrReplaceAll(
        hlo_template, {{"$dtype", type_params.in_dtype},
                       {"$operand_shape", config.operand_shape},
                       {"$broadcast_dims", config.broadcast_dims}});
    bool bf16_compute = absl::StrContains(hlo_text, "bf16");
    double tolerance = bf16_compute ? 1e-2 : 1e-7;
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        hlo_text, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
  }
};

TEST_P(AddWithBroadcastTest, Run) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule add_with_broadcast

    ynn_fusion {
      %lhs = $dtype[8, 10] parameter(0)
      %rhs = $dtype[$operand_shape] parameter(1)
      %broadcast = $dtype[8, 10] broadcast(%rhs), dimensions={$broadcast_dims}
      ROOT %add = $dtype[8, 10] add(%lhs, %broadcast)
    }

    ENTRY entry {
      %p0 = $dtype[8, 10] parameter(0)
      %p1 = $dtype[$operand_shape] parameter(1)
      ROOT %fusion = $dtype[8, 10] fusion(%p0, %p1), kind=kCustom,
        calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, DotWithConstant) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule dot_with_constant

    ynn_fusion {
      %lhs = f32[10, 10] parameter(0)
      %rhs = f32[10, 10] parameter(1), frontend_attributes={is_constant="true"}
      ROOT %dot = f32[10, 10] dot(%lhs, %rhs), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }

    ENTRY entry {
      %p0 = f32[10, 10] parameter(0)
      %p1 = f32[10, 10] parameter(1)
      ROOT %fusion = f32[10, 10] fusion(%p0, %p1), kind=kCustom,
        calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, SliceWithStrides) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule slice_with_strides

    ynn_fusion {
      %input = $dtype[16, 32] parameter(0)
      ROOT %slice = $dtype[4, 8] slice(%input), slice={[4:12:2], [2:26:3]}
    }

    ENTRY entry {
      %p0 = $dtype[16, 32] parameter(0)
      ROOT %fusion = $dtype[4, 8] fusion(%p0), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionTest, SliceReduce) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule m

    add {
      p0 = $dtype[] parameter(0)
      p1 = $dtype[] parameter(1)
      ROOT add = $dtype[] add(p0, p1)
    }

    ynn_fusion {
      p1 = $dtype[3,4,4,4]{3,2,1,0} parameter(1)
      slice = $dtype[1,4,4,4]{3,2,1,0} slice(p1),
        slice={[2:3], [0:4], [0:4], [0:4]}
      p0 = $dtype[] parameter(0)
      ROOT reduce = $dtype[4,4]{1,0} reduce(slice, p0), dimensions={0,2},
        to_apply=add
    }

    ENTRY entry {
      p0 = $dtype[] parameter(0)
      p1 = $dtype[3,4,4,4] parameter(1)
      ROOT fusion = $dtype[4,4] fusion(p0, p1), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

std::vector<YnnFusionTestParams> GetSameTypeTestCases() {
  return std::vector<YnnFusionTestParams>({
      YnnFusionTestParams{"bf16", "bf16"},
      YnnFusionTestParams{"f32", "f32"},
      YnnFusionTestParams{"f64", "f64"},
  });
}

INSTANTIATE_TEST_SUITE_P(YnnFusionTestInstantiation, YnnFusionTest,
                         ::testing::ValuesIn(GetSameTypeTestCases()),
                         YnnFusionTest::Name);

INSTANTIATE_TEST_SUITE_P(
    AddWithBroadcastTestInstantiation, AddWithBroadcastTest,
    ::testing::Combine(::testing::ValuesIn(GetSameTypeTestCases()),
                       ::testing::Values(AddWithBroadcastConfig{"8", "0"},
                                         AddWithBroadcastConfig{"10", "1"},
                                         AddWithBroadcastConfig{"", ""})),
    AddWithBroadcastTest::Name);

using YnnFusionReduceWindowTest = YnnFusionTest;

TEST_P(YnnFusionReduceWindowTest, ReduceWindow) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule reduce_window

    %add {
      %lhs = $out_dtype[] parameter(0)
      %rhs = $out_dtype[] parameter(1)
      ROOT %add = $out_dtype[] add(%lhs, %rhs)
    }

    ynn_fusion {
      %input = $dtype[4] parameter(0)
      %zero = $out_dtype[] constant(0)
      %converted = $out_dtype[4] convert(%input)
      ROOT %reduce_window = $out_dtype[2] reduce-window(%converted, %zero),
        window={size=3 stride=3 pad=1_1}, to_apply=%add
    }

    ENTRY entry {
      %p0 = $dtype[4] parameter(0)
      ROOT %fusion = $out_dtype[2] fusion(%p0), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionReduceWindowTest, ReduceWindowWithInit) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule reduce_window_with_init

    %add {
      %lhs = f32[] parameter(0)
      %rhs = f32[] parameter(1)
      ROOT %add = f32[] add(%lhs, %rhs)
    }

    ynn_fusion {
      %param_0 = f32[4,6]{1,0} parameter(0)
      %c.0 = f32[] constant(1)
      ROOT %reduce_window.0 = f32[3,9]{1,0} reduce-window(%param_0, %c.0),
        window={size=2x1 stride=2x1 pad=0_3x1_2 rhs_dilate=1x2}, to_apply=%add
    }

    ENTRY entry {
      %p0 = f32[4,6]{1,0} parameter(0)
      ROOT %fusion = f32[3,9]{1,0} fusion(%p0), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionReduceWindowTest, ReduceWindowAndReduce) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule reduce_window_and_reduce

    %add {
      %lhs = $out_dtype[] parameter(0)
      %rhs = $out_dtype[] parameter(1)
      ROOT %add = $out_dtype[] add(%lhs, %rhs)
    }

    ynn_fusion {
      %input = $dtype[4] parameter(0)
      %zero = $out_dtype[] constant(0)
      %converted = $out_dtype[4] convert(%input)
      %rw = $out_dtype[2] reduce-window(%converted, %zero),
          window={size=3 stride=3 pad=1_1}, to_apply=%add
      ROOT %reduce = $out_dtype[] reduce(%rw, %zero), dimensions={0},
          to_apply=%add
    }

    ENTRY entry {
      %p0 = $dtype[4] parameter(0)
      ROOT %fusion = $out_dtype[] fusion(%p0), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionReduceWindowTest, ReduceConvert) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule reduce_convert

    %add {
      %lhs = $dtype[] parameter(0)
      %rhs = $dtype[] parameter(1)
      ROOT %add = $dtype[] add(%lhs, %rhs)
    }

    ynn_fusion {
      %input = $dtype[64, 2] parameter(0)
      %zero = $dtype[] constant(0)
      %reduced = $dtype[64] reduce(%input, %zero), dimensions={1}, to_apply=%add
      ROOT %convert = bf16[64] convert(%reduced)
    }

    ENTRY entry {
      %p0 = $dtype[64, 2] parameter(0)
      ROOT %fusion = bf16[64] fusion(%p0), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionReduceWindowTest, ConvertReduce) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule convert_reduce

    %add {
      %lhs = $dtype[] parameter(0)
      %rhs = $dtype[] parameter(1)
      ROOT %add = $dtype[] add(%lhs, %rhs)
    }

    ynn_fusion {
      %input = bf16[64, 2] parameter(0)
      %zero = $dtype[] constant(0)
      %converted = $dtype[64, 2] convert(%input)
      ROOT %reduce = $dtype[64] reduce(%converted, %zero), dimensions={1}, to_apply=%add
    }

    ENTRY entry {
      %p0 = bf16[64, 2] parameter(0)
      ROOT %fusion = $dtype[64] fusion(%p0), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionReduceWindowTest, MixedPrecisionReduce) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule convert_reduce

    %add {
      %lhs = $dtype[] parameter(0)
      %rhs = $dtype[] parameter(1)
      ROOT %add = $dtype[] add(%lhs, %rhs)
    }

    ynn_fusion {
      %input = bf16[64, 2] parameter(0)
      %zero = $dtype[] constant(0)
      ROOT %reduce = $dtype[64] reduce(%input, %zero), dimensions={1},
        to_apply=%add
    }

    ENTRY entry {
      %p0 = bf16[64, 2] parameter(0)
      ROOT %fusion = $dtype[64] fusion(%p0), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

TEST_P(YnnFusionReduceWindowTest, ReduceWindowMax) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule reduce_window_max

    %max {
      %lhs = $dtype[] parameter(0)
      %rhs = $dtype[] parameter(1)
      ROOT %max = $dtype[] maximum(%lhs, %rhs)
    }

    ynn_fusion {
      %param_0 = $dtype[3,2,4,6] parameter(0)
      %constant.0 = $dtype[] constant(-inf)
      ROOT %reduce_window_max.0 = $dtype[4,2,4,8] reduce-window(%param_0, %constant.0),
        window={size=1x1x2x1 stride=1x2x2x1 pad=0_1x1_0x2_3x0_2}, to_apply=%max
    }

    ENTRY entry {
      %p0 = $dtype[3,2,4,6] parameter(0)
      ROOT %fusion = $dtype[4,2,4,8] fusion(%p0), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

INSTANTIATE_TEST_SUITE_P(YnnFusionReduceWindowTestInstantiation,
                         YnnFusionReduceWindowTest,
                         ::testing::Values(YnnFusionTestParams{"bf16", "f32"},
                                           YnnFusionTestParams{"f32", "f32"},
                                           YnnFusionTestParams{"f64", "f64"}),
                         YnnFusionTest::Name);

template <typename T>
absl::Span<const T> GetUnaryOpTestInputs() {
  static absl::NoDestructor<std::vector<T>> values{[]() {
    constexpr size_t kN = 10000;
    std::vector<T> values;
    values.reserve(kN * 2 + 20);
    values.push_back(0.0f);
    values.push_back(-0.0f);
    values.push_back(1.0f);
    values.push_back(-1.0f);
    values.push_back(std::numeric_limits<T>::max());
    values.push_back(std::numeric_limits<T>::min());
    values.push_back(-std::numeric_limits<T>::max());
    values.push_back(-std::numeric_limits<T>::min());
    values.push_back(std::numeric_limits<T>::infinity());
    values.push_back(-std::numeric_limits<T>::infinity());
    values.push_back(std::numeric_limits<T>::quiet_NaN());

    double log_max = std::log2(std::numeric_limits<T>::max());
    for (size_t i = 0; i < kN; ++i) {
      values.push_back(std::exp2((log_max * i) / kN));
      values.push_back(-std::exp2((log_max * i) / kN));
    }

    return values;
  }()};
  return *values;
}

Literal GetUnaryOpTestInputs(PrimitiveType type) {
  switch (type) {
    case F32:
      return LiteralUtil::CreateR1(GetUnaryOpTestInputs<float>());
    case F64:
      return LiteralUtil::CreateR1(GetUnaryOpTestInputs<double>());
    default:
      LOG(FATAL) << "Unsupported type: " << PrimitiveType_Name(type);
  }
}

struct YnnUnaryOpTestParams {
  HloOpcode op;
  PrimitiveType in_dtype;
  PrimitiveType out_dtype;
  ErrorSpec error_spec{0.0};
};

class YnnUnaryOpTest
    : public HloPjRtInterpreterReferenceMixin<HloTestBase>,
      public ::testing::WithParamInterface<YnnUnaryOpTestParams> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<YnnUnaryOpTestParams>& info) {
    return absl::StrCat(
        absl::StrReplaceAll(xla::HloOpcodeString(info.param.op), {{"-", "_"}}),
        "_", absl::AsciiStrToLower(PrimitiveType_Name(info.param.in_dtype)),
        "_", absl::AsciiStrToLower(PrimitiveType_Name(info.param.out_dtype)));
  }
};

TEST_P(YnnUnaryOpTest, Run) {
  HloOpcode op = GetParam().op;
  PrimitiveType in_type = GetParam().in_dtype;
  PrimitiveType out_type = GetParam().out_dtype;
  ErrorSpec error_spec = GetParam().error_spec;

  absl::string_view hlo = R"(
    HloModule convert_reduce

    ynn_fusion {
      %input = $in_dtype[$d0] parameter(0)
      ROOT %output = $out_dtype[$d0] $op(%input)
    }

    ENTRY entry {
      %p0 = $in_dtype[$d0] parameter(0)
      ROOT %fusion = $out_dtype[$d0] fusion(%p0), kind=kCustom,
        calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  Literal p0 = GetUnaryOpTestInputs(in_type);

  std::vector<const Literal*> args = {&p0};
  std::string hlo_text = absl::StrReplaceAll(
      hlo, {{"$in_dtype", absl::AsciiStrToLower(PrimitiveType_Name(in_type))},
            {"$out_dtype", absl::AsciiStrToLower(PrimitiveType_Name(out_type))},
            {"$op", xla::HloOpcodeString(op)},
            {"$d0", absl::StrCat(p0.shape().dimensions(0))}});
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), args, error_spec));
}

ErrorSpec F32_ErrorSpec{/*aabs=*/1e-7, /*arel=*/1e-7};
ErrorSpec F64_ErrorSpec{/*aabs=*/1e-15, /*arel=*/1e-15};

static YnnUnaryOpTestParams unary_op_test_params[] = {
    {HloOpcode::kConvert, F32, BF16},
    {HloOpcode::kConvert, F64, F32},

    {HloOpcode::kAbs, F32, F32},
    {HloOpcode::kCeil, F32, F32},
    {HloOpcode::kErf, F32, F32, ErrorSpec{/*aabs=*/2e-7, /*arel=*/3e-7}},
    {HloOpcode::kExp, F32, F32, ErrorSpec{/*aabs=*/2e-38, /*arel=*/4e-6}},
    {HloOpcode::kExpm1, F32, F32, ErrorSpec{/*aabs=*/2e-38, /*arel=*/4e-6}},
    {HloOpcode::kFloor, F32, F32},
    {
        HloOpcode::kLog,
        F32,
        F32,
        // On ARM, XLA's log returns NaN when the input is close to infinity.
        ErrorSpec{/*aabs=*/0, /*arel=*/4e-7, /*relaxed_nans=*/true},
    },
    // TODO(b/515053903): This test is not reliably passing.
    // {HloOpcode::kLog1p, F32, F32, ErrorSpec{/*aabs=*/2e-7, /*arel=*/2e-7}},
    {HloOpcode::kLogistic, F32, F32, ErrorSpec{/*aabs=*/2e-7, /*arel=*/2e-7}},
    {HloOpcode::kNegate, F32, F32},
    {HloOpcode::kRoundNearestEven, F32, F32},
    {HloOpcode::kRsqrt, F32, F32, F32_ErrorSpec},
    {HloOpcode::kSign, F32, F32},
    {HloOpcode::kSqrt, F32, F32, F32_ErrorSpec},
    {HloOpcode::kTanh, F32, F32, ErrorSpec{/*aabs=*/2e-7, /*arel=*/4e-7}},

    {HloOpcode::kAbs, F64, F64},
    {HloOpcode::kCeil, F64, F64},
    {HloOpcode::kErf, F64, F64, F64_ErrorSpec},
    {HloOpcode::kExp, F64, F64, ErrorSpec(/*aabs=*/2e-308, /*arel=*/4e-14)},
    {HloOpcode::kExpm1, F64, F64, F64_ErrorSpec},
    {HloOpcode::kFloor, F64, F64},
    {HloOpcode::kLog, F64, F64, F64_ErrorSpec},
    {HloOpcode::kLog1p, F64, F64, F64_ErrorSpec},
    {HloOpcode::kLogistic, F64, F64, F64_ErrorSpec},
    {HloOpcode::kNegate, F64, F64},
    {HloOpcode::kRoundNearestEven, F64, F64},
    {HloOpcode::kRsqrt, F64, F64, F64_ErrorSpec},
    {HloOpcode::kSign, F64, F64},
    {HloOpcode::kSqrt, F64, F64, F64_ErrorSpec},
    {HloOpcode::kTanh, F64, F64, F64_ErrorSpec},
};

INSTANTIATE_TEST_SUITE_P(YnnUnaryOpTestInstantiation, YnnUnaryOpTest,
                         ::testing::ValuesIn(unary_op_test_params),
                         YnnUnaryOpTest::Name);

}  // namespace
}  // namespace xla::cpu
