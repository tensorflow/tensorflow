/* Copyright 2017 The OpenXLA Authors.

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
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace {

class BitcastConvertTest
    : public ClientLibraryTestRunnerMixin<
          HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>> {
 public:
  BitcastConvertTest() {
    mutable_debug_options()->add_xla_disable_hlo_passes("algsimp");
    mutable_debug_options()->add_xla_disable_hlo_passes("inline");
  }
};

TEST_F(BitcastConvertTest, ConvertR1S32ToR1S32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {42, 64});
  BitcastConvertType(a, S32);

  std::vector<int32_t> expected = {42, 64};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertR1F32ToR1F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.0f, 64.0f});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, BitcastR1S32ToR1F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder,
                               {0, static_cast<int32_t>(0x80000000), 0x3F800000,
                                static_cast<int32_t>(0xBF800000), 0x3F000000,
                                static_cast<int32_t>(0xBF000000)});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertR1S0S32ToR1S0F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertR1F32ToR1S32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.6, 64.4});
  BitcastConvertType(a, S32);

  std::vector<int32_t> expected = {0x422a6666, 0x4280cccd};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertR1F8e4m3fnToR1U8) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint8_t>(&builder, {0x38, 0xC0});
  BitcastConvertType(a, F8E4M3FN);

  std::vector<tsl::float8_e4m3fn> expected = {tsl::float8_e4m3fn{1.0},
                                              tsl::float8_e4m3fn{-2.0}};
  ComputeAndCompareR1<tsl::float8_e4m3fn>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertS32Extremes) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {std::numeric_limits<int32_t>::min(),
                                          std::numeric_limits<int32_t>::max()});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {-0.0f, NAN};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0, 0));
}

TEST_F(BitcastConvertTest, ConvertMapToS32) {
  XlaBuilder builder(TestName());
  auto b = builder.CreateSubBuilder("convert");
  auto param = Parameter(b.get(), 0, ShapeUtil::MakeShape(F32, {}), "in");
  BitcastConvertType(param, S32);
  auto a = ConstantR1<float>(&builder, {42.0f, 64.0f});
  Map(&builder, {a}, b->BuildAndNoteError(), {0});

  std::vector<int32_t> expected = {0x42280000, 0x42800000};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertMapToF32) {
  XlaBuilder builder(TestName());
  auto b = builder.CreateSubBuilder("convert");
  auto param = Parameter(b.get(), 0, ShapeUtil::MakeShape(S32, {}), "in");
  BitcastConvertType(param, F32);
  auto a = ConstantR1<int32_t>(&builder, {0x42280000, 0x42800000});
  Map(&builder, {a}, b->BuildAndNoteError(), {0});

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

// Regression test for b/31758660. When ReshapeMover transforms
//   input -> reshape -> convert
// to
//   input -> convert -> reshape
// the new convert should have the same element type as the old convert.
TEST_F(BitcastConvertTest, ConvertReshape) {
  XlaBuilder builder(TestName());
  auto input = ConstantR1<int32_t>(&builder, {0x42280000});
  auto reshape = Reshape(input, /*dimensions=*/{});
  BitcastConvertType(reshape, F32);

  ComputeAndCompareR0<float>(&builder, 42.0f, {});
}

class BitcastConvertHloTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {};

TEST_F(BitcastConvertHloTest, S32to4S8) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_smaller

ENTRY main {
  p = s32[10] parameter(0)
  ROOT out = s8[10,4] bitcast-convert(p)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{0, 0}));
}

TEST_F(BitcastConvertHloTest, FourS8toS32) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_larger

ENTRY main {
  p = s8[10,4] parameter(0)
  ROOT out = s32[10] bitcast-convert(p)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{0, 0}));
}

TEST_F(BitcastConvertHloTest, F32to2F16) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_smaller

ENTRY main {
  p = f32[10] parameter(0)
  ROOT out = f16[10,2] bitcast-convert(p)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(BitcastConvertHloTest, TwoF16toF32) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_smaller

ENTRY main {
  p = f16[10,2] parameter(0)
  ROOT out = f32[10] bitcast-convert(p)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla
