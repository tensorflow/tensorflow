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

#include <array>
#include <cstdint>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/array4d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/client_library_test_runner_utils.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

constexpr std::array<PrimitiveType, 4> kPrimitiveTypeParams{
    F32,
    BF16,
    F8E5M2,
    F8E4M3FN,
};

struct ReverseSpec {
  std::vector<int64_t> input_dims;
  std::vector<int64_t> reversal;
  PrimitiveType test_type;

  std::string ToTestCaseName() const {
    return absl::StrFormat(
        "reverse_%s_in_dims_%s_%s", absl::StrJoin(input_dims, "x"),
        absl::StrJoin(reversal, "x"),
        primitive_util::LowercasePrimitiveTypeName(test_type));
  }
};

static std::vector<ReverseSpec> GetTestCases() {
  // clang-format off
  return ExpandTestType<ReverseSpec>(
      kPrimitiveTypeParams,
      {{{}, {}},
        {{0, 0}, {0, 1}},
        {{0, 1}, {0, 1}},
        {{1, 0}, {0, 1}},
        {{1, 1}, {0, 1}},
        {{2, 0, 4, 3}, {0, 2}},
        {{2, 0, 4, 3}, {1, 3}},
        {{1, 2, 3, 4}, {0, 3}},
        {{4, 3, 2, 1}, {0, 1}},
      });
  // clang-format on
}

void PrintTo(const ReverseSpec& spec, std::ostream* os) {
  *os << spec.ToTestCaseName();
}

class FloatReverseTest : public ClientLibraryTestRunnerMixin<
                             HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>,
                         public ::testing::WithParamInterface<ReverseSpec> {
 public:
  FloatReverseTest() { set_float_type(GetParam().test_type); }
};

TEST_P(FloatReverseTest, Reverses) {
  const ReverseSpec& spec = GetParam();
  std::vector<float> input_vector(ShapeUtil::ElementsIn(
      ShapeUtil::MakeValidatedShape(F32, spec.input_dims).value()));
  std::iota(input_vector.begin(), input_vector.end(), 0.0);
  const Literal r1_literal = LiteralUtil::CreateR1<float>(input_vector);
  TF_ASSERT_OK_AND_ASSIGN(const Literal input_literal,
                          r1_literal.Reshape(spec.input_dims));
  const Literal conv_input_literal =
      MaybeConvertLiteralToTestType(input_literal);

  XlaBuilder builder(TestName());
  XlaOp a = Parameter(&builder, 0, conv_input_literal.shape(), "input");
  Rev(a, spec.reversal);

  Literal expected = input_literal.Clone();
  std::vector<int64_t> output_indices(spec.input_dims.size());
  expected.EachCell<float>([&](absl::Span<const int64_t> indices, float) {
    for (int64_t i = 0; i < indices.size(); ++i) {
      output_indices[i] = indices[i];
    }
    float value = input_literal.Get<float>(indices);
    for (int64_t dim : spec.reversal) {
      output_indices[dim] = (spec.input_dims[dim] - 1) - indices[dim];
    }
    expected.Set<float>(output_indices, value);
  });
  ComputeAndCompareLiteral(&builder, expected, {&conv_input_literal});
}

INSTANTIATE_TEST_CASE_P(FloatReverseInstance, FloatReverseTest,
                        ::testing::ValuesIn(GetTestCases()),
                        ::testing::PrintToStringParamName());

// A simple test class which not templated by float precision.
using ReverseTest = ClientLibraryTestRunnerMixin<
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>;

// Tests the reverse operation on a 4D U8 array on dimension 0 and 3.
TEST_F(ReverseTest, Reverse4DU8ArrayOnDim23) {
  XlaBuilder b(TestName());
  // Input shape is U8[1x2x3x4].
  // clang-format off
  Array4D<uint8_t> input({{
    {{1, 2, 3, 4},
     {5, 6, 7, 8},
     {9, 10, 11, 12}},
    {{13, 14, 15, 16},
     {17, 18, 19, 20},
     {21, 22, 23, 24}},
  }});
  // clang-format on

  Rev(ConstantR4FromArray4D<uint8_t>(&b, input), {0, 3});

  // clang-format off
  Array4D<uint8_t> expected({{
    {{4, 3, 2, 1},
     {8, 7, 6, 5},
     {12, 11, 10, 9}},
    {{16, 15, 14, 13},
     {20, 19, 18, 17},
     {24, 23, 22, 21}},
  }});
  // clang-format on
  ComputeAndCompareR4<uint8_t>(&b, expected, {});
}

// Tests the reverse operation on a 4D float array on dimension 0 and 1.
TEST_F(ReverseTest, Reverse4DFloatArrayOnDim01) {
  XlaBuilder b(TestName());
  // Input shape is float[4x3x2x1].
  // clang-format off
  Array4D<float> input({
    {{{1.0f}, {2.0f}},
     {{3.0f}, {4.0f}},
     {{5.0f}, {6.0f}}},
    {{{7.0f}, {8.0f}},
     {{9.0f}, {10.0f}},
     {{11.0f}, {12.0f}}},
    {{{13.0f}, {14.0f}},
     {{15.0f}, {16.0f}},
     {{17.0f}, {18.0f}}},
    {{{19.0f}, {20.0f}},
     {{21.0f}, {22.0f}},
     {{23.0f}, {24.0f}}},
  });
  // clang-format on

  Rev(ConstantR4FromArray4D<float>(&b, input), {0, 1});

  // clang-format off
  Array4D<float> expected({
    {{{23.0f}, {24.0f}},
     {{21.0f}, {22.0f}},
     {{19.0f}, {20.0f}}},
    {{{17.0f}, {18.0f}},
     {{15.0f}, {16.0f}},
     {{13.0f}, {14.0f}}},
    {{{11.0f}, {12.0f}},
     {{9.0f}, {10.0f}},
     {{7.0f}, {8.0f}}},
    {{{5.0f}, {6.0f}},
     {{3.0f}, {4.0f}},
     {{1.0f}, {2.0f}}},
  });
  // clang-format on
  ComputeAndCompareR4<float>(&b, expected, {}, ErrorSpec(0.0001));
}

}  // namespace
}  // namespace xla
