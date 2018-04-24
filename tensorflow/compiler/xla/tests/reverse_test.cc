/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

#ifdef XLA_BACKEND_SUPPORTS_BFLOAT16
// Tests both F32 and BF16.
static std::array<bool, 2> use_bfloat16_params{false, true};
#else
// Only tests F32.
static std::array<bool, 1> use_bfloat16_params{false};
#endif

struct ReverseSpec {
  tensorflow::gtl::ArraySlice<int64> input_dims;
  tensorflow::gtl::ArraySlice<int64> reversal;
  bool use_bfloat16;

  string ToTestCaseName() const {
    return tensorflow::strings::Printf(
        "reverse_%s_in_dims_%s_%s",
        tensorflow::str_util::Join(input_dims, "x").c_str(),
        tensorflow::str_util::Join(reversal, "x").c_str(),
        use_bfloat16 ? "bf16" : "f32");
  }
};

static std::vector<ReverseSpec> GetTestCases() {
  // clang-format off
  return ExpandUseBfloat16<ReverseSpec>(
      use_bfloat16_params,
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

class FloatReverseTest : public ClientLibraryTestBase,
                         public ::testing::WithParamInterface<ReverseSpec> {
 public:
  FloatReverseTest() { set_use_bfloat16(GetParam().use_bfloat16); }
};

TEST_P(FloatReverseTest, Reverses) {
  const ReverseSpec& spec = GetParam();
  std::vector<float> input_vector(
      ShapeUtil::ElementsIn(ShapeUtil::MakeShape(F32, spec.input_dims)));
  std::iota(input_vector.begin(), input_vector.end(), 0.0);
  auto r1_literal = Literal::CreateR1<float>(input_vector);
  auto input_literal = r1_literal->Reshape(spec.input_dims).ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  auto a = AddParam(*input_literal, &builder);
  builder.Rev(a, spec.reversal);

  std::unique_ptr<Literal> expected = input_literal->CloneToUnique();
  std::vector<int64> output_indices(spec.input_dims.size());
  expected->EachCell<float>(
      [&](tensorflow::gtl::ArraySlice<int64> indices, float) {
        for (int64 i = 0; i < indices.size(); ++i) {
          output_indices[i] = indices[i];
        }
        float value = input_literal->Get<float>(indices);
        for (int64 dim : spec.reversal) {
          output_indices[dim] = (spec.input_dims[dim] - 1) - indices[dim];
        }
        expected->Set<float>(output_indices, value);
      });
  ComputeAndCompareLiteral(&builder, *expected, {});
}

INSTANTIATE_TEST_CASE_P(FloatReverseInstance, FloatReverseTest,
                        ::testing::ValuesIn(GetTestCases()),
                        ::testing::PrintToStringParamName());

// A simple test class which not templated by float precision.
class ReverseTest : public ClientLibraryTestBase {};

// Tests the reverse operation on a 4D U8 array on dimension 0 and 3.
XLA_TEST_F(ReverseTest, Reverse4DU8ArrayOnDim23) {
  ComputationBuilder b(client_, TestName());
  // Input shape is U8[1x2x3x4].
  // clang-format off
  Array4D<uint8> input({{
    {{1, 2, 3, 4},
     {5, 6, 7, 8},
     {9, 10, 11, 12}},
    {{13, 14, 15, 16},
     {17, 18, 19, 20},
     {21, 22, 23, 24}},
  }});
  // clang-format on

  b.Rev(b.ConstantR4FromArray4D<uint8>(input), {0, 3});

  // clang-format off
  Array4D<uint8> expected({{
    {{4, 3, 2, 1},
     {8, 7, 6, 5},
     {12, 11, 10, 9}},
    {{16, 15, 14, 13},
     {20, 19, 18, 17},
     {24, 23, 22, 21}},
  }});
  // clang-format on
  ComputeAndCompareR4<uint8>(&b, expected, {});
}

// Tests the reverse operation on a 4D float array on dimension 0 and 1.
TEST_F(ReverseTest, Reverse4DFloatArrayOnDim01) {
  ComputationBuilder b(client_, TestName());
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

  b.Rev(b.ConstantR4FromArray4D<float>(input), {0, 1});

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
