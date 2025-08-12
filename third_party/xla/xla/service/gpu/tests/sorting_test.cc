/* Copyright 2020 The OpenXLA Authors.

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
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "Eigen/Core"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

class TypeSupportTest : public GpuCodegenTest,
                        public ::testing::WithParamInterface<PrimitiveType> {};

TEST_P(TypeSupportTest, SortSupportsType) {
  constexpr char kHloTemplate[] = R"(
compare {
p.0.lhs = $0[] parameter(0)
p.0.rhs = $0[] parameter(1)
ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY test {
p0 = $0[32]{0} parameter(0)
ROOT sort = $0[32]{0} sort(p0), dimensions={0}, is_stable=true,
to_apply=compare
})";
  std::string hlo = absl::Substitute(
      kHloTemplate, primitive_util::LowercasePrimitiveTypeName(GetParam()));
  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{0, 0}));
}

INSTANTIATE_TEST_SUITE_P(
    , TypeSupportTest,
    // 4bit types like U4, S4, or F4E2M1FN are currently not supported.
    // F8E8M0FNU cannot represent NaNs and fails the test below.
    ::testing::ValuesIn({
        PRED,                               // boolean
        S8,         S16,    S32,      S64,  // signed
        U8,         U16,    U32,      U64,  // unsigned
        F8E5M2,     F8E4M3, F8E4M3FN, F8E4M3B11FNUZ, F8E3M4, F8E5M2FNUZ,
        F8E4M3FNUZ, F16,    BF16,     F32,           F64  // floating point
    }),
    [](const ::testing::TestParamInfo<TypeSupportTest::ParamType>& info) {
      return primitive_util::LowercasePrimitiveTypeName(info.param);
    });

class SortingTest : public GpuCodegenTest {
 protected:
  SortingTest() {}
};

TEST_F(SortingTest, Regression1) {
  const char* hlo_text = R"(
HloModule TestModule

compare {
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}


ENTRY TestComputation {
  x = f32[3, 2]{1, 0} parameter(0)
  tr = f32[2, 3]{1, 0} transpose(x), dimensions={1,0}
  b = f32[3, 2]{0, 1} bitcast(tr)
  ROOT sort = f32[3, 2]{0, 1} sort(b), dimensions={1}, to_apply=compare
}

)";

  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

// Test that verifies the IgnoreMemorySpace option works correctly
TEST_F(SortingTest, LayoutsInShapesEqualWithIgnoreMemorySpace) {
  const char* hlo_text = R"(
HloModule TestModule

compare {
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  p.1.lhs = f32[] parameter(2)
  p.1.rhs = f32[] parameter(3)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY TestComputation {
  data = f32[6] parameter(0)

  // Create two copies in different memory spaces
  keys = f32[6] copy(data)
  values = f32[6] copy(data)

  // Sort operation with operands in different memory spaces
  ROOT sort = (f32[6], f32[6]) sort(keys, values), dimensions={0}, to_apply=compare
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  HloInstruction* values =
      module->entry_computation()->GetInstructionWithName("values");
  Shape values_shape = values->shape();
  values_shape.mutable_layout()->set_memory_space(1);
  *values->mutable_shape() = values_shape;

  const HloInstruction* sort = module->entry_computation()->root_instruction();
  EXPECT_EQ(sort->opcode(), HloOpcode::kSort);

  const HloInstruction* keys = sort->operand(0);

  EXPECT_FALSE(
      LayoutUtil::LayoutsInShapesEqual(keys->shape(), values->shape()));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(
      keys->shape(), values->shape(), Layout::Equal().IgnoreMemorySpace()));

  auto literal = LiteralUtil::CreateR1<float>({1.0, 6.0, 7.0, 0.0, 2.0, 5.0});
  absl::StatusOr<Literal> executed = Execute(std::move(module), {&literal});
  EXPECT_TRUE(executed.ok()) << executed.status().message();
}

// Size of the radix sort tests.
static constexpr int kRadixSortTestSize = 100000;

template <typename T>
bool CheckOrder(T lhs, T rhs, bool asc, int pos) {
  if (asc) {
    EXPECT_TRUE(lhs <= rhs) << lhs << " > " << rhs << " @" << pos;
  } else {
    EXPECT_TRUE(lhs >= rhs) << lhs << " < " << rhs << " @" << pos;
  }
  return lhs != rhs;
}

bool CompareAdjacentValues(const Literal& literal, int index, bool ascending) {
  if (primitive_util::IsFloatingPointType(literal.shape().element_type())) {
    return CheckOrder(*literal.GetAsDouble({index - 1}),
                      *literal.GetAsDouble({index}), ascending, index);
  } else {
    return CheckOrder(*literal.GetIntegralAsS64({index - 1}),
                      *literal.GetIntegralAsS64({index}), ascending, index);
  }
}

std::string GetTypeName(PrimitiveType type) {
  return absl::AsciiStrToLower(PrimitiveType_Name(type));
}

// Test cub::DeviceRadixSort::SortKeys in XLA
class CubSortKeysTest : public GpuCodegenTest,
                        public ::testing::WithParamInterface<
                            std::tuple<std::shared_ptr<Literal>, bool>> {};

TEST_P(CubSortKeysTest, SortKeys) {
  constexpr char kHloTemplate[] = R"(
HloModule TestModule

ENTRY %main {
  %input = $0[$1] parameter(0)
  %sort = ($0[$1], u8[$2]) custom-call(%input),
      custom_call_target="__cub$$DeviceRadixSort",
      backend_config="{\"descending\": $3}"
  ROOT %gte = get-tuple-element(%sort), index=0
}
)";

  bool ascending = std::get<1>(GetParam());
  std::string hlo = absl::Substitute(
      kHloTemplate,
      GetTypeName(std::get<0>(GetParam())->shape().element_type()),
      kRadixSortTestSize,
      kRadixSortTestSize * 10,  // added scratch buffer size
      ascending ? "false" : "true");

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  std::vector<Literal*> literals = {std::get<0>(GetParam()).get()};
  auto result = ExecuteAndTransfer(std::move(module), literals);

  bool has_diff = false;
  for (int i = 1; i < kRadixSortTestSize; ++i) {
    has_diff |= CompareAdjacentValues(result, i, ascending);
  }
  EXPECT_TRUE(has_diff) << "uninitialized output";
}

// Test cub::DeviceRadixSort::SortPairs in XLA
class CubSortPairsTest
    : public GpuCodegenTest,
      public ::testing::WithParamInterface<
          std::tuple<std::shared_ptr<Literal>, PrimitiveType, bool>> {};

TEST_P(CubSortPairsTest, SortPairs) {
  // TODO(b/380814507): Remove the disabling part once fixed.
  auto cc = backend()
                .default_stream_executor()
                ->GetDeviceDescription()
                .cuda_compute_capability();
  if (cc.IsAtLeastHopper() &&
      std::get<0>(GetParam())->shape().element_type() == U16 &&
      std::get<1>(GetParam()) == F64) {
    GTEST_SKIP()
        << "CUB sort does not work for pair sorting (U16, F64) on Hopper.";
  }

  constexpr char kHloTemplate[] = R"(
HloModule TestModule

ENTRY %main {
  %keys = $0[$2] parameter(0)
  %values = $1[$2] convert(%keys)
  ROOT %sort = ($0[$2], $1[$2], u8[$3]) custom-call(%keys, %values),
      custom_call_target="__cub$$DeviceRadixSort",
      backend_config="{\"descending\": $4}"
}
)";

  bool ascending = std::get<2>(GetParam());
  std::string hlo = absl::Substitute(
      kHloTemplate,
      GetTypeName(std::get<0>(GetParam())->shape().element_type()),
      GetTypeName(std::get<1>(GetParam())), kRadixSortTestSize,
      kRadixSortTestSize * 20,  // added scratch buffer size
      ascending ? "false" : "true");

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  std::vector<Literal*> literals = {std::get<0>(GetParam()).get()};
  auto result_tuple = ExecuteAndTransfer(std::move(module), literals);
  std::vector<Literal> result = result_tuple.DecomposeTuple();

  bool has_diff = false;
  for (int i = 1; i < kRadixSortTestSize; ++i) {
    has_diff |= CompareAdjacentValues(result[0], i, ascending);
    has_diff |= CompareAdjacentValues(result[1], i, ascending);
  }
  EXPECT_TRUE(has_diff) << "uninitialized output";
}

// Literal creation helper.
template <PrimitiveType P, typename T>
std::shared_ptr<Literal> CreateRandomLiteral(T mean, T stddev) {
  Shape shape = ShapeUtil::MakeShape(P, {kRadixSortTestSize});
  auto maybe_literal =
      LiteralUtil::CreateRandomLiteral<P, T>(shape, mean, stddev);
  CHECK_OK(maybe_literal);
  auto shared_literal = std::make_shared<Literal>(shape);
  CHECK_OK(shared_literal->MoveFrom(std::move(*maybe_literal)));
  return shared_literal;
}

INSTANTIATE_TEST_SUITE_P(
    TestRadixSort, CubSortKeysTest,
    ::testing::Combine(
        ::testing::Values(
            // TODO(b/300112551): upgrade CUB to version 1.13
            // CreateRandomLiteral<BF16, bfloat16>(
            //   bfloat16(),
            //   Eigen::bfloat16_impl::float_to_bfloat16_rtne<true>(1)),
            CreateRandomLiteral<F16, half>(
                half(), Eigen::half_impl::float_to_half_rtne(1)),
            CreateRandomLiteral<F32, float>(0, 1),
            CreateRandomLiteral<F64, double>(0, 1),
            CreateRandomLiteral<S8, int8_t>(0, 10),
            CreateRandomLiteral<S16, int16_t>(0, 1000),
            CreateRandomLiteral<S32, int32_t>(0, 1000000),
            CreateRandomLiteral<U8, uint8_t>(128, 10),
            CreateRandomLiteral<U16, uint16_t>(32768, 1000),
            CreateRandomLiteral<U32, uint32_t>(1 << 30, 1000000)),
        ::testing::Bool()),
    [](const ::testing::TestParamInfo<CubSortKeysTest::ParamType>& info) {
      return absl::StrCat(
          PrimitiveType_Name(std::get<0>(info.param)->shape().element_type()),
          "_", std::get<1>(info.param) ? "asc" : "desc");
    });

INSTANTIATE_TEST_SUITE_P(
    TestRadixSort, CubSortPairsTest,
    ::testing::Combine(
        ::testing::Values(CreateRandomLiteral<U16, uint16_t>(32768, 1000),
                          CreateRandomLiteral<U32, uint32_t>(32768, 1000),
                          CreateRandomLiteral<U64, uint64_t>(32768, 1000)),
        ::testing::Values(F16, F32, F64), ::testing::Bool()),
    [](const ::testing::TestParamInfo<CubSortPairsTest::ParamType>& info) {
      return absl::StrCat(
          PrimitiveType_Name(std::get<0>(info.param)->shape().element_type()),
          "_", PrimitiveType_Name(std::get<1>(info.param)), "_",
          std::get<2>(info.param) ? "asc" : "desc");
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
