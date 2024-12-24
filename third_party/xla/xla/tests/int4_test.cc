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

#include <optional>
#include <string>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

XLA_TEST_F(HloTestBase, InputIsOutput) {
  const std::string hlo_text = R"(
  HloModule InputIsOutput
  ENTRY main {
    ROOT p = s4[8] parameter(0)
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, Reshape) {
  const std::string hlo_text = R"(
  HloModule Reshape
  ENTRY main {
    x = s4[2,3] parameter(0)
    ROOT reshape = s4[3,2] reshape(x)
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, MultiReshape) {
  // Test reshaping multiple arrays to the same shape.
  const std::string hlo_text = R"(
  HloModule MultiReshape

  ENTRY main {
    x = s4[2,3,4] parameter(0)
    x_reshaped = s4[4,6] reshape(x)
    y = s4[4,3,2] parameter(1)
    y_reshaped = s4[4,6] reshape(y)
    ROOT z = s4[4,6] add(x_reshaped, y_reshaped)
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, Slice) {
  // Tests indexing s4 arrays in the presence of a slice instruction. On
  // CPUs/GPUs, the slice is fused with the s4 array.
  const std::string hlo_text = R"(
  HloModule Slice
  ENTRY main {
    x = s4[5,5] parameter(0)
    y = s8[5,5] convert(x)
    ROOT s = s8[3,3] slice(y), slice={[0:3],[1:4]}
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, Add) {
  const std::string hlo_text = R"(
  HloModule Add

  ENTRY main {
    x = s4[5,5] parameter(0)
    ROOT y = add(x, x)
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, Dot) {
  const std::string hlo_text = R"(
  HloModule Dot

  ENTRY main {
    x = s4[5,5] parameter(0)
    y = s4[5,5] parameter(1)
    ROOT z = dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, MixedTypeDot) {
  const std::string hlo_text = R"(
  HloModule Dot

  ENTRY main {
    x = s4[5,5] parameter(0)
    x_bf16 = bf16[5,5] convert(x)
    y = bf16[5,5] parameter(1)
    ROOT z = dot(x_bf16, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, NonMajorToMinorLayout) {
  // Tests transposing a matrix with a non-major-to-minor layout.
  const std::string hlo_text = R"(
  HloModule NonMajorToMinorLayout
  ENTRY main {
    x = s4[2,2]{0,1} parameter(0)
    ROOT transpose = s4[2,2]{0,1} transpose(x), dimensions={1,0}
  })";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, Transpose4d) {
  const std::string hlo_text = R"(
  ENTRY main {
    x = s4[2,2,2,2] parameter(0)
    ROOT y = s4[2,2,2,2] transpose(x), dimensions={3,2,1,0}
  })";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, TransposeDot) {
  const std::string hlo_text = R"(
  ENTRY main {
    x = s4[32,32,32] parameter(0)
    y = s4[32,32,32] parameter(1)
    xt = s4[32,32,32] transpose(x), dimensions={2,1,0}
    yt = s4[32,32,32] transpose(y), dimensions={2,0,1}
    ROOT output = dot(xt, yt), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  })";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, Int4Output2d) {
  // Tests outputting a 2D int4 array.
  const std::string hlo_text = R"(
  HloModule Int4Output2d
  ENTRY main {
    x = s8[2,2] parameter(0)
    ROOT y = s4[2,2] convert(x)
  })";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, TupleOutput) {
  // Tests tuple output with an int4 array
  const std::string hlo_text = R"(
  HloModule TupleOutput
  ENTRY main {
    x = s4[2,2] parameter(0)
    y = s8[2,2] convert(x)
    ROOT t = (s4[2,2], s8[2,2]) tuple(x, y)
  })";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, OddNumberOfElements) {
  // Tests writing to s4 arrays with an odd number of elements
  const std::string hlo_text = R"(
  HloModule OddNumberOfElements
  ENTRY main {
    x = s8[3,5] parameter(0)
    ROOT y = s4[3,5] convert(x)
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, Scalar) {
  // Tests reading an int4 scalar value
  const std::string hlo_text = R"(
  HloModule Scalar
  ENTRY main {
    x = s4[] parameter(0)
    y = s8[] convert(x)
    ROOT z = s8[3, 3] broadcast(y), dimensions={}
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, HorizontalLoopFusion) {
  // Tests an HLO module where horizontal loop fusion can be done on GPUs
  const std::string hlo_text = R"(
  HloModule HorizontalLoopFusion

  ENTRY main {
    x4 = s4[10] parameter(0)
    x8 = s8[10] convert(x4)
    y8 = s8[10] add(x8, x8)
    y4 = s4[10] convert(y8)

    x4_b = s4[13] parameter(1)
    x8_b = s8[13] convert(x4_b)
    y8_b = s8[13] add(x8_b, x8_b)
    y4_b = s4[13] convert(y8_b)

    ROOT t = (s4[10], s4[13]) tuple(y4, y4_b)
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, ReduceMultipleDimensions) {
  const std::string hlo_text = R"(
  add_computation {
    x = s4[] parameter(0)
    y = s4[] parameter(1)
    ROOT z = s4[] add(x, y)
  }

  ENTRY main {
    x = s4[3,4,5] parameter(0)
    zero = s4[] constant(0)
    ROOT reduce.10 = s4[5] reduce(x, zero), dimensions={0,1}, to_apply=add_computation
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, ReduceToScalar) {
  const std::string hlo_text = R"(
  add_computation {
    x = s4[] parameter(0)
    y = s4[] parameter(1)
    ROOT z = s4[] add(x, y)
  }

  ENTRY main {
    x = s4[30,40,50] parameter(0)
    zero = s4[] constant(0)
    ROOT reduce = s4[] reduce(x, zero), dimensions={0,1,2}, to_apply=add_computation
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, dynamic_slice_fusion) {
  const std::string hlo_text = R"(
  ENTRY main {
    input = s4[101,9] parameter(0)
    index = s32[] parameter(1)
    input_doubled = add(input, input)
    ROOT dus = s4[51,6] dynamic-slice(input_doubled, index, index), dynamic_slice_sizes={51,6}
  })";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

XLA_TEST_F(HloTestBase, dynamic_update_slice_fusion) {
  const std::string hlo_text = R"(
  ENTRY main {
    input = s4[101,9] parameter(0)
    slice = s4[101,6] parameter(1)
    index = s32[] parameter(2)
    input_doubled = add(input, input)
    ROOT dus = s4[101,9] dynamic-update-slice(input_doubled, slice, index, index)
  })";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

class HloTestBaseWithAlgsimpDisabled : public HloTestBase {
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = HloTestBase::GetDebugOptionsForTest();
    options.add_xla_disable_hlo_passes("algsimp");
    return options;
  }
};

XLA_TEST_F(HloTestBaseWithAlgsimpDisabled, TwoDots) {
  // This tests a regression that occured when a non-parameter non-ROOT
  // instruction was s4 as the input or output of a fusion. Fusion passes tend
  // to make any int4 instructions only internal to a fusion, but this HLO, at
  // the time it is written, has an int4 tensor existing between fusions when
  // algebraic simplifier is disabled.
  const std::string hlo_text = R"(
  HloModule TwoDots

  ENTRY main {
    x = s8[25,20,10,5] parameter(0)
    y = s8[25,20,10,5] parameter(1)
    z = s8[5,20] parameter(2)
    dot0 = s8[25,20,10,5] dot(x, y), lhs_batch_dims={0,1,2,3}, lhs_contracting_dims={}, rhs_batch_dims={0,1,2,3}, rhs_contracting_dims={}
    dot0_4 = s4[25,20,10,5] convert(dot0)
    dot0_8 = s8[25,20,10,5] convert(dot0_4)
    dot1 = s8[5,25,10] dot(z, dot0_8), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={3}, rhs_contracting_dims={1}
  }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

class ElementwiseTest : public HloTestBase,
                        public ::testing::WithParamInterface<
                            std::tuple<HloOpcode, PrimitiveType>> {
 public:
  static std::vector<HloOpcode> GetElementwiseOpcodesWithIntSupportWithArity(
      int arity) {
    std::vector<HloOpcode> opcodes;
    for (int i = 0; i < HloOpcodeCount(); ++i) {
      HloOpcode opcode = static_cast<HloOpcode>(i);
      auto opcode_arity = HloOpcodeArity(opcode);
      if (HloInstruction::IsOpElementwise(opcode) &&
          !IsFloatingPointOnly(opcode) && opcode_arity.has_value() &&
          *opcode_arity == arity) {
        opcodes.push_back(opcode);
      }
    }
    return opcodes;
  }

  static std::string FormatTestName(
      const ::testing::TestParamInfo<std::tuple<HloOpcode, PrimitiveType>>&
          info) {
    auto opcode_str = absl::StrReplaceAll(
        HloOpcodeString(std::get<0>(info.param)), {{"-", "_"}});
    auto type_str =
        primitive_util::LowercasePrimitiveTypeName(std::get<1>(info.param));
    return absl::StrCat(opcode_str, "_", type_str);
  }

 private:
  static bool IsFloatingPointOnly(HloOpcode opcode) {
    switch (opcode) {
      case HloOpcode::kAtan2:
      case HloOpcode::kCbrt:
      case HloOpcode::kCeil:
      case HloOpcode::kComplex:
      case HloOpcode::kCos:
      case HloOpcode::kSin:
      case HloOpcode::kErf:
      case HloOpcode::kExp:
      case HloOpcode::kExpm1:
      case HloOpcode::kFloor:
      case HloOpcode::kImag:
      case HloOpcode::kIsFinite:
      case HloOpcode::kLog:
      case HloOpcode::kLog1p:
      case HloOpcode::kLogistic:
      case HloOpcode::kPower:
      case HloOpcode::kReal:
      case HloOpcode::kReducePrecision:
      case HloOpcode::kRoundNearestAfz:
      case HloOpcode::kRoundNearestEven:
      case HloOpcode::kRsqrt:
      case HloOpcode::kSqrt:
      case HloOpcode::kStochasticConvert:
      case HloOpcode::kTan:
      case HloOpcode::kTanh:
        return true;
      default:
        return false;
    }
  }
};

using UnaryElementwiseTest = ElementwiseTest;
using BinaryElementwiseTest = ElementwiseTest;
using TernaryElementwiseTest = ElementwiseTest;

TEST_P(UnaryElementwiseTest, Unary) {
  auto [opcode, type] = GetParam();
  auto opcode_name = HloOpcodeString(opcode);

  if (primitive_util::IsSignedIntegralType(type) &&
      (opcode == HloOpcode::kClz || opcode == HloOpcode::kPopulationCount)) {
    GTEST_SKIP() << "Unsupported op on signed type: " << opcode_name;
  }
  if (primitive_util::IsUnsignedIntegralType(type) &&
      (opcode == HloOpcode::kAbs || opcode == HloOpcode::kSign)) {
    GTEST_SKIP() << "Unsupported op on unsigned type: " << opcode_name;
  }

  const std::string hlo_text = absl::Substitute(
      R"(
  HloModule unary

  ENTRY main {
    a = $0[100] parameter(0)
    ROOT r = $0[100] $1(a)
  }
)",
      primitive_util::LowercasePrimitiveTypeName(type),
      HloOpcodeString(opcode));
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

TEST_P(BinaryElementwiseTest, Binary) {
  auto [opcode, type] = GetParam();
  auto type_name = primitive_util::LowercasePrimitiveTypeName(type);

  const std::string hlo_text = absl::Substitute(
      R"(
  HloModule binary

  ENTRY main {
    a = $0[100] parameter(0)
    b = $0[100] parameter(1)
    ROOT r = $2[100] $1(a, b) $3
  }
)",
      type_name, HloOpcodeString(opcode),
      opcode == HloOpcode::kCompare ? "pred" : type_name,
      opcode == HloOpcode::kCompare ? ", direction=LT" : "");
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

TEST_P(TernaryElementwiseTest, Ternary) {
  auto [opcode, type] = GetParam();
  auto type_name = primitive_util::LowercasePrimitiveTypeName(type);

  const std::string hlo_text =
      absl::Substitute(R"(
  HloModule ternary

  ENTRY main {
    a = $2[100] parameter(0)
    b = $0[100] parameter(1)
    c = $0[100] parameter(2)
    ROOT r = $0[100] $1(a, b, c)
  }
)",
                       type_name, HloOpcodeString(opcode),
                       opcode == HloOpcode::kSelect ? "pred" : type_name);
  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

INSTANTIATE_TEST_SUITE_P(
    Elementwise, UnaryElementwiseTest,
    ::testing::Combine(
        ::testing::ValuesIn(
            ElementwiseTest::GetElementwiseOpcodesWithIntSupportWithArity(1)),
        ::testing::Values(U4, S4)),
    ElementwiseTest::FormatTestName);

INSTANTIATE_TEST_SUITE_P(
    Elementwise, BinaryElementwiseTest,
    ::testing::Combine(
        ::testing::ValuesIn(
            ElementwiseTest::GetElementwiseOpcodesWithIntSupportWithArity(2)),
        ::testing::Values(U4, S4)),
    ElementwiseTest::FormatTestName);

INSTANTIATE_TEST_SUITE_P(
    Elementwise, TernaryElementwiseTest,
    ::testing::Combine(
        ::testing::ValuesIn(
            ElementwiseTest::GetElementwiseOpcodesWithIntSupportWithArity(3)),
        ::testing::Values(U4, S4)),
    ElementwiseTest::FormatTestName);

}  // namespace
}  // namespace xla
