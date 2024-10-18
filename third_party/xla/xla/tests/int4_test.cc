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
  // Tests that the convert is not moved after the reshape. Currently reshape
  // and most other ops are unsupported in int4
  const std::string hlo_text = R"(
  HloModule Reshape
  ENTRY main {
    x = s4[2,3] parameter(0)
    y = s8[2,3] convert(x)
    ROOT reshape = s8[3,2] reshape(y)
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
  HloModule HorizontalLoopFusion

  ENTRY main {
    x = s4[5,5] parameter(0)
    x8 = s8[5,5] convert(x)
    y8 = add(x8, x8)
    ROOT y = s4[5,5] convert(y8)
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
    y = s8[2,2]{0,1} convert(x)
    ROOT transpose = s8[2,2]{0,1} transpose(y), dimensions={1,0}
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

class HloTestBaseWithAlgsimpDisabled : public HloTestBase {
  DebugOptions GetDebugOptionsForTest() override {
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
}  // namespace
}  // namespace xla
