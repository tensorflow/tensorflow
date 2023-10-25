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
  // CPUs/GPUs, the slice is fused with the s4 array
  const std::string hlo_text = R"(
  HloModule Slice
  ENTRY main {
    x = s4[4,5] parameter(0)
    y = s8[4,5] convert(x)
    ROOT s = s8[3,2] slice(y), slice={[0:3],[2:4]}
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

}  // namespace
}  // namespace xla
