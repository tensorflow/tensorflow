/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_get_dimension_size_rewriter.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class HloGetDimensionSizeRewriterTest : public HloTestBase {
 protected:
  HloGetDimensionSizeRewriterTest() {}
};

TEST_F(HloGetDimensionSizeRewriterTest, Ok) {
  auto module = ParseHloString(R"(
HloModule _
ENTRY gds {
  p = s32[3,4] parameter(0)
  size0 = u32[] get-dimension-size(p), dimensions={0}
  size1 = u32[] get-dimension-size(p), dimensions={1}
  ROOT mul = u32[] multiply(size0, size1)
})")
                    .ValueOrDie();
  HloGetDimensionSizeRewriter pass;
  EXPECT_TRUE(pass.Run(module.get()).ValueOrDie());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Multiply(op::Constant(), op::Constant()));
}

TEST_F(HloGetDimensionSizeRewriterTest, IllegalType) {
  auto module = ParseHloString(R"(
HloModule _
ENTRY gds {
  p = s32[3]{0} parameter(0)
  ROOT gds = s64[] get-dimension-size(p), dimensions={0}
})")
                    .ValueOrDie();
  HloGetDimensionSizeRewriter pass;
  EXPECT_FALSE(pass.Run(module.get()).ok());
}

TEST_F(HloGetDimensionSizeRewriterTest, IllegalDimension) {
  auto module = ParseHloString(R"(
HloModule _
ENTRY gds {
  p = f32[2,5] parameter(0)
  ROOT gds = u32[] get-dimension-size(p), dimensions={2}
})")
                    .ValueOrDie();
  HloGetDimensionSizeRewriter pass;
  EXPECT_FALSE(pass.Run(module.get()).ok());
}

}  // namespace
}  // namespace xla
