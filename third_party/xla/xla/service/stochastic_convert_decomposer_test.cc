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

#include "xla/service/stochastic_convert_decomposer.h"

#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using StochasticConvertDecomposerTest = HloTestBase;
using ::testing::HasSubstr;

TEST_F(StochasticConvertDecomposerTest, DecomposeStochasticConvertF32ToS32) {
  const std::string module_str = R"(
HloModule module

ENTRY entry {
  %arg_param.1 = f32[65536]{0} parameter(0)
  %random_param.2 = u32[65536]{0} parameter(1)
  ROOT %stochastic-convert.3 = s32[65536]{0} stochastic-convert(f32[65536]{0} %arg_param.1, u32[65536]{0} %random_param.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  StochasticConvertDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Select(op::Compare(), op::Broadcast(),
                         op::Select(op::Compare(), op::Broadcast(),
                                    op::Select(op::Compare(), op::Negate(),
                                               op::Select()))));
}

TEST_F(StochasticConvertDecomposerTest, DecomposeStochasticConvertBF16ToS8) {
  const std::string module_str = R"(
HloModule module

ENTRY entry {
  %arg_param.1 = bf16[65536]{0} parameter(0)
  %random_param.2 = u16[65536]{0} parameter(1)
  ROOT %stochastic-convert.3 = s8[65536]{0} stochastic-convert(bf16[65536]{0} %arg_param.1, u16[65536]{0} %random_param.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  StochasticConvertDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Select(op::Compare(), op::Broadcast(),
                         op::Select(op::Compare(), op::Broadcast(),
                                    op::Select(op::Compare(), op::Negate(),
                                               op::Select()))));
}

TEST_F(StochasticConvertDecomposerTest, WrongRandomBitWidth) {
  const std::string module_str = R"(
HloModule module

ENTRY entry {
  %arg_param.1 = bf16[65536]{0} parameter(0)
  %random_param.2 = u32[65536]{0} parameter(1)
  ROOT %stochastic-convert.3 = s32[65536]{0} stochastic-convert(bf16[65536]{0} %arg_param.1, u32[65536]{0} %random_param.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  StochasticConvertDecomposer decomposer;

  auto result = decomposer.Run(module.get());
  EXPECT_NE(OkStatus(), result.status());
  EXPECT_THAT(result.status().message(), HasSubstr("have same bits"));
}

TEST_F(StochasticConvertDecomposerTest, WrongRandomType) {
  const std::string module_str = R"(
HloModule module

ENTRY entry {
  %arg_param.1 = f32[65536]{0} parameter(0)
  %random_param.2 = s32[65536]{0} parameter(1)
  ROOT %stochastic-convert.3 = s32[65536]{0} stochastic-convert(f32[65536]{0} %arg_param.1, s32[65536]{0} %random_param.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  StochasticConvertDecomposer decomposer;

  auto result = decomposer.Run(module.get());
  EXPECT_NE(OkStatus(), result.status());
  EXPECT_THAT(result.status().message(),
              HasSubstr("must be unsigned integers"));
}

}  // namespace
}  // namespace xla
